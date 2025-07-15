import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.optim as optim
from metrics import get_eval_metrics  # 导入metrics.py中的函数
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import logging
import shutil
import os
# Set up logging
log_file = 'training_validation_metrics-new-new.log'
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# 数据预处理
w = 384
h = 384
train_transforms = transforms.Compose([
        transforms.Resize((w, h)), 
        # transforms.RandomHorizontalFlip(),# 随机旋转
        transforms.RandomVerticalFlip(),# 随机翻转
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
val_transforms = transforms.Compose([
        transforms.Resize((w, h)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

DATA_DIR = "F:\COD_dataset\TrainDataset\Imgs"
# 加载数据集
train_dir = os.path.join(DATA_DIR, "")
# test_dir = os.path.join(DATA_DIR, "test")
test_dir = "F:\RefCOD\img/test"

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#num_workers=8
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# 打印数据信息
print("len of train dataset:" + str(len(train_dataset)))
print("len of val dataset:" + str(len(test_dataset)))
print(train_dataset.classes)
print(train_dataset.class_to_idx)

# 加载ResNet50模型
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("F:\RefCOD\checkpoint-new-384.pt")
model.load_state_dict(checkpoint)
model = model.to(device)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=0.00001)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint-new-384-new.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
def train_and_validate(model, train_loader, test_loader, criterion, optimizer,scheduler, num_epochs=150, patience=15):
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        all_preds_train = []
        all_targets_train = []
        correct_train = 0
        total_train = 0
        
        logging.info(f'Epoch {epoch+1}/{num_epochs}')
        train_bar = tqdm(train_loader, desc='Training', leave=False)
        
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds_train.extend(predicted.cpu().numpy())
            all_targets_train.extend(labels.cpu().numpy())
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            train_bar.set_postfix(loss=running_loss/len(train_loader), acc=100 * correct_train / total_train)
        
        train_accuracy = 100 * correct_train / total_train
        train_loss = running_loss / len(train_loader)
        if epoch % 5 == 0:
            # 验证阶段
            model.eval()
            running_loss_val = 0.0
            all_preds_val = []
            all_targets_val = []
            all_probs_val = []
            correct_val = 0
            total_val = 0

            val_bar = tqdm(test_loader, desc='Validating', leave=False)
            
            with torch.no_grad():
                for inputs, labels in val_bar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    running_loss_val += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    all_probs_val.extend(probs.cpu().numpy())
                    all_preds_val.extend(predicted.cpu().numpy())
                    all_targets_val.extend(labels.cpu().numpy())
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
                    
                    val_bar.set_postfix(loss=running_loss_val/len(test_loader), acc=100 * correct_val / total_val)
            
            val_accuracy = 100 * correct_val / total_val
            val_loss = running_loss_val / len(test_loader)
            
            # 计算并输出指标
            train_metrics = get_eval_metrics(all_targets_train, all_preds_train)
            val_metrics = get_eval_metrics(all_targets_val, all_preds_val, probs_all=all_probs_val)
            cm = confusion_matrix(all_targets_val, all_preds_val)
            
            # Log metrics
            logging.info(f'Epoch [{epoch+1}/{num_epochs}] - '
                        f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%\n'
                        f'Validation Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            
            # Log training metrics per class
            logging.info("Training Metrics (Overall):")
            for metric, value in train_metrics.items():
                if "report" not in metric:
                    logging.info(f"{metric}: {value}")
            
            logging.info("Training Metrics (Per Class):")
            if "report" in train_metrics:
                class_report = train_metrics["report"]
                for class_label, metrics in class_report.items():
                    if class_label.isdigit():  # 确保只处理具体类别的指标
                        logging.info(f"Class {class_label}: "
                                    f"Precision: {metrics['precision']:.4f}, "
                                    f"Recall: {metrics['recall']:.4f}, "
                                    f"F1-score: {metrics['f1-score']:.4f}")
            
            # Log validation metrics per class
            logging.info("\nValidation Metrics (Overall):")
            for metric, value in val_metrics.items():
                if "report" not in metric:
                    logging.info(f"{metric}: {value}")
            
            logging.info("\nValidation Metrics (Per Class):")
            if "report" in val_metrics:
                class_report = val_metrics["report"]
                for class_label, metrics in class_report.items():
                    if class_label.isdigit():
                        logging.info(f"Class {class_label}: "
                                    f"Precision: {metrics['precision']:.4f}, "
                                    f"Recall: {metrics['recall']:.4f}, "
                                    f"F1-score: {metrics['f1-score']:.4f}")
            logging.info("\nconfusion_matrix")
            print(cm)
            for  i in range(len(cm)):
                logging.info(cm[i])
            draw_ROC(epoch,all_probs_val,all_targets_val)
            
            # 早停检查
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break



def draw_ROC(epoch,all_probs_val,all_targets_val):
        # 绘制ROC曲线
    n_classes = len(np.unique(all_targets_val))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(all_targets_val) == i, np.array(all_probs_val)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制ROC曲线图
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.5f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    # 保存ROC曲线图
    save_fold = os.path.join('./', 'roc_curves')
    if not os.path.exists(save_fold):
        os.makedirs(save_fold)

    save_path = os.path.join(save_fold, f'roc_curve_epoch_{epoch}.png')
    plt.savefig(save_path,dpi=300)
    plt.close()

#推理，移动文件夹中的所有图片至同一个文件夹后，分类移动至对应文件夹
def inference(model, data_loader, device):
    model.eval()
    #列表
    class_list = ['墙房间-indoor', '树干-trunk', '水下-underwater', '池塘-pond', '花叶枝-Flower leaf branch', '草地砂石-gravel and grass', '雪地-snowfield']
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            for i, pred in enumerate(predicted):
                # 获取当前图片的文件名
                filename = data_loader.dataset.samples[i][0].split('/')[-1]
                # 移动图片到对应文件夹
                shutil.move(data_loader.dataset.samples[i][0], os.path.join('./', class_list[int(pred.item())], filename))
if  __name__ == '__main__':
    # 调用训练和验证过程
    train_and_validate(model, train_loader, test_loader, criterion, optimizer,scheduler)
