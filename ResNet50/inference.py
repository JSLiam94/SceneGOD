import os
import shutil
import torch
from torchvision import datasets
from torchvision import datasets, transforms, models
from tqdm import tqdm

# 定义类名和对应的文件夹名
class_list = ['墙房间-indoor', '树干-trunk', '水下-underwater', '池塘-pond', 
              '花叶枝-Flower leaf branch', '草地砂石-gravel and grass', '雪地-snowfield']

# 定义图片预处理
transform = transforms.Compose([
    transforms.Resize((352, 352)),  # 调整图片大小
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 使用ImageNet均值和标准差
])

# 加载图片数据，不打乱顺序
def load_images_from_folder(folder):
    # 创建一个自定义的Dataset
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, folder, transform=None):
            self.folder = folder
            self.transform = transform
            self.image_paths = []
            # 遍历所有子文件夹和图片
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        self.image_paths.append(os.path.join(root, file))
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = datasets.folder.default_loader(image_path)
            if self.transform:
                image = self.transform(image)
            return image, image_path  # 返回图片和路径
    
    dataset = CustomDataset(folder, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

# 推理并移动图片
def inference_and_move(model, data_loader, device):
    model.eval()
    infer_bar = tqdm(data_loader, desc="Inference")
    with torch.no_grad():
        for inputs, image_paths in infer_bar:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            for i, pred in enumerate(predicted):
                # 获取当前图片的文件名
                image_path = image_paths[i]
                filename = os.path.basename(image_path)
                
                # 获取目标文件夹
                target_dir = os.path.join('F:\COD_dataset\SOD\ECSSD/images-sc', class_list[pred.item()])

                os.makedirs(target_dir, exist_ok=True)  # 创建目标文件夹
                
                # 移动图片
                try:
                    shutil.move(image_path, os.path.join(target_dir, filename))
                    # print(f"Moved {filename} to {class_list[pred.item()]}")
                except Exception as e:
                    print(f"Error moving {filename}: {e}")

if __name__ == '__main__':
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 7)
    model.load_state_dict(torch.load("F:\RefCOD\ResNet50\checkpoint.pt"))  # 加载你的模型权重
    model = model.to(device)
    
    # 加载图片
    image_folder = "F:\COD_dataset\SOD\ECSSD/images"  # 替换为你的图片文件夹路径
    data_loader = load_images_from_folder(image_folder)
    
    # 推理并移动
    inference_and_move(model, data_loader, device)