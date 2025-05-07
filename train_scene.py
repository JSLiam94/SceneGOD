'''
the codes for training the model.
created by Xuying Zhang (zhangxuying1004@gmail.com) on 2023-06-23
'''

import os
import numpy as np
from time import time
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
# from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn

import utils.metrics as Measure
from utils.utils import set_gpu, structure_loss, clip_gradient

from models.scene import Network
from data import get_dataloader


def train(train_loader, model, optimizer, epoch, save_path,log_file):
    """
    train function
    """
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    train_bar = tqdm(train_loader)
    count = 0
    # class_loss_all = 0
    total_train = 0
    correct_train = 0
    try:
        for i, (images, gts, scene_label, _) in enumerate(train_bar):
            optimizer.zero_grad()

            images = images.cuda()
            scene_label = scene_label.cuda()
            gts = gts.cuda()
            # supp_feats = supp_feats.cuda()
            # print("label",scene_label)
            
            preds,  scene_cls = model(images)
            _, predicted = torch.max(scene_cls, 1)
            correct_train += (predicted == scene_label).sum().item()
            # print("pre",predicted)
            total_train +=  scene_label.size(0)
            class_loss = class_criterion(scene_cls, scene_label)
            acc=100 * correct_train / total_train
            main_loss = structure_loss(preds, gts)
            # aux_loss = structure_loss(inner_preds[0], gts)
            # inner_num = len(inner_preds)
            # for inner_idx in range(1, inner_num):
            #     aux_loss = aux_loss + structure_loss(inner_preds[inner_idx], gts)
            # aux_loss /= inner_num
            loss = main_loss + class_loss
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            
            optimizer.step()
            cosine_schedule.step()

            step += 1
            epoch_step += 1
            # class_loss_all  += class_loss.item()
            loss_all += loss.data
            count += 1

            # if i % 20 == 0 or i == total_step or i == 1:
            #     print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
            #           format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))
            train_bar.set_description('Epoch [{:d}/{:d}] seg loss: {:.4f} acc:{:.2f}'.format(epoch, opt.epoch,loss_all/count,acc))

        loss_all /= epoch_step
        # writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch % 10 == 0:
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch
            }, save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save({
            'state_dict': model.state_dict(),
            'epoch': epoch
        }, save_path + 'Net_Interrupt_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path,log_file):
    """
    validation function
    """
    global best_mae, best_epoch, best_score, best_other_epoch
    WFM = Measure.WeightedFmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    MAE = Measure.MAE()

    model.eval()
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for i,(image, gt, sf, _) in enumerate(test_bar):
                gt = gt.numpy().astype(np.float32).squeeze()
                gt /= (gt.max() + 1e-8)  # 标准化处理, 把数值范围控制到(0,1)
                image = image.cuda()
                # sf = sf.cuda()
                res, _ = model(image)

                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)  # 标准化处理, 把数值范围控制到(0,1)

                WFM.step(pred=res*255, gt=gt*255)
                SM.step(pred=res*255, gt=gt*255)
                EM.step(pred=res*255, gt=gt*255)
                MAE.step(pred=res*255, gt=gt*255)
                
              
                # pbar.update()

        sm1 = SM.get_results()['sm'].round(4)
        adpem1 = EM.get_results()['em']['adp'].round(4)
        wfm1 = WFM.get_results()['wfm'].round(4)
        mae1 = MAE.get_results()['mae'].round(4)


        if epoch == 1:
            best_mae = mae1
            best_score = sm1 + adpem1 + wfm1
        else:
            if mae1 < best_mae:
                best_mae = mae1
                best_epoch = epoch
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch
                }, save_path + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
            
            score = sm1 + adpem1 + wfm1
            if score > best_score:
                best_score = score
                best_other_epoch = epoch
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch
                }, save_path + 'Net_epoch_other_best.pth')
                print('Save state_dict successfully! Best other epoch:{}.'.format(epoch))
        with open(log_file, 'a') as f:
            f.write('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.\n'.format(epoch, mae1, best_mae, best_epoch))
            f.write('Sm:{:.4f}\n'.format(torch.tensor(sm1)))
            f.write('AdpEm:{:.4f}\n'.format(torch.tensor(adpem1)))
            f.write('Wfm:{:.4f}\n'.format(torch.tensor(wfm1)))
            f.write('MAE:{:.4f}\n'.format(torch.tensor(mae1)))
        # writer.add_scalar('Sm', torch.tensor(sm1), global_step=epoch)
        # writer.add_scalar('adaEm', torch.tensor(adpem1), global_step=epoch)
        # writer.add_scalar('wF', torch.tensor(wfm1), global_step=epoch)
        # writer.add_scalar('MAE', torch.tensor(mae1), global_step=epoch)
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae1, best_mae, best_epoch))  # 修改为 mae1

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='r2cnet_scene')
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--dim', type=int, default=64, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of workers in dataloader')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')

    parser.add_argument('--train_data_root', type=str, default='F:\COD_dataset\TrainDataset', help='the path to put dataset')
    parser.add_argument('--test_data_root', type=str, default='F:\COD_dataset\TestDataset\COD10K', help='the path to put dataset')   
    parser.add_argument('--save_root', type=str, default='./snapshot/', help='the path to save model params and log')


    opt = parser.parse_args()
    print(opt)

    # set the device for training
    set_gpu(opt.gpu_id)
    cudnn.benchmark = True

    start_time = time()
    class_num = 7
    model = Network(channel=opt.dim,class_num=class_num)
    load_classifier = False
    resume = r"F:/RefCOD\snapshot\saved_models/r2cnet_scene/Net_epoch_100.pth"
    start_epoch = 1

    if load_classifier:
        classifier_checkpoint_path = "F:\RefCOD\checkpoint-new.pt"
        classifier_checkpoint = torch.load(classifier_checkpoint_path)
        model.resnet_classifier.load_state_dict(classifier_checkpoint)
    elif resume is not None:
        check_point = torch.load(resume)
        model.load_state_dict(check_point["state_dict"])
        start_epoch = check_point["epoch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    base,classer,body = [], [],[]
    for name, param in model.named_parameters():
        if 'resnet' in name and "class" not in name:
            base.append(param)   
        elif 'resnet' in name and "class" in name:
            classer.append(param)
        else:
            body.append(param)

    params_dict = [{'params': base, 'lr': opt.lr * 0.1},    
                   {'params': classer, 'lr': opt.lr*0.01},       
                {'params': body, 'lr': opt.lr}]
    optimizer = torch.optim.SGD(params_dict)
    cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100)

    print('load data...')
    train_loader = get_dataloader(opt.train_data_root, opt.shot, opt.trainsize, opt.batchsize, opt.num_workers, mode='train')
    val_loader = get_dataloader(opt.test_data_root, opt.shot, opt.trainsize, opt.num_workers, mode='val')
    total_step = len(train_loader)

    save_path = opt.save_root + 'saved_models/' + opt.model_name + '/'
    save_logs_path = opt.save_root + 'logs/'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_logs_path, exist_ok=True)
    class_criterion = torch.nn.CrossEntropyLoss()
    
    # writer = SummaryWriter(save_logs_path + opt.model_name)
    
    step = 0

    best_mae = 1
    best_epoch = 0

    best_score = 0.
    best_other_epoch = 0

    log_file = os.path.join(save_logs_path, opt.model_name + '.txt')

    print("Start train...")
    for epoch in range(start_epoch+1, start_epoch+opt.epoch + 1):  # 修改为 opt.epoch + 1 以包含最后一个 epoch
        
        
        # writer.add_scalar('lr_base', cosine_schedule.get_lr()[0], global_step=epoch)
        # writer.add_scalar('lr_body', cosine_schedule.get_lr()[1], global_step=epoch)

        # train
        train(train_loader, model, optimizer, epoch, save_path,log_file)
        # schedule
        if epoch % 5 == 0:
            val(val_loader, model, epoch, save_path,log_file)


    end_time = time()

    print('it costs {} h to train'.format((end_time - start_time)/3600))