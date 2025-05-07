import os
import numpy as np
import random
import json

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class SceneDataset(Dataset):

    def __init__(self, data_root, mode='train', shot=5, image_size=352):
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.data_root = data_root
        self.shot = shot
        self.image_size = image_size

        # 收集数据列表
        if mode == 'train':
            self.img_dir = os.path.join(data_root, 'imgs')
            self.gt_dir = os.path.join(data_root, 'GT')
        else:
            # 验证集和测试集的结构可能不同，根据实际情况修改
            self.img_dir = os.path.join(data_root,  'imgs')
            self.gt_dir = os.path.join(data_root,'GT')

        self.data_list = []
        self.scene_labels = []  # 用于存储场景标签

        # 根据数据集结构加载图片和标签
        if mode == 'train':
            # 训练集：img目录下有场景子文件夹
            scene_dirs = os.listdir(self.img_dir)
            # 对场景目录进行排序
            sorted_scene_dirs = sorted(scene_dirs, key=lambda x: x.lower())
            # 创建场景名称到标签的映射
            scene_to_label = {scene: idx for idx, scene in enumerate(sorted_scene_dirs)}

            for scene_dir in scene_dirs:
                scene_path = os.path.join(self.img_dir, scene_dir)
                if not os.path.isdir(scene_path):
                    continue  # 跳过非目录项

                img_names = os.listdir(scene_path)
                for img_name in img_names:
                    if img_name.lower().endswith(('.jpg', '.jpeg')):
                        img_path = os.path.join(scene_path, img_name)
                        gt_name = img_name.split('.')[0] + '.png'
                        gt_path = os.path.join(self.gt_dir, gt_name)

                        if os.path.exists(gt_path):
                            self.data_list.append((img_path, gt_path))
                            # 获取场景标签
                            self.scene_labels.append(scene_to_label.get(scene_dir, len(sorted_scene_dirs)))

        else:
            # 测试集和验证集：img目录下直接是图片文件
            img_names = os.listdir(self.img_dir)
            for img_name in img_names:
                if img_name.lower().endswith(('.jpg', '.jpeg')):
                    img_path = os.path.join(self.img_dir, img_name)
                    gt_name = img_name.split('.')[0] + '.png'
                    gt_path = os.path.join(self.gt_dir, gt_name)

                    if os.path.exists(gt_path):
                        self.data_list.append((img_path, gt_path))
                        self.scene_labels.append(0)  # 测试集场景标签默认为0

        print(f"{mode} dataset size: {len(self.data_list)}")
        assert len(self.data_list) == len(self.scene_labels)

        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        scene_label = self.scene_labels[index]

        name = os.path.basename(image_path).split('.')[0]

        # 读取图片和标签
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        # 数据增强
        # if self.mode == 'train':
        #     image, label = self.aug_data(image, label)

        # 转换
        image = self.img_transform(image)
        if self.mode == 'train':
            label = self.gt_transform(label)
        else:
            label = np.array(label, dtype=np.float32) / 255.0  # 转换为0-1范围

        # 这里可以添加参考特征的逻辑，根据实际需求实现
        # sal_f = torch.zeros((3, self.image_size, self.image_size))  # 示例默认值

        return image, label, scene_label, name

    def aug_data(self, image, label):
        # 实现数据增强逻辑，如随机翻转、旋转等
        if random.random() > 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            label = label.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return image, label