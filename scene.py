import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.utils import BasicConv2d
from models.modules import FeatFusion, RFE
from torchvision import  models as resnet_model

class Network(nn.Module):
    def __init__(self, channel=64, imagenet_pretrained=True, class_num=7):
        super(Network, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = torchvision.models.resnet50(pretrained=imagenet_pretrained)
        
        self.x2_down_channel = BasicConv2d(512, channel, 1)
        self.x3_down_channel = BasicConv2d(1024, channel, 1)
        self.x4_down_channel = BasicConv2d(2048, channel, 1)

        self.ref_proj = BasicConv2d(2048, channel, 1)

        # dsf + msf
        self.feat_fusion = FeatFusion(channel=channel)

        # target matching
        self.relevance_norm = nn.BatchNorm2d(1)
        self.relevance_acti = nn.LeakyReLU(0.1, inplace=True)

        # rfe
        self.rfe = RFE(d_model=channel)

        self.cls = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.1), 
            nn.Conv2d(channel, 1, 1)
        )
        self.resnet_classifier = resnet_model.resnet50(pretrained=True)
        num_ftrs = self.resnet_classifier.fc.in_features
        self.resnet_classifier.fc = torch.nn.Linear(num_ftrs, class_num)
        
        # 新增部分：场景分类特征融合模块
        self.scene_embedding = nn.Embedding(class_num, channel)  # 场景类别嵌入
        self.scene_attn = nn.MultiheadAttention(embed_dim=channel, num_heads=8)
        self.scene_norm = nn.LayerNorm(channel)
        self.scene_mlp = nn.Sequential(
            nn.Linear(channel, 2*channel),
            nn.GELU(),
            nn.Linear(2*channel, channel),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        bs, _, H, W = x.shape

        # Feature Extraction
        old_x = x
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88

        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11

        # Down Channel
        x2 = self.x2_down_channel(x2)    # bs, 64, 44, 44
        x3 = self.x3_down_channel(x3)    # bs, 64, 22, 22   
        x4 = self.x4_down_channel(x4)    # bs, 64, 11, 11

        # 新增部分：场景分类
        scene_logits = self.resnet_classifier(old_x)  # 获取场景分类结果
        scene_labels = torch.argmax(scene_logits, dim=1)  # 获取预测的场景类别
        
        # 嵌入场景特征
        scene_embed = self.scene_embedding(scene_labels).unsqueeze(1)  # shape: (bs, 1, channel)
        
        # 特征图展平
        x4_flat = x4.permute(0, 2, 3, 1).flatten(1, 2)  # shape: (bs, 121, channel)
        
        # 注意力融合场景特征
        attn_output, _ = self.scene_attn(
            query=scene_embed.expand(-1, x4_flat.shape[1], -1),
            key=x4_flat,
            value=x4_flat
        )
        
        # 恢复特征图形状
        attn_output = attn_output.permute(0, 2, 1).view(bs, -1, 11, 11)  # 恢复到 (bs, channel, 11, 11)
        
        # 特征融合
        x2_h = attn_output

        # Conv Head
        S_g = self.cls(x2_h)
        S_g_pred = F.interpolate(S_g, size=(H, W), mode='bilinear', align_corners=True)      # (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        return S_g_pred, scene_logits