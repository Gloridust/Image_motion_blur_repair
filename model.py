import torch
import torch.nn as nn
from torchvision import models
from transformers import ViTModel

class FeatureExtractor(nn.Module):
    """特征提取模块，融合 CNN 和 ViT"""
    def __init__(self, img_size=256):
        super(FeatureExtractor, self).__init__()
        # CNN 部分，使用预训练的 ResNet18
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # 去掉全连接层
        # ViT 部分，使用预训练的 ViT
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        # 调整输入尺寸
        self.patch_size = 16
        self.img_size = img_size

    def forward(self, x):
        # CNN 特征提取
        cnn_feat = self.cnn(x)
        # ViT 特征提取，需要调整输入尺寸到 224
        vit_input = nn.functional.interpolate(x, size=224, mode='bilinear', align_corners=False)
        vit_feat = self.vit(vit_input).last_hidden_state  # (batch_size, seq_len, hidden_size)
        vit_feat = vit_feat[:, 0, :]  # 取 [CLS] token 的特征
        # 特征融合
        fused_feat = torch.cat([cnn_feat, vit_feat], dim=1)
        return fused_feat

class ConditionalUNet(nn.Module):
    """条件扩散模型中的UNet结构，接受融合特征作为条件"""
    def __init__(self, img_channels=3, base_channels=64, time_embed_dim=256, cond_dim=768+512):
        super(ConditionalUNet, self).__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.context_conv = nn.Conv2d(time_embed_dim, img_channels, kernel_size=1)  # 新增卷积层
        self.enc1 = nn.Sequential(
            nn.Conv2d(img_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
        )
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
        )
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
        )
        self.conv_out = nn.Conv2d(base_channels, img_channels, kernel_size=3, padding=1)

    def forward(self, x, t, cond):
        # x: 输入的噪声图像
        # t: 时间步（噪声级别）
        # cond: 条件特征（融合后的特征）
        t_embed = self.time_embed(t.unsqueeze(-1))
        cond_embed = self.cond_embed(cond)
        context = t_embed + cond_embed  # 融合时间和条件信息
        context = self.context_conv(context[:, :, None, None])  # 调整通道数

        x1 = self.enc1(x + context)
        x2 = self.down1(x1)
        x3 = self.enc2(x2 + context)
        x4 = self.up1(x3)
        x5 = self.dec1(x4 + x1 + context)
        out = self.conv_out(x5)
        return out