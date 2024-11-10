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
        cnn_feat = self.cnn(x)  # [batch_size, 512]
        # ViT 特征提取，需要调整输入尺寸到 224
        vit_input = nn.functional.interpolate(x, size=224, mode='bilinear', align_corners=False)
        vit_feat = self.vit(vit_input).last_hidden_state  # [batch_size, seq_len, 768]
        vit_feat = vit_feat[:, 0, :]  # 取 [CLS] token 的特征 [batch_size, 768]
        # 特征融合
        fused_feat = torch.cat([cnn_feat, vit_feat], dim=1)  # [batch_size, 512+768]
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
        
        # 初始输入处理
        self.init_conv = nn.Conv2d(img_channels, base_channels, kernel_size=3, padding=1)
        
        # 编码器部分
        self.enc1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
        )
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
        )
        
        # 解码器部分
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),  # 注意：输入通道数是base_channels*2，因为有skip connection
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
        )
        
        # 条件嵌入投影
        self.context_proj1 = nn.Sequential(
            nn.Linear(time_embed_dim, base_channels),
            nn.ReLU(),
        )
        self.context_proj2 = nn.Sequential(
            nn.Linear(time_embed_dim, base_channels * 2),
            nn.ReLU(),
        )
        
        # 输出层
        self.conv_out = nn.Conv2d(base_channels, img_channels, kernel_size=3, padding=1)

    def forward(self, x, t, cond):
        # 时间和条件嵌入
        t_embed = self.time_embed(t.unsqueeze(-1))  # [batch_size, time_embed_dim]
        cond_embed = self.cond_embed(cond)  # [batch_size, time_embed_dim]
        context = t_embed + cond_embed  # [batch_size, time_embed_dim]
        
        # 投影条件嵌入
        context1 = self.context_proj1(context)  # [batch_size, base_channels]
        context2 = self.context_proj2(context)  # [batch_size, base_channels*2]
        
        # 初始特征提取
        x = self.init_conv(x)  # [batch_size, base_channels, H, W]
        
        # 编码器路径
        x1 = self.enc1(x + context1.view(context1.shape[0], -1, 1, 1))
        x2 = self.down1(x1)
        x3 = self.enc2(x2 + context2.view(context2.shape[0], -1, 1, 1))
        
        # 解码器路径
        x4 = self.up1(x3)
        x5 = torch.cat([x4, x1], dim=1)  # Skip connection
        x6 = self.dec1(x5)
        
        # 输出
        out = self.conv_out(x6)
        return out