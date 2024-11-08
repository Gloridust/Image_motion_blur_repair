import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """残差块定义"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out

class SelfAttention(nn.Module):
    """自注意力机制"""
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=channels, out_channels=channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=channels, out_channels=channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch_size, C, width, height = x.size()
        # 生成查询、键、值矩阵
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0,2,1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

class DeblurTransformer(nn.Module):
    """去模糊模型主体"""
    def __init__(self):
        super(DeblurTransformer, self).__init__()
        self.conv_input = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu = nn.PReLU()
        # 添加多个残差块
        self.res_blocks = self._make_layer(ResidualBlock, 5, channels=64)
        # 自注意力模块
        self.attention = SelfAttention(channels=64)
        # 重建层
        self.conv_output = nn.Conv2d(64, 3, kernel_size=9, padding=4)
    
    def _make_layer(self, block, num_blocks, channels):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.relu(self.conv_input(x))
        out = self.res_blocks(out)
        out = self.attention(out)
        out = self.conv_output(out)
        out = torch.tanh(out)
        return out 