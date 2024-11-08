import torch
import torchvision.utils as vutils

def save_image_tensor(input_tensor: torch.Tensor, filename):
    """保存Tensor格式的图像到文件"""
    vutils.save_image(input_tensor, filename)

def noise_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """生成噪声调度表"""
    return torch.linspace(beta_start, beta_end, timesteps)

def extract(a, t, x_shape):
    """从数组a中提取第t项，并重塑为x_shape的形状"""
    batch_size = t.size(0)
    out = a.gather(-1, t).reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    return out 