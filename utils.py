import torchvision.utils as vutils

def save_image_tensor(input_tensor: torch.Tensor, filename):
    """保存Tensor格式的图像到文件"""
    vutils.save_image(input_tensor, filename) 