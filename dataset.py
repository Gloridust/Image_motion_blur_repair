import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class DeblurDataset(Dataset):
    """数据集加载"""
    def __init__(self, split='train'):
        super(DeblurDataset, self).__init__()
        self.split = split
        if self.split == 'train':
            self.blur_dir = 'data/blur_train'
            self.sharp_dir = 'data/sharp_train'
        elif self.split == 'val':
            self.blur_dir = 'data/blur_val'
            self.sharp_dir = 'data/sharp_val'
        elif self.split == 'test':
            self.blur_dir = 'data/blur_test'
            self.sharp_dir = None  # 测试集无需清晰图像
        else:
            raise ValueError("split must be 'train', 'val', or 'test'.")

        # 只保留图像文件
        self.blur_images = sorted([f for f in os.listdir(self.blur_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        if self.sharp_dir:
            self.sharp_images = sorted([f for f in os.listdir(self.sharp_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        else:
            self.sharp_images = None

        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.blur_images)
    
    def __getitem__(self, idx):
        blur_img = Image.open(os.path.join(self.blur_dir, self.blur_images[idx])).convert('RGB')
        blur_img = self.transform(blur_img)
        if self.sharp_images:
            sharp_img = Image.open(os.path.join(self.sharp_dir, self.sharp_images[idx])).convert('RGB')
            sharp_img = self.transform(sharp_img)
        else:
            sharp_img = None
        return blur_img, sharp_img