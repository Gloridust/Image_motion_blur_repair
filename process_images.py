import os
import torch
from torchvision import transforms
from PIL import Image
from model import FeatureExtractor, ConditionalUNet
from utils import save_image_tensor, noise_schedule, extract

def load_model(device):
    # 加载预训练的模型
    feature_extractor = FeatureExtractor().to(device)
    unet = ConditionalUNet().to(device)
    checkpoint = torch.load('best_model.pth', map_location=device)
    feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
    unet.load_state_dict(checkpoint['unet_state_dict'])
    feature_extractor.eval()
    unet.eval()
    return feature_extractor, unet

def process_image(image_path, feature_extractor, unet, device, timesteps=1000):
    # 加载和预处理图像
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # 提取条件特征
    cond_feat = feature_extractor(image)
    x = torch.randn_like(image)
    betas = noise_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # 逐步去噪
    with torch.no_grad():
        for t in reversed(range(timesteps)):
            t_tensor = torch.tensor([t], device=device, dtype=torch.float32) / timesteps
            a_t = alphas_cumprod[t]
            a_prev = alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=device)
            beta_t = betas[t]
            x = (1 / torch.sqrt(alphas[t])) * (x - (beta_t / torch.sqrt(1 - a_t)) * unet(x, t_tensor, cond_feat))
            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt((1 - a_prev) / (1 - a_t) * beta_t)
                x += sigma * noise
    return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor, unet = load_model(device)

    blur_dir = 'blur_use'
    sharp_dir = 'sharp_use'
    if not os.path.exists(sharp_dir):
        os.makedirs(sharp_dir)

    for filename in os.listdir(blur_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(blur_dir, filename)
            output_image = process_image(image_path, feature_extractor, unet, device)
            save_image_tensor(output_image.cpu(), os.path.join(sharp_dir, filename))
            print(f"Processed and saved: {filename}")

if __name__ == '__main__':
    main() 