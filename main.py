import torch
from torch.utils.data import DataLoader
from dataset import DeblurDataset
from model import FeatureExtractor, ConditionalUNet
import torch.optim as optim
import torch.nn as nn
from utils import save_image_tensor, noise_schedule, extract
import os
from tqdm import tqdm

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 模型和优化器
    feature_extractor = FeatureExtractor().to(device)
    unet = ConditionalUNet().to(device)
    optimizer = optim.Adam(list(feature_extractor.parameters()) + list(unet.parameters()), lr=1e-4)
    mse_loss = nn.MSELoss()
    num_epochs = 50
    batch_size = 4
    timesteps = 1000  # 总的时间步数

    # 噪声调度
    betas = noise_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # 数据集加载
    train_dataset = DeblurDataset(split='train')
    val_dataset = DeblurDataset(split='val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        feature_extractor.train()
        unet.train()
        epoch_loss = 0.0
        for blur_img, sharp_img in tqdm(train_loader):
            blur_img = blur_img.to(device)
            sharp_img = sharp_img.to(device)

            # 提取条件特征
            cond_feat = feature_extractor(blur_img)

            # 随机选择时间步
            batch_size = blur_img.size(0)
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            # 计算对应的 α_t
            a_t = extract(alphas_cumprod, t, sharp_img.shape)

            # 添加噪声
            noise = torch.randn_like(sharp_img)
            noisy_img = torch.sqrt(a_t) * sharp_img + torch.sqrt(1 - a_t) * noise

            # 前向传播
            output = unet(noisy_img, t.float() / timesteps, cond_feat)
            loss = mse_loss(output, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # 验证模型
        val_loss = validate(feature_extractor, unet, val_loader, device, mse_loss, timesteps)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'feature_extractor_state_dict': feature_extractor.state_dict(),
                'unet_state_dict': unet.state_dict(),
            }, "best_model.pth")
            print("Best model saved.")

def validate(feature_extractor, unet, val_loader, device, criterion, timesteps):
    feature_extractor.eval()
    unet.eval()
    val_loss = 0.0
    with torch.no_grad():
        for blur_img, sharp_img in val_loader:
            blur_img = blur_img.to(device)
            sharp_img = sharp_img.to(device)

            # 提取条件特征
            cond_feat = feature_extractor(blur_img)

            # 随机选择时间步
            batch_size = blur_img.size(0)
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            # 计算对应的 α_t
            a_t = extract(alphas_cumprod, t, sharp_img.shape)

            # 添加噪声
            noise = torch.randn_like(sharp_img)
            noisy_img = torch.sqrt(a_t) * sharp_img + torch.sqrt(1 - a_t) * noise

            # 前向传播
            output = unet(noisy_img, t.float() / timesteps, cond_feat)
            loss = criterion(output, noise)

            val_loss += loss.item()

    return val_loss / len(val_loader)

def sample(feature_extractor, unet, blur_img, device, timesteps=1000):
    feature_extractor.eval()
    unet.eval()
    with torch.no_grad():
        # 提取条件特征
        cond_feat = feature_extractor(blur_img)
        x = torch.randn_like(blur_img)
        betas = noise_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
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

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = FeatureExtractor().to(device)
    unet = ConditionalUNet().to(device)
    checkpoint = torch.load('best_model.pth')
    feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
    unet.load_state_dict(checkpoint['unet_state_dict'])

    test_dataset = DeblurDataset(split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if not os.path.exists('results'):
        os.makedirs('results')

    for idx, (blur_img, _) in enumerate(test_loader):
        blur_img = blur_img.to(device)
        sample_img = sample(feature_extractor, unet, blur_img, device)
        save_image_tensor(sample_img.cpu(), f'results/deblurred_{idx}.png')
        print(f"Saved results/deblurred_{idx}.png")

if __name__ == '__main__':
    train()
    test()
