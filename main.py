import torch
from torch.utils.data import DataLoader
from dataset import DeblurDataset
from model import FeatureExtractor, ConditionalUNet
import torch.optim as optim
import torch.nn as nn
from utils import save_image_tensor, noise_schedule, extract
import os
from tqdm import tqdm

def validate(feature_extractor, unet, val_loader, device, criterion, timesteps, alphas_cumprod):
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

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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
    
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Please check the data directory.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty. Please check the data directory.")

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    patience = 10  # 早停耐心值
    patience_counter = 0

    # 创建保存模型的目录
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    try:
        for epoch in range(num_epochs):
            # 训练阶段
            feature_extractor.train()
            unet.train()
            epoch_loss = 0.0
            for blur_img, sharp_img in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}")

            # 验证阶段
            val_loss = validate(feature_extractor, unet, val_loader, device, mse_loss, timesteps, alphas_cumprod)
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

            # 保存检查点
            checkpoint_path = os.path.join('checkpoints', f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'feature_extractor_state_dict': feature_extractor.state_dict(),
                'unet_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'val_loss': val_loss,
            }, checkpoint_path)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'feature_extractor_state_dict': feature_extractor.state_dict(),
                    'unet_state_dict': unet.state_dict(),
                }, "checkpoints/best_model.pth")
                print("Best model saved.")
                patience_counter = 0
            else:
                patience_counter += 1

            # 早停检查
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    except KeyboardInterrupt:
        print("Training interrupted by user")
        # 保存当前模型
        torch.save({
            'feature_extractor_state_dict': feature_extractor.state_dict(),
            'unet_state_dict': unet.state_dict(),
        }, "interrupted_model.pth")
        print("Model saved at interrupted state.")

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = FeatureExtractor().to(device)
    unet = ConditionalUNet().to(device)
    
    # 加载最佳模型
    try:
        checkpoint = torch.load('best_model.pth', map_location=device)
        feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        unet.load_state_dict(checkpoint['unet_state_dict'])
        print("Loaded best model successfully")
    except FileNotFoundError:
        print("Best model not found. Please train the model first.")
        return

    test_dataset = DeblurDataset(split='test')
    if len(test_dataset) == 0:
        print("Test dataset is empty. Please check the data directory.")
        return

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if not os.path.exists('results'):
        os.makedirs('results')

    feature_extractor.eval()
    unet.eval()
    
    with torch.no_grad():
        for idx, (blur_img, _) in enumerate(tqdm(test_loader, desc="Testing")):
            blur_img = blur_img.to(device)
            sample_img = sample(feature_extractor, unet, blur_img, device)
            save_image_tensor(sample_img.cpu(), f'results/deblurred_{idx}.png')

def sample(feature_extractor, unet, blur_img, device, timesteps=1000):
    with torch.no_grad():
        # 提取条件特征
        cond_feat = feature_extractor(blur_img)
        x = torch.randn_like(blur_img)
        betas = noise_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # 逐步去噪
        for t in tqdm(reversed(range(timesteps)), desc="Sampling", leave=False):
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

if __name__ == '__main__':
    try:
        train()
        test()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
