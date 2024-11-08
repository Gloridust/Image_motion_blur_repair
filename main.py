import torch
from torch.utils.data import DataLoader
from dataset import DeblurDataset
from model import DeblurTransformer
import torch.optim as optim
import torch.nn as nn
from utils import save_image_tensor
import os

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 实例化模型
    model = DeblurTransformer().to(device)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # 学习率调度器，每20个epoch，将学习率减半
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # 训练设置
    num_epochs = 50
    early_stop_patience = 5  # 早停的耐心值
    best_val_loss = float('inf')
    patience_counter = 0

    # 加载数据集
    train_dataset = DeblurDataset(split='train')
    val_dataset = DeblurDataset(split='val')
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (blur_img, sharp_img) in enumerate(train_loader):
            blur_img = blur_img.to(device)
            sharp_img = sharp_img.to(device)

            # 前向传播
            outputs = model(blur_img)
            loss = criterion(outputs, sharp_img)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')

        # 学习率调整
        scheduler.step()

        # 验证模型
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('Best model saved.')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print('Early stopping triggered.')
                break

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for blur_img, sharp_img in data_loader:
            blur_img = blur_img.to(device)
            sharp_img = sharp_img.to(device)
            outputs = model(blur_img)
            loss = criterion(outputs, sharp_img)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeblurTransformer().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    test_dataset = DeblurDataset(split='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    if not os.path.exists('results'):
        os.makedirs('results')

    with torch.no_grad():
        for idx, (blur_img, _) in enumerate(test_loader):
            blur_img = blur_img.to(device)
            output = model(blur_img)
            save_image_tensor(output.cpu(), f'results/deblurred_{idx}.png')

if __name__ == '__main__':
    train()
    test()
