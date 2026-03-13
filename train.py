"""
GNN4EEG Training Script

用法:
    python train.py --config config/deap.yaml
"""
import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        if batch_idx % 10 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                              'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), 100. * correct / total


def main():
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    print(f"加载配置：{args.config}")
    
    # 设置随机种子
    set_seed(config.get('misc', {}).get('seed', 42))
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")
    
    # TODO: 初始化数据集
    # from data.deap import get_deap_loaders
    # train_loader, val_loader, test_loader = get_deap_loaders(...)
    
    # TODO: 初始化模型
    # from models import GNN4EEG
    # model = GNN4EEG(...).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    
    # TensorBoard
    writer = SummaryWriter(log_dir=config['misc']['save_dir'])
    
    # 训练循环
    best_acc = 0
    patience = config['training']['early_stopping']['patience']
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        print(f'\nEpoch {epoch+1}/{config["training"]["epochs"]}')
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                            optimizer, device)
        print(f'训练集 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        
        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f'验证集 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        
        # 记录 TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 
                       os.path.join(config['misc']['save_dir'], 'best_model.pth'))
            print(f'保存最佳模型 (Acc: {val_acc:.2f}%)')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'早停触发 ({patience} epochs 无改进)')
                break
    
    # 测试
    print('\n在测试集上评估最佳模型...')
    model.load_state_dict(torch.load(os.path.join(config['misc']['save_dir'], 
                                                   'best_model.pth')))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f'测试集 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')
    
    writer.close()
    print('训练完成!')


if __name__ == '__main__':
    main()
