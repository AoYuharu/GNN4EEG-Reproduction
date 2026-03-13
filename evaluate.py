"""
GNN4EEG 评估脚本

用法:
    python evaluate.py --config config.yaml --checkpoint experiments/best_model.pth.tar
"""
import os
import yaml
import argparse
import numpy as np
import torch
from tqdm import tqdm


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=2):
    """
    评估模型
    
    Args:
        model: 模型
        loader: 数据加载器
        criterion: 损失函数
        device: 计算设备
        num_classes: 类别数
    
    Returns:
        metrics: 评估指标字典
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    total_loss = 0
    
    pbar = tqdm(loader, desc='Evaluating')
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        loss = criterion(output, target)
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # 计算指标
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    metrics = {
        'loss': total_loss / len(loader),
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision_macro': precision_score(all_targets, all_preds, average='macro', zero_division=0),
        'recall_macro': recall_score(all_targets, all_preds, average='macro', zero_division=0),
        'f1_macro': f1_score(all_targets, all_preds, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    }
    
    if num_classes == 2:
        metrics['precision'] = precision_score(all_targets, all_preds, zero_division=0)
        metrics['recall'] = recall_score(all_targets, all_preds, zero_division=0)
        metrics['f1'] = f1_score(all_targets, all_preds, zero_division=0)
    
    return metrics


def print_metrics(metrics, num_classes=2):
    """打印评估指标"""
    print("\n" + "="*50)
    print("评估结果")
    print("="*50)
    print(f"Loss:     {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"F1 Score (macro):  {metrics['f1_macro']:.4f}")
    
    if num_classes == 2:
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
    
    print("\n混淆矩阵:")
    print(metrics['confusion_matrix'])
    print("="*50)


def main():
    # 解析参数
    parser = argparse.ArgumentParser(description='GNN4EEG 评估脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    parser.add_argument('--split', type=str, default='test', 
                        choices=['train', 'val', 'test'], help='评估数据集划分')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    print(f"加载配置：{args.config}")
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")
    
    # TODO: 加载数据集
    # from data import get_dataloaders
    # train_loader, val_loader, test_loader = get_dataloaders(...)
    
    # TODO: 初始化模型
    # from models import GNN4EEG
    # model = GNN4EEG(...).to(device)
    
    # 加载检查点
    print(f"加载模型检查点：{args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"模型已加载 (epoch {checkpoint.get('epoch', 'N/A')})")
    
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    
    # 选择评估的数据集
    # if args.split == 'test':
    #     loader = test_loader
    # elif args.split == 'val':
    #     loader = val_loader
    # else:
    #     loader = train_loader
    
    # 评估
    print(f"\n在 {args.split} 集上评估...")
    # metrics = evaluate(model, loader, criterion, device, 
    #                    num_classes=config['data']['num_classes'])
    
    # 打印结果
    # print_metrics(metrics, num_classes=config['data']['num_classes'])
    
    # 保存结果
    # results_dir = os.path.join(config['experiment']['log_dir'], 'results')
    # os.makedirs(results_dir, exist_ok=True)
    # np.save(os.path.join(results_dir, f'{args.split}_metrics.npy'), metrics)
    # print(f"\n结果已保存：{results_dir}")
    
    print("\n评估脚本框架已完成，待模型实现后启用")


if __name__ == '__main__':
    main()
