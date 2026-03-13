"""
辅助工具函数
"""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    设置所有随机种子以确保可复现性
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNN 确定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"随机种子已设置：{seed}")


def count_parameters(model: torch.nn.Module) -> int:
    """
    计算模型可训练参数数量
    
    Args:
        model: PyTorch 模型
    
    Returns:
        可训练参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: torch.nn.Module) -> int:
    """
    计算模型总参数数量 (包括不可训练的)
    
    Args:
        model: PyTorch 模型
    
    Returns:
        总参数数量
    """
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(state: dict, is_best: bool, save_dir: str, filename: str = 'checkpoint.pth.tar'):
    """
    保存训练检查点
    
    Args:
        state: 要保存的状态字典
        is_best: 是否是最佳模型
        save_dir: 保存目录
        filename: 文件名
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存当前检查点
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(state, checkpoint_path)
    print(f"检查点已保存：{checkpoint_path}")
    
    # 如果是最佳模型，复制一份
    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth.tar')
        torch.save(state, best_path)
        print(f"最佳模型已保存：{best_path}")


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, 
                    optimizer=None, device: str = 'cpu'):
    """
    加载训练检查点
    
    Args:
        checkpoint_path: 检查点文件路径
        model: 要加载权重的模型
        optimizer: 优化器 (可选)
        device: 计算设备
    
    Returns:
        checkpoint: 检查点字典
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在：{checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['state_dict'])
    print(f"模型权重已加载：{checkpoint_path}")
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("优化器状态已加载")
    
    return checkpoint


class EarlyStopping:
    """
    早停机制
    
    当验证指标在指定轮数内没有改善时停止训练
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 mode: str = 'max', verbose: bool = True):
        """
        Args:
            patience: 容忍轮数
            min_delta: 最小改善阈值
            mode: 'max' (越大越好) 或 'min' (越小越好)
            verbose: 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            score: 当前轮次的指标值
        
        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        # 判断是否改善
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f"指标改善：{self.best_score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"早停计数：{self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"触发早停 (最佳指标：{self.best_score:.4f})")
        
        return self.early_stop
    
    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class AverageMeter:
    """
    计算并存储当前值、平均值、总和的辅助类
    """
    
    def __init__(self, name: str = 'Metric'):
        self.name = name
        self.reset()
    
    def reset(self):
        """重置所有统计值"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        更新统计值
        
        Args:
            val: 当前值
            n: 样本数量
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"
