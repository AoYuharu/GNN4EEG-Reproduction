"""
评估指标计算
"""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class Metric:
    """评估指标基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    def reset(self):
        """重置指标状态"""
        self.predictions = []
        self.targets = []
    
    def update(self, preds: np.ndarray, targets: np.ndarray):
        """更新指标"""
        self.predictions.extend(preds.tolist())
        self.targets.extend(targets.tolist())
    
    def compute(self) -> float:
        """计算指标值"""
        raise NotImplementedError
    
    def __call__(self, preds: np.ndarray, targets: np.ndarray) -> float:
        """直接计算指标"""
        self.update(preds, targets)
        return self.compute()
    
    def __repr__(self) -> str:
        return f"{self.name}: {self.compute():.4f}"


class Accuracy(Metric):
    """准确率"""
    
    def __init__(self):
        super().__init__('Accuracy')
    
    def compute(self) -> float:
        if not self.predictions:
            return 0.0
        return accuracy_score(self.targets, self.predictions)


class Precision(Metric):
    """精确率"""
    
    def __init__(self, average='macro'):
        super().__init__('Precision')
        self.average = average
    
    def compute(self) -> float:
        if not self.predictions:
            return 0.0
        unique_labels = len(set(self.targets))
        if unique_labels < 2:
            return 0.0
        return precision_score(self.targets, self.predictions, 
                               average=self.average, zero_division=0)


class Recall(Metric):
    """召回率"""
    
    def __init__(self, average='macro'):
        super().__init__('Recall')
        self.average = average
    
    def compute(self) -> float:
        if not self.predictions:
            return 0.0
        unique_labels = len(set(self.targets))
        if unique_labels < 2:
            return 0.0
        return recall_score(self.targets, self.predictions,
                            average=self.average, zero_division=0)


class F1Score(Metric):
    """F1 分数"""
    
    def __init__(self, average='macro'):
        super().__init__('F1')
        self.average = average
    
    def compute(self) -> float:
        if not self.predictions:
            return 0.0
        unique_labels = len(set(self.targets))
        if unique_labels < 2:
            return 0.0
        return f1_score(self.targets, self.predictions,
                        average=self.average, zero_division=0)


class ConfusionMatrix:
    """混淆矩阵"""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """重置混淆矩阵"""
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, preds: np.ndarray, targets: np.ndarray):
        """更新混淆矩阵"""
        cm = confusion_matrix(targets, preds, labels=list(range(self.num_classes)))
        self.matrix += cm
    
    def compute(self) -> np.ndarray:
        """返回混淆矩阵"""
        return self.matrix
    
    def plot(self, class_names=None, save_path=None):
        """绘制混淆矩阵"""
        import matplotlib.pyplot as plt
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(self.num_classes)]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(self.matrix, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(self.num_classes),
               yticks=np.arange(self.num_classes),
               xticklabels=class_names, yticklabels=class_names,
               xlabel='Predicted label',
               ylabel='True label')
        
        # 添加数值标注
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                ax.text(j, i, str(self.matrix[i, j]),
                        ha='center', va='center',
                        color='white' if self.matrix[i, j] > self.matrix.max() / 2 else 'black')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        
        return fig


def compute_metrics(predictions, targets, num_classes=2):
    """
    计算所有评估指标
    
    Args:
        predictions: 预测标签
        targets: 真实标签
        num_classes: 类别数
    
    Returns:
        包含所有指标的字典
    """
    metrics = {
        'accuracy': accuracy_score(targets, predictions),
        'precision_macro': precision_score(targets, predictions, average='macro', zero_division=0),
        'recall_macro': recall_score(targets, predictions, average='macro', zero_division=0),
        'f1_macro': f1_score(targets, predictions, average='macro', zero_division=0),
        'precision_weighted': precision_score(targets, predictions, average='weighted', zero_division=0),
        'recall_weighted': recall_score(targets, predictions, average='weighted', zero_division=0),
        'f1_weighted': f1_score(targets, predictions, average='weighted', zero_division=0),
    }
    
    if num_classes == 2:
        metrics['precision'] = precision_score(targets, predictions, zero_division=0)
        metrics['recall'] = recall_score(targets, predictions, zero_division=0)
        metrics['f1'] = f1_score(targets, predictions, zero_division=0)
    
    return metrics
