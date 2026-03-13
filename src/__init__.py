"""
GNN4EEG 源代码模块

包含模型实现、训练逻辑等核心代码
"""

# 模型导入
from models import GNN4EEG, GraphConstructor, SpatialGCN, TemporalTCN

# 数据导入
from data import EEGDataset, DEAPDataset, SEEDDataset, get_dataloaders

# 工具导入
from utils import (
    Accuracy, Precision, Recall, F1Score, ConfusionMatrix,
    set_seed, count_parameters, save_checkpoint, load_checkpoint,
    EarlyStopping, AverageMeter
)

__version__ = '0.1.0'
__author__ = 'GNN4EEG Reproduction Team'

__all__ = [
    # 模型
    'GNN4EEG',
    'GraphConstructor',
    'SpatialGCN',
    'TemporalTCN',
    
    # 数据
    'EEGDataset',
    'DEAPDataset',
    'SEEDDataset',
    'get_dataloaders',
    
    # 工具
    'Accuracy',
    'Precision',
    'Recall',
    'F1Score',
    'ConfusionMatrix',
    'set_seed',
    'count_parameters',
    'save_checkpoint',
    'load_checkpoint',
    'EarlyStopping',
    'AverageMeter'
]
