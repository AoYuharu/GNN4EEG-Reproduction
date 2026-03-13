"""
GNN4EEG 核心模型实现

状态：框架完成，等待论文原文确认以下参数：
- 图构建方法 (PLV/相关系数/其他)
- GCN 层数和隐藏维度
- 注意力机制配置
- 具体超参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_constructor import GraphConstructor
from .layers import SpatialGCN, TemporalTCN, AttentionPool


class GNN4EEG(nn.Module):
    """
    GNN4EEG 模型 - 基于图神经网络的 EEG 情绪识别
    
    架构:
    1. 图构建：从 EEG 通道构建功能连接图
    2. 空间特征提取：GCN 处理通道间关系
    3. 时间特征提取：TCN 处理时间序列
    4. 注意力池化：聚合时间维度特征
    5. 分类头：输出情绪类别
    """
    
    def __init__(self, 
                 num_channels=32,
                 in_features=1,
                 hidden_dim=64,
                 num_classes=2,
                 num_gcn_layers=3,
                 dropout=0.5,
                 graph_type='correlation',
                 use_attention=True):
        """
        Args:
            num_channels: EEG 通道数 (DEAP=32, SEED=62)
            in_features: 输入特征维度
            hidden_dim: 隐藏层维度
            num_classes: 分类类别数
            num_gcn_layers: GCN 层数
            dropout: Dropout 率
            graph_type: 图构建方法 ('correlation' | 'plv' | 'learnable')
            use_attention: 是否使用注意力池化
        """
        super(GNN4EEG, self).__init__()
        
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 1. 图构建模块
        self.graph_constructor = GraphConstructor(method=graph_type)
        
        # 2. 空间特征提取 - 多层 GCN
        self.gcn_layers = nn.ModuleList()
        for i in range(num_gcn_layers):
            in_dim = in_features if i == 0 else hidden_dim
            out_dim = hidden_dim
            self.gcn_layers.append(SpatialGCN(in_dim, out_dim, dropout=dropout))
        
        # 3. 时间特征提取 - TCN
        self.temporal_tcn = TemporalTCN(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            num_layers=2,
            dropout=dropout
        )
        
        # 4. 注意力池化 (可选)
        self.use_attention = use_attention
        if use_attention:
            self.attention_pool = AttentionPool(hidden_dim)
        
        # 5. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入 EEG 数据
               - 格式 1: (batch_size, num_channels, sequence_length) - 原始 EEG
               - 格式 2: (batch_size, num_channels, in_features) - DE 特征
        
        Returns:
            logits: 分类输出 (batch_size, num_classes)
        """
        batch_size, num_channels, seq_len = x.size()
        
        # 1. 构建功能连接图
        # x: (batch, channels, time) -> adj: (batch, channels, channels)
        adj = self.graph_constructor(x)
        
        # 2. 空间特征提取 - GCN
        # 对每个时间步应用 GCN
        # x: (batch, channels, time) -> (batch, time, channels)
        x_trans = x.transpose(1, 2)  # (batch, time, channels)
        
        # 对每个时间步应用 GCN
        h_list = []
        for t in range(seq_len):
            h_t = x_trans[:, t, :].unsqueeze(1)  # (batch, 1, channels)
            for gcn in self.gcn_layers:
                h_t = gcn(h_t, adj)  # (batch, 1, hidden_dim)
            h_list.append(h_t)
        
        h = torch.cat(h_list, dim=1)  # (batch, time, hidden_dim)
        
        # 3. 时间特征提取 - TCN
        # 转换维度：(batch, time, hidden_dim) -> (batch, hidden_dim, time)
        h = h.transpose(1, 2)  # (batch, hidden_dim, time)
        h = self.temporal_tcn(h)  # (batch, hidden_dim, time')
        
        # 4. 注意力池化或全局平均池化
        if self.use_attention:
            # h: (batch, hidden_dim, time) -> (batch, hidden_dim)
            h = self.attention_pool(h)
        else:
            h = h.mean(dim=-1)  # 全局平均池化
        
        # 5. 分类
        logits = self.classifier(h)  # (batch, num_classes)
        
        return logits
    
    def reset_parameters(self):
        """初始化模型参数"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def count_parameters(self):
        """计算可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(name='gnn4eeg', **kwargs):
    """
    创建模型工厂函数
    
    Args:
        name: 模型名称
        **kwargs: 模型参数
    
    Returns:
        model: PyTorch 模型
    """
    if name.lower() == 'gnn4eeg':
        return GNN4EEG(**kwargs)
    else:
        raise ValueError(f"不支持的模型：{name}")


# 预定义模型配置
MODEL_CONFIGS = {
    'gnn4eeg_small': {
        'hidden_dim': 32,
        'num_gcn_layers': 2,
        'dropout': 0.3,
    },
    'gnn4eeg_base': {
        'hidden_dim': 64,
        'num_gcn_layers': 3,
        'dropout': 0.5,
    },
    'gnn4eeg_large': {
        'hidden_dim': 128,
        'num_gcn_layers': 4,
        'dropout': 0.5,
    },
}


def get_model_config(config_name='gnn4eeg_base'):
    """获取预定义模型配置"""
    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"未知配置：{config_name}")
    return MODEL_CONFIGS[config_name]
