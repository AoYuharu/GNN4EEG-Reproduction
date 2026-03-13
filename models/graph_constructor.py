"""
图构建模块

实现多种 EEG 功能连接图构建方法:
- 皮尔逊相关系数 (Pearson Correlation)
- 相位锁定值 (PLV)
- 可学习图构建
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConstructor(nn.Module):
    """
    图构建模块 - 从 EEG 数据构建功能连接图
    
    支持的构建方法:
    - correlation: 皮尔逊相关系数
    - plv: 相位锁定值
    - learnable: 可学习的邻接矩阵
    - identity: 单位矩阵 (固定图)
    """
    
    def __init__(self, method='correlation', num_channels=None, threshold=0.3):
        """
        Args:
            method: 图构建方法
            num_channels: 通道数 (仅 learnable 方法需要)
            threshold: 相关性阈值 (用于稀疏化)
        """
        super(GraphConstructor, self).__init__()
        self.method = method.lower()
        self.threshold = threshold
        
        if self.method == 'learnable':
            if num_channels is None:
                raise ValueError("learnable 方法需要指定 num_channels")
            # 可学习的邻接矩阵参数
            self.adj = nn.Parameter(torch.randn(num_channels, num_channels) * 0.01)
        elif self.method == 'identity':
            # 固定为单位矩阵
            self.register_buffer('identity_adj', torch.eye(num_channels))
    
    def forward(self, x):
        """
        构建功能连接图
        
        Args:
            x: EEG 数据 (batch_size, num_channels, sequence_length)
        
        Returns:
            adj: 邻接矩阵 (batch_size, num_channels, num_channels)
        """
        if self.method == 'correlation':
            return self._build_correlation_graph(x)
        elif self.method == 'plv':
            return self._build_plv_graph(x)
        elif self.method == 'learnable':
            return self._build_learnable_graph(x)
        elif self.method == 'identity':
            return self._build_identity_graph(x)
        else:
            raise ValueError(f"不支持的图构建方法：{self.method}")
    
    def _build_correlation_graph(self, x):
        """
        使用皮尔逊相关系数构建功能连接图
        
        Args:
            x: EEG 数据 (batch_size, num_channels, sequence_length)
        
        Returns:
            adj: 相关系数矩阵 (batch_size, num_channels, num_channels)
        """
        batch_size, num_channels, seq_len = x.size()
        
        # 标准化数据 (均值为 0，方差为 1)
        x_normalized = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)
        
        # 计算相关系数矩阵：(batch, channels, seq) @ (batch, seq, channels) -> (batch, channels, channels)
        adj = torch.bmm(x_normalized, x_normalized.transpose(1, 2)) / seq_len
        
        # 应用阈值稀疏化 (可选)
        if self.threshold > 0:
            mask = torch.abs(adj) < self.threshold
            adj = adj.masked_fill(mask, 0)
        
        # 确保对角线为 0 (无自环)
        for i in range(batch_size):
            adj[i].fill_diagonal_(0)
        
        return adj
    
    def _build_correlation_graph(self, x):
        """
        使用皮尔逊相关系数构建功能连接图
        
        Args:
            x: EEG 数据 (batch_size, num_channels, sequence_length)
        
        Returns:
            adj: 相关系数矩阵 (batch_size, num_channels, num_channels)
        """
        batch_size, num_channels, seq_len = x.size()
        
        # 标准化数据 (均值为 0，方差为 1)
        x_normalized = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)
        
        # 计算相关系数矩阵：(batch, channels, seq) @ (batch, seq, channels) -> (batch, channels, channels)
        adj = torch.bmm(x_normalized, x_normalized.transpose(1, 2)) / seq_len
        
        # 应用阈值稀疏化 (可选)
        if self.threshold > 0:
            mask = torch.abs(adj) < self.threshold
            adj = adj.masked_fill(mask, 0)
        
        # 确保对角线为 0 (无自环)
        for i in range(batch_size):
            adj[i].fill_diagonal_(0)
        
        return adj
    
    def _build_plv_graph(self, x):
        """
        使用相位锁定值 (PLV) 构建功能连接图
        
        PLV 衡量两个信号相位的一致性
        
        Args:
            x: EEG 数据 (batch_size, num_channels, sequence_length)
        
        Returns:
            adj: PLV 矩阵 (batch_size, num_channels, num_channels)
        """
        batch_size, num_channels, seq_len = x.size()
        
        # 使用 Hilbert 变换提取瞬时相位 (简化版本：使用 FFT)
        # 实际应用中应使用 scipy.signal.hilbert
        x_fft = torch.fft.fft(x, dim=-1)
        phase = torch.angle(x_fft)
        
        # 计算相位差
        # phase: (batch, channels, freq)
        phase_diff = phase.unsqueeze(2) - phase.unsqueeze(1)  # (batch, channels, channels, freq)
        
        # PLV = |mean(exp(i * phase_diff))|
        plv = torch.abs(torch.mean(torch.exp(1j * phase_diff), dim=-1))
        
        # 确保对角线为 0
        for i in range(batch_size):
            plv[i].fill_diagonal_(0)
        
        return plv.real
    
    def _build_learnable_graph(self, x):
        """
        使用可学习参数构建图
        
        Args:
            x: EEG 数据
        
        Returns:
            adj: 邻接矩阵 (batch_size, num_channels, num_channels)
        """
        batch_size = x.size(0)
        num_channels = self.adj.size(0)
        
        # 使用 softmax 归一化
        adj = F.softmax(self.adj, dim=-1)
        
        # 扩展到 batch
        adj = adj.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 确保对角线为 0
        for i in range(batch_size):
            adj[i].fill_diagonal_(0)
        
        return adj
    
    def _build_identity_graph(self, x):
        """
        返回单位矩阵 (固定图结构)
        
        Args:
            x: EEG 数据
        
        Returns:
            adj: 单位矩阵 (batch_size, num_channels, num_channels)
        """
        batch_size = x.size(0)
        num_channels = self.identity_adj.size(0)
        
        return self.identity_adj.unsqueeze(0).repeat(batch_size, 1, 1)


class DynamicGraphConstructor(nn.Module):
    """
    动态图构建器 - 为每个时间步构建不同的图
    
    适用于时间序列中功能连接动态变化的场景
    """
    
    def __init__(self, in_features, hidden_dim, num_channels):
        """
        Args:
            in_features: 输入特征维度
            hidden_dim: 隐藏层维度
            num_channels: 通道数
        """
        super(DynamicGraphConstructor, self).__init__()
        
        self.num_channels = num_channels
        
        # 使用神经网络学习图结构
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.adj_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: EEG 数据 (batch_size, num_channels, sequence_length)
        
        Returns:
            adj: 动态邻接矩阵 (batch_size, num_channels, num_channels)
        """
        batch_size, num_channels, seq_len = x.size()
        
        # 提取节点特征
        node_features = self.feature_extractor(x.transpose(1, 2))  # (batch, seq, hidden_dim)
        
        # 计算节点对之间的边权重
        adj = torch.zeros(batch_size, num_channels, num_channels, device=x.device)
        
        for i in range(num_channels):
            for j in range(i + 1, num_channels):
                # 拼接两个节点的特征
                pair_features = torch.cat([
                    node_features[:, :, i:i+1],  # (batch, seq, hidden_dim)
                    node_features[:, :, j:j+1]
                ], dim=-1)  # (batch, seq, hidden_dim*2)
                
                # 预测边权重
                edge_weight = self.adj_predictor(pair_features).mean(dim=1)  # (batch, 1)
                adj[:, i, j] = edge_weight.squeeze()
                adj[:, j, i] = edge_weight.squeeze()  # 无向图
        
        # 应用 softmax 归一化
        adj = F.softmax(adj, dim=-1)
        
        return adj
