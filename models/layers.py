"""
GNN4EEG 模型层定义

包含:
- SpatialGCN: 空间图卷积层
- TemporalTCN: 时间卷积网络
- AttentionPool: 注意力池化层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGCN(nn.Module):
    """
    空间图卷积层 (Graph Convolutional Network)
    
    公式：H' = σ(AHW)
    - A: 邻接矩阵 (batch, channels, channels)
    - H: 节点特征 (batch, channels, features)
    - W: 可学习权重
    """
    
    def __init__(self, in_features, out_features, dropout=0.5, use_bias=True):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            dropout: Dropout 率
            use_bias: 是否使用偏置
        """
        super(SpatialGCN, self).__init__()
        
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.linear.weight)
        if use_bias:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x, adj):
        """
        Args:
            x: 节点特征 (batch_size, num_channels, in_features) 或 (batch_size, in_features, num_channels)
            adj: 邻接矩阵 (batch_size, num_channels, num_channels)
        
        Returns:
            更新后的节点特征 (batch_size, num_channels, out_features)
        """
        # 确保 x 的维度是 (batch, channels, features)
        if x.dim() == 3 and x.size(1) != adj.size(1):
            # 如果是 (batch, features, channels)，转置为 (batch, channels, features)
            x = x.transpose(1, 2)
        
        # 图卷积：聚合邻居信息
        # H' = A * H
        h = torch.matmul(adj, x)  # (batch, channels, features)
        
        # 线性变换
        h = self.linear(h)
        
        # 归一化
        h = self.norm(h)
        
        # 激活函数
        h = F.relu(h)
        
        # Dropout
        h = self.dropout(h)
        
        return h


class TemporalTCN(nn.Module):
    """
    时间卷积网络 (Temporal Convolutional Network)
    
    使用因果卷积处理时间序列
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 num_layers=2, dropout=0.5, dilation=1):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            num_layers: TCN 层数
            dropout: Dropout 率
            dilation: 膨胀卷积因子
        """
        super(TemporalTCN, self).__init__()
        
        self.num_layers = num_layers
        
        # 构建多层 TCN
        layers = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            out_ch = out_channels
            
            # 因果卷积：padding 确保输出长度不变
            padding = (kernel_size - 1) * dilation // 2
            
            layers.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size, 
                         padding=padding, dilation=dilation),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        self.tcn = nn.Sequential(*layers)
        
        # 残差连接 (如果输入输出维度不同)
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = None
    
    def forward(self, x):
        """
        Args:
            x: 输入 (batch_size, channels, sequence_length)
        
        Returns:
            时间特征 (batch_size, out_channels, sequence_length)
        """
        residual = x
        
        # TCN 处理
        out = self.tcn(x)
        
        # 残差连接
        if self.residual is not None:
            residual = self.residual(residual)
        
        # 调整维度以匹配
        if residual.size(2) > out.size(2):
            residual = residual[:, :, :out.size(2)]
        
        out = out + residual
        
        return out


class AttentionPool(nn.Module):
    """
    注意力池化层
    
    学习不同时间步的重要性权重，加权聚合时间维度特征
    """
    
    def __init__(self, hidden_dim):
        """
        Args:
            hidden_dim: 隐藏层维度
        """
        super(AttentionPool, self).__init__()
        
        # 注意力权重计算
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        """
        Args:
            x: 输入特征 (batch_size, hidden_dim, sequence_length)
        
        Returns:
            聚合后的特征 (batch_size, hidden_dim)
        """
        # 转换维度：(batch, hidden, seq) -> (batch, seq, hidden)
        x_trans = x.transpose(1, 2)
        
        # 计算注意力权重：(batch, seq, 1)
        attn_weights = self.attention(x_trans)
        
        # 加权求和
        # x: (batch, hidden, seq), attn: (batch, seq, 1)
        # 结果：(batch, hidden)
        pooled = torch.sum(x * attn_weights.transpose(1, 2), dim=-1)
        
        return pooled


class GraphAttentionLayer(nn.Module):
    """
    图注意力层 (Graph Attention Layer)
    
    使用注意力机制学习邻居节点的重要性
    """
    
    def __init__(self, in_features, out_features, num_heads=1, dropout=0.5):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度 (每个注意力头)
            num_heads: 注意力头数
            dropout: Dropout 率
        """
        super(GraphAttentionLayer, self).__init__()
        
        self.num_heads = num_heads
        self.out_features = out_features
        
        # 线性变换
        self.linear = nn.Linear(in_features, num_heads * out_features)
        
        # 注意力参数
        self.attention = nn.Parameter(torch.randn(1, num_heads, out_features * 2))
        nn.init.xavier_uniform_(self.attention)
        
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
        # 初始化
        nn.init.xavier_uniform_(self.linear.weight)
    
    def forward(self, x, adj):
        """
        Args:
            x: 节点特征 (batch_size, num_channels, in_features)
            adj: 邻接矩阵 (batch_size, num_channels, num_channels)
        
        Returns:
            更新后的节点特征 (batch_size, num_channels, num_heads * out_features)
        """
        batch_size, num_channels, _ = x.size()
        
        # 线性变换
        h = self.linear(x)  # (batch, channels, heads * out_features)
        h = h.view(batch_size, num_channels, self.num_heads, self.out_features)
        h = h.transpose(1, 2)  # (batch, heads, channels, out_features)
        
        # 计算注意力分数
        # 拼接节点对特征
        h_i = h.unsqueeze(2).expand(-1, -1, num_channels, -1, -1)  # (batch, heads, channels, channels, out_features)
        h_j = h.unsqueeze(1).expand(-1, -1, num_channels, -1, -1)  # (batch, heads, channels, channels, out_features)
        h_pair = torch.cat([h_i, h_j], dim=-1)  # (batch, heads, channels, channels, out_features*2)
        
        # 注意力分数：(batch, heads, channels, channels)
        e = (h_pair * self.attention).sum(dim=-1)
        e = self.leakyrelu(e)
        
        # 应用邻接矩阵掩码
        adj_expanded = adj.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        e = e.masked_fill(adj_expanded == 0, -1e9)
        
        # Softmax 归一化
        attention_scores = F.softmax(e, dim=-1)
        attention_scores = self.dropout(attention_scores)
        
        # 加权聚合
        h_out = torch.matmul(attention_scores, h)  # (batch, heads, channels, out_features)
        
        # 合并多头
        h_out = h_out.transpose(1, 2).contiguous().view(
            batch_size, num_channels, self.num_heads * self.out_features
        )
        
        return h_out


class PositionalEncoding(nn.Module):
    """
    位置编码
    
    为时间序列添加位置信息
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout 率
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: 输入 (batch_size, sequence_length, d_model)
        
        Returns:
            添加位置编码后的输出
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
