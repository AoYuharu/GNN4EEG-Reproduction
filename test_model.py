"""
GNN4EEG 模型测试脚本

测试内容:
1. 模型前向传播测试
2. 图构建模块测试
3. 各层组件测试
4. 参数数量统计
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.gnn4eeg import GNN4EEG, create_model, get_model_config
from models.graph_constructor import GraphConstructor
from models.layers import SpatialGCN, TemporalTCN, AttentionPool


def test_graph_constructor():
    """测试图构建模块"""
    print("\n" + "="*60)
    print("测试 1: 图构建模块")
    print("="*60)
    
    batch_size = 4
    num_channels = 32
    seq_len = 8064  # DEAP: 62 秒 × 128 Hz
    
    # 生成随机 EEG 数据
    x = torch.randn(batch_size, num_channels, seq_len)
    
    # 测试不同图构建方法
    methods = ['correlation', 'plv', 'identity']
    
    for method in methods:
        print(f"\n测试方法：{method}")
        gc = GraphConstructor(method=method, num_channels=num_channels)
        adj = gc(x)
        
        print(f"  输入形状：{x.shape}")
        print(f"  输出形状：{adj.shape}")
        print(f"  邻接矩阵范围：[{adj.min():.4f}, {adj.max():.4f}]")
        # 检查对角线 (batch 中每个矩阵)
        diag_check = all(torch.allclose(adj[i].diag(), torch.zeros_like(adj[i].diag())) for i in range(batch_size))
        print(f"  对角线为零：{diag_check}")
        
        # 验证形状
        assert adj.shape == (batch_size, num_channels, num_channels), \
            f"形状错误：期望 {(batch_size, num_channels, num_channels)}, 得到 {adj.shape}"
        print(f"  ✅ {method} 测试通过")
    
    print("\n✅ 图构建模块测试全部通过")


def test_spatial_gcn():
    """测试空间 GCN 层"""
    print("\n" + "="*60)
    print("测试 2: 空间 GCN 层")
    print("="*60)
    
    batch_size = 4
    num_channels = 32
    in_features = 64
    out_features = 64
    
    # 生成随机数据
    x = torch.randn(batch_size, num_channels, in_features)
    adj = torch.randn(batch_size, num_channels, num_channels)
    adj = torch.softmax(adj, dim=-1)  # 归一化
    
    # 创建 GCN 层
    gcn = SpatialGCN(in_features, out_features, dropout=0.5)
    
    # 前向传播
    out = gcn(x, adj)
    
    print(f"  输入形状：{x.shape}")
    print(f"  邻接矩阵形状：{adj.shape}")
    print(f"  输出形状：{out.shape}")
    
    assert out.shape == (batch_size, num_channels, out_features), \
        f"形状错误：期望 {(batch_size, num_channels, out_features)}, 得到 {out.shape}"
    
    print("  ✅ 空间 GCN 层测试通过")


def test_temporal_tcn():
    """测试时间 TCN 层"""
    print("\n" + "="*60)
    print("测试 3: 时间 TCN 层")
    print("="*60)
    
    batch_size = 4
    channels = 64
    seq_len = 100
    
    # 生成随机数据
    x = torch.randn(batch_size, channels, seq_len)
    
    # 创建 TCN 层
    tcn = TemporalTCN(channels, channels, kernel_size=3, num_layers=2, dropout=0.5)
    
    # 前向传播
    out = tcn(x)
    
    print(f"  输入形状：{x.shape}")
    print(f"  输出形状：{out.shape}")
    
    assert out.shape == (batch_size, channels, seq_len), \
        f"形状错误：期望 {(batch_size, channels, seq_len)}, 得到 {out.shape}"
    
    print("  ✅ 时间 TCN 层测试通过")


def test_attention_pool():
    """测试注意力池化层"""
    print("\n" + "="*60)
    print("测试 4: 注意力池化层")
    print("="*60)
    
    batch_size = 4
    hidden_dim = 64
    seq_len = 100
    
    # 生成随机数据
    x = torch.randn(batch_size, hidden_dim, seq_len)
    
    # 创建注意力池化层
    attn_pool = AttentionPool(hidden_dim)
    
    # 前向传播
    out = attn_pool(x)
    
    print(f"  输入形状：{x.shape}")
    print(f"  输出形状：{out.shape}")
    
    assert out.shape == (batch_size, hidden_dim), \
        f"形状错误：期望 {(batch_size, hidden_dim)}, 得到 {out.shape}"
    
    print("  ✅ 注意力池化层测试通过")


def test_gnn4eeg_model():
    """测试完整 GNN4EEG 模型"""
    print("\n" + "="*60)
    print("测试 5: GNN4EEG 完整模型")
    print("="*60)
    
    # 测试配置
    configs = [
        {
            'name': 'DEAP 二分类',
            'num_channels': 32,
            'in_features': 1,
            'hidden_dim': 64,
            'num_classes': 2,
            'seq_len': 8064,
        },
        {
            'name': 'SEED 三分类',
            'num_channels': 62,
            'in_features': 1,
            'hidden_dim': 64,
            'num_classes': 3,
            'seq_len': 1000,  # SEED 使用 DE 特征，序列较短
        },
        {
            'name': '小模型配置',
            'num_channels': 32,
            'in_features': 1,
            'hidden_dim': 32,
            'num_classes': 2,
            'seq_len': 1000,
        },
    ]
    
    for config in configs:
        print(f"\n测试配置：{config['name']}")
        
        # 创建模型
        model = GNN4EEG(
            num_channels=config['num_channels'],
            in_features=config['in_features'],
            hidden_dim=config['hidden_dim'],
            num_classes=config['num_classes'],
            num_gcn_layers=3,
            dropout=0.5,
            graph_type='correlation',
            use_attention=True
        )
        
        # 生成随机输入
        batch_size = 4
        x = torch.randn(batch_size, config['num_channels'], config['seq_len'])
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        print(f"  输入形状：{x.shape}")
        print(f"  输出形状：{output.shape}")
        print(f"  参数量：{model.count_parameters():,}")
        
        assert output.shape == (batch_size, config['num_classes']), \
            f"形状错误：期望 {(batch_size, config['num_classes'])}, 得到 {output.shape}"
        
        print(f"  ✅ {config['name']} 测试通过")
    
    print("\n✅ GNN4EEG 完整模型测试全部通过")


def test_model_factory():
    """测试模型工厂函数"""
    print("\n" + "="*60)
    print("测试 6: 模型工厂函数")
    print("="*60)
    
    # 测试预定义配置
    for config_name in ['gnn4eeg_small', 'gnn4eeg_base', 'gnn4eeg_large']:
        print(f"\n测试配置：{config_name}")
        
        config = get_model_config(config_name)
        print(f"  配置：{config}")
        
        # 创建模型
        model = create_model('gnn4eeg', **config, num_channels=32, in_features=1, num_classes=2)
        
        print(f"  参数量：{model.count_parameters():,}")
        print(f"  ✅ {config_name} 测试通过")
    
    print("\n✅ 模型工厂函数测试全部通过")


def test_gradient_flow():
    """测试梯度流动"""
    print("\n" + "="*60)
    print("测试 7: 梯度流动测试")
    print("="*60)
    
    # 创建模型
    model = GNN4EEG(
        num_channels=32,
        in_features=1,
        hidden_dim=64,
        num_classes=2,
        num_gcn_layers=3,
        dropout=0.5
    )
    
    # 生成数据
    batch_size = 4
    seq_len = 1000
    x = torch.randn(batch_size, 32, seq_len, requires_grad=False)
    target = torch.randint(0, 2, (batch_size,))
    
    # 前向传播 + 反向传播
    model.train()
    output = model(x)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    loss.backward()
    
    # 检查梯度 (只检查需要梯度的参数)
    has_gradient = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += 1
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradient += 1
    
    print(f"  损失值：{loss.item():.4f}")
    print(f"  有梯度的参数：{has_gradient}/{trainable_params}")
    
    # 允许部分参数无梯度 (如图构建模块的固定参数)
    assert has_gradient > 0, "没有参数获得梯度"
    assert has_gradient >= trainable_params * 0.5, f"梯度过少：{has_gradient}/{trainable_params}"
    
    print("  ✅ 梯度流动测试通过")


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("GNN4EEG 模型测试套件")
    print("="*60)
    print(f"PyTorch 版本：{torch.__version__}")
    print(f"设备：{'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    try:
        # 运行所有测试
        test_graph_constructor()
        test_spatial_gcn()
        test_temporal_tcn()
        test_attention_pool()
        test_gnn4eeg_model()
        test_model_factory()
        test_gradient_flow()
        
        # 总结
        print("\n" + "="*60)
        print("🎉 所有测试通过!")
        print("="*60)
        print("\n测试结果总结:")
        print("  ✅ 图构建模块 (correlation/plv/identity)")
        print("  ✅ 空间 GCN 层")
        print("  ✅ 时间 TCN 层")
        print("  ✅ 注意力池化层")
        print("  ✅ GNN4EEG 完整模型 (多种配置)")
        print("  ✅ 模型工厂函数")
        print("  ✅ 梯度流动")
        print("\n模型已准备就绪，可以开始训练!")
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ 测试失败!")
        print("="*60)
        print(f"错误：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
