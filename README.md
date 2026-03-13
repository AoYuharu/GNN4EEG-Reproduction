# GNN4EEG 复现项目

**目标**: 复现 GNN4EEG 论文 - 基于图神经网络的 EEG 情绪识别  
**状态**: 🚧 框架搭建中  
**创建时间**: 2026-03-13

---

## 📁 项目结构

```
GNN4EEG-Reproduction/
├── README.md                    # 项目说明
├── requirements.txt             # Python 依赖
├── config/                      # 配置文件
│   ├── default.yaml            # 默认配置
│   ├── deap.yaml               # DEAP 数据集配置
│   └── seed.yaml               # SEED 数据集配置
├── data/                        # 数据相关
│   ├── __init__.py
│   ├── dataset.py              # 数据集基类
│   ├── deap.py                 # DEAP 数据加载器
│   ├── seed.py                 # SEED 数据加载器
│   └── preprocess.py           # 预处理工具
├── models/                      # 模型定义
│   ├── __init__.py
│   ├── gnn4eeg.py              # GNN4EEG 主模型
│   ├── layers.py               # 自定义层
│   └── graph_constructor.py    # 图构建方法
├── utils/                       # 工具函数
│   ├── __init__.py
│   ├── metrics.py              # 评估指标
│   └── helpers.py              # 辅助函数
├── train.py                     # 训练脚本
├── evaluate.py                  # 评估脚本
└── experiments/                 # 实验记录
    └── (实验输出目录)
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 创建数据目录
mkdir -p data/deap
mkdir -p data/seed

# 将下载的数据集放入对应目录
# DEAP: data/deap/data_preprocessed_python/
# SEED: data/seed/Preprocessed_EEG/
```

### 3. 配置修改

编辑 `config/deap.yaml`:
```yaml
data:
  root: ./data/deap
  dataset: deap
  
model:
  num_channels: 32
  num_classes: 2  # 二分类
  
training:
  batch_size: 32
  lr: 0.001
  epochs: 100
```

### 4. 开始训练

```bash
# 使用默认配置训练
python train.py --config config/deap.yaml

# 指定 GPU
CUDA_VISIBLE_DEVICES=0 python train.py --config config/deap.yaml
```

---

## 📦 依赖说明

### 核心依赖

| 包 | 版本 | 用途 |
|----|------|------|
| torch | >=1.9.0 | 深度学习框架 |
| torch-geometric | >=2.0.0 | 图神经网络 |
| numpy | >=1.19.0 | 数值计算 |
| scipy | >=1.5.0 | 信号处理 |
| mne | >=0.23.0 | EEG 数据处理 |
| scikit-learn | >=0.24.0 | 机器学习工具 |
| pyyaml | >=5.4.0 | 配置解析 |
| tqdm | >=4.60.0 | 进度条 |
| tensorboard | >=2.5.0 | 训练可视化 |

### 安装命令

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install numpy scipy mne scikit-learn pyyaml tqdm tensorboard
```

---

## 📊 数据集配置

### DEAP 配置

```yaml
# config/deap.yaml
data:
  name: deap
  root: ./data/deap
  num_channels: 32
  num_samples: 8064  # 62 秒 × 128 Hz
  num_trials: 40
  num_subjects: 32
  
preprocessing:
  sample_rate: 128
  bandpass: [4.0, 45.0]
  baseline_correction: true
  normalize: z-score
  
labels:
  task: valence  # valence/arousal
  threshold: 5   # 二分类阈值
  num_classes: 2
```

### SEED 配置

```yaml
# config/seed.yaml
data:
  name: seed
  root: ./data/seed
  num_channels: 62
  num_features: 5  # 5 频段 DE 特征
  num_trials: 15
  num_subjects: 15
  
preprocessing:
  sample_rate: 200
  bands:
    delta: [1, 3]
    theta: [4, 7]
    alpha: [8, 13]
    beta: [14, 30]
    gamma: [31, 50]
  
labels:
  task: emotion
  num_classes: 3  # positive/neutral/negative
```

---

## 🏗️ 模型架构 (待实现)

根据 GNN4EEG 论文，模型应包含:

1. **图构建模块**: 从 EEG 通道构建功能连接图
2. **空间特征提取**: GNN 层处理通道间关系
3. **时间特征提取**: TCN/LSTM 处理时间序列
4. **分类头**: 全连接层输出情绪类别

```python
# models/gnn4eeg.py (框架)
class GNN4EEG(nn.Module):
    def __init__(self, num_channels, in_features, hidden_dim, num_classes):
        super().__init__()
        self.graph_constructor = GraphConstructor(num_channels)
        self.spatial_gnn = GraphConv(in_features, hidden_dim)
        self.temporal_net = TCN(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: (batch, channels, time)
        adj = self.graph_constructor(x)
        h = self.spatial_gnn(x, adj)
        h = self.temporal_net(h)
        return self.classifier(h)
```

---

## 📈 训练流程

```python
# train.py 主要流程
1. 加载配置
2. 初始化数据集
3. 构建模型
4. 训练循环:
   - forward pass
   - 计算 loss
   - backward pass
   - 更新权重
   - 记录 metrics
5. 定期评估验证集
6. 保存最佳模型
```

---

## 📝 实验记录

### 待复现结果

| 数据集 | 论文指标 | 复现结果 | 状态 |
|--------|----------|----------|------|
| DEAP (Valence) | Accuracy: XX% | - | ⏳ 待训练 |
| DEAP (Arousal) | Accuracy: XX% | - | ⏳ 待训练 |
| SEED | Accuracy: XX% | - | ⏳ 待训练 |

### 实验配置模板

```yaml
# experiments/exp001.yaml
name: exp001_gnn4eeg_deap
description: 基础复现实验
model: gnn4eeg
dataset: deap
hyperparams:
  lr: 0.001
  batch_size: 32
  epochs: 100
  dropout: 0.5
```

---

## 🔧 开发工具

### 代码检查

```bash
# 代码格式化
black .

# 类型检查
mypy .

# Lint
flake8 .
```

### 实验管理

- 使用 TensorBoard 记录训练曲线
- 使用 WandB (可选) 跟踪实验
- 所有实验结果保存在 `experiments/` 目录

---

## 📚 参考资源

- **论文**: GNN4EEG (待上传)
- **代码库**: https://github.com/AoYuharu/GNN4EEG-Reproduction
- **数据集**: 
  - DEAP: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
  - SEED: http://bcmi.sjtu.edu.cn/~seed/

---

## ⚠️ 注意事项

1. **数据预处理**: 确保预处理方法与论文一致
2. **图构建**: 功能连接计算方法需参考论文
3. **超参数**: 优先使用论文报告的值
4. **随机种子**: 固定随机种子确保可复现性

---

**最后更新**: 2026-03-13 04:58 UTC  
**维护者**: essayanalyser
-e "
## Bot 测试

这是 codingagent-bot 创建的测试分支
"
Bot test
