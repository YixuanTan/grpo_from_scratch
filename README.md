# GRPO 训练与推理项目

本项目实现了 GRPO (Group Relative Policy Optimization) 算法用于训练语言模型。

## 📁 项目结构

```
.
├── train.py              # GRPO 训练脚本
├── reward_func.py        # 奖励函数定义
├── inference.py          # 统一推理脚本 ⭐
├── output/              # 训练输出目录
│   ├── checkpoint_10/   # 第10步的检查点
│   ├── checkpoint_20/   # 第20步的检查点
│   └── ...
└── runs/                # TensorBoard 日志
```

## 🚀 快速开始

### 1. 训练模型

```bash
# 分布式训练（2个GPU）
torchrun --standalone --nproc_per_node=2 train.py

# 单GPU训练（必须使用 torchrun）
torchrun --standalone --nproc_per_node=1 train.py

# 指定使用第0号GPU进行单GPU训练
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py
```

### 2. 测试模型

使用 `inference.py` 进行推理（推荐）：

```bash
# 单次提问
python inference.py "小明有10个苹果，吃了3个，还剩几个？"

# 批量测试（使用预定义的测试用例）
python inference.py

# 指定不同的 checkpoint
python inference.py "你的问题"  # 默认使用 checkpoint_20
```

### 3. 查看训练日志

```bash
tensorboard --logdir=./runs
```

## 📝 配置说明

### 训练参数 (train.py 中的 GRPOArguments)

```python
output_dir = './output'              # 输出目录
lr = 0.000001                        # 学习率
save_steps = 100                     # 保存间隔
epoch = 3                            # 训练轮数
num_generations = 4                  # 每组生成的样本数
max_prompt_length = 256              # 最大输入长度
max_generate_length = 256            # 最大生成长度
beta = 0.0                           # KL散度系数（0=不使用参考模型）
clip_eps = 0.2                       # PPO裁剪系数
gradient_accumulation_steps = 2      # 梯度累积步数
batch_size = 1                       # 批次大小
```

### 奖励函数 (reward_func.py)

项目使用多个奖励函数组合：
- `correctness_reward`: 答案正确性奖励
- `digit_reward`: 数字提取奖励
- `hard_format_reward`: 格式匹配奖励
- `mark_reward`: 标记符号奖励

## 🛠️ 环境要求

```bash
pip install torch transformers datasets tensorboard
```

## ⚠️ 注意事项

1. **内存管理**：训练过程中会自动进行显存管理和梯度裁剪，防止 OOM 和梯度爆炸。

2. **数值稳定性**：代码中包含多重数值稳定性保护，包括：
   - Logits 裁剪
   - 优势值归一化和裁剪
   - 梯度范数裁剪
   - NaN/Inf 检测和处理

## 📊 模型输出格式

模型按照以下格式回答：

```
<think>
思考过程
</think>
<answer>
最终答案
</answer>
```

## 🤝 贡献

欢迎提交问题和改进建议！

## 📄 许可证

MIT License

