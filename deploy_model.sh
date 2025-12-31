#!/bin/bash
# 使用 vLLM 部署训练好的模型

MODEL_PATH="./output/checkpoint_20"
PORT=8037

echo "Deploying model from $MODEL_PATH on port $PORT..."
echo "Make sure vLLM is installed: pip install vllm"
echo ""
echo "注意: vLLM 部署可能因为 CUDA 工具链兼容性问题失败"
echo "如果失败，建议直接使用 Python 脚本: python batch_test.py"
echo ""

# 不设置后端，让 vLLM 自动选择
# 或者你的环境可能需要禁用某些功能来避免兼容性问题

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --port $PORT \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --disable-log-stats \
    --gpu-memory-utilization 0.8 \
    --enforce-eager

# enforce-eager 禁用 CUDA graphs，可以避免一些兼容性问题

# 然后在 test.py 中修改：
# base_url='http://localhost:8037/v1'
# model='./output/checkpoint_20'  # 或者使用完整路径

