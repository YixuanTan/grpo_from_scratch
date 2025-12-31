#!/usr/bin/env python3
"""
统一推理脚本 - 支持单次测试和批量测试
使用方法: 
  python inference.py "你的问题"
  python inference.py  # 批量测试模式
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys

SYSTEM_PROMPT = """按照如下格式回答问题：
<think>
你的思考过程
</think>
<answer>
你的回答
</answer>"""

def load_model(checkpoint_path="./output/checkpoint_20"):
    """加载模型"""
    print(f"📦 Loading model from {checkpoint_path}...", file=sys.stderr)
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}\n", file=sys.stderr)
    return model, tokenizer, device

def generate(model, tokenizer, device, question):
    """生成回复"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    return response

def main():
    model, tokenizer, device = load_model()
    
    # 如果提供了命令行参数，则进行单次测试
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"❓ 问题: {question}\n")
        response = generate(model, tokenizer, device, question)
        print(f"💡 回答:\n{response}")
    else:
        # 批量测试模式
        test_cases = [
            "天上五只鸟，地上五只鸡，一共几只鸭？",
            "小明有10个苹果，吃了3个，还剩几个？",
            "1 + 1 = ?",
            "一个数加上5等于12，这个数是多少？",
        ]
        
        print("="*80)
        for i, question in enumerate(test_cases, 1):
            print(f"\n[测试 {i}/{len(test_cases)}]")
            print(f"❓ 问题: {question}")
            print("-"*80)
            response = generate(model, tokenizer, device, question)
            print(f"💡 回答:\n{response}")
            print("="*80)

if __name__ == "__main__":
    main()

