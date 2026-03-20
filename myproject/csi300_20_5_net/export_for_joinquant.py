#!/usr/bin/env python
"""
导出模型参数供 JoinQuant 使用
从 Qlib 保存的模型文件中提取 state_dict，保存为 pickle 格式
"""

import os
import sys
import pickle
import argparse


def extract_state_dict(input_path, output_path, model_type="short"):
    """
    从 Qlib 模型文件中提取 state_dict
    
    Args:
        input_path: 输入的 .pkl 文件路径 (Qlib 保存的完整模型)
        output_path: 输出的 .pkl 文件路径 (仅 state_dict，pickle格式)
        model_type: "short" 或 "long"
    """
    print(f"\n加载模型: {input_path}")
    
    with open(input_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"模型类型: {type(model)}")
    
    # 尝试提取 state_dict
    state_dict = None
    
    # 情况1: 加载的已经是 state_dict (OrderedDict/dict)
    if isinstance(model, dict):
        state_dict = model
        print(f"  ✓ 加载的是 state_dict，包含 {len(state_dict)} 个参数")
    
    # 情况2: Qlib GATs 使用 GAT_model
    elif hasattr(model, 'GAT_model'):
        print(f"  找到 GAT_model: {type(model.GAT_model)}")
        if hasattr(model.GAT_model, 'state_dict'):
            state_dict = model.GAT_model.state_dict()
            print(f"  ✓ 从 GAT_model 提取 state_dict")
    
    # 情况3: 某些版本可能使用 model
    elif hasattr(model, 'model'):
        print(f"  找到 model: {type(model.model)}")
        if hasattr(model.model, 'state_dict'):
            state_dict = model.model.state_dict()
            print(f"  ✓ 从 model 提取 state_dict")
    
    # 情况4: 直接检查是否是 nn.Module
    elif hasattr(model, 'state_dict'):
        state_dict = model.state_dict()
        print(f"  ✓ 从 model 直接提取 state_dict")
    
    if state_dict is None:
        print(f"  ✗ 无法提取 state_dict")
        print(f"  模型属性: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        return False
    
    # 转换为纯 Python 字典（避免 PyTorch 特定的 pickle 问题）
    import torch
    state_dict_cpu = {}
    for key, value in state_dict.items():
        # 确保所有张量都在 CPU 上
        if isinstance(value, torch.Tensor):
            state_dict_cpu[key] = value.cpu()
        else:
            state_dict_cpu[key] = value
    
    # 保存为 pickle 格式（协议 4，兼容性最好）
    with open(output_path, 'wb') as f:
        pickle.dump(state_dict_cpu, f, protocol=4)
    
    print(f"  ✓ 已保存到: {output_path}")
    
    # 验证
    with open(output_path, 'rb') as f:
        loaded = pickle.load(f)
    print(f"  ✓ 验证成功，包含 {len(loaded)} 个参数")
    for key in list(loaded.keys())[:5]:
        print(f"    - {key}: {loaded[key].shape}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='导出 JoinQuant 模型')
    parser.add_argument('--short-input', type=str, help='短期模型输入文件 (.pkl)')
    parser.add_argument('--long-input', type=str, help='长期模型输入文件 (.pkl)')
    parser.add_argument('--output-dir', type=str, default='.', help='输出目录')
    
    args = parser.parse_args()
    
    success = True
    
    if args.short_input and os.path.exists(args.short_input):
        output = os.path.join(args.output_dir, 'short_term_state.pkl')
        if not extract_state_dict(args.short_input, output, "short"):
            success = False
    elif args.short_input:
        print(f"错误: 找不到文件 {args.short_input}")
        success = False
    
    if args.long_input and os.path.exists(args.long_input):
        output = os.path.join(args.output_dir, 'long_term_state.pkl')
        if not extract_state_dict(args.long_input, output, "long"):
            success = False
    elif args.long_input:
        print(f"错误: 找不到文件 {args.long_input}")
        success = False
    
    if success:
        print("\n" + "="*50)
        print("✓ 导出完成，请上传以下文件到 JoinQuant:")
        print("  - short_term_state.pkl")
        print("  - long_term_state.pkl")
        print("="*50)
    else:
        print("\n✗ 导出失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
