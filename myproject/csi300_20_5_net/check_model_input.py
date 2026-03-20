#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查模型实际期望的输入维度
"""

import pickle
import torch

# 加载模型参数
model_path = 'mlruns/667112072301421907/d5fea5ecca7c4fecaa9ddac8600e388f/artifacts/long_term_params.pkl'

with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

if isinstance(model_data, dict) and 'state_dict' in model_data:
    state_dict = model_data['state_dict']
else:
    state_dict = model_data

print("模型参数信息:")
print("=" * 60)

# 检查第一层权重，确定输入维度
for name, param in state_dict.items():
    print(f"{name}: {param.shape}")
    if 'weight' in name and 'rnn' in name:
        # LSTM/GRU 第一层权重形状: (hidden_size*num_directions, input_size)
        # 或者 (hidden_size, input_size) 取决于实现
        print(f"  -> 输入特征维度: {param.shape[1]}")
        break

print("\n" + "=" * 60)
print("配置对比:")
print(f"  模型期望输入: 查看上面的权重形状")
print(f"  factor_config.yaml 中的因子数: 698")
print(f"  workflow_config 中的 d_feat: 150")
