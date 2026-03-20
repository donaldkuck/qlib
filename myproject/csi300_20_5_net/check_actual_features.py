#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查实际训练使用的特征名称和数量
"""

import sys
sys.path.insert(0, '/Users/duckdonald/workspace/myqlib')
sys.path.insert(0, '/Users/duckdonald/workspace/myqlib/myproject')

import qlib
from qlib.utils import init_instance_by_config
import yaml

# 初始化 Qlib
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn')

# 加载配置
with open('workflow_config_long_short_strategy.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("=" * 80)
print("检查短期模型实际使用的特征")
print("=" * 80)

# 初始化短期数据集
short_dataset = init_instance_by_config(config['task']['short_term_dataset'])
short_dataset.setup_data()

# 获取特征列
if hasattr(short_dataset.handler, '_learn') and short_dataset.handler._learn is not None:
    feature_cols = short_dataset.handler._learn['feature'].columns
    print(f"\n实际特征数量: {len(feature_cols)}")
    print(f"\n前30个特征名称:")
    for i, col in enumerate(feature_cols[:30]):
        print(f"  {i+1}. {col}")
    
    # 保存完整列表
    with open('actual_short_features.txt', 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    print(f"\n完整特征列表已保存到 actual_short_features.txt")
else:
    print("无法获取短期特征")

print("\n" + "=" * 80)
print("检查长期模型实际使用的特征")
print("=" * 80)

# 初始化长期数据集
long_dataset = init_instance_by_config(config['task']['long_term_dataset'])
long_dataset.setup_data()

if hasattr(long_dataset.handler, '_learn') and long_dataset.handler._learn is not None:
    feature_cols = long_dataset.handler._learn['feature'].columns
    print(f"\n实际特征数量: {len(feature_cols)}")
    print(f"\n前30个特征名称:")
    for i, col in enumerate(feature_cols[:30]):
        print(f"  {i+1}. {col}")
    
    with open('actual_long_features.txt', 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    print(f"\n完整特征列表已保存到 actual_long_features.txt")
else:
    print("无法获取长期特征")

print("\n" + "=" * 80)
print("模型配置 vs 实际特征数量对比")
print("=" * 80)
short_config = config['task']['short_term_model']['kwargs']['d_feat']
long_config = config['task']['long_term_model']['kwargs']['d_feat']

# 获取实际数量
short_actual = len(short_dataset.handler._learn['feature'].columns) if hasattr(short_dataset.handler, '_learn') and short_dataset.handler._learn is not None else 0
long_actual = len(long_dataset.handler._learn['feature'].columns) if hasattr(long_dataset.handler, '_learn') and long_dataset.handler._learn is not None else 0

print(f"\n短期模型:")
print(f"  配置 d_feat: {short_config}")
print(f"  实际特征数: {short_actual}")
print(f"  匹配: {'✓' if short_config == short_actual else '✗'}")

print(f"\n长期模型:")
print(f"  配置 d_feat: {long_config}")
print(f"  实际特征数: {long_actual}")
print(f"  匹配: {'✓' if long_config == long_actual else '✗'}")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)
if short_config != short_actual or long_config != long_actual:
    print("警告: 模型配置的 d_feat 与实际特征数量不匹配!")
    print("这可能导致模型加载失败或预测不准确。")
    print("\n建议:")
    print("1. 使用实际特征数量重新创建模型")
    print("2. 或者使用预导出的特征进行预测（推荐）")
else:
    print("模型配置与实际特征数量匹配。")
