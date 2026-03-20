#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 Qlib 导出实际训练使用的特征名称和顺序
"""

import sys
import pickle
import qlib
from qlib.utils import init_instance_by_config
import yaml

# 初始化 Qlib
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn')

# 加载配置
with open('workflow_config_long_short_strategy.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 初始化短期数据集
print("=" * 60)
print("获取短期特征列表")
print("=" * 60)
short_dataset = init_instance_by_config(config['task']['short_term_dataset'])
short_dataset.setup_data()

# 获取特征列
if hasattr(short_dataset.handler, '_learn'):
    feature_cols = short_dataset.handler._learn['feature'].columns
    print(f"\n短期特征数量: {len(feature_cols)}")
    print("\n前20个特征:")
    for i, col in enumerate(feature_cols[:20]):
        print(f"  {i+1}. {col}")
    
    # 保存完整列表
    with open('short_term_features.txt', 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    print(f"\n完整列表已保存到 short_term_features.txt")

# 初始化长期数据集
print("\n" + "=" * 60)
print("获取长期特征列表")
print("=" * 60)
long_dataset = init_instance_by_config(config['task']['long_term_dataset'])
long_dataset.setup_data()

if hasattr(long_dataset.handler, '_learn'):
    feature_cols = long_dataset.handler._learn['feature'].columns
    print(f"\n长期特征数量: {len(feature_cols)}")
    print("\n前20个特征:")
    for i, col in enumerate(feature_cols[:20]):
        print(f"  {i+1}. {col}")
    
    with open('long_term_features.txt', 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    print(f"\n完整列表已保存到 long_term_features.txt")
