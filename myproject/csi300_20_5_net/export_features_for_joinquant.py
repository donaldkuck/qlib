#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 Qlib 导出特征值用于 JoinQuant
绕过复杂的因子计算，直接导出模型需要的输入特征
"""

import sys
import pickle
import numpy as np
import pandas as pd
import qlib
from qlib.utils import init_instance_by_config
import yaml
from datetime import datetime

# 初始化 Qlib
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn')

# 加载配置
with open('workflow_config_long_short_strategy.yaml', 'r') as f:
    config = yaml.safe_load(f)

def export_features(dataset_config, output_name, step_len):
    """导出特征"""
    print(f"\n{'='*60}")
    print(f"导出 {output_name} 特征")
    print(f"{'='*60}")
    
    # 初始化数据集
    dataset = init_instance_by_config(dataset_config)
    dataset.setup_data()
    
    # 获取测试集数据
    test_slice = dataset.segments['test']
    print(f"测试集时间段: {test_slice}")
    
    # 获取特征数据
    if hasattr(dataset.handler, '_infer'):
        infer_data = dataset.handler._infer
        feature_data = infer_data['feature']
        
        print(f"\n特征数据形状: {feature_data.shape}")
        print(f"特征数量: {len(feature_data.columns)}")
        
        # 获取股票列表
        instruments = dataset.handler.instruments
        if isinstance(instruments, dict):
            stock_list = list(instruments.keys())
        else:
            stock_list = instruments
        
        print(f"股票数量: {len(stock_list)}")
        print(f"时间步长: {step_len}")
        
        # 导出为字典格式 {stock: {date: features}}
        features_dict = {}
        
        for stock in stock_list:
            try:
                # 获取该股票的所有特征
                stock_data = feature_data.loc[stock] if stock in feature_data.index else None
                if stock_data is None or len(stock_data) == 0:
                    continue
                
                stock_features = {}
                for date, row in stock_data.iterrows():
                    # 获取时间序列特征（最近 step_len 天）
                    date_idx = stock_data.index.get_loc(date)
                    start_idx = max(0, date_idx - step_len + 1)
                    
                    seq_features = stock_data.iloc[start_idx:date_idx+1].values
                    
                    # 如果序列长度不足，用前面的数据填充
                    if len(seq_features) < step_len:
                        padding = np.tile(seq_features[0], (step_len - len(seq_features), 1))
                        seq_features = np.vstack([padding, seq_features])
                    
                    stock_features[date.strftime('%Y-%m-%d')] = seq_features.astype(np.float32)
                
                features_dict[stock] = stock_features
                
            except Exception as e:
                print(f"处理 {stock} 失败: {e}")
                continue
        
        # 保存
        output_file = f'joinquant_{output_name}_features.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(features_dict, f)
        
        print(f"\n✓ 特征已导出到 {output_file}")
        print(f"  包含 {len(features_dict)} 只股票")
        
        return features_dict
    
    return None

# 导出短期特征
short_features = export_features(
    config['task']['short_term_dataset'],
    'short_term',
    config['task']['short_term_dataset']['kwargs'].get('step_len', 20)
)

# 导出长期特征
long_features = export_features(
    config['task']['long_term_dataset'],
    'long_term',
    config['task']['long_term_dataset']['kwargs'].get('step_len', 40)
)

print("\n" + "="*60)
print("导出完成")
print("="*60)
print("\n使用说明:")
print("1. 将生成的 .pkl 文件上传到 JoinQuant 研究环境")
print("2. 在策略中加载特征而不是实时计算")
print("3. 修改 get_predictions 函数直接使用导出的特征")
