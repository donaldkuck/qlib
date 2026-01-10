"""
分析自定义因子在沪深500上的相关性
支持自定义label表达式
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from ruamel.yaml import YAML
warnings.filterwarnings('ignore')

# 使用系统安装的qlib
import sys
from pathlib import Path

# 移除当前目录下的qlib路径（如果存在）
current_dir = Path(__file__).parent.absolute()
qlib_local_path = current_dir.parent / 'qlib'
# 从sys.path中移除本地qlib目录
if str(current_dir.parent) in sys.path:
    sys.path.remove(str(current_dir.parent))
if str(qlib_local_path) in sys.path:
    sys.path.remove(str(qlib_local_path))

import qlib
from qlib.constant import REG_CN
from qlib.contrib.eva.alpha import calc_ic  # 使用qlib自带的calc_ic函数
from qlib.data import D
from qlib.utils import exists_qlib_data
from qlib.log import get_module_logger

logger = get_module_logger("自定义因子分析")


def load_factor_config(config_path=None):
    """加载因子配置文件"""
    if config_path is None:
        config_path = Path(__file__).parent / "factor_config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        # 如果配置文件不存在，创建默认配置
        default_config = {
            'good_factors': {},
            'bad_factors': {},
            'min_ic_threshold': 0.02,
            'min_icir_threshold': 0.1
        }
        save_factor_config(default_config, config_path)
        return default_config
    
    yaml = YAML(typ="safe", pure=True)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f)
    
    # 如果config是None，创建默认配置
    if config is None:
        config = {}
    
    # 确保所有键都存在，并且不是None
    if 'good_factors' not in config or config.get('good_factors') is None:
        config['good_factors'] = {}
    if 'bad_factors' not in config or config.get('bad_factors') is None:
        config['bad_factors'] = {}
    if 'min_ic_threshold' not in config or config.get('min_ic_threshold') is None:
        config['min_ic_threshold'] = 0.02
    if 'min_icir_threshold' not in config or config.get('min_icir_threshold') is None:
        config['min_icir_threshold'] = 0.1
    
    # 确保good_factors和bad_factors是字典类型
    if not isinstance(config['good_factors'], dict):
        config['good_factors'] = {}
    if not isinstance(config['bad_factors'], dict):
        config['bad_factors'] = {}
    
    return config


def save_factor_config(config, config_path=None):
    """保存因子配置文件"""
    if config_path is None:
        config_path = Path(__file__).parent / "factor_config.yaml"
    else:
        config_path = Path(config_path)
    
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    print(f"✓ 因子配置已保存到: {config_path}")


def update_factor_config(new_good_factors=None, new_bad_factors=None, config_path=None):
    """更新因子配置文件"""
    config = load_factor_config(config_path)
    
    if new_good_factors:
        config['good_factors'].update(new_good_factors)
        print(f"✓ 添加了 {len(new_good_factors)} 个好因子到配置")
    
    if new_bad_factors:
        config['bad_factors'].update(new_bad_factors)
        print(f"✓ 添加了 {len(new_bad_factors)} 个坏因子到配置")
    
    save_factor_config(config, config_path)
    return config


def deduplicate_good_factors_by_correlation(
    good_factors,
    instruments_list,
    start_time="2020-01-01",
    end_time="2025-01-01",
    freq="day",
    correlation_threshold=0.85,
    ic_results=None
):
    """
    基于相关性去除冗余的好因子，只保留效果最好的
    
    Parameters
    ----------
    good_factors : dict
        好因子字典
    instruments_list : list
        股票列表
    start_time : str
        开始时间
    end_time : str
        结束时间
    freq : str
        数据频率
    correlation_threshold : float
        相关性阈值，超过此值的因子对视为冗余（默认0.85）
    ic_results : pd.DataFrame, optional
        因子IC结果，如果提供则使用，否则需要重新计算
        
    Returns
    -------
    dict
        去重后的好因子字典
    """
    if len(good_factors) <= 1:
        return good_factors, {}
    
    print(f"\n开始分析好因子相关性（阈值: {correlation_threshold}）...")
    print(f"原始好因子数量: {len(good_factors)}")
    
    # 如果没有提供IC结果，需要先计算IC（简化版，只计算IC用于比较）
    if ic_results is None:
        print("正在计算因子IC值用于比较...")
        factor_ic_map = {}
        factor_names = list(good_factors.keys())
        
        # 批量获取因子数据
        try:
            # 支持字典格式和列表格式（方法1：固定股票池）
            all_factor_data = D.features(
                instruments_list,  # 可以是字典或列表
                list(good_factors.values()),
                start_time=start_time,
                end_time=end_time,
                freq=freq
            )
            
            # 获取标签数据用于计算IC
            label_data = D.features(
                instruments_list,  # 可以是字典或列表
                ["Ref($close, -2)/Ref($open, -1) - 1"],
                start_time=start_time,
                end_time=end_time,
                freq=freq
            )
            
            if len(label_data) > 0:
                if isinstance(label_data.columns, pd.MultiIndex):
                    labels = label_data[label_data.columns[0]]
                else:
                    labels = label_data.iloc[:, 0]
                
                # 计算每个因子的IC
                for i, factor_name in enumerate(factor_names):
                    try:
                        if isinstance(all_factor_data.columns, pd.MultiIndex):
                            factor_col = all_factor_data.columns[i]
                            factor_values = all_factor_data[factor_col]
                        else:
                            factor_values = all_factor_data.iloc[:, i]
                        
                        # 对齐数据
                        common_index = factor_values.index.intersection(labels.index)
                        if len(common_index) > 0:
                            aligned_factor = factor_values.loc[common_index]
                            aligned_label = labels.loc[common_index]
                            valid_mask = aligned_factor.notna() & aligned_label.notna()
                            aligned_factor = aligned_factor[valid_mask]
                            aligned_label = aligned_label[valid_mask]
                            
                            if len(aligned_factor) > 10:
                                ic_series, _ = calc_ic(
                                    pred=aligned_factor,
                                    label=aligned_label,
                                    date_col="datetime",
                                    dropna=True
                                )
                                if len(ic_series) > 0:
                                    factor_ic_map[factor_name] = abs(ic_series.mean())
                    except Exception as e:
                        print(f"  计算 {factor_name} 的IC失败: {e}")
                        factor_ic_map[factor_name] = 0.0
        except Exception as e:
            print(f"⚠️  批量计算IC失败: {e}，使用默认IC值")
            # 如果批量计算失败，给每个因子一个默认IC值
            factor_ic_map = {name: 0.03 for name in factor_names}
    else:
        # 使用提供的IC结果
        factor_ic_map = {}
        for _, row in ic_results.iterrows():
            factor_name = row['factor']
            if factor_name in good_factors:
                factor_ic_map[factor_name] = abs(row.get('IC', 0.0))
    
    # 获取因子数据计算相关性
    print("正在计算因子相关性矩阵...")
    factor_names = list(good_factors.keys())
    
    # 构建因子值矩阵（按日期和股票）
    factor_matrix_dict = {}
    
    try:
        # 批量获取所有因子数据（方法1：固定股票池）
        all_factor_data = D.features(
            instruments_list,  # 可以是字典或列表
            list(good_factors.values()),
            start_time=start_time,
            end_time=end_time,
            freq=freq
        )
        
        if len(all_factor_data) == 0:
            print("⚠️  无法获取因子数据，跳过相关性分析")
            return good_factors, {}
        
        # 提取每个因子的数据并构建矩阵
        for i, factor_name in enumerate(factor_names):
            try:
                if isinstance(all_factor_data.columns, pd.MultiIndex):
                    factor_col = all_factor_data.columns[i]
                    factor_values = all_factor_data[factor_col]
                else:
                    factor_values = all_factor_data.iloc[:, i]
                
                # 如果是MultiIndex，转换为DataFrame便于计算相关性
                if isinstance(factor_values.index, pd.MultiIndex):
                    # 转换为宽表：日期为行，股票为列
                    factor_df = factor_values.reset_index()
                    if 'datetime' in factor_df.columns and 'instrument' in factor_df.columns:
                        factor_pivot = factor_df.pivot(index='datetime', columns='instrument', values=factor_values.name if hasattr(factor_values, 'name') else factor_df.columns[-1])
                        # 展平为Series（按日期平均，用于简化相关性计算）
                        factor_matrix_dict[factor_name] = factor_pivot.mean(axis=1)
                    else:
                        # 如果列名不对，尝试其他方式
                        factor_matrix_dict[factor_name] = factor_values.groupby(level='datetime').mean()
                else:
                    factor_matrix_dict[factor_name] = factor_values
            except Exception as e:
                print(f"  ⚠️  处理因子 {factor_name} 数据失败: {e}")
                continue
    except Exception as e:
        print(f"⚠️  批量获取因子数据失败: {e}")
        print("跳过相关性分析，返回原始因子")
        return good_factors, {}
    
    # 计算相关性矩阵
    if len(factor_matrix_dict) < 2:
        print("因子数据不足，跳过相关性分析")
        return good_factors, {}
    
    # 对齐所有因子的时间索引
    common_dates = None
    for factor_name, factor_series in factor_matrix_dict.items():
        dates = set(factor_series.index)
        if common_dates is None:
            common_dates = dates
        else:
            common_dates = common_dates.intersection(dates)
    
    if len(common_dates) < 10:
        print("共同日期不足，跳过相关性分析")
        return good_factors, {}
    
    # 构建对齐后的因子矩阵
    aligned_factor_matrix = {}
    for factor_name, factor_series in factor_matrix_dict.items():
        aligned_factor_matrix[factor_name] = factor_series.loc[factor_series.index.intersection(common_dates)]
    
    # 计算相关性矩阵
    try:
        correlation_matrix = pd.DataFrame(aligned_factor_matrix).corr()
    except Exception as e:
        print(f"⚠️  计算相关性矩阵失败: {e}")
        print("跳过相关性分析，返回原始因子")
        return good_factors, {}
    
    # 找出高相关性的因子对
    factors_to_remove = set()
    factor_pairs_checked = set()
    
    print(f"\n发现高相关性因子对（相关性 > {correlation_threshold}）:")
    for i, factor1 in enumerate(correlation_matrix.index):
        for j, factor2 in enumerate(correlation_matrix.columns):
            if i >= j:  # 只检查上三角矩阵
                continue
            
            if factor1 not in good_factors or factor2 not in good_factors:
                continue
            
            pair_key = tuple(sorted([factor1, factor2]))
            if pair_key in factor_pairs_checked:
                continue
            factor_pairs_checked.add(pair_key)
            
            corr_value = correlation_matrix.loc[factor1, factor2]
            if abs(corr_value) > correlation_threshold:
                # 比较IC绝对值，保留IC绝对值更大的因子
                # 注意：factor_ic_map中存储的已经是IC的绝对值（见第211行）
                ic1_abs = factor_ic_map.get(factor1, 0.0)  # IC绝对值
                ic2_abs = factor_ic_map.get(factor2, 0.0)  # IC绝对值
                
                if ic1_abs > ic2_abs:
                    # factor1的IC绝对值更大，保留factor1，移除factor2
                    factors_to_remove.add(factor2)
                    print(f"  {factor1:40s} (|IC|={ic1_abs:.6f}) vs {factor2:40s} (|IC|={ic2_abs:.6f}) | 相关性: {corr_value:.4f} -> 移除 {factor2}（保留IC更大的）")
                elif ic2_abs > ic1_abs:
                    # factor2的IC绝对值更大，保留factor2，移除factor1
                    factors_to_remove.add(factor1)
                    print(f"  {factor1:40s} (|IC|={ic1_abs:.6f}) vs {factor2:40s} (|IC|={ic2_abs:.6f}) | 相关性: {corr_value:.4f} -> 移除 {factor1}（保留IC更大的）")
                else:
                    # IC绝对值相同，保留名称更短的（通常更基础）
                    if len(factor1) <= len(factor2):
                        factors_to_remove.add(factor2)
                        print(f"  {factor1:40s} vs {factor2:40s} | 相关性: {corr_value:.4f} | |IC|相同（{ic1_abs:.6f}）-> 移除 {factor2}（保留名称更短的）")
                    else:
                        factors_to_remove.add(factor1)
                        print(f"  {factor1:40s} vs {factor2:40s} | 相关性: {corr_value:.4f} | |IC|相同（{ic1_abs:.6f}）-> 移除 {factor1}（保留名称更短的）")
    
    # 创建去重后的因子字典
    deduplicated_factors = {
        name: expr for name, expr in good_factors.items()
        if name not in factors_to_remove
    }
    
    # 创建被去除的因子字典（用于加入到bad_factors）
    removed_factors = {
        name: expr for name, expr in good_factors.items()
        if name in factors_to_remove
    }
    
    removed_count = len(good_factors) - len(deduplicated_factors)
    print(f"\n✓ 相关性去重完成:")
    print(f"  - 移除因子数: {removed_count}")
    print(f"  - 保留因子数: {len(deduplicated_factors)}")
    
    return deduplicated_factors, removed_factors


def generate_new_factors_from_good_ones(good_factors, bad_factors, max_new_factors=200):
    """
    基于好因子生成新的组合因子（发散思路，生成多样化因子）
    
    Parameters
    ----------
    good_factors : dict
        好因子字典
    bad_factors : dict
        坏因子字典（避免重复）
    max_new_factors : int
        最多生成的新因子数量
        
    Returns
    -------
    dict
        新生成的因子字典
    """
    new_factors = {}
    tried_factors = {**good_factors, **bad_factors}  # 所有已尝试的因子
    
    print(f"  已尝试因子数: {len(tried_factors)}")
    print(f"  好因子数: {len(good_factors)}")
    print(f"  坏因子数: {len(bad_factors)}")
    
    # 提取基础因子组件
    base_components = {
        'Price_Change_1': '($close - Ref($close, 1)) / Ref($close, 1)',
        'Price_Change_2': '($close - Ref($close, 2)) / Ref($close, 2)',
        'Price_Change_3': '($close - Ref($close, 3)) / Ref($close, 3)',
        'Price_Change_4': '($close - Ref($close, 4)) / Ref($close, 4)',
        'Price_Change_5': '($close - Ref($close, 5)) / Ref($close, 5)',
        'MA_Ratio_3': '$close / Mean($close, 3) - 1',
        'MA_Ratio_5': '$close / Mean($close, 5) - 1',
        'MA_Ratio_7': '$close / Mean($close, 7) - 1',
        'MA_Ratio_10': '$close / Mean($close, 10) - 1',
        'MA_Ratio_15': '$close / Mean($close, 15) - 1',
        'MA_Ratio_20': '$close / Mean($close, 20) - 1',
        'Slope_Close_3': 'Slope($close, 3) / $close',
        'Slope_Close_5': 'Slope($close, 5) / $close',
        'Slope_Close_7': 'Slope($close, 7) / $close',
        'CumSum_Return_3': 'Sum(($close - Ref($close, 1)) / Ref($close, 1), 3)',
        'CumSum_Return_5': 'Sum(($close - Ref($close, 1)) / Ref($close, 1), 5)',
        'CumSum_Return_7': 'Sum(($close - Ref($close, 1)) / Ref($close, 1), 7)',
        'If_Up_Return_3': 'If($close > Ref($close, 3), ($close - Ref($close, 3)) / Ref($close, 3), 0)',
        'If_Up_Return_5': 'If($close > Ref($close, 5), ($close - Ref($close, 5)) / Ref($close, 5), 0)',
        'Max_Gain_5': '(($close - Min($close, 5)) / Min($close, 5))',
        'Max_Gain_7': '(($close - Min($close, 7)) / Min($close, 7))',
        'Resi_Close_7': 'Resi($close, 7) / $close',
        'Resi_Close_5': 'Resi($close, 5) / $close',
    }
    
    # 基于TOP好因子生成新组合
    top_good_factors = [
        'Price_Change_1_MA3_MA5',
        'MA_Ratio_3_5_Price_Change_1',
        'CumSum_Return_3_MA3_Price_Change_1',
        'Price_Change_1_MA3_CumSum3',
        'Price_Change_1_MA3_Slope3',
        'Price_Change_1_3_MA3',
        'Price_Change_1_3_MA5',
        'Slope_Close_3_MA3_MA5',
        'MA_Ratio_3_5_Slope3',
    ]
    
    # 策略1: 基于TOP因子添加新的时间窗口
    for top_factor_name in top_good_factors[:5]:  # 只取前5个TOP因子
        if top_factor_name in good_factors:
            expr = good_factors[top_factor_name]
            # 尝试不同的时间窗口组合
            for ma_period in [20, 30, 60]:
                new_name = f"{top_factor_name}_MA{ma_period}"
                if new_name not in tried_factors:
                    new_expr = f"{expr} * ($close / Mean($close, {ma_period}) - 1)"
                    new_factors[new_name] = new_expr
                    if len(new_factors) >= max_new_factors:
                        return new_factors
    
    # 策略2: 基于Price_Change_1_MA3_MA5生成更深层组合
    if 'Price_Change_1_MA3_MA5' in good_factors:
        base_expr = good_factors['Price_Change_1_MA3_MA5']
        # 添加更多组件
        combinations = [
            ('Slope3', '(Slope($close, 3) / $close)'),
            ('CumSum3', 'Sum(($close - Ref($close, 1)) / Ref($close, 1), 3)'),
            ('Price_Change_2', '(($close - Ref($close, 2)) / Ref($close, 2))'),
            ('MA7', '($close / Mean($close, 7) - 1)'),
            ('MA10', '($close / Mean($close, 10) - 1)'),
            ('Max_Gain_5', '(($close - Min($close, 5)) / Min($close, 5))'),
            ('Resi_7', '(Resi($close, 7) / $close)'),
        ]
        for suffix, component_expr in combinations:
            new_name = f"Price_Change_1_MA3_MA5_{suffix}"
            if new_name not in tried_factors:
                new_expr = f"{base_expr} * {component_expr}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略3: 基于MA_Ratio_3_5_Price_Change_1生成新组合
    if 'MA_Ratio_3_5_Price_Change_1' in good_factors:
        base_expr = good_factors['MA_Ratio_3_5_Price_Change_1']
        combinations = [
            ('Slope3', '(Slope($close, 3) / $close)'),
            ('CumSum3', 'Sum(($close - Ref($close, 1)) / Ref($close, 1), 3)'),
            ('Price_Change_2', '(($close - Ref($close, 2)) / Ref($close, 2))'),
            ('MA7', '($close / Mean($close, 7) - 1)'),
            ('Max_Gain_5', '(($close - Min($close, 5)) / Min($close, 5))'),
        ]
        for suffix, component_expr in combinations:
            new_name = f"MA_Ratio_3_5_Price_Change_1_{suffix}"
            if new_name not in tried_factors:
                new_expr = f"{base_expr} * {component_expr}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略4: 生成新的Price_Change组合
    price_change_combos = [
        ('Price_Change_1', 'Price_Change_5', 'Price_Change_1_5'),
        ('Price_Change_2', 'Price_Change_4', 'Price_Change_2_4'),
        ('Price_Change_1', 'Price_Change_2', 'Price_Change_3', 'Price_Change_4', 'Price_Change_1_2_3_4'),
    ]
    for combo in price_change_combos:
        if len(combo) == 3:
            name1, name2, new_name = combo
            if new_name not in tried_factors and name1 in base_components and name2 in base_components:
                new_expr = f"{base_components[name1]} * {base_components[name2]}"
                new_factors[new_name] = new_expr
        elif len(combo) == 5:
            name1, name2, name3, name4, new_name = combo
            if new_name not in tried_factors:
                if all(n in base_components for n in [name1, name2, name3, name4]):
                    new_expr = f"{base_components[name1]} * {base_components[name2]} * {base_components[name3]} * {base_components[name4]}"
                    new_factors[new_name] = new_expr
        if len(new_factors) >= max_new_factors:
            return new_factors
    
    # 策略5: 生成新的MA_Ratio组合
    ma_combos = [
        ('MA_Ratio_3', 'MA_Ratio_5', 'MA_Ratio_10', 'MA_Ratio_3_5_10'),
        ('MA_Ratio_5', 'MA_Ratio_7', 'MA_Ratio_10', 'MA_Ratio_5_7_10'),
        ('MA_Ratio_3', 'MA_Ratio_7', 'MA_Ratio_15', 'MA_Ratio_3_7_15'),
        ('MA_Ratio_5', 'MA_Ratio_10', 'MA_Ratio_15', 'MA_Ratio_5_10_15'),
    ]
    for combo in ma_combos:
        if len(combo) == 4:
            name1, name2, name3, new_name = combo
            if new_name not in tried_factors:
                expr1 = base_components.get(name1, '')
                expr2 = base_components.get(name2, '')
                expr3 = base_components.get(name3, '')
                if expr1 and expr2 and expr3:
                    new_expr = f"{expr1} * {expr2} * {expr3}"
                    new_factors[new_name] = new_expr
        if len(new_factors) >= max_new_factors:
            return new_factors
    
    # 策略6: Price_Change + MA_Ratio + 其他技术指标的新组合
    tech_indicators = [
        ('Slope_Close_5', 'Slope($close, 5) / $close'),
        ('Slope_Close_7', 'Slope($close, 7) / $close'),
        ('CumSum_Return_5', 'Sum(($close - Ref($close, 1)) / Ref($close, 1), 5)'),
        ('CumSum_Return_7', 'Sum(($close - Ref($close, 1)) / Ref($close, 1), 7)'),
        ('If_Up_Return_5', 'If($close > Ref($close, 5), ($close - Ref($close, 5)) / Ref($close, 5), 0)'),
        ('Max_Gain_7', '(($close - Min($close, 7)) / Min($close, 7))'),
        ('Resi_Close_5', 'Resi($close, 5) / $close'),
    ]
    
    for pc_name, pc_expr in [('Price_Change_1', base_components['Price_Change_1']),
                              ('Price_Change_2', base_components['Price_Change_2']),
                              ('Price_Change_3', base_components['Price_Change_3'])]:
        for ma_name, ma_expr in [('MA_Ratio_3', base_components['MA_Ratio_3']),
                                 ('MA_Ratio_5', base_components['MA_Ratio_5']),
                                 ('MA_Ratio_7', base_components['MA_Ratio_7'])]:
            for tech_name, tech_expr in tech_indicators:
                new_name = f"{pc_name}_{ma_name}_{tech_name}"
                if new_name not in tried_factors:
                    new_expr = f"{pc_expr} * {ma_expr} * {tech_expr}"
                    new_factors[new_name] = new_expr
                    if len(new_factors) >= max_new_factors:
                        return new_factors
    
    # 策略7: 基于表现好的因子，尝试除法组合（而非乘法）
    if 'Price_Change_1_MA3_MA5' in good_factors:
        base_expr = good_factors['Price_Change_1_MA3_MA5']
        # 尝试除法组合
        div_combos = [
            ('Slope3', '(Slope($close, 3) / $close)'),
            ('CumSum3', 'Sum(($close - Ref($close, 1)) / Ref($close, 1), 3)'),
        ]
        for suffix, component_expr in div_combos:
            new_name = f"Price_Change_1_MA3_MA5_Div_{suffix}"
            if new_name not in tried_factors:
                new_expr = f"{base_expr} / ({component_expr} + 0.0001)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略8: 尝试不同的时间窗口（5, 7, 10, 15, 20）
    time_windows = [5, 7, 10, 15, 20]
    for window in time_windows:
        # Price_Change + MA_Ratio + Slope
        new_name = f"Price_Change_1_MA{window}_Slope{window}"
        if new_name not in tried_factors:
            new_expr = f"($close - Ref($close, 1)) / Ref($close, 1) * ($close / Mean($close, {window}) - 1) * (Slope($close, {window}) / $close)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # MA_Ratio组合
        for window2 in time_windows:
            if window < window2:
                new_name = f"MA_Ratio_{window}_{window2}"
                if new_name not in tried_factors:
                    new_expr = f"($close / Mean($close, {window}) - 1) * ($close / Mean($close, {window2}) - 1)"
                    new_factors[new_name] = new_expr
                    if len(new_factors) >= max_new_factors:
                        return new_factors
    
    # === 发散思路：生成更多样化的因子 ===
    
    # 策略9: 减法组合（而非乘法）- 捕捉差异
    for pc_name, pc_expr in [('Price_Change_1', base_components['Price_Change_1']),
                              ('Price_Change_2', base_components['Price_Change_2'])]:
        for ma_name, ma_expr in [('MA_Ratio_3', base_components['MA_Ratio_3']),
                                 ('MA_Ratio_5', base_components['MA_Ratio_5'])]:
            new_name = f"{pc_name}_Minus_{ma_name}"
            if new_name not in tried_factors:
                new_expr = f"{pc_expr} - {ma_expr}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略10: 除法组合（捕捉比率关系）
    for pc_name, pc_expr in [('Price_Change_1', base_components['Price_Change_1']),
                              ('Price_Change_2', base_components['Price_Change_2'])]:
        for ma_name, ma_expr in [('MA_Ratio_3', base_components['MA_Ratio_3']),
                                 ('MA_Ratio_5', base_components['MA_Ratio_5'])]:
            new_name = f"{pc_name}_Div_{ma_name}"
            if new_name not in tried_factors:
                new_expr = f"{pc_expr} / ({ma_expr} + 0.0001)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略11: 价格位置因子（RSV变种）
    rsv_windows = [3, 5, 7, 10, 15, 20]
    for window in rsv_windows:
        # 标准RSV
        new_name = f"RSV_{window}"
        if new_name not in tried_factors:
            new_expr = f"($close - Min($low, {window})) / (Max($high, {window}) - Min($low, {window}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # RSV与MA_Ratio组合
        for ma_window in [3, 5, 7]:
            new_name = f"RSV_{window}_MA{ma_window}"
            if new_name not in tried_factors:
                new_expr = f"(($close - Min($low, {window})) / (Max($high, {window}) - Min($low, {window}) + 0.0001)) * ($close / Mean($close, {ma_window}) - 1)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略12: 波动率相关因子
    vol_windows = [3, 5, 7, 10, 20]
    for window in vol_windows:
        # 波动率标准化
        new_name = f"Volatility_Norm_{window}"
        if new_name not in tried_factors:
            new_expr = f"Std($close, {window}) / (Mean($close, {window}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 波动率与价格变化组合
        new_name = f"Price_Change_1_Vol_{window}"
        if new_name not in tried_factors:
            new_expr = f"($close - Ref($close, 1)) / Ref($close, 1) / (Std($close, {window}) / $close + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略13: 相对强度因子（与市场/平均比较）
    for window in [5, 10, 20]:
        # 价格相对位置
        new_name = f"Price_Position_{window}"
        if new_name not in tried_factors:
            new_expr = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 价格位置与MA_Ratio组合
        new_name = f"Price_Position_{window}_MA{window}"
        if new_name not in tried_factors:
            new_expr = f"(($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)) * ($close / Mean($close, {window}) - 1)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略14: 动量与反转的组合
    momentum_windows = [3, 5, 7, 10]
    reversal_windows = [1, 2, 3]
    for mom_win in momentum_windows:
        for rev_win in reversal_windows:
            new_name = f"Momentum{mom_win}_Reversal{rev_win}"
            if new_name not in tried_factors:
                new_expr = f"(($close - Ref($close, {mom_win})) / Ref($close, {mom_win})) * (Ref($close, {rev_win}) / $close - 1)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略15: 分位数相关因子
    quantile_windows = [5, 10, 20]
    quantile_levels = [0.2, 0.25, 0.3, 0.7, 0.75, 0.8]
    for window in quantile_windows:
        for level in quantile_levels:
            level_name = str(int(level * 100))
            new_name = f"Quantile_{level_name}_{window}"
            if new_name not in tried_factors:
                new_expr = f"Quantile($close, {window}, {level}) / $close"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
            
            # 分位数与MA_Ratio组合
            new_name = f"Quantile_{level_name}_{window}_MA{window}"
            if new_name not in tried_factors:
                new_expr = f"(Quantile($close, {window}, {level}) / $close) * ($close / Mean($close, {window}) - 1)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略16: 价格加速度（二阶导数）
    for window in [3, 5, 7]:
        new_name = f"Price_Acceleration_{window}"
        if new_name not in tried_factors:
            new_expr = f"(($close - Ref($close, 1)) / Ref($close, 1)) - (Ref($close, 1) - Ref($close, 2)) / Ref($close, 2)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 加速度与MA_Ratio组合
        new_name = f"Price_Acceleration_{window}_MA{window}"
        if new_name not in tried_factors:
            new_expr = f"((($close - Ref($close, 1)) / Ref($close, 1)) - (Ref($close, 1) - Ref($close, 2)) / Ref($close, 2)) * ($close / Mean($close, {window}) - 1)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略17: 价格变化率的变化（变化的变化）
    for window in [3, 5]:
        new_name = f"Price_Change_Rate_{window}"
        if new_name not in tried_factors:
            new_expr = f"(($close - Ref($close, {window})) / Ref($close, {window})) - (Ref($close, {window}) - Ref($close, {window * 2})) / Ref($close, {window * 2})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略18: MA交叉信号
    ma_pairs = [(3, 5), (5, 10), (10, 20), (3, 10), (5, 20)]
    for ma1, ma2 in ma_pairs:
        new_name = f"MA_Cross_{ma1}_{ma2}"
        if new_name not in tried_factors:
            new_expr = f"(Mean($close, {ma1}) - Mean($close, {ma2})) / (Mean($close, {ma2}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # MA交叉与价格变化组合
        new_name = f"MA_Cross_{ma1}_{ma2}_Price_Change_1"
        if new_name not in tried_factors:
            new_expr = f"((Mean($close, {ma1}) - Mean($close, {ma2})) / (Mean($close, {ma2}) + 0.0001)) * (($close - Ref($close, 1)) / Ref($close, 1))"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略19: 价格与均线的距离（标准化）
    for window in [3, 5, 7, 10, 20]:
        new_name = f"Price_MA_Distance_{window}"
        if new_name not in tried_factors:
            new_expr = f"($close - Mean($close, {window})) / (Std($close, {window}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略20: 条件因子（基于价格位置的条件组合）
    for window in [5, 10]:
        # 如果价格在低位，则使用动量；如果在高位，则使用反转
        new_name = f"Conditional_Momentum_Reversal_{window}"
        if new_name not in tried_factors:
            price_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            momentum = f"($close - Ref($close, 3)) / Ref($close, 3)"
            reversal = f"(Ref($close, 3) / $close - 1)"
            new_expr = f"If({price_pos} < 0.3, {momentum}, If({price_pos} > 0.7, {reversal}, 0))"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略21: 价格变化的一致性（连续上涨/下跌）
    for window in [3, 5]:
        new_name = f"Price_Consistency_{window}"
        if new_name not in tried_factors:
            new_expr = f"Mean(($close > Ref($close, 1)), {window}) - Mean(($close < Ref($close, 1)), {window})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略22: 价格变化幅度（而非方向）
    for window in [3, 5, 7]:
        new_name = f"Price_Change_Magnitude_{window}"
        if new_name not in tried_factors:
            new_expr = f"Mean(Abs(($close - Ref($close, 1)) / Ref($close, 1)), {window})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略23: 价格与成交量的协同性（如果有成交量数据）
    # 注意：这里假设有$volume字段
    for window in [5, 10]:
        new_name = f"Price_Volume_Sync_{window}"
        if new_name not in tried_factors:
            new_expr = f"Corr(($close - Ref($close, 1)) / Ref($close, 1), ($volume - Ref($volume, 1)) / Ref($volume, 1), {window})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略24: 价格变化的分位数（在历史变化中的位置）
    for window in [10, 20]:
        new_name = f"Price_Change_Quantile_{window}"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            new_expr = f"Quantile({price_change}, {window}, 0.5)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略25: 多时间框架组合（短期vs长期）
    short_windows = [3, 5]
    long_windows = [10, 20]
    for short_win in short_windows:
        for long_win in long_windows:
            new_name = f"Short_Long_Ratio_{short_win}_{long_win}"
            if new_name not in tried_factors:
                new_expr = f"($close / Mean($close, {short_win}) - 1) / ($close / Mean($close, {long_win}) - 1 + 0.0001)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略26: 价格变化的累积效应
    for window in [5, 10]:
        new_name = f"Cumulative_Price_Change_{window}"
        if new_name not in tried_factors:
            new_expr = f"Sum(($close - Ref($close, 1)) / Ref($close, 1), {window}) / {window}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略27: 价格变化的稳定性（低波动时的信号更强）
    for window in [5, 10]:
        new_name = f"Price_Change_Stability_{window}"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            volatility = f"Std($close, {window}) / $close"
            new_expr = f"{price_change} / ({volatility} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略28: 价格突破因子（突破近期高低点，改进：始终有值）
    for window in [5, 10, 20]:
        # 突破高点（突破时为正，未突破时为负，始终有值）
        new_name = f"Breakout_High_{window}"
        if new_name not in tried_factors:
            new_expr = f"($close - Max($close, {window})) / (Max($close, {window}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 突破低点（突破时为正，未突破时为负，始终有值）
        new_name = f"Breakout_Low_{window}"
        if new_name not in tried_factors:
            new_expr = f"(Min($close, {window}) - $close) / (Min($close, {window}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略29: 价格变化的滞后效应
    for lag in [1, 2, 3]:
        for window in [3, 5]:
            new_name = f"Lag{lag}_Price_Change_{window}"
            if new_name not in tried_factors:
                new_expr = f"Ref(($close - Ref($close, {window})) / Ref($close, {window}), {lag})"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略30: 价格变化的周期性（不同周期的组合）
    cycles = [(3, 7), (5, 10), (7, 14)]
    for cycle1, cycle2 in cycles:
        new_name = f"Price_Cycle_{cycle1}_{cycle2}"
        if new_name not in tried_factors:
            change1 = f"($close - Ref($close, {cycle1})) / Ref($close, {cycle1})"
            change2 = f"($close - Ref($close, {cycle2})) / Ref($close, {cycle2})"
            new_expr = f"{change1} - {change2}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # === 策略31-40: 顶背离和底背离相关因子 ===
    
    # 策略31: 顶背离 - 价格创新高但动量减弱（改进：始终有值）
    for window in [5, 10, 20]:
        # 价格创新高，但价格变化率下降
        new_name = f"Top_Divergence_Momentum_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越高值越大）
            price_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            momentum_current = f"($close - Ref($close, 3)) / (Ref($close, 3) + 0.0001)"
            momentum_prev = f"(Ref($close, 3) - Ref($close, 6)) / (Ref($close, 6) + 0.0001)"
            # 价格高且动量下降，值越大
            new_expr = f"{price_pos} * ({momentum_prev} - {momentum_current})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 价格创新高，但MA_Ratio下降
        new_name = f"Top_Divergence_MA_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越高值越大）
            price_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            ma_ratio_current = f"$close / Mean($close, 5) - 1"
            ma_ratio_prev = f"Ref($close, 5) / Mean(Ref($close, 5), 5) - 1"
            # 价格高且MA_Ratio下降，值越大
            new_expr = f"{price_pos} * ({ma_ratio_prev} - {ma_ratio_current})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 价格创新高，但斜率下降
        new_name = f"Top_Divergence_Slope_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越高值越大）
            price_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            slope_current = f"Slope($close, 5) / ($close + 0.0001)"
            slope_prev = f"Ref(Slope($close, 5), 5) / (Ref($close, 5) + 0.0001)"
            # 价格高且斜率下降，值越大
            new_expr = f"{price_pos} * ({slope_prev} - {slope_current})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略32: 底背离 - 价格创新低但动量增强（改进：始终有值）
    for window in [5, 10, 20]:
        # 价格创新低，但价格变化率上升
        new_name = f"Bottom_Divergence_Momentum_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越低值越大）
            price_pos = f"(Min($close, {window}) - $close) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            momentum_current = f"($close - Ref($close, 3)) / (Ref($close, 3) + 0.0001)"
            momentum_prev = f"(Ref($close, 3) - Ref($close, 6)) / (Ref($close, 6) + 0.0001)"
            # 价格低且动量上升，值越大
            new_expr = f"{price_pos} * ({momentum_current} - {momentum_prev})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 价格创新低，但MA_Ratio上升
        new_name = f"Bottom_Divergence_MA_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越低值越大）
            price_pos = f"(Min($close, {window}) - $close) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            ma_ratio_current = f"$close / Mean($close, 5) - 1"
            ma_ratio_prev = f"Ref($close, 5) / Mean(Ref($close, 5), 5) - 1"
            # 价格低且MA_Ratio上升，值越大
            new_expr = f"{price_pos} * ({ma_ratio_current} - {ma_ratio_prev})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 价格创新低，但斜率上升
        new_name = f"Bottom_Divergence_Slope_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越低值越大）
            price_pos = f"(Min($close, {window}) - $close) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            slope_current = f"Slope($close, 5) / ($close + 0.0001)"
            slope_prev = f"Ref(Slope($close, 5), 5) / (Ref($close, 5) + 0.0001)"
            # 价格低且斜率上升，值越大
            new_expr = f"{price_pos} * ({slope_current} - {slope_prev})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略33: 顶背离强度 - 价格创新高但指标下降的幅度（改进：始终有值）
    for window in [5, 10]:
        new_name = f"Top_Divergence_Strength_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越高值越大）
            price_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            ma_ratio_current = f"$close / Mean($close, 5) - 1"
            ma_ratio_prev = f"Ref($close, 5) / Mean(Ref($close, 5), 5) - 1"
            # 始终计算背离强度，价格越高且MA_Ratio下降越多，值越大
            new_expr = f"{price_pos} * ({ma_ratio_prev} - {ma_ratio_current})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略34: 底背离强度 - 价格创新低但指标上升的幅度（改进：始终有值，条件满足时值更大）
    for window in [5, 10]:
        new_name = f"Bottom_Divergence_Strength_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越低值越大）
            price_pos = f"(Min($close, {window}) - $close) / (Min($close, {window}) + 0.0001)"
            ma_ratio_current = f"$close / Mean($close, 5) - 1"
            ma_ratio_prev = f"Ref($close, 5) / Mean(Ref($close, 5), 5) - 1"
            # 始终计算背离强度，价格越低且MA_Ratio上升越多，值越大
            new_expr = f"{price_pos} * ({ma_ratio_current} - {ma_ratio_prev})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略35: 价格与MA_Ratio的背离（改进：始终有值，条件满足时值更大）
    for window in [5, 10, 20]:
        # 顶背离：价格相对高位，但MA_Ratio相对位置下降
        new_name = f"Price_MA_Divergence_Top_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越高值越大）
            price_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            ma_ratio_current = f"$close / Mean($close, {window}) - 1"
            ma_ratio_prev = f"Ref($close, {window // 2}) / Mean(Ref($close, {window // 2}), {window}) - 1"
            # 价格高但MA_Ratio下降，值越大
            new_expr = f"{price_pos} * ({ma_ratio_prev} - {ma_ratio_current})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 底背离：价格相对低位，但MA_Ratio相对位置上升
        new_name = f"Price_MA_Divergence_Bottom_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越低值越大，用1减去）
            price_pos = f"1 - ($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            ma_ratio_current = f"$close / Mean($close, {window}) - 1"
            ma_ratio_prev = f"Ref($close, {window // 2}) / Mean(Ref($close, {window // 2}), {window}) - 1"
            # 价格低但MA_Ratio上升，值越大
            new_expr = f"{price_pos} * ({ma_ratio_current} - {ma_ratio_prev})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略36: 价格与动量的背离（改进：始终有值）
    for window in [5, 10]:
        # 顶背离：价格相对高位，但动量减弱
        new_name = f"Price_Momentum_Divergence_Top_{window}"
        if new_name not in tried_factors:
            # 价格相对位置
            price_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            momentum_current = f"($close - Ref($close, 3)) / Ref($close, 3)"
            momentum_prev = f"(Ref($close, 3) - Ref($close, 6)) / Ref($close, 6)"
            # 价格高但动量减弱，值越大
            new_expr = f"{price_pos} * ({momentum_prev} - {momentum_current})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 底背离：价格相对低位，但动量增强
        new_name = f"Price_Momentum_Divergence_Bottom_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越低值越大）
            price_pos = f"1 - ($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            momentum_current = f"($close - Ref($close, 3)) / Ref($close, 3)"
            momentum_prev = f"(Ref($close, 3) - Ref($close, 6)) / Ref($close, 6)"
            # 价格低但动量增强，值越大
            new_expr = f"{price_pos} * ({momentum_current} - {momentum_prev})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略37: 价格与累积收益的背离（改进：始终有值）
    for window in [5, 10]:
        # 顶背离：价格相对高位，但累积收益下降
        new_name = f"Price_CumSum_Divergence_Top_{window}"
        if new_name not in tried_factors:
            # 价格相对位置
            price_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            cumsum_current = f"Sum(($close - Ref($close, 1)) / Ref($close, 1), 5)"
            cumsum_prev = f"Ref(Sum(($close - Ref($close, 1)) / Ref($close, 1), 5), 5)"
            # 价格高但累积收益下降，值越大
            new_expr = f"{price_pos} * ({cumsum_prev} - {cumsum_current})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 底背离：价格相对低位，但累积收益上升
        new_name = f"Price_CumSum_Divergence_Bottom_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越低值越大）
            price_pos = f"1 - ($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            cumsum_current = f"Sum(($close - Ref($close, 1)) / Ref($close, 1), 5)"
            cumsum_prev = f"Ref(Sum(($close - Ref($close, 1)) / Ref($close, 1), 5), 5)"
            # 价格低但累积收益上升，值越大
            new_expr = f"{price_pos} * ({cumsum_current} - {cumsum_prev})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略38: 价格与RSV的背离（改进：始终有值）
    for window in [5, 10, 20]:
        # 顶背离：价格创新高，但RSV下降
        new_name = f"Price_RSV_Divergence_Top_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越高值越大）
            price_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            rsv_current = f"($close - Min($low, {window})) / (Max($high, {window}) - Min($low, {window}) + 0.0001)"
            rsv_prev = f"Ref(($close - Min($low, {window})) / (Max($high, {window}) - Min($low, {window}) + 0.0001), {window // 2})"
            # 价格高且RSV下降，值越大
            new_expr = f"{price_pos} * ({rsv_prev} - {rsv_current})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 底背离：价格创新低，但RSV上升
        new_name = f"Price_RSV_Divergence_Bottom_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越低值越大）
            price_pos = f"(Min($close, {window}) - $close) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            rsv_current = f"($close - Min($low, {window})) / (Max($high, {window}) - Min($low, {window}) + 0.0001)"
            rsv_prev = f"Ref(($close - Min($low, {window})) / (Max($high, {window}) - Min($low, {window}) + 0.0001), {window // 2})"
            # 价格低且RSV上升，值越大
            new_expr = f"{price_pos} * ({rsv_current} - {rsv_prev})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略39: 价格与斜率的背离（改进：始终有值）
    for window in [5, 10]:
        # 顶背离：价格创新高，但斜率下降
        new_name = f"Price_Slope_Divergence_Top_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越高值越大）
            price_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            slope_current = f"Slope($close, 5) / ($close + 0.0001)"
            slope_prev = f"Ref(Slope($close, 5), 5) / (Ref($close, 5) + 0.0001)"
            # 价格高且斜率下降，值越大
            new_expr = f"{price_pos} * ({slope_prev} - {slope_current})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 底背离：价格创新低，但斜率上升
        new_name = f"Price_Slope_Divergence_Bottom_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越低值越大）
            price_pos = f"(Min($close, {window}) - $close) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            slope_current = f"Slope($close, 5) / ($close + 0.0001)"
            slope_prev = f"Ref(Slope($close, 5), 5) / (Ref($close, 5) + 0.0001)"
            # 价格低且斜率上升，值越大
            new_expr = f"{price_pos} * ({slope_current} - {slope_prev})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略40: 综合背离因子（改进：始终有值）
    for window in [5, 10]:
        # 顶背离综合：价格相对高位 + MA_Ratio下降 + 动量减弱
        new_name = f"Comprehensive_Top_Divergence_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越高值越大）
            price_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            ma_ratio_current = f"$close / Mean($close, 5) - 1"
            ma_ratio_prev = f"Ref($close, 5) / Mean(Ref($close, 5), 5) - 1"
            momentum_current = f"($close - Ref($close, 3)) / Ref($close, 3)"
            momentum_prev = f"(Ref($close, 3) - Ref($close, 6)) / Ref($close, 6)"
            # 价格高且MA_Ratio下降且动量减弱，值越大（直接计算，允许负值）
            ma_divergence = f"{ma_ratio_prev} - {ma_ratio_current}"
            momentum_divergence = f"{momentum_prev} - {momentum_current}"
            new_expr = f"{price_pos} * ({ma_divergence} + {momentum_divergence})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 底背离综合：价格相对低位 + MA_Ratio上升 + 动量增强
        new_name = f"Comprehensive_Bottom_Divergence_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越低值越大）
            price_pos = f"1 - ($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            ma_ratio_current = f"$close / Mean($close, 5) - 1"
            ma_ratio_prev = f"Ref($close, 5) / Mean(Ref($close, 5), 5) - 1"
            momentum_current = f"($close - Ref($close, 3)) / Ref($close, 3)"
            momentum_prev = f"(Ref($close, 3) - Ref($close, 6)) / Ref($close, 6)"
            # 价格低且MA_Ratio上升且动量增强，值越大（直接计算，允许负值）
            ma_divergence = f"{ma_ratio_current} - {ma_ratio_prev}"
            momentum_divergence = f"{momentum_current} - {momentum_prev}"
            new_expr = f"{price_pos} * ({ma_divergence} + {momentum_divergence})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略41: 背离强度因子（改进：始终有值）
    for window in [5, 10]:
        # 顶背离强度：价格相对高位的幅度 vs MA_Ratio下降的幅度
        new_name = f"Top_Divergence_Magnitude_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越高值越大）
            price_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            ma_ratio_current = f"$close / Mean($close, 5) - 1"
            ma_ratio_prev = f"Ref($close, 5) / Mean(Ref($close, 5), 5) - 1"
            ma_ratio_drop = f"{ma_ratio_prev} - {ma_ratio_current}"
            # 价格高且MA_Ratio下降，值越大（直接计算，允许负值）
            new_expr = f"{price_pos} * {ma_ratio_drop}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 底背离强度：价格相对低位的幅度 vs MA_Ratio上升的幅度
        new_name = f"Bottom_Divergence_Magnitude_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越低值越大）
            price_pos = f"1 - ($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            ma_ratio_current = f"$close / Mean($close, 5) - 1"
            ma_ratio_prev = f"Ref($close, 5) / Mean(Ref($close, 5), 5) - 1"
            ma_ratio_rise = f"{ma_ratio_current} - {ma_ratio_prev}"
            # 价格低且MA_Ratio上升，值越大（直接计算，允许负值）
            new_expr = f"{price_pos} * {ma_ratio_rise}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略42: 背离确认因子（改进：始终有值）
    for window in [5, 10]:
        # 顶背离确认：价格相对高位 + MA_Ratio下降 + 价格下跌
        new_name = f"Top_Divergence_Confirmation_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越高值越大）
            price_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            ma_ratio_current = f"$close / Mean($close, 5) - 1"
            ma_ratio_prev = f"Ref($close, 5) / Mean(Ref($close, 5), 5) - 1"
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            # 价格高且MA_Ratio下降且价格下跌，值越大（直接计算，允许负值）
            ma_divergence = f"{ma_ratio_prev} - {ma_ratio_current}"
            price_fall = f"0 - {price_change}"  # 价格下跌为负，取负号（使用0-而不是-）
            new_expr = f"{price_pos} * {ma_divergence} * {price_fall}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 底背离确认：价格相对低位 + MA_Ratio上升 + 价格上涨
        new_name = f"Bottom_Divergence_Confirmation_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越低值越大）
            price_pos = f"1 - ($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            ma_ratio_current = f"$close / Mean($close, 5) - 1"
            ma_ratio_prev = f"Ref($close, 5) / Mean(Ref($close, 5), 5) - 1"
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            # 价格低且MA_Ratio上升且价格上涨，值越大（直接计算，允许负值）
            ma_divergence = f"{ma_ratio_current} - {ma_ratio_prev}"
            price_rise = f"{price_change}"  # 价格上涨为正
            new_expr = f"{price_pos} * {ma_divergence} * {price_rise}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略43: 更简化的背离因子（改进：始终有值）
    for window in [5, 10, 20]:
        # 顶背离简化版：价格相对高位但MA_Ratio相对下降
        new_name = f"Simple_Top_Divergence_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越高值越大）
            price_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            ma_ratio_current = f"$close / Mean($close, {window}) - 1"
            ma_ratio_prev = f"Ref($close, {window // 2}) / Mean(Ref($close, {window // 2}), {window}) - 1"
            # 价格高但MA_Ratio下降，值越大
            new_expr = f"{price_pos} * ({ma_ratio_prev} - {ma_ratio_current})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 底背离简化版：价格相对低位但MA_Ratio相对上升
        new_name = f"Simple_Bottom_Divergence_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越低值越大）
            price_pos = f"1 - ($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            ma_ratio_current = f"$close / Mean($close, {window}) - 1"
            ma_ratio_prev = f"Ref($close, {window // 2}) / Mean(Ref($close, {window // 2}), {window}) - 1"
            # 价格低但MA_Ratio上升，值越大
            new_expr = f"{price_pos} * ({ma_ratio_current} - {ma_ratio_prev})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略44: 价格与动量的背离（简化版，改进：始终有值）
    for window in [5, 10]:
        # 顶背离：价格相对高位，但动量相对减弱
        new_name = f"Price_Momentum_Div_Top_{window}"
        if new_name not in tried_factors:
            # 价格相对位置
            price_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            momentum_current = f"($close - Ref($close, 3)) / Ref($close, 3)"
            momentum_prev = f"(Ref($close, 3) - Ref($close, 6)) / Ref($close, 6)"
            # 价格高但动量减弱，值越大
            new_expr = f"{price_pos} * ({momentum_prev} - {momentum_current})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 底背离：价格相对低位，但动量相对增强
        new_name = f"Price_Momentum_Div_Bottom_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越低值越大）
            price_pos = f"1 - ($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            momentum_current = f"($close - Ref($close, 3)) / Ref($close, 3)"
            momentum_prev = f"(Ref($close, 3) - Ref($close, 6)) / Ref($close, 6)"
            # 价格低但动量增强，值越大
            new_expr = f"{price_pos} * ({momentum_current} - {momentum_prev})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略45: 价格与斜率的背离（改进：始终有值）
    for window in [5, 10]:
        # 顶背离：价格相对高位，但斜率下降
        new_name = f"Price_Slope_Div_Top_{window}"
        if new_name not in tried_factors:
            # 价格相对位置
            price_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            slope_current = f"Slope($close, 5) / $close"
            slope_prev = f"Ref(Slope($close, 5), 5) / Ref($close, 5)"
            # 价格高但斜率下降，值越大
            new_expr = f"{price_pos} * ({slope_prev} - {slope_current})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 底背离：价格相对低位，但斜率上升
        new_name = f"Price_Slope_Div_Bottom_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越低值越大）
            price_pos = f"1 - ($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            slope_current = f"Slope($close, 5) / $close"
            slope_prev = f"Ref(Slope($close, 5), 5) / Ref($close, 5)"
            # 价格低但斜率上升，值越大
            new_expr = f"{price_pos} * ({slope_current} - {slope_prev})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略46: 价格与累积收益的背离（改进：始终有值）
    for window in [5, 10]:
        # 顶背离：价格相对高位，但累积收益下降
        new_name = f"Price_CumSum_Div_Top_{window}"
        if new_name not in tried_factors:
            # 价格相对位置
            price_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            cumsum_current = f"Sum(($close - Ref($close, 1)) / Ref($close, 1), 5)"
            cumsum_prev = f"Ref(Sum(($close - Ref($close, 1)) / Ref($close, 1), 5), 5)"
            # 价格高但累积收益下降，值越大
            new_expr = f"{price_pos} * ({cumsum_prev} - {cumsum_current})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 底背离：价格相对低位，但累积收益上升
        new_name = f"Price_CumSum_Div_Bottom_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越低值越大）
            price_pos = f"1 - ($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            cumsum_current = f"Sum(($close - Ref($close, 1)) / Ref($close, 1), 5)"
            cumsum_prev = f"Ref(Sum(($close - Ref($close, 1)) / Ref($close, 1), 5), 5)"
            # 价格低但累积收益上升，值越大
            new_expr = f"{price_pos} * ({cumsum_current} - {cumsum_prev})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略47: 背离强度因子（改进：始终有值）
    for window in [5, 10]:
        # 顶背离强度：价格相对高位的幅度 × MA_Ratio下降的幅度
        new_name = f"Top_Divergence_Power_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越高值越大）
            price_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            # 价格相对高位的幅度（当前价格相对于窗口最高价的位置）
            price_gain_pct = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            ma_ratio_current = f"$close / Mean($close, 5) - 1"
            ma_ratio_prev = f"Ref($close, 5) / Mean(Ref($close, 5), 5) - 1"
            ma_ratio_drop = f"{ma_ratio_prev} - {ma_ratio_current}"
            # 价格高且MA_Ratio下降，值越大
            new_expr = f"{price_gain_pct} * Abs({ma_ratio_drop})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 底背离强度：价格相对低位的幅度 × MA_Ratio上升的幅度
        new_name = f"Bottom_Divergence_Power_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越低值越大）
            price_pos = f"1 - ($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            # 价格相对低位的幅度（当前价格相对于窗口最低价的位置）
            price_drop_pct = f"(Max($close, {window}) - $close) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            ma_ratio_current = f"$close / Mean($close, 5) - 1"
            ma_ratio_prev = f"Ref($close, 5) / Mean(Ref($close, 5), 5) - 1"
            ma_ratio_rise = f"{ma_ratio_current} - {ma_ratio_prev}"
            # 价格低且MA_Ratio上升，值越大
            new_expr = f"{price_drop_pct} * Abs({ma_ratio_rise})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # === 新的因子挖掘策略（48-60）===
    
    # 策略48: 基于好因子的时间窗口扩展
    if 'Price_Change_1_MA3_MA5' in good_factors:
        base_expr = good_factors['Price_Change_1_MA3_MA5']
        for ma_window in [7, 10, 15]:
            new_name = f"Price_Change_1_MA3_MA{ma_window}"
            if new_name not in tried_factors:
                new_expr = f"($close - Ref($close, 1)) / Ref($close, 1) * ($close / Mean($close, 3) - 1) * ($close / Mean($close, {ma_window}) - 1)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略49: 价格突破与回撤的组合（改进：始终有值）
    for window in [5, 10, 20]:
        # 价格相对高位后的回撤幅度
        new_name = f"Breakout_Retracement_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越高值越大）
            price_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            # 回撤幅度（相对于前一日）
            retracement = f"(Ref($close, 1) - $close) / (Ref($close, 1) + 0.0001)"
            # 价格高且回撤，值越大
            new_expr = f"{price_pos} * {retracement}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 价格相对低位后的反弹幅度
        new_name = f"Breakdown_Rebound_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越低值越大）
            price_pos = f"1 - ($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            # 反弹幅度（相对于前一日）
            rebound = f"($close - Ref($close, 1)) / (Ref($close, 1) + 0.0001)"
            # 价格低且反弹，值越大
            new_expr = f"{price_pos} * {rebound}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略50: 价格波动率的相对位置
    for window in [5, 10, 20]:
        # 当前波动率在历史波动率中的分位数
        new_name = f"Volatility_Quantile_{window}"
        if new_name not in tried_factors:
            vol_current = f"Std($close, 5) / $close"
            new_expr = f"Quantile({vol_current}, {window}, 0.5) / {vol_current}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略51: 价格与成交量的背离（改进：始终有值）
    for window in [5, 10]:
        # 价格相对高位但成交量下降
        new_name = f"Price_Volume_Divergence_Top_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越高值越大）
            price_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            vol_current = f"$volume / Mean($volume, 5)"
            vol_prev = f"Ref($volume, 5) / Mean(Ref($volume, 5), 5)"
            # 价格高但成交量下降，值越大
            new_expr = f"{price_pos} * ({vol_prev} - {vol_current})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 价格相对低位但成交量上升
        new_name = f"Price_Volume_Divergence_Bottom_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越低值越大）
            price_pos = f"1 - ($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            vol_current = f"$volume / Mean($volume, 5)"
            vol_prev = f"Ref($volume, 5) / Mean(Ref($volume, 5), 5)"
            # 价格低但成交量上升，值越大
            new_expr = f"{price_pos} * ({vol_current} - {vol_prev})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略52: 价格变化的二阶导数（加速度的变化率）
    for window in [3, 5]:
        new_name = f"Price_Jerk_{window}"
        if new_name not in tried_factors:
            accel_current = f"(($close - Ref($close, 1)) / Ref($close, 1)) - ((Ref($close, 1) - Ref($close, 2)) / Ref($close, 2))"
            accel_prev = f"((Ref($close, 1) - Ref($close, 2)) / Ref($close, 2)) - ((Ref($close, 2) - Ref($close, 3)) / Ref($close, 3))"
            new_expr = f"{accel_current} - {accel_prev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略53: 价格与均线的距离变化率
    for ma_window in [5, 10, 20]:
        new_name = f"MA_Distance_Change_{ma_window}"
        if new_name not in tried_factors:
            dist_current = f"($close - Mean($close, {ma_window})) / Mean($close, {ma_window})"
            dist_prev = f"(Ref($close, {ma_window}) - Mean(Ref($close, {ma_window}), {ma_window})) / Mean(Ref($close, {ma_window}), {ma_window})"
            new_expr = f"{dist_current} - {dist_prev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略54: 价格在高低点区间中的位置变化
    for window in [5, 10, 20]:
        new_name = f"Price_Position_Change_{window}"
        if new_name not in tried_factors:
            pos_current = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            pos_prev = f"(Ref($close, {window // 2}) - Min(Ref($close, {window // 2}), {window})) / (Max(Ref($close, {window // 2}), {window}) - Min(Ref($close, {window // 2}), {window}) + 0.0001)"
            new_expr = f"{pos_current} - {pos_prev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略55: 价格趋势的一致性（连续N天同向）
    for window in [3, 5, 7]:
        new_name = f"Trend_Consistency_{window}"
        if new_name not in tried_factors:
            up_days = f"Sum(($close > Ref($close, 1)), {window})"
            down_days = f"Sum(($close < Ref($close, 1)), {window})"
            new_expr = f"({up_days} - {down_days}) / {window}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略56: 价格变化的平滑度（波动率的倒数）
    for window in [5, 10]:
        new_name = f"Price_Smoothness_{window}"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            volatility = f"Std({price_change}, {window})"
            new_expr = f"1 / ({volatility} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略57: 价格与均线的交叉信号强度
    for short_ma, long_ma in [(3, 5), (5, 10), (10, 20)]:
        new_name = f"MA_Cross_Strength_{short_ma}_{long_ma}"
        if new_name not in tried_factors:
            ma_short = f"Mean($close, {short_ma})"
            ma_long = f"Mean($close, {long_ma})"
            cross_signal = f"({ma_short} - {ma_long}) / ({ma_long} + 0.0001)"
            cross_prev = f"(Ref({ma_short}, 1) - Ref({ma_long}, 1)) / (Ref({ma_long}, 1) + 0.0001)"
            new_expr = f"{cross_signal} - {cross_prev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略58: 价格相对强度的变化
    for window in [5, 10, 20]:
        new_name = f"Relative_Strength_Change_{window}"
        if new_name not in tried_factors:
            rs_current = f"($close - Ref($close, {window})) / Ref($close, {window})"
            rs_prev = f"(Ref($close, {window}) - Ref($close, {window * 2})) / Ref($close, {window * 2})"
            new_expr = f"{rs_current} - {rs_prev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略59: 价格突破的确认（改进：始终有值）
    for window in [5, 10]:
        # 价格相对高位后是否继续上涨
        new_name = f"Breakout_Confirmation_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越高值越大）
            price_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            # 价格变化（上涨为正，下跌为负）
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            # 价格高且继续上涨，值越大（正值）；价格高但下跌，值为负
            new_expr = f"{price_pos} * {price_change}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 价格相对低位后是否继续下跌
        new_name = f"Breakdown_Confirmation_{window}"
        if new_name not in tried_factors:
            # 价格相对位置（越低值越大）
            price_pos = f"1 - ($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            # 价格变化（上涨为正，下跌为负）
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            # 价格低且继续下跌，值越大（负值，取负号使其为正）；价格低但上涨，值为正
            new_expr = f"{price_pos} * (0 - {price_change})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略60: 价格与均线的相对位置变化率
    for ma_window in [5, 10, 20]:
        new_name = f"Price_MA_Relative_Change_{ma_window}"
        if new_name not in tried_factors:
            ratio_current = f"$close / Mean($close, {ma_window})"
            ratio_prev = f"Ref($close, {ma_window // 2}) / Mean(Ref($close, {ma_window // 2}), {ma_window})"
            new_expr = f"({ratio_current} - {ratio_prev}) / ({ratio_prev} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略61: 价格变化的周期性特征（自相关）
    for lag in [3, 5, 7]:
        new_name = f"Price_Autocorr_{lag}"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            price_change_lag = f"Ref({price_change}, {lag})"
            new_expr = f"Corr({price_change}, {price_change_lag}, 10)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略62: 价格变化的非对称性（上涨幅度 vs 下跌幅度）
    for window in [5, 10]:
        new_name = f"Price_Asymmetry_{window}"
        if new_name not in tried_factors:
            up_changes = f"Sum(If(($close - Ref($close, 1)) / Ref($close, 1) > 0, ($close - Ref($close, 1)) / Ref($close, 1), 0), {window})"
            down_changes = f"Sum(If(($close - Ref($close, 1)) / Ref($close, 1) < 0, Abs(($close - Ref($close, 1)) / Ref($close, 1)), 0), {window})"
            new_expr = f"({up_changes} - {down_changes}) / ({up_changes} + {down_changes} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略63: 价格与均线的偏离度（标准化）
    for ma_window in [5, 10, 20]:
        new_name = f"Price_MA_Deviation_{ma_window}"
        if new_name not in tried_factors:
            deviation = f"($close - Mean($close, {ma_window})) / (Std($close, {ma_window}) + 0.0001)"
            new_expr = f"{deviation}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略64: 价格变化的动量衰减
    for window in [3, 5, 7]:
        new_name = f"Momentum_Decay_{window}"
        if new_name not in tried_factors:
            momentum_short = f"($close - Ref($close, {window // 2})) / Ref($close, {window // 2})"
            momentum_long = f"($close - Ref($close, {window})) / Ref($close, {window})"
            new_expr = f"{momentum_short} - {momentum_long}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略65: 价格与均线的交叉频率
    for ma_window in [5, 10]:
        new_name = f"MA_Cross_Frequency_{ma_window}"
        if new_name not in tried_factors:
            above_ma = f"$close > Mean($close, {ma_window})"
            above_ma_prev = f"Ref($close, 1) > Mean(Ref($close, 1), {ma_window})"
            cross_count = f"Sum(If({above_ma} != {above_ma_prev}, 1, 0), 10)"
            new_expr = f"{cross_count}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略66: 价格变化的稳定性（方差的变化）
    for window in [5, 10]:
        new_name = f"Price_Stability_{window}"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            volatility_current = f"Std({price_change}, {window})"
            volatility_prev = f"Ref(Std({price_change}, {window}), {window})"
            new_expr = f"{volatility_prev} - {volatility_current}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略67: 价格与均线的距离标准化后的变化
    for ma_window in [5, 10, 20]:
        new_name = f"MA_Distance_Norm_Change_{ma_window}"
        if new_name not in tried_factors:
            dist_current = f"($close - Mean($close, {ma_window})) / (Std($close, {ma_window}) + 0.0001)"
            dist_prev = f"(Ref($close, {ma_window // 2}) - Mean(Ref($close, {ma_window // 2}), {ma_window})) / (Std(Ref($close, {ma_window // 2}), {ma_window}) + 0.0001)"
            new_expr = f"{dist_current} - {dist_prev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略68: 价格变化的趋势强度（斜率的变化）
    for window in [5, 10]:
        new_name = f"Trend_Strength_{window}"
        if new_name not in tried_factors:
            slope_current = f"Slope($close, {window}) / $close"
            slope_prev = f"Ref(Slope($close, {window}), {window}) / Ref($close, {window})"
            new_expr = f"{slope_current} - {slope_prev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略69: 价格突破的强度（突破幅度 × 持续时间）
    for window in [5, 10]:
        new_name = f"Breakout_Intensity_{window}"
        if new_name not in tried_factors:
            price_new_high = f"$close > Max($close, {window})"
            breakout_magnitude = f"($close - Max($close, {window})) / Max($close, {window})"
            days_above = f"Sum($close > Max($close, {window}), {window})"
            new_expr = f"If({price_new_high}, {breakout_magnitude} * {days_above}, 0)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略70: 价格与均线的相对位置（分位数）
    for ma_window in [5, 10, 20]:
        new_name = f"Price_MA_Quantile_{ma_window}"
        if new_name not in tried_factors:
            price_ma_ratio = f"$close / Mean($close, {ma_window})"
            new_expr = f"Quantile({price_ma_ratio}, {ma_window * 2}, 0.5) / {price_ma_ratio}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # === 新的因子挖掘策略（71-85）===
    
    # 策略71: 价格变化的偏度（Skewness）- 捕捉价格分布的不对称性
    for window in [10, 20]:
        new_name = f"Price_Skewness_{window}"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            # 使用分位数近似偏度：中位数与均值的差异
            median = f"Quantile({price_change}, {window}, 0.5)"
            mean_val = f"Mean({price_change}, {window})"
            std_val = f"Std({price_change}, {window})"
            new_expr = f"({median} - {mean_val}) / ({std_val} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略72: 价格变化的峰度（Kurtosis）- 捕捉价格分布的尖峰程度
    for window in [10, 20]:
        new_name = f"Price_Kurtosis_{window}"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            # 使用分位数近似峰度：75分位与25分位的差异
            q75 = f"Quantile({price_change}, {window}, 0.75)"
            q25 = f"Quantile({price_change}, {window}, 0.25)"
            std_val = f"Std({price_change}, {window})"
            new_expr = f"({q75} - {q25}) / ({std_val} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略73: 价格突破后的确认（突破后N天的表现，改进：始终有值）
    for window in [5, 10]:
        for confirm_days in [1, 2, 3]:
            new_name = f"Breakout_Confirm_{window}_{confirm_days}"
            if new_name not in tried_factors:
                # 计算N天前的突破幅度（突破时为正，未突破时为负）
                breakout_magnitude = f"(Ref($close, {confirm_days}) - Max(Ref($close, {confirm_days}), {window})) / (Max(Ref($close, {confirm_days}), {window}) + 0.0001)"
                # 计算当前相对于N天前的价格变化
                price_change = f"($close - Ref($close, {confirm_days})) / (Ref($close, {confirm_days}) + 0.0001)"
                # 突破幅度越大且价格继续上涨，值越大
                new_expr = f"{breakout_magnitude} * {price_change}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略74: 价格与均线的交叉强度（交叉时的价格变化幅度）
    for short_ma, long_ma in [(3, 5), (5, 10), (10, 20)]:
        new_name = f"MA_Cross_Intensity_{short_ma}_{long_ma}"
        if new_name not in tried_factors:
            ma_short = f"Mean($close, {short_ma})"
            ma_long = f"Mean($close, {long_ma})"
            ma_short_prev = f"Ref(Mean($close, {short_ma}), 1)"
            ma_long_prev = f"Ref(Mean($close, {long_ma}), 1)"
            cross_now = f"({ma_short} - {ma_long}) > 0"
            cross_prev = f"({ma_short_prev} - {ma_long_prev}) > 0"
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            new_expr = f"If({cross_now} != {cross_prev}, {price_change}, 0)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略75: 价格在历史分位数中的位置变化率
    for window in [10, 20]:
        new_name = f"Price_Quantile_Change_{window}"
        if new_name not in tried_factors:
            quantile_current = f"Quantile($close, {window}, 0.5) / $close"
            quantile_prev = f"Ref(Quantile($close, {window}, 0.5), {window // 2}) / Ref($close, {window // 2})"
            new_expr = f"{quantile_current} - {quantile_prev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略76: 价格变化的持续性（连续同向变化的天数）
    for window in [3, 5, 7]:
        new_name = f"Price_Persistence_{window}"
        if new_name not in tried_factors:
            up_days = f"Sum(($close > Ref($close, 1)), {window})"
            down_days = f"Sum(($close < Ref($close, 1)), {window})"
            new_expr = f"({up_days} - {down_days}) / {window}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略77: 价格与成交量的相关性变化（如果有成交量数据）
    for window in [5, 10]:
        new_name = f"Price_Volume_Corr_Change_{window}"
        if new_name not in tried_factors:
            corr_current = f"Corr(($close - Ref($close, 1)) / Ref($close, 1), ($volume - Ref($volume, 1)) / Ref($volume, 1), {window})"
            corr_prev = f"Ref(Corr(($close - Ref($close, 1)) / Ref($close, 1), ($volume - Ref($volume, 1)) / Ref($volume, 1), {window}), {window})"
            new_expr = f"{corr_current} - {corr_prev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略78: 价格波动的聚集性（波动率聚类）
    for window in [5, 10]:
        new_name = f"Volatility_Clustering_{window}"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            vol_current = f"Std({price_change}, {window})"
            vol_prev = f"Ref(Std({price_change}, {window}), {window})"
            new_expr = f"{vol_current} * {vol_prev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略79: 价格突破的强度（突破幅度 × 成交量变化）
    for window in [5, 10]:
        new_name = f"Breakout_Strength_Volume_{window}"
        if new_name not in tried_factors:
            price_new_high = f"$close > Max($close, {window})"
            breakout_magnitude = f"($close - Max($close, {window})) / Max($close, {window})"
            volume_change = f"$volume / Mean($volume, {window}) - 1"
            new_expr = f"If({price_new_high}, {breakout_magnitude} * {volume_change}, 0)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略80: 价格与均线的偏离度标准化后的变化率
    for ma_window in [5, 10, 20]:
        new_name = f"MA_Deviation_Change_Rate_{ma_window}"
        if new_name not in tried_factors:
            deviation_current = f"($close - Mean($close, {ma_window})) / (Std($close, {ma_window}) + 0.0001)"
            deviation_prev = f"(Ref($close, {ma_window // 2}) - Mean(Ref($close, {ma_window // 2}), {ma_window})) / (Std(Ref($close, {ma_window // 2}), {ma_window}) + 0.0001)"
            new_expr = f"({deviation_current} - {deviation_prev}) / (Abs({deviation_prev}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略81: 价格变化的动量衰减率
    for short_window, long_window in [(3, 5), (5, 10)]:
        new_name = f"Momentum_Decay_Rate_{short_window}_{long_window}"
        if new_name not in tried_factors:
            momentum_short = f"($close - Ref($close, {short_window})) / Ref($close, {short_window})"
            momentum_long = f"($close - Ref($close, {long_window})) / Ref($close, {long_window})"
            new_expr = f"{momentum_short} / ({momentum_long} + 0.0001) - 1"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略82: 价格与均线的交叉频率（单位时间内的交叉次数）
    for ma_window in [5, 10]:
        new_name = f"MA_Cross_Frequency_{ma_window}"
        if new_name not in tried_factors:
            above_ma = f"$close > Mean($close, {ma_window})"
            above_ma_prev = f"Ref($close, 1) > Mean(Ref($close, 1), {ma_window})"
            cross_signal = f"If({above_ma} != {above_ma_prev}, 1, 0)"
            cross_count = f"Sum({cross_signal}, {ma_window * 2})"
            new_expr = f"{cross_count} / ({ma_window * 2} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略83: 价格变化的稳定性变化（波动率的变化率）
    for window in [5, 10]:
        new_name = f"Price_Stability_Change_{window}"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            volatility_current = f"Std({price_change}, {window})"
            volatility_prev = f"Ref(Std({price_change}, {window}), {window})"
            new_expr = f"({volatility_prev} - {volatility_current}) / ({volatility_prev} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略84: 价格与均线的距离标准化后的变化（改进版）
    for ma_window in [5, 10, 20]:
        new_name = f"MA_Distance_Norm_Change_Improved_{ma_window}"
        if new_name not in tried_factors:
            dist_current = f"($close - Mean($close, {ma_window})) / (Std($close, {ma_window}) + 0.0001)"
            dist_prev = f"(Ref($close, {ma_window // 2}) - Mean(Ref($close, {ma_window // 2}), {ma_window})) / (Std(Ref($close, {ma_window // 2}), {ma_window}) + 0.0001)"
            # 使用相对变化率
            new_expr = f"({dist_current} - {dist_prev}) / (Abs({dist_prev}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略85: 价格变化的趋势强度变化（斜率的变化率）
    for window in [5, 10]:
        new_name = f"Trend_Strength_Change_{window}"
        if new_name not in tried_factors:
            slope_current = f"Slope($close, {window}) / $close"
            slope_prev = f"Ref(Slope($close, {window}), {window}) / Ref($close, {window})"
            new_expr = f"({slope_current} - {slope_prev}) / (Abs({slope_prev}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略86: 价格突破的强度（突破幅度 × 持续时间）- 改进版
    for window in [5, 10]:
        new_name = f"Breakout_Intensity_Improved_{window}"
        if new_name not in tried_factors:
            price_new_high = f"$close > Max($close, {window})"
            breakout_magnitude = f"($close - Max($close, {window})) / Max($close, {window})"
            # 计算连续创新高的天数（直接使用天数，避免使用可能不支持的函数）
            days_above = f"Sum($close > Max($close, {window}), {window})"
            new_expr = f"If({price_new_high}, {breakout_magnitude} * ({days_above} + 1), 0)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略87: 价格与均线的相对位置（分位数）- 改进版
    for ma_window in [5, 10, 20]:
        new_name = f"Price_MA_Quantile_Improved_{ma_window}"
        if new_name not in tried_factors:
            price_ma_ratio = f"$close / Mean($close, {ma_window})"
            quantile_val = f"Quantile({price_ma_ratio}, {ma_window * 2}, 0.5)"
            new_expr = f"({price_ma_ratio} - {quantile_val}) / ({quantile_val} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略88: 价格变化的相对强度（与历史平均变化的比较）
    for window in [5, 10, 20]:
        new_name = f"Price_Relative_Strength_{window}"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            avg_change = f"Mean({price_change}, {window})"
            std_change = f"Std({price_change}, {window})"
            new_expr = f"({price_change} - {avg_change}) / ({std_change} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略89: 价格与均线的交叉后的价格行为
    for short_ma, long_ma in [(3, 5), (5, 10)]:
        new_name = f"MA_Cross_Aftermath_{short_ma}_{long_ma}"
        if new_name not in tried_factors:
            ma_short = f"Mean($close, {short_ma})"
            ma_long = f"Mean($close, {long_ma})"
            ma_short_prev = f"Ref(Mean($close, {short_ma}), 1)"
            ma_long_prev = f"Ref(Mean($close, {long_ma}), 1)"
            # 使用嵌套If实现"且"逻辑
            cross_up = f"({ma_short} > {ma_long})"
            cross_prev = f"(Ref({ma_short_prev}, 1) <= Ref({ma_long_prev}, 1))"
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            new_expr = f"If({cross_up}, If({cross_prev}, {price_change}, 0), 0)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略90: 价格在历史区间中的位置变化（RSV的变化）
    for window in [5, 10, 20]:
        new_name = f"RSV_Change_{window}"
        if new_name not in tried_factors:
            rsv_current = f"($close - Min($low, {window})) / (Max($high, {window}) - Min($low, {window}) + 0.0001)"
            rsv_prev = f"Ref(($close - Min($low, {window})) / (Max($high, {window}) - Min($low, {window}) + 0.0001), {window // 2})"
            new_expr = f"{rsv_current} - {rsv_prev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略91: 多周期价格变化的一致性（短期与长期方向一致时信号更强）
    for short_window in [3, 5]:
        for long_window in [10, 20]:
            new_name = f"Multi_Timeframe_Consistency_{short_window}_{long_window}"
            if new_name not in tried_factors:
                short_change = f"($close - Ref($close, {short_window})) / Ref($close, {short_window})"
                long_change = f"($close - Ref($close, {long_window})) / Ref($close, {long_window})"
                # 方向一致时值为正，不一致时为负（使用If代替Sign）
                short_sign = f"If({short_change} > 0, 1, If({short_change} < 0, -1, 0))"
                long_sign = f"If({long_change} > 0, 1, If({long_change} < 0, -1, 0))"
                new_expr = f"{short_sign} * {long_sign} * (Abs({short_change}) + Abs({long_change}))"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略92: 价格变化的相对强度（相对于历史波动率）
    for window in [5, 10, 20]:
        new_name = f"Price_Change_Relative_Volatility_{window}"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            volatility = f"Std({price_change}, {window})"
            new_expr = f"{price_change} / ({volatility} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略93: 价格与均线的角度（斜率比，捕捉趋势强度）
    for ma_window in [5, 10, 20]:
        new_name = f"Price_MA_Angle_{ma_window}"
        if new_name not in tried_factors:
            price_slope = f"Slope($close, {ma_window}) / $close"
            ma_slope = f"Slope(Mean($close, {ma_window}), {ma_window}) / Mean($close, {ma_window})"
            new_expr = f"{price_slope} / ({ma_slope} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略94: 价格加速度的变化率（三阶导数，捕捉加速度的变化）
    for window in [3, 5]:
        new_name = f"Price_Jerk_{window}"
        if new_name not in tried_factors:
            accel_current = f"(($close - Ref($close, 1)) / Ref($close, 1)) - ((Ref($close, 1) - Ref($close, 2)) / Ref($close, 2))"
            accel_prev = f"((Ref($close, 1) - Ref($close, 2)) / Ref($close, 2)) - ((Ref($close, 2) - Ref($close, 3)) / Ref($close, 3))"
            new_expr = f"{accel_current} - {accel_prev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略95: 价格在多个时间窗口中的分位数一致性
    for window1, window2 in [(5, 10), (10, 20)]:
        new_name = f"Price_Quantile_Consistency_{window1}_{window2}"
        if new_name not in tried_factors:
            quantile1 = f"Quantile($close, {window1}, 0.5) / $close"
            quantile2 = f"Quantile($close, {window2}, 0.5) / $close"
            # 两个分位数越接近，值越大
            new_expr = f"1 / (Abs({quantile1} - {quantile2}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略96: 价格变化的非线性组合（平方项，捕捉极端变化）
    if 'Price_Change_1' in base_components:
        for ma_window in [3, 5, 10]:
            new_name = f"Price_Change_Squared_MA{ma_window}"
            if new_name not in tried_factors:
                price_change = base_components['Price_Change_1']
                ma_ratio = f"$close / Mean($close, {ma_window}) - 1"
                # 平方项捕捉极端变化
                new_expr = f"{price_change} * {price_change} * {ma_ratio}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略97: 价格与均线的相对位置变化率（标准化）
    for ma_window in [5, 10, 20]:
        new_name = f"Price_MA_Position_Change_Rate_{ma_window}"
        if new_name not in tried_factors:
            dist_current = f"($close - Mean($close, {ma_window})) / (Std($close, {ma_window}) + 0.0001)"
            dist_prev = f"(Ref($close, {ma_window // 2}) - Mean(Ref($close, {ma_window // 2}), {ma_window})) / (Std(Ref($close, {ma_window // 2}), {ma_window}) + 0.0001)"
            new_expr = f"({dist_current} - {dist_prev}) / (Abs({dist_prev}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略98: 价格变化的累积效应（多日累积收益的标准化）
    for window in [3, 5, 7]:
        new_name = f"Cumulative_Return_Normalized_{window}"
        if new_name not in tried_factors:
            cum_return = f"Sum(($close - Ref($close, 1)) / Ref($close, 1), {window})"
            volatility = f"Std(($close - Ref($close, 1)) / Ref($close, 1), {window})"
            new_expr = f"{cum_return} / ({volatility} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略99: 价格突破的持续性（突破后连续保持的时间）
    for window in [5, 10]:
        new_name = f"Breakout_Persistence_{window}"
        if new_name not in tried_factors:
            is_breakout = f"$close > Max($close, {window})"
            # 计算连续突破的天数（简化版：使用Sum）
            days_above = f"Sum({is_breakout}, {window})"
            breakout_magnitude = f"($close - Max($close, {window})) / (Max($close, {window}) + 0.0001)"
            new_expr = f"{days_above} * {breakout_magnitude}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略100: 价格与均线的交叉后的价格行为（交叉后N天的表现）
    for short_ma, long_ma in [(3, 5), (5, 10)]:
        for days_after in [1, 2, 3]:
            new_name = f"MA_Cross_After_{short_ma}_{long_ma}_{days_after}"
            if new_name not in tried_factors:
                ma_short = f"Mean($close, {short_ma})"
                ma_long = f"Mean($close, {long_ma})"
                ma_short_prev = f"Ref(Mean($close, {short_ma}), {days_after})"
                ma_long_prev = f"Ref(Mean($close, {long_ma}), {days_after})"
                # 当前交叉且之前未交叉
                cross_now = f"({ma_short} > {ma_long})"
                cross_prev = f"({ma_short_prev} <= {ma_long_prev})"
                price_change = f"($close - Ref($close, {days_after})) / Ref($close, {days_after})"
                new_expr = f"If({cross_now}, If({cross_prev}, {price_change}, 0), 0)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略101: 价格变化的非对称性（上涨幅度 vs 下跌幅度的差异）
    for window in [5, 10, 20]:
        new_name = f"Price_Asymmetry_{window}"
        if new_name not in tried_factors:
            up_moves = f"Sum(If(($close - Ref($close, 1)) / Ref($close, 1) > 0, ($close - Ref($close, 1)) / Ref($close, 1), 0), {window})"
            down_moves = f"Sum(If(($close - Ref($close, 1)) / Ref($close, 1) < 0, Abs(($close - Ref($close, 1)) / Ref($close, 1)), 0), {window})"
            new_expr = f"({up_moves} - {down_moves}) / ({up_moves} + {down_moves} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略102: 价格与均线的距离的标准化变化（改进版，使用相对变化率）
    for ma_window in [5, 10, 20]:
        new_name = f"MA_Distance_Relative_Change_{ma_window}"
        if new_name not in tried_factors:
            dist_current = f"($close - Mean($close, {ma_window})) / Mean($close, {ma_window})"
            dist_prev = f"(Ref($close, {ma_window // 2}) - Mean(Ref($close, {ma_window // 2}), {ma_window})) / Mean(Ref($close, {ma_window // 2}), {ma_window})"
            new_expr = f"({dist_current} - {dist_prev}) / (Abs({dist_prev}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略103: 价格变化的动量与反转的交互
    for momentum_window in [3, 5]:
        for reversal_window in [1, 2]:
            new_name = f"Momentum_Reversal_Interaction_{momentum_window}_{reversal_window}"
            if new_name not in tried_factors:
                momentum = f"($close - Ref($close, {momentum_window})) / Ref($close, {momentum_window})"
                reversal = f"(Ref($close, {reversal_window}) / $close - 1)"
                new_expr = f"{momentum} * {reversal} * ({momentum} + {reversal})"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略104: 价格在历史分位数中的位置变化（捕捉分位数突破）
    for window in [10, 20]:
        for quantile in [0.25, 0.5, 0.75]:
            new_name = f"Price_Quantile_Position_{window}_{int(quantile*100)}"
            if new_name not in tried_factors:
                quantile_value = f"Quantile($close, {window}, {quantile})"
                quantile_prev = f"Ref(Quantile($close, {window}, {quantile}), {window // 2})"
                price_relative = f"$close / {quantile_value}"
                price_relative_prev = f"Ref($close, {window // 2}) / {quantile_prev}"
                new_expr = f"{price_relative} - {price_relative_prev}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略105: 价格变化的平滑度（低波动时的信号强度）
    for window in [5, 10]:
        new_name = f"Price_Smoothness_Signal_{window}"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            volatility = f"Std({price_change}, {window})"
            smoothness = f"1 / ({volatility} + 0.0001)"
            ma_ratio = f"$close / Mean($close, {window}) - 1"
            new_expr = f"{smoothness} * {ma_ratio}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # === 市场相关因子（106-150）===
    # 注意：这些因子使用市场指数数据，需要确保qlib数据中包含指数数据
    # 使用 ChangeInstrument 函数获取市场指数数据
    # 市场指数代码：CSI300=SH000300, CSI500=SH000905, CSI100=SH000903
    
    # 策略106: 个股相对市场指数的收益率（相对强度）
    market_indices = [
        ('CSI300', 'SH000300'),
        ('CSI500', 'SH000905'),
        ('CSI100', 'SH000903'),
    ]
    
    for market_name, market_code in market_indices:
        for window in [1, 3, 5, 10, 20]:
            # 个股收益率 vs 市场收益率
            new_name = f"Stock_Market_Return_{market_name}_{window}"
            if new_name not in tried_factors:
                stock_return = f"($close - Ref($close, {window})) / Ref($close, {window})"
                market_return = f"ChangeInstrument('{market_code}', ($close - Ref($close, {window})) / Ref($close, {window}))"
                new_expr = f"{stock_return} - {market_return}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
            
            # 个股相对市场指数的超额收益（标准化）
            new_name = f"Stock_Market_Excess_Return_{market_name}_{window}"
            if new_name not in tried_factors:
                stock_return = f"($close - Ref($close, {window})) / Ref($close, {window})"
                market_return = f"ChangeInstrument('{market_code}', ($close - Ref($close, {window})) / Ref($close, {window}))"
                excess_return = f"{stock_return} - {market_return}"
                market_return_daily = f"ChangeInstrument('{market_code}', ($close - Ref($close, 1)) / Ref($close, 1))"
                market_vol = f"Std({market_return_daily}, {window})"
                new_expr = f"{excess_return} / ({market_vol} + 0.0001)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略107: 个股与市场指数的相关性（简化版：使用收益率差异的标准化）
    # 注意：Corr函数在处理ChangeInstrument时可能有形状不匹配问题，改用其他方式
    for market_name, market_code in market_indices:
        for window in [5, 10, 20]:
            # 个股与市场指数的收益率同步性（使用差异的绝对值）
            new_name = f"Stock_Market_Return_Sync_{market_name}_{window}"
            if new_name not in tried_factors:
                stock_return = f"($close - Ref($close, 1)) / Ref($close, 1)"
                market_return = f"ChangeInstrument('{market_code}', ($close - Ref($close, 1)) / Ref($close, 1))"
                return_diff = f"{stock_return} - {market_return}"
                # 使用差异的标准差来衡量同步性（差异越小，同步性越高）
                sync_score = f"1 / (Std({return_diff}, {window}) + 0.0001)"
                new_expr = sync_score
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
            
            # 个股与市场指数的收益率方向一致性（使用乘积符号）
            new_name = f"Stock_Market_Direction_Match_{market_name}_{window}"
            if new_name not in tried_factors:
                stock_return = f"($close - Ref($close, 1)) / Ref($close, 1)"
                market_return = f"ChangeInstrument('{market_code}', ($close - Ref($close, 1)) / Ref($close, 1))"
                # 使用收益率乘积的符号：同向为正，反向为负
                # 乘积 > 0 表示同向，乘积 < 0 表示反向
                return_product = f"{stock_return} * {market_return}"
                # 计算窗口内的平均乘积（标准化后）
                # 使用乘积的均值来衡量方向一致性
                new_expr = f"Mean({return_product}, {window})"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
            
            # 个股与市场指数的收益率方向一致性（标准化版）
            new_name = f"Stock_Market_Direction_Score_{market_name}_{window}"
            if new_name not in tried_factors:
                stock_return = f"($close - Ref($close, 1)) / Ref($close, 1)"
                market_return = f"ChangeInstrument('{market_code}', ($close - Ref($close, 1)) / Ref($close, 1))"
                # 使用收益率乘积的符号：同向为正，反向为负
                return_product = f"{stock_return} * {market_return}"
                # 标准化：除以两个收益率的绝对值乘积，得到方向分数（-1到1之间）
                abs_product = f"Abs({stock_return}) * Abs({market_return})"
                direction_score = f"{return_product} / ({abs_product} + 0.0001)"
                # 计算窗口内的平均方向分数
                new_expr = f"Mean({direction_score}, {window})"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略108: 个股相对市场指数的Beta（简化版：使用波动率比率和方向匹配）
    # 注意：Corr函数在处理ChangeInstrument时可能有形状不匹配问题，改用其他方式
    for market_name, market_code in market_indices:
        for window in [10, 20, 60]:
            # Beta近似：使用波动率比率和方向一致性
            new_name = f"Stock_Market_Beta_Approx_{market_name}_{window}"
            if new_name not in tried_factors:
                stock_return = f"($close - Ref($close, 1)) / Ref($close, 1)"
                market_return = f"ChangeInstrument('{market_code}', ($close - Ref($close, 1)) / Ref($close, 1))"
                stock_vol = f"Std({stock_return}, {window})"
                market_vol = f"Std({market_return}, {window})"
                # 方向一致性：使用收益率乘积的均值（同向为正，反向为负）
                return_product = f"{stock_return} * {market_return}"
                direction_score = f"Mean({return_product}, {window})"
                # 标准化方向分数（-1到1之间）
                abs_product = f"Mean(Abs({stock_return}) * Abs({market_return}), {window})"
                normalized_direction = f"{direction_score} / ({abs_product} + 0.0001)"
                # Beta近似 = 波动率比率 * 标准化方向一致性
                new_expr = f"({stock_vol} / ({market_vol} + 0.0001)) * {normalized_direction}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
            
            # 个股相对市场指数的敏感度（收益率比率）
            new_name = f"Stock_Market_Sensitivity_{market_name}_{window}"
            if new_name not in tried_factors:
                stock_return = f"($close - Ref($close, {window})) / Ref($close, {window})"
                market_return = f"ChangeInstrument('{market_code}', ($close - Ref($close, {window})) / Ref($close, {window}))"
                # 敏感度 = 个股收益率 / 市场收益率（当市场收益率不为0时）
                new_expr = f"{stock_return} / ({market_return} + 0.0001)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略109: 个股相对市场指数的相对强度（价格比）
    for market_name, market_code in market_indices:
        for window in [3, 5, 10, 20]:
            # 个股价格/市场指数价格的比率变化
            new_name = f"Stock_Market_Price_Ratio_{market_name}_{window}"
            if new_name not in tried_factors:
                market_close = f"ChangeInstrument('{market_code}', $close)"
                price_ratio = f"$close / {market_close}"
                price_ratio_prev = f"Ref($close, {window}) / Ref({market_close}, {window})"
                new_expr = f"({price_ratio} - {price_ratio_prev}) / ({price_ratio_prev} + 0.0001)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
            
            # 个股价格相对市场指数的位置（分位数）
            new_name = f"Stock_Market_Price_Position_{market_name}_{window}"
            if new_name not in tried_factors:
                market_close = f"ChangeInstrument('{market_code}', $close)"
                price_ratio = f"$close / {market_close}"
                price_ratio_min = f"Min({price_ratio}, {window})"
                price_ratio_max = f"Max({price_ratio}, {window})"
                new_expr = f"({price_ratio} - {price_ratio_min}) / ({price_ratio_max} - {price_ratio_min} + 0.0001)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略110: 个股与市场指数的动量差异
    for market_name, market_code in market_indices:
        for window in [3, 5, 10]:
            # 个股动量 vs 市场动量
            new_name = f"Stock_Market_Momentum_Diff_{market_name}_{window}"
            if new_name not in tried_factors:
                stock_momentum = f"($close - Ref($close, {window})) / Ref($close, {window})"
                market_momentum = f"ChangeInstrument('{market_code}', ($close - Ref($close, {window})) / Ref($close, {window}))"
                new_expr = f"{stock_momentum} - {market_momentum}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
            
            # 个股动量相对市场动量的比率
            new_name = f"Stock_Market_Momentum_Ratio_{market_name}_{window}"
            if new_name not in tried_factors:
                stock_momentum = f"($close - Ref($close, {window})) / Ref($close, {window})"
                market_momentum = f"ChangeInstrument('{market_code}', ($close - Ref($close, {window})) / Ref($close, {window}))"
                new_expr = f"{stock_momentum} / ({market_momentum} + 0.0001)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略111: 个股与市场指数的波动率差异
    for market_name, market_code in market_indices:
        for window in [5, 10, 20]:
            # 个股波动率 vs 市场波动率
            new_name = f"Stock_Market_Volatility_Diff_{market_name}_{window}"
            if new_name not in tried_factors:
                stock_return = f"($close - Ref($close, 1)) / Ref($close, 1)"
                market_return = f"ChangeInstrument('{market_code}', ($close - Ref($close, 1)) / Ref($close, 1))"
                stock_vol = f"Std({stock_return}, {window})"
                market_vol = f"Std({market_return}, {window})"
                new_expr = f"{stock_vol} - {market_vol}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
            
            # 个股波动率相对市场波动率的比率
            new_name = f"Stock_Market_Volatility_Ratio_{market_name}_{window}"
            if new_name not in tried_factors:
                stock_return = f"($close - Ref($close, 1)) / Ref($close, 1)"
                market_return = f"ChangeInstrument('{market_code}', ($close - Ref($close, 1)) / Ref($close, 1))"
                stock_vol = f"Std({stock_return}, {window})"
                market_vol = f"Std({market_return}, {window})"
                new_expr = f"{stock_vol} / ({market_vol} + 0.0001)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略112: 个股与市场指数的MA比率差异
    for market_name, market_code in market_indices:
        for ma_window in [5, 10, 20]:
            # 个股MA比率 vs 市场MA比率
            new_name = f"Stock_Market_MA_Ratio_Diff_{market_name}_{ma_window}"
            if new_name not in tried_factors:
                stock_ma_ratio = f"$close / Mean($close, {ma_window}) - 1"
                market_ma_ratio = f"ChangeInstrument('{market_code}', $close / Mean($close, {ma_window}) - 1)"
                new_expr = f"{stock_ma_ratio} - {market_ma_ratio}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略113: 个股与市场指数的斜率差异
    for market_name, market_code in market_indices:
        for window in [5, 10, 20]:
            # 个股斜率 vs 市场斜率
            new_name = f"Stock_Market_Slope_Diff_{market_name}_{window}"
            if new_name not in tried_factors:
                stock_slope = f"Slope($close, {window}) / $close"
                market_slope = f"ChangeInstrument('{market_code}', Slope($close, {window}) / $close)"
                new_expr = f"{stock_slope} - {market_slope}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略114: 个股相对市场指数的突破信号
    for market_name, market_code in market_indices:
        for window in [5, 10, 20]:
            # 个股创新高但市场未创新高（相对强势）
            new_name = f"Stock_Market_Breakout_{market_name}_{window}"
            if new_name not in tried_factors:
                stock_new_high = f"$close > Max($close, {window})"
                market_close = f"ChangeInstrument('{market_code}', $close)"
                market_new_high = f"{market_close} > Max({market_close}, {window})"
                stock_breakout = f"($close - Max($close, {window})) / Max($close, {window})"
                # 使用 1 - market_new_high 来检查市场未创新高（布尔转数值后取反）
                market_not_new_high = f"1 - ({market_new_high})"
                new_expr = f"If({stock_new_high}, {market_not_new_high} * {stock_breakout}, 0)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略115: 个股与市场指数的背离（价格背离）
    for market_name, market_code in market_indices:
        for window in [5, 10, 20]:
            # 个股价格上涨但市场下跌（相对强势）
            new_name = f"Stock_Market_Divergence_Up_{market_name}_{window}"
            if new_name not in tried_factors:
                stock_return = f"($close - Ref($close, {window})) / Ref($close, {window})"
                market_return = f"ChangeInstrument('{market_code}', ($close - Ref($close, {window})) / Ref($close, {window}))"
                # 使用条件相乘来避免嵌套If的广播问题
                condition1 = f"If({stock_return} > 0, 1, 0)"
                condition2 = f"If({market_return} < 0, 1, 0)"
                divergence_value = f"{stock_return} - {market_return}"
                new_expr = f"{condition1} * {condition2} * {divergence_value}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
            
            # 个股价格下跌但市场上涨（相对弱势）
            new_name = f"Stock_Market_Divergence_Down_{market_name}_{window}"
            if new_name not in tried_factors:
                stock_return = f"($close - Ref($close, {window})) / Ref($close, {window})"
                market_return = f"ChangeInstrument('{market_code}', ($close - Ref($close, {window})) / Ref($close, {window}))"
                # 使用条件相乘来避免嵌套If的广播问题
                condition1 = f"If({stock_return} < 0, 1, 0)"
                condition2 = f"If({market_return} > 0, 1, 0)"
                divergence_value = f"{stock_return} - {market_return}"
                new_expr = f"{condition1} * {condition2} * {divergence_value}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略116: 个股与市场指数的相对强度变化率
    for market_name, market_code in market_indices:
        for window in [5, 10]:
            # 相对强度的变化（加速度）
            new_name = f"Stock_Market_Relative_Strength_Change_{market_name}_{window}"
            if new_name not in tried_factors:
                stock_return_short = f"($close - Ref($close, {window // 2})) / Ref($close, {window // 2})"
                market_return_short = f"ChangeInstrument('{market_code}', ($close - Ref($close, {window // 2})) / Ref($close, {window // 2}))"
                stock_return_long = f"($close - Ref($close, {window})) / Ref($close, {window})"
                market_return_long = f"ChangeInstrument('{market_code}', ($close - Ref($close, {window})) / Ref($close, {window}))"
                rs_short = f"{stock_return_short} - {market_return_short}"
                rs_long = f"{stock_return_long} - {market_return_long}"
                new_expr = f"{rs_short} - {rs_long}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略117: 个股与市场指数的累积收益差异
    for market_name, market_code in market_indices:
        for window in [5, 10, 20]:
            # 个股累积收益 vs 市场累积收益
            new_name = f"Stock_Market_CumReturn_Diff_{market_name}_{window}"
            if new_name not in tried_factors:
                stock_cum_return = f"Sum(($close - Ref($close, 1)) / Ref($close, 1), {window})"
                market_cum_return = f"ChangeInstrument('{market_code}', Sum(($close - Ref($close, 1)) / Ref($close, 1), {window}))"
                new_expr = f"{stock_cum_return} - {market_cum_return}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略118: 个股与市场指数的价格位置差异
    for market_name, market_code in market_indices:
        for window in [5, 10, 20]:
            # 个股价格位置 vs 市场价格位置
            new_name = f"Stock_Market_Price_Position_Diff_{market_name}_{window}"
            if new_name not in tried_factors:
                stock_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
                market_close = f"ChangeInstrument('{market_code}', $close)"
                market_pos = f"({market_close} - Min({market_close}, {window})) / (Max({market_close}, {window}) - Min({market_close}, {window}) + 0.0001)"
                new_expr = f"{stock_pos} - {market_pos}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略119: 个股与市场指数的组合因子（个股因子 × 市场因子）
    for market_name, market_code in market_indices:
        # 个股价格变化 × 市场价格变化
        new_name = f"Stock_Market_Price_Change_Product_{market_name}"
        if new_name not in tried_factors:
            stock_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            market_change = f"ChangeInstrument('{market_code}', ($close - Ref($close, 1)) / Ref($close, 1))"
            new_expr = f"{stock_change} * {market_change}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 个股MA比率 × 市场MA比率
        for ma_window in [5, 10]:
            new_name = f"Stock_Market_MA_Ratio_Product_{market_name}_{ma_window}"
            if new_name not in tried_factors:
                stock_ma_ratio = f"$close / Mean($close, {ma_window}) - 1"
                market_ma_ratio = f"ChangeInstrument('{market_code}', $close / Mean($close, {ma_window}) - 1)"
                new_expr = f"{stock_ma_ratio} * {market_ma_ratio}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略120: 个股相对市场指数的标准化超额收益
    for market_name, market_code in market_indices:
        for window in [5, 10, 20]:
            # 超额收益 / 市场波动率
            new_name = f"Stock_Market_Sharpe_Diff_{market_name}_{window}"
            if new_name not in tried_factors:
                stock_return = f"($close - Ref($close, {window})) / Ref($close, {window})"
                market_return = f"ChangeInstrument('{market_code}', ($close - Ref($close, {window})) / Ref($close, {window}))"
                excess_return = f"{stock_return} - {market_return}"
                market_return_daily = f"ChangeInstrument('{market_code}', ($close - Ref($close, 1)) / Ref($close, 1))"
                market_vol = f"Std({market_return_daily}, {window})"
                new_expr = f"{excess_return} / ({market_vol} + 0.0001)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    return new_factors


def analyze_custom_factors_csi500(
    instruments="csi300",
    label_expr="Ref($close, -1)/$close - 1",
    custom_factors=None,
    start_time=None,
    end_time=None,
    freq="day",
    limit_data_days=None,
    min_valid_days=10,
    output_dir=None,
):
    """
    分析自定义因子在指定市场（默认沪深300）上的相关性
    
    Parameters
    ----------
    instruments : str or list
        标的列表，可以是"csi500"、"csi300"等预定义市场，或股票代码列表
    label_expr : str
        标签表达式，例如："Ref($close, -1)/$close - 1"（未来1日收益率）
    custom_factors : dict, optional
        自定义因子字典，格式：{"因子名": "因子表达式"}
        如果为None，使用默认因子列表
    start_time : str, optional
        开始时间，格式："YYYY-MM-DD"
    end_time : str, optional
        结束时间，格式："YYYY-MM-DD"
    freq : str
        数据频率，默认"day"（日线）
    limit_data_days : int, optional
        限制数据天数（用于加速测试）
    min_valid_days : int
        至少需要的有效交易日数，默认10
    output_dir : str, optional
        输出目录，如果为None，使用当前目录
        
    Returns
    -------
    pd.DataFrame
        因子分析结果
    """
    # 初始化qlib（使用系统数据）
    qlib_data_path = os.path.expanduser("~/.qlib/qlib_data/cn_data")
    if not exists_qlib_data(qlib_data_path):
        # 尝试其他可能的路径
        qlib_data_path = "~/.qlib/qlib_data/cn_data"
    
    try:
        qlib.init(provider_uri=qlib_data_path, region=REG_CN)
        print(f"Qlib已初始化，数据目录: {qlib_data_path}")
    except Exception as e:
        print(f"初始化qlib失败: {e}")
        print("请确保已安装qlib数据，运行: python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data --region cn")
        return None
    
    # 使用固定股票池（方法1：获取当前成分股列表，确保股票池一致）
    # 显式获取当前成分股列表，固定使用这个列表（约300只）
    if isinstance(instruments, str):
        if instruments.lower() in ["csi500", "csi300", "csi100", "all"]:
            try:
                # 获取当前成分股（使用最近的交易日，比如2024-12-31）
                # 使用相同的日期作为start_time和end_time，只获取该日期的成分股
                # 注意：使用一个固定的最近交易日，确保获取的是当前成分股（约300只）
                current_date = "2024-12-31"  # 使用最近的交易日
                instruments_config = D.instruments(market=instruments.lower())
                instruments_list = D.list_instruments(
                    instruments=instruments_config,
                    start_time=current_date,  # 使用固定日期
                    end_time=current_date,    # 使用固定日期，只获取该日期的成分股
                    as_list=True
                )
                
                if not isinstance(instruments_list, list):
                    instruments_list = list(instruments_list)
                
                # 验证结果：确保是股票代码列表
                if len(instruments_list) > 0:
                    first_item = str(instruments_list[0])
                    if not (first_item.startswith('SH') or first_item.startswith('SZ')):
                        print(f"⚠️  警告: 获取的标的列表格式可能不正确")
                        print(f"  第一个元素: {first_item}")
                        print(f"  前5个元素: {instruments_list[:5]}")
                        # 过滤掉非股票代码的项
                        instruments_list = [s for s in instruments_list if str(s).startswith('SH') or str(s).startswith('SZ')]
                
                if len(instruments_list) == 0:
                    raise ValueError(f"获取{instruments}当前成分股列表为空")
                
                # 验证：CSI300应该只有约300只股票，如果超过500只，可能是获取了历史成分股
                expected_count = {"csi300": 300, "csi500": 500, "csi100": 100}.get(instruments.lower(), None)
                if expected_count and len(instruments_list) > expected_count * 1.5:
                    print(f"⚠️  警告: 获取的成分股数量({len(instruments_list)})明显超过预期({expected_count})")
                    print(f"  可能获取了历史成分股，尝试使用end_time方式...")
                    # 尝试只使用end_time，不传start_time
                    try:
                        instruments_list_v2 = D.list_instruments(
                            instruments=instruments_config,
                            end_time=current_date,  # 只使用end_time
                            as_list=True
                        )
                        if not isinstance(instruments_list_v2, list):
                            instruments_list_v2 = list(instruments_list_v2)
                        if len(instruments_list_v2) < len(instruments_list) and len(instruments_list_v2) <= expected_count * 1.2:
                            instruments_list = instruments_list_v2
                            print(f"  ✓ 使用end_time方式获取: {len(instruments_list)}只股票")
                    except Exception as e2:
                        print(f"  ⚠️  使用end_time方式失败: {e2}")
                
                print(f"✓ 获取{instruments}当前成分股列表: {len(instruments_list)}只股票（固定股票池）")
                print(f"  说明: 使用日期{current_date}的成分股，不包含历史成分股")
                
            except Exception as e:
                print(f"获取{instruments}当前成分股列表失败: {e}")
                import traceback
                traceback.print_exc()
                print("\n建议:")
                print("1. 检查qlib数据是否正确安装")
                print("2. 尝试手动指定股票代码列表")
                return None
        else:
            instruments_list = [instruments]
            print(f"✓ 使用单个股票: {instruments}")
    elif isinstance(instruments, dict):
        # 如果是字典格式，尝试获取当前成分股列表
        if "market" in instruments:
            try:
                # 使用最近的交易日（2024-12-31）获取当前成分股
                current_date = "2024-12-31"  # 使用最近的交易日
                market_name = instruments.get("market", "csi300")
                instruments_config = D.instruments(market=market_name)
                instruments_list = D.list_instruments(
                    instruments=instruments_config,
                    start_time=current_date,  # 使用固定日期
                    end_time=current_date,    # 使用固定日期，只获取该日期的成分股
                    as_list=True
                )
                if not isinstance(instruments_list, list):
                    instruments_list = list(instruments_list)
                print(f"✓ 从字典配置获取{market_name}当前成分股: {len(instruments_list)}只股票（固定股票池）")
            except Exception as e:
                print(f"⚠️  从字典配置获取当前成分股失败: {e}，使用字典格式")
                instruments_list = None  # 回退到字典格式
        else:
            instruments_list = None  # 使用字典格式
            print(f"✓ 使用字典格式股票池: {instruments}")
    else:
        # 列表格式，直接使用
        instruments_list = list(instruments)
        print(f"✓ 使用股票列表: {len(instruments_list)}只股票")
    
    # 验证并过滤股票代码
    if instruments_list is not None:
        valid_instruments = []
        for inst in instruments_list:
            inst_str = str(inst)
            if inst_str.startswith('SH') or inst_str.startswith('SZ'):
                valid_instruments.append(inst_str)
            else:
                print(f"⚠️  跳过非股票代码项: {inst_str}")
        
        if len(valid_instruments) == 0:
            print("⚠️  没有有效的股票代码")
            return None
        
        instruments_list = valid_instruments
        print(f"使用标的数量: {len(instruments_list)}只（固定股票池）")
        if len(instruments_list) <= 10:
            print(f"标的列表: {instruments_list}")
        else:
            print(f"前10个标的: {instruments_list[:10]}")
    
    # 统一使用instruments_param变量
    instruments_param = instruments_list if instruments_list is not None else instruments
    
    # 获取数据时间范围
    if start_time is None or end_time is None:
        # 尝试从数据中获取时间范围
        try:
            # 使用固定股票池列表获取测试数据
            test_instruments = instruments_list[:min(10, len(instruments_list))] if instruments_list else instruments_param
            test_data = D.features(
                test_instruments,
                ["$close"],
                start_time="2020-01-01",
                end_time="2025-01-01",
                freq=freq
            )
            if len(test_data) > 0:
                if isinstance(test_data.index, pd.MultiIndex):
                    data_times = test_data.index.get_level_values('datetime')
                else:
                    data_times = test_data.index
                if start_time is None:
                    start_time = data_times.min().strftime('%Y-%m-%d')
                if end_time is None:
                    end_time = data_times.max().strftime('%Y-%m-%d')
                print(f"从数据获取时间范围: {start_time} 到 {end_time}")
        except Exception as e:
            print(f"获取数据时间范围失败: {e}")
    
    if start_time is None:
        start_time = "2020-01-01"
    if end_time is None:
        end_time = "2025-01-01"
    
    # 如果指定了limit_data_days，限制数据范围
    if limit_data_days is not None:
        start_ts = pd.Timestamp(start_time)
        end_ts = pd.Timestamp(end_time)
        limited_start_ts = end_ts - pd.Timedelta(days=limit_data_days)
        if limited_start_ts > start_ts:
            start_time = limited_start_ts.strftime('%Y-%m-%d')
            print(f"⚠️  限制数据范围: 使用最近{limit_data_days}天的数据")
    
    print(f"\n使用时间范围: {start_time} 到 {end_time}")
    print(f"标签表达式: {label_expr}")
    
    # 加载因子配置
    factor_config = load_factor_config()
    good_factors = factor_config.get('good_factors', {})
    bad_factors = factor_config.get('bad_factors', {})
    min_ic_threshold = factor_config.get('min_ic_threshold', 0.02)
    min_icir_threshold = factor_config.get('min_icir_threshold', 0.1)
    
    print(f"\n从配置文件加载:")
    print(f"  - 好因子数量: {len(good_factors)}")
    print(f"  - 坏因子数量: {len(bad_factors)}")
    print(f"  - IC阈值: {min_ic_threshold}")
    print(f"  - ICIR阈值: {min_icir_threshold}")
    
    # 如果没有提供自定义因子，基于配置文件生成新因子
    if custom_factors is None:
        # 基于好因子生成新的组合因子
        print("\n基于好因子生成新的组合因子...")
        new_factors = generate_new_factors_from_good_ones(
            good_factors=good_factors,
            bad_factors=bad_factors,
            max_new_factors=150  # 生成最多150个新因子
        )
        
        print(f"✓ 生成了 {len(new_factors)} 个新因子（已排除重复因子）")
        
        # 如果生成了新因子，使用新因子；否则使用好因子
        if len(new_factors) > 0:
            custom_factors = new_factors
        else:
            # 如果没有生成新因子，使用配置文件中的好因子
            custom_factors = good_factors.copy()
            print("⚠️  没有生成新因子，使用配置文件中的好因子")
        
        # 如果配置文件中没有好因子且没有生成新因子，使用默认的TOP因子
        if len(custom_factors) == 0:
            print("⚠️  配置文件中没有好因子，使用默认TOP因子")
            custom_factors = {
            # === TOP50核心因子（按IC绝对值排序，保留IC>0.02的因子）===
            # 1-5: 价格变化和反转因子
            "Price_Change_1": "($close - Ref($close, 1)) / Ref($close, 1)",
            "Reversal_1": "Ref($close, 1)/$close - 1",
            "CLOSE1": "Ref($close, 1) / $close",
            "Price_Change_2": "($close - Ref($close, 2)) / Ref($close, 2)",
            "Price_Change_3": "($close - Ref($close, 3)) / Ref($close, 3)",
            
            # 6-10: 移动平均因子
            "MA_Ratio_3": "$close / Mean($close, 3) - 1",
            "MA_Ratio_5": "$close / Mean($close, 5) - 1",
            "MA_Ratio_7": "$close / Mean($close, 7) - 1",
            "MA_Ratio_10": "$close / Mean($close, 10) - 1",
            "MA_Ratio_15": "$close / Mean($close, 15) - 1",
            
            # 11-15: 分位数因子
            "Quantile_25_5": "Quantile($close, 5, 0.25) / $close",
            "Quantile_20_5": "Quantile($close, 5, 0.2) / $close",
            "Quantile_75_5": "Quantile($close, 5, 0.75) / $close",
            "Quantile_80_5": "Quantile($close, 5, 0.8) / $close",
            "Quantile_75_10": "Quantile($close, 10, 0.75) / $close",
            
            # 16-20: 价格相对特征
            "HIGH1": "Ref($high, 1) / $close",
            "HIGH2": "Ref($high, 2) / $close",
            "HIGH3": "Ref($high, 3) / $close",
            "OPEN2": "Ref($open, 2) / $close",
            "LOW2": "Ref($low, 2) / $close",
            
            # 21-25: 技术指标因子
            "Max_Gain_5": "($close - Min($close, 5)) / Min($close, 5)",
            "Slope_Close_3": "Slope($close, 3) / $close",
            "CumSum_Return_3": "Sum(($close - Ref($close, 1)) / Ref($close, 1), 3)",
            "If_Up_Return_3": "If($close > Ref($close, 3), ($close - Ref($close, 3)) / Ref($close, 3), 0)",
            "KLOW": "(If($open < $close, $open, $close) - $low) / $open",
            
            # 26-30: 反转和残差因子
            "Reversal_2": "Ref($close, 2)/$close - 1",
            "CLOSE2": "Ref($close, 2) / $close",
            "Reversal_3": "Ref($close, 3)/$close - 1",
            "CLOSE3": "Ref($close, 3) / $close",
            "Resi_Close_7": "Resi($close, 7) / $close",
            "Price_Change_4": "($close - Ref($close, 4)) / Ref($close, 4)",
            "Quantile_80_10": "Quantile($close, 10, 0.8) / $close",
            
            # === 基于TOP30因子挖掘的新组合因子 ===
            # 价格变化组合
            "Price_Change_1_3": "($close - Ref($close, 1)) / Ref($close, 1) * (($close - Ref($close, 3)) / Ref($close, 3))",
            "Price_Change_1_MA3": "($close - Ref($close, 1)) / Ref($close, 1) * ($close / Mean($close, 3) - 1)",
            "Price_Change_2_MA5": "($close - Ref($close, 2)) / Ref($close, 2) * ($close / Mean($close, 5) - 1)",
            
            # MA_Ratio组合
            "MA_Ratio_3_5": "($close / Mean($close, 3) - 1) * ($close / Mean($close, 5) - 1)",
            "MA_Ratio_3_7": "($close / Mean($close, 3) - 1) * ($close / Mean($close, 7) - 1)",
            "MA_Ratio_5_10": "($close / Mean($close, 5) - 1) * ($close / Mean($close, 10) - 1)",
            "MA_Ratio_Delta_3": "($close / Mean($close, 3) - 1) - (Ref($close, 3) / Mean(Ref($close, 3), 3) - 1)",
            "MA_Ratio_Delta_7": "($close / Mean($close, 7) - 1) - (Ref($close, 7) / Mean(Ref($close, 7), 7) - 1)",
            
            # 分位数组合
            "Quantile_Range_5": "(Quantile($close, 5, 0.8) - Quantile($close, 5, 0.2)) / $close",
            "Quantile_Range_10": "(Quantile($close, 10, 0.8) - Quantile($close, 10, 0.2)) / $close",
            "Quantile_25_75_5": "(Quantile($close, 5, 0.25) / $close) * (Quantile($close, 5, 0.75) / $close)",
            "Quantile_20_80_5": "(Quantile($close, 5, 0.2) / $close) * (Quantile($close, 5, 0.8) / $close)",
            
            # 价格相对特征组合
            "HIGH1_LOW2": "(Ref($high, 1) / $close) * (Ref($low, 2) / $close)",
            "HIGH2_OPEN2": "(Ref($high, 2) / $close) * (Ref($open, 2) / $close)",
            "HIGH3_LOW2": "(Ref($high, 3) / $close) * (Ref($low, 2) / $close)",
            "CLOSE1_CLOSE2": "(Ref($close, 1) / $close) * (Ref($close, 2) / $close)",
            "CLOSE1_CLOSE3": "(Ref($close, 1) / $close) * (Ref($close, 3) / $close)",
            
            # 技术指标组合
            "Slope_Close_3_MA3": "(Slope($close, 3) / $close) * ($close / Mean($close, 3) - 1)",
            "Slope_Close_3_CumSum3": "(Slope($close, 3) / $close) * Sum(($close - Ref($close, 1)) / Ref($close, 1), 3)",
            "CumSum_Return_3_MA3": "Sum(($close - Ref($close, 1)) / Ref($close, 1), 3) * ($close / Mean($close, 3) - 1)",
            "If_Up_Return_3_MA3": "If($close > Ref($close, 3), ($close - Ref($close, 3)) / Ref($close, 3), 0) * ($close / Mean($close, 3) - 1)",
            "Max_Gain_5_MA5": "(($close - Min($close, 5)) / Min($close, 5)) * ($close / Mean($close, 5) - 1)",
            
            # 残差组合
            "Resi_Close_7_MA7": "(Resi($close, 7) / $close) * ($close / Mean($close, 7) - 1)",
            "Resi_Close_7_Slope3": "(Resi($close, 7) / $close) * (Slope($close, 3) / $close)",
            
            # 反转组合
            "Reversal_1_2": "(Ref($close, 1)/$close - 1) * (Ref($close, 2)/$close - 1)",
            "Reversal_1_3": "(Ref($close, 1)/$close - 1) * (Ref($close, 3)/$close - 1)",
            "Reversal_2_3": "(Ref($close, 2)/$close - 1) * (Ref($close, 3)/$close - 1)",
            "CLOSE1_CLOSE2_Reversal": "(Ref($close, 1) / $close) * (Ref($close, 2) / $close) * (Ref($close, 3)/$close - 1)",
            
            # === 基于TOP30因子与成交量的组合 ===
            # 价格变化 + 成交量
            "Price_Change_1_Volume": "($close - Ref($close, 1)) / Ref($close, 1) * ($volume / Mean($volume, 5) - 1)",
            "Price_Change_2_Volume": "($close - Ref($close, 2)) / Ref($close, 2) * ($volume / Mean($volume, 5) - 1)",
            "Price_Change_3_Volume": "($close - Ref($close, 3)) / Ref($close, 3) * ($volume / Mean($volume, 5) - 1)",
            
            # MA_Ratio + 成交量
            "MA_Ratio_3_Volume": "($close / Mean($close, 3) - 1) * ($volume / Mean($volume, 3) - 1)",
            "MA_Ratio_5_Volume": "($close / Mean($close, 5) - 1) * ($volume / Mean($volume, 5) - 1)",
            "MA_Ratio_7_Volume": "($close / Mean($close, 7) - 1) * ($volume / Mean($volume, 7) - 1)",
            
            # 分位数 + 成交量
            "Quantile_25_5_Volume": "(Quantile($close, 5, 0.25) / $close) * ($volume / Mean($volume, 5) - 1)",
            "Quantile_75_5_Volume": "(Quantile($close, 5, 0.75) / $close) * ($volume / Mean($volume, 5) - 1)",
            "Quantile_80_5_Volume": "(Quantile($close, 5, 0.8) / $close) * ($volume / Mean($volume, 5) - 1)",
            
            # 价格相对特征 + 成交量
            "HIGH1_Volume": "(Ref($high, 1) / $close) * ($volume / Mean($volume, 5) - 1)",
            "HIGH2_Volume": "(Ref($high, 2) / $close) * ($volume / Mean($volume, 5) - 1)",
            "CLOSE1_Volume": "(Ref($close, 1) / $close) * ($volume / Mean($volume, 5) - 1)",
            "CLOSE2_Volume": "(Ref($close, 2) / $close) * ($volume / Mean($volume, 5) - 1)",
            
            # 技术指标 + 成交量
            "Slope_Close_3_Volume": "(Slope($close, 3) / $close) * ($volume / Mean($volume, 3) - 1)",
            "CumSum_Return_3_Volume": "Sum(($close - Ref($close, 1)) / Ref($close, 1), 3) * ($volume / Mean($volume, 3) - 1)",
            "If_Up_Return_3_Volume": "If($close > Ref($close, 3), ($close - Ref($close, 3)) / Ref($close, 3), 0) * ($volume / Mean($volume, 3) - 1)",
            "Max_Gain_5_Volume": "(($close - Min($close, 5)) / Min($close, 5)) * ($volume / Mean($volume, 5) - 1)",
            "KLOW_Volume": "((If($open < $close, $open, $close) - $low) / $open) * ($volume / Mean($volume, 5) - 1)",
            
            # 残差 + 成交量
            "Resi_Close_7_Volume": "(Resi($close, 7) / $close) * ($volume / Mean($volume, 7) - 1)",
            
            # === 基于TOP50因子挖掘更深层的五重组合因子 ===
            # 基于Price_Change_1_MA3_MA5 (IC=0.055991) 的扩展
            "Price_Change_1_MA3_MA5_Slope3": "($close - Ref($close, 1)) / Ref($close, 1) * ($close / Mean($close, 3) - 1) * ($close / Mean($close, 5) - 1) * (Slope($close, 3) / $close)",
            "Price_Change_1_MA3_MA5_CumSum3": "($close - Ref($close, 1)) / Ref($close, 1) * ($close / Mean($close, 3) - 1) * ($close / Mean($close, 5) - 1) * Sum(($close - Ref($close, 1)) / Ref($close, 1), 3)",
            "Price_Change_1_MA3_MA5_Price_Change_2": "($close - Ref($close, 1)) / Ref($close, 1) * ($close / Mean($close, 3) - 1) * ($close / Mean($close, 5) - 1) * (($close - Ref($close, 2)) / Ref($close, 2))",
            "Price_Change_1_MA3_MA5_MA7": "($close - Ref($close, 1)) / Ref($close, 1) * ($close / Mean($close, 3) - 1) * ($close / Mean($close, 5) - 1) * ($close / Mean($close, 7) - 1)",
            
            # 基于MA_Ratio_3_5_Price_Change_1 (IC=0.055991) 的扩展
            "MA_Ratio_3_5_Price_Change_1_Slope3": "($close / Mean($close, 3) - 1) * ($close / Mean($close, 5) - 1) * (($close - Ref($close, 1)) / Ref($close, 1)) * (Slope($close, 3) / $close)",
            "MA_Ratio_3_5_Price_Change_1_CumSum3": "($close / Mean($close, 3) - 1) * ($close / Mean($close, 5) - 1) * (($close - Ref($close, 1)) / Ref($close, 1)) * Sum(($close - Ref($close, 1)) / Ref($close, 1), 3)",
            "MA_Ratio_3_5_Price_Change_1_MA7": "($close / Mean($close, 3) - 1) * ($close / Mean($close, 5) - 1) * (($close - Ref($close, 1)) / Ref($close, 1)) * ($close / Mean($close, 7) - 1)",
            "MA_Ratio_3_5_Price_Change_1_Price_Change_2": "($close / Mean($close, 3) - 1) * ($close / Mean($close, 5) - 1) * (($close - Ref($close, 1)) / Ref($close, 1)) * (($close - Ref($close, 2)) / Ref($close, 2))",
            
            # 基于CumSum_Return_3_MA3_Price_Change_1 (IC=0.055479) 的扩展
            "CumSum_Return_3_MA3_Price_Change_1_Slope3": "Sum(($close - Ref($close, 1)) / Ref($close, 1), 3) * ($close / Mean($close, 3) - 1) * (($close - Ref($close, 1)) / Ref($close, 1)) * (Slope($close, 3) / $close)",
            "CumSum_Return_3_MA3_Price_Change_1_MA5": "Sum(($close - Ref($close, 1)) / Ref($close, 1), 3) * ($close / Mean($close, 3) - 1) * (($close - Ref($close, 1)) / Ref($close, 1)) * ($close / Mean($close, 5) - 1)",
            "CumSum_Return_3_MA3_Price_Change_1_Price_Change_2": "Sum(($close - Ref($close, 1)) / Ref($close, 1), 3) * ($close / Mean($close, 3) - 1) * (($close - Ref($close, 1)) / Ref($close, 1)) * (($close - Ref($close, 2)) / Ref($close, 2))",
            
            # 基于Price_Change_1_MA3_CumSum3 (IC=0.055479) 的扩展
            "Price_Change_1_MA3_CumSum3_Slope3": "($close - Ref($close, 1)) / Ref($close, 1) * ($close / Mean($close, 3) - 1) * Sum(($close - Ref($close, 1)) / Ref($close, 1), 3) * (Slope($close, 3) / $close)",
            "Price_Change_1_MA3_CumSum3_MA5": "($close - Ref($close, 1)) / Ref($close, 1) * ($close / Mean($close, 3) - 1) * Sum(($close - Ref($close, 1)) / Ref($close, 1), 3) * ($close / Mean($close, 5) - 1)",
            "Price_Change_1_MA3_CumSum3_Price_Change_2": "($close - Ref($close, 1)) / Ref($close, 1) * ($close / Mean($close, 3) - 1) * Sum(($close - Ref($close, 1)) / Ref($close, 1), 3) * (($close - Ref($close, 2)) / Ref($close, 2))",
            
            # 基于MA_Ratio_3_7_Price_Change_1 (IC=0.055424) 的扩展
            "MA_Ratio_3_7_Price_Change_1_Slope3": "($close / Mean($close, 3) - 1) * ($close / Mean($close, 7) - 1) * (($close - Ref($close, 1)) / Ref($close, 1)) * (Slope($close, 3) / $close)",
            "MA_Ratio_3_7_Price_Change_1_CumSum3": "($close / Mean($close, 3) - 1) * ($close / Mean($close, 7) - 1) * (($close - Ref($close, 1)) / Ref($close, 1)) * Sum(($close - Ref($close, 1)) / Ref($close, 1), 3)",
            "MA_Ratio_3_7_Price_Change_1_MA5": "($close / Mean($close, 3) - 1) * ($close / Mean($close, 7) - 1) * (($close - Ref($close, 1)) / Ref($close, 1)) * ($close / Mean($close, 5) - 1)",
            
            # 基于Price_Change_1_MA3_Slope3 (IC=0.054551) 的扩展
            "Price_Change_1_MA3_Slope3_CumSum3": "($close - Ref($close, 1)) / Ref($close, 1) * ($close / Mean($close, 3) - 1) * (Slope($close, 3) / $close) * Sum(($close - Ref($close, 1)) / Ref($close, 1), 3)",
            "Price_Change_1_MA3_Slope3_MA5": "($close - Ref($close, 1)) / Ref($close, 1) * ($close / Mean($close, 3) - 1) * (Slope($close, 3) / $close) * ($close / Mean($close, 5) - 1)",
            "Price_Change_1_MA3_Slope3_Price_Change_2": "($close - Ref($close, 1)) / Ref($close, 1) * ($close / Mean($close, 3) - 1) * (Slope($close, 3) / $close) * (($close - Ref($close, 2)) / Ref($close, 2))",
            
            # 基于Price_Change_1_3_MA3 (IC=0.054143) 的扩展
            "Price_Change_1_3_MA3_Slope3": "($close - Ref($close, 1)) / Ref($close, 1) * (($close - Ref($close, 3)) / Ref($close, 3)) * ($close / Mean($close, 3) - 1) * (Slope($close, 3) / $close)",
            "Price_Change_1_3_MA3_CumSum3": "($close - Ref($close, 1)) / Ref($close, 1) * (($close - Ref($close, 3)) / Ref($close, 3)) * ($close / Mean($close, 3) - 1) * Sum(($close - Ref($close, 1)) / Ref($close, 1), 3)",
            "Price_Change_1_3_MA3_MA5": "($close - Ref($close, 1)) / Ref($close, 1) * (($close - Ref($close, 3)) / Ref($close, 3)) * ($close / Mean($close, 3) - 1) * ($close / Mean($close, 5) - 1)",
            "Price_Change_1_3_MA3_Price_Change_2": "($close - Ref($close, 1)) / Ref($close, 1) * (($close - Ref($close, 3)) / Ref($close, 3)) * ($close / Mean($close, 3) - 1) * (($close - Ref($close, 2)) / Ref($close, 2))",
            
            # 基于Price_Change_1_3_MA5 (IC=0.053333) 的扩展
            "Price_Change_1_3_MA5_Slope3": "($close - Ref($close, 1)) / Ref($close, 1) * (($close - Ref($close, 3)) / Ref($close, 3)) * ($close / Mean($close, 5) - 1) * (Slope($close, 3) / $close)",
            "Price_Change_1_3_MA5_CumSum3": "($close - Ref($close, 1)) / Ref($close, 1) * (($close - Ref($close, 3)) / Ref($close, 3)) * ($close / Mean($close, 5) - 1) * Sum(($close - Ref($close, 1)) / Ref($close, 1), 3)",
            "Price_Change_1_3_MA5_MA7": "($close - Ref($close, 1)) / Ref($close, 1) * (($close - Ref($close, 3)) / Ref($close, 3)) * ($close / Mean($close, 5) - 1) * ($close / Mean($close, 7) - 1)",
            
            # 基于Slope_Close_3_MA3_MA5 (IC=0.053096) 的扩展
            "Slope_Close_3_MA3_MA5_Price_Change_1": "(Slope($close, 3) / $close) * ($close / Mean($close, 3) - 1) * ($close / Mean($close, 5) - 1) * (($close - Ref($close, 1)) / Ref($close, 1))",
            "Slope_Close_3_MA3_MA5_CumSum3": "(Slope($close, 3) / $close) * ($close / Mean($close, 3) - 1) * ($close / Mean($close, 5) - 1) * Sum(($close - Ref($close, 1)) / Ref($close, 1), 3)",
            "Slope_Close_3_MA3_MA5_MA7": "(Slope($close, 3) / $close) * ($close / Mean($close, 3) - 1) * ($close / Mean($close, 5) - 1) * ($close / Mean($close, 7) - 1)",
            
            # 基于MA_Ratio_3_5_Slope3 (IC=0.053096) 的扩展
            "MA_Ratio_3_5_Slope3_CumSum3": "($close / Mean($close, 3) - 1) * ($close / Mean($close, 5) - 1) * (Slope($close, 3) / $close) * Sum(($close - Ref($close, 1)) / Ref($close, 1), 3)",
            "MA_Ratio_3_5_Slope3_MA7": "($close / Mean($close, 3) - 1) * ($close / Mean($close, 5) - 1) * (Slope($close, 3) / $close) * ($close / Mean($close, 7) - 1)",
            "MA_Ratio_3_5_Slope3_Price_Change_2": "($close / Mean($close, 3) - 1) * ($close / Mean($close, 5) - 1) * (Slope($close, 3) / $close) * (($close - Ref($close, 2)) / Ref($close, 2))",
            
            # 基于Price_Change_1_2_3 (IC=0.052819) 的扩展
            "Price_Change_1_2_3_MA3": "($close - Ref($close, 1)) / Ref($close, 1) * (($close - Ref($close, 2)) / Ref($close, 2)) * (($close - Ref($close, 3)) / Ref($close, 3)) * ($close / Mean($close, 3) - 1)",
            "Price_Change_1_2_3_MA5": "($close - Ref($close, 1)) / Ref($close, 1) * (($close - Ref($close, 2)) / Ref($close, 2)) * (($close - Ref($close, 3)) / Ref($close, 3)) * ($close / Mean($close, 5) - 1)",
            "Price_Change_1_2_3_Slope3": "($close - Ref($close, 1)) / Ref($close, 1) * (($close - Ref($close, 2)) / Ref($close, 2)) * (($close - Ref($close, 3)) / Ref($close, 3)) * (Slope($close, 3) / $close)",
            "Price_Change_1_2_3_CumSum3": "($close - Ref($close, 1)) / Ref($close, 1) * (($close - Ref($close, 2)) / Ref($close, 2)) * (($close - Ref($close, 3)) / Ref($close, 3)) * Sum(($close - Ref($close, 1)) / Ref($close, 1), 3)",
            
            # 基于Price_Change_2_MA3_MA5 (IC=0.052723) 的扩展
            "Price_Change_2_MA3_MA5_Slope3": "($close - Ref($close, 2)) / Ref($close, 2) * ($close / Mean($close, 3) - 1) * ($close / Mean($close, 5) - 1) * (Slope($close, 3) / $close)",
            "Price_Change_2_MA3_MA5_CumSum3": "($close - Ref($close, 2)) / Ref($close, 2) * ($close / Mean($close, 3) - 1) * ($close / Mean($close, 5) - 1) * Sum(($close - Ref($close, 1)) / Ref($close, 1), 3)",
            "Price_Change_2_MA3_MA5_Price_Change_1": "($close - Ref($close, 2)) / Ref($close, 2) * ($close / Mean($close, 3) - 1) * ($close / Mean($close, 5) - 1) * (($close - Ref($close, 1)) / Ref($close, 1))",
            
            # === 基于TOP50因子与成交量的五重组合 ===
            # 表现最好的因子 + 成交量
            "Price_Change_1_MA3_MA5_Volume": "($close - Ref($close, 1)) / Ref($close, 1) * ($close / Mean($close, 3) - 1) * ($close / Mean($close, 5) - 1) * ($volume / Mean($volume, 3) - 1)",
            "MA_Ratio_3_5_Price_Change_1_Volume": "($close / Mean($close, 3) - 1) * ($close / Mean($close, 5) - 1) * (($close - Ref($close, 1)) / Ref($close, 1)) * ($volume / Mean($volume, 3) - 1)",
            "CumSum_Return_3_MA3_Price_Change_1_Volume": "Sum(($close - Ref($close, 1)) / Ref($close, 1), 3) * ($close / Mean($close, 3) - 1) * (($close - Ref($close, 1)) / Ref($close, 1)) * ($volume / Mean($volume, 3) - 1)",
            "Price_Change_1_MA3_CumSum3_Volume": "($close - Ref($close, 1)) / Ref($close, 1) * ($close / Mean($close, 3) - 1) * Sum(($close - Ref($close, 1)) / Ref($close, 1), 3) * ($volume / Mean($volume, 3) - 1)",
            "Price_Change_1_MA3_Slope3_Volume": "($close - Ref($close, 1)) / Ref($close, 1) * ($close / Mean($close, 3) - 1) * (Slope($close, 3) / $close) * ($volume / Mean($volume, 3) - 1)",
            "Price_Change_1_3_MA3_Volume": "($close - Ref($close, 1)) / Ref($close, 1) * (($close - Ref($close, 3)) / Ref($close, 3)) * ($close / Mean($close, 3) - 1) * ($volume / Mean($volume, 3) - 1)",
            "Price_Change_1_3_MA5_Volume": "($close - Ref($close, 1)) / Ref($close, 1) * (($close - Ref($close, 3)) / Ref($close, 3)) * ($close / Mean($close, 5) - 1) * ($volume / Mean($volume, 3) - 1)",
            "Slope_Close_3_MA3_MA5_Volume": "(Slope($close, 3) / $close) * ($close / Mean($close, 3) - 1) * ($close / Mean($close, 5) - 1) * ($volume / Mean($volume, 3) - 1)",
            "MA_Ratio_3_5_Slope3_Volume": "($close / Mean($close, 3) - 1) * ($close / Mean($close, 5) - 1) * (Slope($close, 3) / $close) * ($volume / Mean($volume, 3) - 1)",
            "Price_Change_1_2_3_Volume": "($close - Ref($close, 1)) / Ref($close, 1) * (($close - Ref($close, 2)) / Ref($close, 2)) * (($close - Ref($close, 3)) / Ref($close, 3)) * ($volume / Mean($volume, 3) - 1)",
        }
    
    # 过滤掉已测试的因子（包括好因子和坏因子），避免重复尝试
    tried_factors = {**good_factors, **bad_factors}
    original_count = len(custom_factors)
    custom_factors = {k: v for k, v in custom_factors.items() if k not in tried_factors}
    filtered_count = original_count - len(custom_factors)
    if filtered_count > 0:
        print(f"\n⚠️  跳过了 {filtered_count} 个已测试的因子（好因子: {sum(1 for k in custom_factors.keys() if k in good_factors)}, 坏因子: {sum(1 for k in custom_factors.keys() if k in bad_factors)}）")
    
    if len(custom_factors) == 0:
        print("\n⚠️  所有因子都已测试过，没有新因子需要测试")
        return None
    
    print(f"\n正在测试 {len(custom_factors)} 个自定义因子...")
    
    # 获取标签数据
    print("\n正在获取标签数据...")
    try:
        # 使用固定股票池列表（方法1：当前成分股）
        label_data = D.features(
            instruments_list,  # 使用固定股票池列表
            [label_expr],
            start_time=start_time,
            end_time=end_time,
            freq=freq
        )
        
        if len(label_data) == 0:
            print("⚠️  无法获取标签数据")
            return None
        
        # 提取标签值
        if isinstance(label_data.columns, pd.MultiIndex):
            label_col = label_data.columns[0]
            labels = label_data[label_col]
        else:
            labels = label_data.iloc[:, 0]
        
        print(f"标签数据量: {len(labels)}")
        if isinstance(labels.index, pd.MultiIndex):
            # 多股票数据，按日期分组统计
            datetime_level = labels.index.get_level_values('datetime')
            print(f"日期范围: {datetime_level.min()} 到 {datetime_level.max()}")
            print(f"交易日数: {datetime_level.nunique()}")
            print(f"股票数: {labels.index.get_level_values('instrument').nunique()}")
        else:
            print(f"标签均值: {labels.mean():.6f}, 标准差: {labels.std():.6f}")
        
    except Exception as e:
        print(f"获取标签数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 测试每个因子
    print("\n开始计算因子IC...")
    factor_results = []
    
    for i, (factor_name, factor_expr) in enumerate(custom_factors.items(), 1):
        try:
            # 再次检查因子是否在已测试列表中（双重保险）
            if factor_name in tried_factors:
                print(f"[{i}/{len(custom_factors)}] 跳过已测试因子: {factor_name}")
                continue
            
            print(f"[{i}/{len(custom_factors)}] 计算因子: {factor_name}")
            
            # 获取因子数据（方法1：固定股票池）
            factor_data = D.features(
                instruments_list,  # 使用固定股票池列表
                [factor_expr],
                start_time=start_time,
                end_time=end_time,
                freq=freq
            )
            
            if len(factor_data) == 0:
                print(f"  ⚠️  {factor_name:30s} | 无数据")
                factor_results.append({
                    'factor': factor_name,
                    'expression': factor_expr,
                    'IC': np.nan,
                    'Rank_IC': np.nan,
                    'ICIR': np.nan,
                    'IC_std': np.nan,
                    'valid_days': 0,
                    'status': 'no_data'
                })
                continue
            
            # 提取因子值
            if isinstance(factor_data.columns, pd.MultiIndex):
                factor_col = factor_data.columns[0]
                factor_values = factor_data[factor_col]
            else:
                factor_values = factor_data.iloc[:, 0]
            
            # 对齐因子和标签（使用相同的索引）
            common_index = factor_values.index.intersection(labels.index)
            if len(common_index) == 0:
                print(f"  ⚠️  {factor_name:30s} | 无共同索引")
                factor_results.append({
                    'factor': factor_name,
                    'expression': factor_expr,
                    'IC': np.nan,
                    'Rank_IC': np.nan,
                    'ICIR': np.nan,
                    'IC_std': np.nan,
                    'valid_days': 0,
                    'status': 'no_common_index'
                })
                continue
            
            # 使用共同索引对齐数据
            aligned_factor = factor_values.loc[common_index]
            aligned_label = labels.loc[common_index]
            
            # 删除NaN值
            valid_mask = aligned_factor.notna() & aligned_label.notna()
            aligned_factor = aligned_factor[valid_mask]
            aligned_label = aligned_label[valid_mask]
            
            if len(aligned_factor) < min_valid_days:
                print(f"  ⚠️  {factor_name:30s} | 数据不足 (有效数据: {len(aligned_factor)} < {min_valid_days})")
                factor_results.append({
                    'factor': factor_name,
                    'expression': factor_expr,
                    'IC': np.nan,
                    'Rank_IC': np.nan,
                    'ICIR': np.nan,
                    'IC_std': np.nan,
                    'valid_days': len(aligned_factor),
                    'status': 'insufficient_data'
                })
                continue
            
            # 使用qlib的calc_ic函数计算IC（自动按日期分组）
            # calc_ic函数支持MultiIndex格式，会自动按datetime层级分组计算IC
            try:
                # 使用qlib的calc_ic函数，它会自动处理MultiIndex格式
                # 对于MultiIndex格式的Series，calc_ic会按datetime层级分组计算IC
                ic_series, ric_series = calc_ic(
                    pred=aligned_factor,
                    label=aligned_label,
                    date_col="datetime",  # MultiIndex的层级名或列名
                    dropna=True
                )
                
                # 计算IC统计量
                if len(ic_series) >= min_valid_days:
                    mean_ic = ic_series.mean()
                    mean_ric = ric_series.mean()
                    ic_std = ic_series.std()
                    
                    # ICIR = 平均IC / IC标准差
                    if ic_std > 1e-10 and not np.isnan(mean_ic):
                        icir = mean_ic / ic_std
                    else:
                        icir = np.nan
                    
                    factor_results.append({
                        'factor': factor_name,
                        'expression': factor_expr,
                        'IC': mean_ic,
                        'Rank_IC': mean_ric,
                        'ICIR': icir,
                        'IC_std': ic_std,
                        'valid_days': len(ic_series),
                        'status': 'success'
                    })
                    
                    print(f"  ✓ {factor_name:30s} | IC: {mean_ic:8.6f} | Rank_IC: {mean_ric:8.6f} | ICIR: {icir:8.4f} | 有效天数: {len(ic_series)}")
                else:
                    print(f"  ⚠️  {factor_name:30s} | IC天数不足 (有效天数: {len(ic_series)} < {min_valid_days})")
                    factor_results.append({
                        'factor': factor_name,
                        'expression': factor_expr,
                        'IC': np.nan,
                        'Rank_IC': np.nan,
                        'ICIR': np.nan,
                        'IC_std': np.nan,
                        'valid_days': len(ic_series),
                        'status': 'insufficient_days'
                    })
            except Exception as e:
                print(f"  ✗ {factor_name:30s} | 计算IC错误: {str(e)[:100]}")
                factor_results.append({
                    'factor': factor_name,
                    'expression': factor_expr,
                    'IC': np.nan,
                    'Rank_IC': np.nan,
                    'ICIR': np.nan,
                    'IC_std': np.nan,
                    'valid_days': 0,
                    'status': f'error: {str(e)[:50]}'
                })
            
        except Exception as e:
            print(f"  ✗ {factor_name:30s} | 错误: {str(e)[:50]}")
            factor_results.append({
                'factor': factor_name,
                'expression': factor_expr,
                'IC': np.nan,
                'Rank_IC': np.nan,
                'ICIR': np.nan,
                'IC_std': np.nan,
                'valid_days': 0,
                'status': f'error: {str(e)[:50]}'
            })
            continue
    
    # 分析结果
    if factor_results:
        results_df = pd.DataFrame(factor_results)
        # 按IC绝对值排序
        results_df['IC_abs'] = results_df['IC'].abs().fillna(0)
        results_df = results_df.sort_values('IC_abs', ascending=False)
        results_df = results_df.drop('IC_abs', axis=1)
        
        print("\n" + "="*100)
        print("自定义因子IC分析结果")
        print("="*100)
        
        # 统计信息
        valid_results = results_df[results_df['status'] == 'success']
        if len(valid_results) > 0:
            print(f"\n总体统计:")
            print(f"  成功测试因子数: {len(valid_results)} / {len(factor_results)}")
            print(f"  平均IC: {valid_results['IC'].mean():.6f}")
            print(f"  IC标准差: {valid_results['IC'].std():.6f}")
            print(f"  平均Rank IC: {valid_results['Rank_IC'].mean():.6f}")
            print(f"  平均ICIR: {valid_results['ICIR'].mean():.6f}")
            
            # 有效因子（根据配置文件中的阈值）
            effective_factors = valid_results[
                (valid_results['IC'].abs() > min_ic_threshold) | (valid_results['ICIR'].abs() > min_icir_threshold)
            ]
            
            # 坏因子（IC绝对值 < min_ic_threshold 且 ICIR绝对值 < min_icir_threshold）
            bad_factors_new = valid_results[
                (valid_results['IC'].abs() < min_ic_threshold) & (valid_results['ICIR'].abs() < min_icir_threshold)
            ]
            
            print(f"\n有效因子数 (|IC|>{min_ic_threshold} 或 |ICIR|>{min_icir_threshold}): {len(effective_factors)}")
            print(f"无效因子数 (|IC|<{min_ic_threshold} 且 |ICIR|<{min_icir_threshold}): {len(bad_factors_new)}")
            
            # 显示IC绝对值最大的前20个因子
            print(f"\nIC绝对值最大的前20个因子:")
            top_factors = valid_results.head(20)
            for idx, row in top_factors.iterrows():
                print(f"  {row['factor']:30s} | IC: {row['IC']:8.6f} | Rank_IC: {row['Rank_IC']:8.6f} | ICIR: {row['ICIR']:8.4f} | 有效天数: {row['valid_days']}")
                print(f"    {row['expression']}")
            
            # 显示有效因子
            if len(effective_factors) > 0:
                print(f"\n有效因子详情:")
                for idx, row in effective_factors.iterrows():
                    print(f"  {row['factor']:30s} | IC: {row['IC']:8.6f} | Rank_IC: {row['Rank_IC']:8.6f} | ICIR: {row['ICIR']:8.4f} | 有效天数: {row['valid_days']}")
                    print(f"    {row['expression']}")
            
            # 保存结果
            if output_dir is None:
                output_dir = Path.cwd()
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / "custom_factors_csi500_results.csv"
            results_df.to_csv(output_file, index=False)
            print(f"\n结果已保存到: {output_file}")
            
            # 保存有效因子
            if len(effective_factors) > 0:
                effective_file = output_dir / "effective_factors_csi500.csv"
                effective_factors.to_csv(effective_file, index=False)
                print(f"有效因子已保存到: {effective_file}")
            
            # 更新因子配置文件
            print("\n更新因子配置文件...")
            new_good_factors = {}
            new_bad_factors = {}
            
            # 收集新发现的好因子（不在配置文件中的）
            for idx, row in effective_factors.iterrows():
                factor_name = row['factor']
                if factor_name not in good_factors:
                    new_good_factors[factor_name] = row['expression']
            
            # 收集新发现的坏因子（不在配置文件中的）
            # 包括：1) IC/ICIR不达标的因子 2) 测试失败的因子（错误、数据不足等）
            for idx, row in bad_factors_new.iterrows():
                factor_name = row['factor']
                if factor_name not in bad_factors and factor_name not in good_factors:
                    new_bad_factors[factor_name] = row['expression']
            
            # 收集所有测试失败的因子（包括错误、数据不足等）
            failed_factors = results_df[results_df['status'] != 'success']
            for idx, row in failed_factors.iterrows():
                factor_name = row['factor']
                # 如果因子不在好因子和坏因子列表中，且测试失败，则加入坏因子
                if factor_name not in bad_factors and factor_name not in good_factors:
                    new_bad_factors[factor_name] = row['expression']
            
            # 确保所有测试过的因子都被记录（包括IC/ICIR不达标的）
            # 对于成功测试但IC/ICIR不达标的因子，也要加入到坏因子列表
            effective_factor_names = set(effective_factors['factor'].values) if len(effective_factors) > 0 else set()
            for idx, row in valid_results.iterrows():
                factor_name = row['factor']
                # 如果因子既不在有效因子列表，也不在好因子列表，也不在坏因子列表，则加入坏因子
                if factor_name not in effective_factor_names and factor_name not in bad_factors and factor_name not in good_factors:
                    new_bad_factors[factor_name] = row['expression']
            
            # 更新配置文件
            if new_good_factors or new_bad_factors:
                update_factor_config(
                    new_good_factors=new_good_factors if new_good_factors else None,
                    new_bad_factors=new_bad_factors if new_bad_factors else None
                )
            else:
                print("✓ 没有新的因子需要更新到配置文件")
            
            # 对好因子进行相关性去重
            print("\n对好因子进行相关性去重...")
            try:
                # 合并原有的好因子和新发现的好因子
                all_good_factors = {**good_factors, **new_good_factors}
                
                if len(all_good_factors) > 1:
                    # 使用IC结果进行相关性去重
                    deduplicated_factors, removed_factors = deduplicate_good_factors_by_correlation(
                        good_factors=all_good_factors,
                        instruments_list=instruments_list,  # 使用固定股票池列表
                        start_time=start_time,
                        end_time=end_time,
                        freq=freq,
                        correlation_threshold=0.85,  # 相关性阈值
                        ic_results=valid_results  # 使用已计算的IC结果
                    )
                    
                    # 更新配置文件中的好因子
                    if len(deduplicated_factors) < len(all_good_factors):
                        config = load_factor_config()
                        config['good_factors'] = deduplicated_factors
                        
                        # 将被去除的因子加入到坏因子列表
                        if removed_factors:
                            if 'bad_factors' not in config or config['bad_factors'] is None:
                                config['bad_factors'] = {}
                            config['bad_factors'].update(removed_factors)
                            print(f"✓ 已将 {len(removed_factors)} 个去重移除的因子加入到坏因子列表")
                        
                        save_factor_config(config)
                        print(f"✓ 已更新配置文件，保留 {len(deduplicated_factors)} 个去重后的好因子")
                    else:
                        print("✓ 没有发现高相关性的因子对，无需去重")
                else:
                    print("✓ 好因子数量不足，跳过相关性分析")
            except Exception as e:
                print(f"⚠️  相关性去重失败: {e}")
                import traceback
                traceback.print_exc()
        
        return results_df
    else:
        print("没有成功测试的因子")
        return None


def main():
    """主函数"""
    print("="*80)
    print("自定义因子相关性分析（沪深300）")
    print("="*80)
    
    # 配置参数
    instruments = "csi300"  # 可以使用 "csi500", "csi300", "csi100" 或股票代码列表
    
    # 自定义label表达式
    # 示例：
    # - "Ref($close, -1)/$close - 1"  # 未来1日收益率
    # - "Ref($close, -5)/$close - 1"  # 未来5日收益率
    # - "Ref($close, -20)/$close - 1"  # 未来20日收益率
    label_expr = "Ref($close, -1)/$close - 1"  # 修复：移除多余空格
    
    # 自定义因子（可选，如果为None则使用默认因子列表）
    custom_factors = None
    # 示例：
    # custom_factors = {
    #     "Momentum_5": "Ref($close, 5)/$close - 1",
    #     "MA_Ratio_20": "$close / Mean($close, 20) - 1",
    # }
    
    # 时间范围
    start_time = "2015-01-01"
    end_time = "2025-01-01"
    
    # 数据频率
    freq = "day"  # 日线数据
    
    # 限制数据天数（可选，用于加速测试）
    limit_data_days = None  # 例如：365 表示只使用最近1年的数据
    
    # 输出目录
    output_dir = Path(__file__).parent / "factor_evaluation_results"
    
    # 执行分析
    results = analyze_custom_factors_csi500(
        instruments=instruments,
        label_expr=label_expr,
        custom_factors=custom_factors,
        start_time=start_time,
        end_time=end_time,
        freq=freq,
        limit_data_days=limit_data_days,
        output_dir=output_dir,
    )
    
    if results is not None:
        print("\n" + "="*80)
        print("分析完成！")
        print("="*80)
    else:
        print("\n分析失败，请检查错误信息。")


if __name__ == "__main__":
    main()

