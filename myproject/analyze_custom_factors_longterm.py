"""
分析自定义因子在沪深500上的相关性（长期目标：未来20日平均价格）
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

logger = get_module_logger("长期因子分析")


def load_factor_config(config_path=None):
    """加载因子配置文件（长期目标专用）"""
    if config_path is None:
        config_path = Path(__file__).parent / "factor_config_longterm.yaml"
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
    """保存因子配置文件（长期目标专用）"""
    if config_path is None:
        config_path = Path(__file__).parent / "factor_config_longterm.yaml"
    else:
        config_path = Path(config_path)
    
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    print(f"✓ 因子配置已保存到: {config_path}")


def update_factor_config(new_good_factors=None, new_bad_factors=None, config_path=None):
    """更新因子配置文件（长期目标专用）"""
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
            all_factor_data = D.features(
                instruments_list,
                list(good_factors.values()),
                start_time=start_time,
                end_time=end_time,
                freq=freq
            )
            
            # 获取标签数据用于计算IC（未来20日平均价格）
            # 手动列出所有Ref，因为Mean函数不支持负数窗口
            ref_list = " + ".join([f"Ref($close, -{i})" for i in range(1, 21)])
            label_expr_for_ic = f"(({ref_list}) / 20) / $close - 1"
            label_data = D.features(
                instruments_list,  # 可以是字典或列表
                [label_expr_for_ic],
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
                # 比较IC值，保留IC绝对值更大的
                ic1 = factor_ic_map.get(factor1, 0.0)
                ic2 = factor_ic_map.get(factor2, 0.0)
                
                if ic1 > ic2:
                    factors_to_remove.add(factor2)
                    print(f"  {factor1:40s} (IC={ic1:.6f}) vs {factor2:40s} (IC={ic2:.6f}) | 相关性: {corr_value:.4f} -> 移除 {factor2}")
                elif ic2 > ic1:
                    factors_to_remove.add(factor1)
                    print(f"  {factor1:40s} (IC={ic1:.6f}) vs {factor2:40s} (IC={ic2:.6f}) | 相关性: {corr_value:.4f} -> 移除 {factor1}")
                else:
                    # IC相同，保留名称更短的（通常更基础）
                    if len(factor1) <= len(factor2):
                        factors_to_remove.add(factor2)
                        print(f"  {factor1:40s} vs {factor2:40s} | 相关性: {corr_value:.4f} | IC相同 -> 移除 {factor2}")
                    else:
                        factors_to_remove.add(factor1)
                        print(f"  {factor1:40s} vs {factor2:40s} | 相关性: {corr_value:.4f} | IC相同 -> 移除 {factor1}")
    
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
    针对长期目标（未来20日平均价格）优化
    
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
    
    # 提取基础因子组件（针对长期目标，使用更长的窗口）
    base_components = {
        'Price_Change_1': '($close - Ref($close, 1)) / Ref($close, 1)',
        'Price_Change_5': '($close - Ref($close, 5)) / Ref($close, 5)',
        'Price_Change_10': '($close - Ref($close, 10)) / Ref($close, 10)',
        'Price_Change_20': '($close - Ref($close, 20)) / Ref($close, 20)',
        'MA_Ratio_5': '$close / Mean($close, 5) - 1',
        'MA_Ratio_10': '$close / Mean($close, 10) - 1',
        'MA_Ratio_20': '$close / Mean($close, 20) - 1',
        'MA_Ratio_30': '$close / Mean($close, 30) - 1',
        'MA_Ratio_60': '$close / Mean($close, 60) - 1',
        'Slope_Close_5': 'Slope($close, 5) / $close',
        'Slope_Close_10': 'Slope($close, 10) / $close',
        'Slope_Close_20': 'Slope($close, 20) / $close',
        'CumSum_Return_5': 'Sum(($close - Ref($close, 1)) / Ref($close, 1), 5)',
        'CumSum_Return_10': 'Sum(($close - Ref($close, 1)) / Ref($close, 1), 10)',
        'CumSum_Return_20': 'Sum(($close - Ref($close, 1)) / Ref($close, 1), 20)',
        'Max_Gain_10': '(($close - Min($close, 10)) / Min($close, 10))',
        'Max_Gain_20': '(($close - Min($close, 20)) / Min($close, 20))',
        'Resi_Close_10': 'Resi($close, 10) / $close',
        'Resi_Close_20': 'Resi($close, 20) / $close',
    }
    
    # 策略1: 基于长期趋势的因子组合
    for ma_window in [20, 30, 60]:
        for slope_window in [10, 20]:
            new_name = f"MA{ma_window}_Slope{slope_window}_LongTerm"
            if new_name not in tried_factors:
                new_expr = f"($close / Mean($close, {ma_window}) - 1) * (Slope($close, {slope_window}) / $close)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略2: 多周期价格变化组合（长期目标需要多周期信息）
    for short_window in [5, 10]:
        for long_window in [20, 30]:
            new_name = f"Price_Change_{short_window}_{long_window}_LongTerm"
            if new_name not in tried_factors:
                short_change = base_components.get(f'Price_Change_{short_window}', f'($close - Ref($close, {short_window})) / Ref($close, {short_window})')
                long_change = base_components.get(f'Price_Change_{long_window}', f'($close - Ref($close, {long_window})) / Ref($close, {long_window})')
                new_expr = f"{short_change} * {long_change}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略3: 长期MA_Ratio组合
    ma_combos = [
        ('MA_Ratio_10', 'MA_Ratio_20', 'MA_Ratio_10_20_LongTerm'),
        ('MA_Ratio_20', 'MA_Ratio_30', 'MA_Ratio_20_30_LongTerm'),
        ('MA_Ratio_30', 'MA_Ratio_60', 'MA_Ratio_30_60_LongTerm'),
    ]
    for combo in ma_combos:
        name1, name2, new_name = combo
        if new_name not in tried_factors:
            expr1 = base_components.get(name1, '')
            expr2 = base_components.get(name2, '')
            if expr1 and expr2:
                new_expr = f"{expr1} * {expr2}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略4: 价格位置因子（长期窗口）
    for window in [20, 30, 60]:
        new_name = f"Price_Position_{window}_LongTerm"
        if new_name not in tried_factors:
            new_expr = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略5: 长期动量因子
    for window in [10, 20, 30]:
        new_name = f"Momentum_{window}_LongTerm"
        if new_name not in tried_factors:
            new_expr = f"($close - Ref($close, {window})) / Ref($close, {window})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略6: 价格与长期均线的距离
    for ma_window in [20, 30, 60]:
        new_name = f"Price_MA_Distance_{ma_window}_LongTerm"
        if new_name not in tried_factors:
            new_expr = f"($close - Mean($close, {ma_window})) / (Std($close, {ma_window}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略7: 长期趋势强度
    for window in [10, 20, 30]:
        new_name = f"Trend_Strength_{window}_LongTerm"
        if new_name not in tried_factors:
            slope_current = f"Slope($close, {window}) / $close"
            slope_prev = f"Ref(Slope($close, {window}), {window}) / Ref($close, {window})"
            new_expr = f"{slope_current} - {slope_prev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略8: 多时间框架一致性（短期与长期方向一致）
    for short_window in [5, 10]:
        for long_window in [20, 30]:
            new_name = f"Multi_Timeframe_{short_window}_{long_window}_LongTerm"
            if new_name not in tried_factors:
                short_change = f"($close - Ref($close, {short_window})) / Ref($close, {short_window})"
                long_change = f"($close - Ref($close, {long_window})) / Ref($close, {long_window})"
                # 方向一致时值为正
                short_sign = f"If({short_change} > 0, 1, If({short_change} < 0, -1, 0))"
                long_sign = f"If({long_change} > 0, 1, If({long_change} < 0, -1, 0))"
                new_expr = f"{short_sign} * {long_sign} * (Abs({short_change}) + Abs({long_change}))"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略9: 长期累积收益
    for window in [10, 20, 30]:
        new_name = f"Cumulative_Return_{window}_LongTerm"
        if new_name not in tried_factors:
            new_expr = f"Sum(($close - Ref($close, 1)) / Ref($close, 1), {window}) / {window}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略10: 价格变化的稳定性（长期）
    for window in [10, 20]:
        new_name = f"Price_Stability_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            volatility = f"Std($close, {window}) / $close"
            new_expr = f"{price_change} / ({volatility} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略11: MA交叉信号（长期）
    for short_ma, long_ma in [(10, 20), (20, 30), (30, 60), (10, 30), (20, 60)]:
        new_name = f"MA_Cross_{short_ma}_{long_ma}_LongTerm"
        if new_name not in tried_factors:
            new_expr = f"(Mean($close, {short_ma}) - Mean($close, {long_ma})) / (Mean($close, {long_ma}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略12: 价格与均线的相对位置变化率（长期）
    for ma_window in [20, 30, 60]:
        new_name = f"Price_MA_Relative_Change_{ma_window}_LongTerm"
        if new_name not in tried_factors:
            ratio_current = f"$close / Mean($close, {ma_window})"
            ratio_prev = f"Ref($close, {ma_window // 2}) / Mean(Ref($close, {ma_window // 2}), {ma_window})"
            new_expr = f"({ratio_current} - {ratio_prev}) / ({ratio_prev} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略13: 长期趋势的加速度（斜率的变化率）
    for window in [10, 20, 30]:
        new_name = f"Trend_Acceleration_{window}_LongTerm"
        if new_name not in tried_factors:
            slope_current = f"Slope($close, {window}) / $close"
            slope_prev = f"Ref(Slope($close, {window}), {window // 2}) / Ref($close, {window // 2})"
            new_expr = f"({slope_current} - {slope_prev}) / (Abs({slope_prev}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略14: 价格在长期分位数中的位置
    for window in [20, 30, 60]:
        for quantile in [0.25, 0.5, 0.75]:
            quantile_name = int(quantile * 100)
            new_name = f"Price_Quantile_{quantile_name}_{window}_LongTerm"
            if new_name not in tried_factors:
                quantile_value = f"Quantile($close, {window}, {quantile})"
                new_expr = f"$close / ({quantile_value} + 0.0001) - 1"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略15: 长期波动率因子
    for window in [20, 30, 60]:
        new_name = f"Volatility_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            volatility = f"Std({price_change}, {window})"
            new_expr = f"{volatility}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略16: 价格与长期均线的偏离度（标准化）
    for ma_window in [20, 30, 60]:
        new_name = f"Price_MA_Deviation_{ma_window}_LongTerm"
        if new_name not in tried_factors:
            deviation = f"($close - Mean($close, {ma_window})) / (Std($close, {ma_window}) + 0.0001)"
            new_expr = f"{deviation}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略17: 长期动量衰减（短期动量 vs 长期动量）
    for short_window in [5, 10]:
        for long_window in [20, 30]:
            new_name = f"Momentum_Decay_{short_window}_{long_window}_LongTerm"
            if new_name not in tried_factors:
                momentum_short = f"($close - Ref($close, {short_window})) / Ref($close, {short_window})"
                momentum_long = f"($close - Ref($close, {long_window})) / Ref($close, {long_window})"
                new_expr = f"{momentum_short} - {momentum_long}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略18: 价格变化的周期性（自相关）
    for lag in [5, 10, 20]:
        new_name = f"Price_Autocorr_{lag}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            price_change_lag = f"Ref({price_change}, {lag})"
            new_expr = f"Corr({price_change}, {price_change_lag}, {lag * 2})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略19: 长期趋势的一致性（连续同向变化）
    for window in [10, 20]:
        new_name = f"Trend_Consistency_{window}_LongTerm"
        if new_name not in tried_factors:
            up_days = f"Sum(($close > Ref($close, 1)), {window})"
            down_days = f"Sum(($close < Ref($close, 1)), {window})"
            new_expr = f"({up_days} - {down_days}) / {window}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略20: 价格与均线的距离变化率
    for ma_window in [20, 30, 60]:
        new_name = f"MA_Distance_Change_{ma_window}_LongTerm"
        if new_name not in tried_factors:
            dist_current = f"($close - Mean($close, {ma_window})) / Mean($close, {ma_window})"
            dist_prev = f"(Ref($close, {ma_window // 2}) - Mean(Ref($close, {ma_window // 2}), {ma_window})) / Mean(Ref($close, {ma_window // 2}), {ma_window})"
            new_expr = f"{dist_current} - {dist_prev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略21: 长期价格突破（突破近期高低点）
    for window in [20, 30, 60]:
        # 突破高点
        new_name = f"Breakout_High_{window}_LongTerm"
        if new_name not in tried_factors:
            new_expr = f"($close - Max($close, {window})) / (Max($close, {window}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
        
        # 突破低点
        new_name = f"Breakout_Low_{window}_LongTerm"
        if new_name not in tried_factors:
            new_expr = f"(Min($close, {window}) - $close) / (Min($close, {window}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略22: 长期RSV因子（价格在高低点区间中的位置）
    for window in [20, 30, 60]:
        new_name = f"RSV_{window}_LongTerm"
        if new_name not in tried_factors:
            new_expr = f"($close - Min($low, {window})) / (Max($high, {window}) - Min($low, {window}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略23: 长期价格变化的平滑度
    for window in [10, 20]:
        new_name = f"Price_Smoothness_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            volatility = f"Std({price_change}, {window})"
            new_expr = f"1 / ({volatility} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略24: 长期价格变化的非对称性（上涨 vs 下跌）
    for window in [20, 30]:
        new_name = f"Price_Asymmetry_{window}_LongTerm"
        if new_name not in tried_factors:
            up_changes = f"Sum(If(($close - Ref($close, 1)) / Ref($close, 1) > 0, ($close - Ref($close, 1)) / Ref($close, 1), 0), {window})"
            down_changes = f"Sum(If(($close - Ref($close, 1)) / Ref($close, 1) < 0, Abs(($close - Ref($close, 1)) / Ref($close, 1)), 0), {window})"
            new_expr = f"({up_changes} - {down_changes}) / ({up_changes} + {down_changes} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略25: 长期价格变化的相对强度（相对于历史波动率）
    for window in [20, 30, 60]:
        new_name = f"Price_Relative_Strength_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            avg_change = f"Mean({price_change}, {window})"
            std_change = f"Std({price_change}, {window})"
            new_expr = f"({price_change} - {avg_change}) / ({std_change} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略26: 长期价格与均线的角度（斜率比）
    for ma_window in [20, 30, 60]:
        new_name = f"Price_MA_Angle_{ma_window}_LongTerm"
        if new_name not in tried_factors:
            price_slope = f"Slope($close, {ma_window}) / $close"
            ma_slope = f"Slope(Mean($close, {ma_window}), {ma_window}) / Mean($close, {ma_window})"
            new_expr = f"{price_slope} / ({ma_slope} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略27: 长期价格变化的累积效应（标准化）
    for window in [10, 20, 30]:
        new_name = f"Cumulative_Return_Norm_{window}_LongTerm"
        if new_name not in tried_factors:
            cum_return = f"Sum(($close - Ref($close, 1)) / Ref($close, 1), {window})"
            volatility = f"Std(($close - Ref($close, 1)) / Ref($close, 1), {window})"
            new_expr = f"{cum_return} / ({volatility} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略28: 长期价格变化的稳定性变化（波动率的变化率）
    for window in [20, 30]:
        new_name = f"Price_Stability_Change_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            volatility_current = f"Std({price_change}, {window})"
            volatility_prev = f"Ref(Std({price_change}, {window}), {window})"
            new_expr = f"({volatility_prev} - {volatility_current}) / ({volatility_prev} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略29: 长期价格与均线的相对位置（分位数）
    for ma_window in [20, 30, 60]:
        new_name = f"Price_MA_Quantile_{ma_window}_LongTerm"
        if new_name not in tried_factors:
            price_ma_ratio = f"$close / Mean($close, {ma_window})"
            quantile_val = f"Quantile({price_ma_ratio}, {ma_window * 2}, 0.5)"
            new_expr = f"({price_ma_ratio} - {quantile_val}) / ({quantile_val} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略30: 长期价格变化的偏度（分布不对称性）
    for window in [20, 30]:
        new_name = f"Price_Skewness_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            median = f"Quantile({price_change}, {window}, 0.5)"
            mean_val = f"Mean({price_change}, {window})"
            std_val = f"Std({price_change}, {window})"
            new_expr = f"({median} - {mean_val}) / ({std_val} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略31: 长期价格变化的峰度（分布尖峰程度）
    for window in [20, 30]:
        new_name = f"Price_Kurtosis_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            q75 = f"Quantile({price_change}, {window}, 0.75)"
            q25 = f"Quantile({price_change}, {window}, 0.25)"
            std_val = f"Std({price_change}, {window})"
            new_expr = f"({q75} - {q25}) / ({std_val} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略32: 长期价格与均线的交叉强度
    for short_ma, long_ma in [(10, 20), (20, 30), (30, 60)]:
        new_name = f"MA_Cross_Intensity_{short_ma}_{long_ma}_LongTerm"
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
    
    # 策略33: 长期价格变化的持续性
    for window in [10, 20]:
        new_name = f"Price_Persistence_{window}_LongTerm"
        if new_name not in tried_factors:
            up_days = f"Sum(($close > Ref($close, 1)), {window})"
            down_days = f"Sum(($close < Ref($close, 1)), {window})"
            new_expr = f"({up_days} - {down_days}) / {window}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略34: 长期价格变化的动量与反转交互
    for momentum_window in [10, 20]:
        for reversal_window in [3, 5]:
            new_name = f"Momentum_Reversal_{momentum_window}_{reversal_window}_LongTerm"
            if new_name not in tried_factors:
                momentum = f"($close - Ref($close, {momentum_window})) / Ref($close, {momentum_window})"
                reversal = f"(Ref($close, {reversal_window}) / $close - 1)"
                new_expr = f"{momentum} * {reversal} * ({momentum} + {reversal})"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略35: 长期价格变化的相对波动率
    for window in [20, 30, 60]:
        new_name = f"Price_Change_Relative_Volatility_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            volatility = f"Std({price_change}, {window})"
            new_expr = f"{price_change} / ({volatility} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # === 继续发散思维：更多创新因子策略（36-70）===
    
    # 策略36: 非线性组合 - 平方项（捕捉极端变化）
    for ma_window in [20, 30, 60]:
        new_name = f"MA_Ratio_Squared_{ma_window}_LongTerm"
        if new_name not in tried_factors:
            ma_ratio = f"$close / Mean($close, {ma_window}) - 1"
            new_expr = f"{ma_ratio} * {ma_ratio}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略37: 价格变化的滞后效应组合
    for lag1, lag2 in [(5, 10), (10, 20), (5, 20)]:
        new_name = f"Price_Change_Lag_{lag1}_{lag2}_LongTerm"
        if new_name not in tried_factors:
            change_lag1 = f"Ref(($close - Ref($close, 1)) / Ref($close, 1), {lag1})"
            change_lag2 = f"Ref(($close - Ref($close, 1)) / Ref($close, 1), {lag2})"
            new_expr = f"{change_lag1} * {change_lag2}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略38: 分位数差（捕捉价格分布的变化）
    for window in [20, 30, 60]:
        new_name = f"Quantile_Range_{window}_LongTerm"
        if new_name not in tried_factors:
            q75 = f"Quantile($close, {window}, 0.75)"
            q25 = f"Quantile($close, {window}, 0.25)"
            new_expr = f"({q75} - {q25}) / $close"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略39: 极值比（最高价与最低价的比率）
    for window in [20, 30, 60]:
        new_name = f"Extreme_Ratio_{window}_LongTerm"
        if new_name not in tried_factors:
            max_price = f"Max($close, {window})"
            min_price = f"Min($close, {window})"
            new_expr = f"{max_price} / ({min_price} + 0.0001) - 1"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略40: 价格变化的差分（一阶差分）
    for window in [10, 20, 30]:
        new_name = f"Price_Change_Diff_{window}_LongTerm"
        if new_name not in tried_factors:
            change_current = f"($close - Ref($close, {window})) / Ref($close, {window})"
            change_prev = f"(Ref($close, {window}) - Ref($close, {window * 2})) / Ref($close, {window * 2})"
            new_expr = f"{change_current} - {change_prev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略41: 多周期移动平均的交互（三均线）
    for ma1, ma2, ma3 in [(10, 20, 30), (20, 30, 60), (10, 30, 60)]:
        new_name = f"MA_Triple_{ma1}_{ma2}_{ma3}_LongTerm"
        if new_name not in tried_factors:
            ma1_ratio = f"$close / Mean($close, {ma1}) - 1"
            ma2_ratio = f"$close / Mean($close, {ma2}) - 1"
            ma3_ratio = f"$close / Mean($close, {ma3}) - 1"
            new_expr = f"{ma1_ratio} * {ma2_ratio} * {ma3_ratio}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略42: 价格与均线的距离的平方（捕捉极端偏离）
    for ma_window in [20, 30, 60]:
        new_name = f"Price_MA_Distance_Squared_{ma_window}_LongTerm"
        if new_name not in tried_factors:
            distance = f"($close - Mean($close, {ma_window})) / Mean($close, {ma_window})"
            new_expr = f"{distance} * {distance}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略43: 价格变化的符号一致性
    for window in [10, 20, 30]:
        new_name = f"Price_Sign_Consistency_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            up_count = f"Sum(If({price_change} > 0, 1, 0), {window})"
            down_count = f"Sum(If({price_change} < 0, 1, 0), {window})"
            new_expr = f"({up_count} - {down_count}) / {window}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略44: 价格变化的幅度一致性
    for window in [10, 20]:
        new_name = f"Price_Magnitude_Consistency_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            abs_change = f"Abs({price_change})"
            avg_abs_change = f"Mean({abs_change}, {window})"
            new_expr = f"{abs_change} / ({avg_abs_change} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略45: 价格与均线的相对位置变化（分位数变化）
    for ma_window in [20, 30]:
        new_name = f"Price_MA_Quantile_Change_{ma_window}_LongTerm"
        if new_name not in tried_factors:
            ratio_current = f"$close / Mean($close, {ma_window})"
            ratio_prev = f"Ref($close, {ma_window // 2}) / Mean(Ref($close, {ma_window // 2}), {ma_window})"
            quantile_current = f"Quantile({ratio_current}, {ma_window}, 0.5)"
            quantile_prev = f"Ref(Quantile({ratio_prev}, {ma_window}, 0.5), {ma_window // 2})"
            new_expr = f"{quantile_current} - {quantile_prev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略46: 价格变化的波动率聚类
    for window in [20, 30]:
        new_name = f"Volatility_Clustering_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            vol_current = f"Std({price_change}, {window})"
            vol_prev = f"Ref(Std({price_change}, {window}), {window})"
            new_expr = f"{vol_current} * {vol_prev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略47: MA交叉频率
    for ma_window in [20, 30]:
        new_name = f"MA_Cross_Frequency_{ma_window}_LongTerm"
        if new_name not in tried_factors:
            above_ma = f"$close > Mean($close, {ma_window})"
            above_ma_prev = f"Ref($close, 1) > Mean(Ref($close, 1), {ma_window})"
            cross_signal = f"If({above_ma} != {above_ma_prev}, 1, 0)"
            cross_count = f"Sum({cross_signal}, {ma_window})"
            new_expr = f"{cross_count} / ({ma_window} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略48: 价格变化的相对强度（相对于历史分位数）
    for window in [20, 30]:
        for quantile in [0.25, 0.5, 0.75]:
            q_name = int(quantile * 100)
            new_name = f"Price_Change_Relative_Q{q_name}_{window}_LongTerm"
            if new_name not in tried_factors:
                price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
                quantile_val = f"Quantile({price_change}, {window}, {quantile})"
                new_expr = f"{price_change} / ({quantile_val} + 0.0001)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略49: 价格与均线的距离的标准化累积
    for ma_window in [20, 30]:
        new_name = f"MA_Distance_Norm_Cumulative_{ma_window}_LongTerm"
        if new_name not in tried_factors:
            distance = f"($close - Mean($close, {ma_window})) / (Std($close, {ma_window}) + 0.0001)"
            cum_distance = f"Sum({distance}, {ma_window // 2})"
            new_expr = f"{cum_distance} / ({ma_window // 2} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略50: 价格变化的相对强度（相对于历史波动率分位数）
    for window in [20, 30]:
        new_name = f"Price_Change_Relative_Vol_Quantile_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            volatility = f"Std({price_change}, {window})"
            vol_quantile = f"Quantile({volatility}, {window * 2}, 0.5)"
            new_expr = f"{price_change} / ({vol_quantile} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略51: 价格与均线的相对位置（多分位数交互）
    for ma_window in [20, 30]:
        new_name = f"Price_MA_Multi_Quantile_Interaction_{ma_window}_LongTerm"
        if new_name not in tried_factors:
            ratio = f"$close / Mean($close, {ma_window})"
            q25 = f"Quantile({ratio}, {ma_window}, 0.25)"
            q50 = f"Quantile({ratio}, {ma_window}, 0.5)"
            q75 = f"Quantile({ratio}, {ma_window}, 0.75)"
            pos_25 = f"({ratio} - {q25}) / ({q50} - {q25} + 0.0001)"
            pos_75 = f"({q75} - {ratio}) / ({q75} - {q50} + 0.0001)"
            new_expr = f"{pos_25} * {pos_75}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略52: 价格变化的相对强度（相对于历史平均绝对变化）
    for window in [20, 30]:
        new_name = f"Price_Change_Relative_AbsMean_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            abs_mean = f"Mean(Abs({price_change}), {window})"
            new_expr = f"{price_change} / ({abs_mean} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略53: 价格与均线的距离的累积变化
    for ma_window in [20, 30]:
        new_name = f"MA_Distance_Cumulative_Change_{ma_window}_LongTerm"
        if new_name not in tried_factors:
            distance_current = f"($close - Mean($close, {ma_window})) / Mean($close, {ma_window})"
            distance_prev = f"(Ref($close, {ma_window // 2}) - Mean(Ref($close, {ma_window // 2}), {ma_window})) / Mean(Ref($close, {ma_window // 2}), {ma_window})"
            cum_current = f"Sum({distance_current}, {ma_window // 2})"
            cum_prev = f"Ref(Sum({distance_prev}, {ma_window // 2}), {ma_window // 2})"
            new_expr = f"{cum_current} - {cum_prev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略54: 价格变化的相对位置（相对于历史分位数范围）
    for window in [20, 30]:
        new_name = f"Price_Change_Relative_Range_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            q75 = f"Quantile({price_change}, {window}, 0.75)"
            q25 = f"Quantile({price_change}, {window}, 0.25)"
            q50 = f"Quantile({price_change}, {window}, 0.5)"
            new_expr = f"({price_change} - {q50}) / ({q75} - {q25} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略55: 价格与均线的相对位置（多时间框架）
    for short_ma, long_ma in [(10, 20), (20, 30), (30, 60)]:
        new_name = f"Price_MA_Multi_Timeframe_{short_ma}_{long_ma}_LongTerm"
        if new_name not in tried_factors:
            ratio_short = f"$close / Mean($close, {short_ma})"
            ratio_long = f"$close / Mean($close, {long_ma})"
            new_expr = f"{ratio_short} / ({ratio_long} + 0.0001) - 1"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略56: 价格变化的相对强度（相对于历史分位数，多分位数组合）
    for window in [20, 30]:
        new_name = f"Price_Change_Multi_Quantile_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            q25 = f"Quantile({price_change}, {window}, 0.25)"
            q50 = f"Quantile({price_change}, {window}, 0.5)"
            q75 = f"Quantile({price_change}, {window}, 0.75)"
            pos_25 = f"({price_change} - {q25}) / ({q50} - {q25} + 0.0001)"
            pos_75 = f"({q75} - {price_change}) / ({q75} - {q50} + 0.0001)"
            new_expr = f"{pos_25} * {pos_75}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略57: 价格与均线的距离的标准化变化率
    for ma_window in [20, 30, 60]:
        new_name = f"MA_Distance_Norm_Change_Rate_{ma_window}_LongTerm"
        if new_name not in tried_factors:
            dist_current = f"($close - Mean($close, {ma_window})) / (Std($close, {ma_window}) + 0.0001)"
            dist_prev = f"(Ref($close, {ma_window // 2}) - Mean(Ref($close, {ma_window // 2}), {ma_window})) / (Std(Ref($close, {ma_window // 2}), {ma_window}) + 0.0001)"
            new_expr = f"({dist_current} - {dist_prev}) / (Abs({dist_prev}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略58: 价格变化的动量衰减率
    for short_window, long_window in [(5, 20), (10, 30), (10, 60)]:
        new_name = f"Momentum_Decay_Rate_{short_window}_{long_window}_LongTerm"
        if new_name not in tried_factors:
            momentum_short = f"($close - Ref($close, {short_window})) / Ref($close, {short_window})"
            momentum_long = f"($close - Ref($close, {long_window})) / Ref($close, {long_window})"
            new_expr = f"{momentum_short} / ({momentum_long} + 0.0001) - 1"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略59: 价格与均线的相对位置（多分位数，标准化）
    for ma_window in [20, 30, 60]:
        for quantile in [0.25, 0.5, 0.75]:
            q_name = int(quantile * 100)
            new_name = f"Price_MA_Q{q_name}_Norm_{ma_window}_LongTerm"
            if new_name not in tried_factors:
                ratio = f"$close / Mean($close, {ma_window})"
                quantile_val = f"Quantile({ratio}, {ma_window}, {quantile})"
                new_expr = f"({ratio} - {quantile_val}) / ({quantile_val} + 0.0001)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略60: 价格变化的周期性（不同周期的相关性）
    for period1, period2 in [(5, 10), (10, 20), (5, 20)]:
        new_name = f"Price_Cycle_{period1}_{period2}_LongTerm"
        if new_name not in tried_factors:
            change1 = f"($close - Ref($close, {period1})) / Ref($close, {period1})"
            change2 = f"($close - Ref($close, {period2})) / Ref($close, {period2})"
            new_expr = f"Corr({change1}, {change2}, {period2})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略61: 价格与均线的距离的累积效应
    for ma_window in [20, 30]:
        new_name = f"MA_Distance_Cumulative_{ma_window}_LongTerm"
        if new_name not in tried_factors:
            distance = f"($close - Mean($close, {ma_window})) / Mean($close, {ma_window})"
            cum_distance = f"Sum({distance}, {ma_window // 2})"
            new_expr = f"{cum_distance} / ({ma_window // 2} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略62: 价格变化的相对位置（在历史变化中的分位数）
    for window in [20, 30]:
        new_name = f"Price_Change_Quantile_Position_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            quantile_50 = f"Quantile({price_change}, {window}, 0.5)"
            quantile_25 = f"Quantile({price_change}, {window}, 0.25)"
            quantile_75 = f"Quantile({price_change}, {window}, 0.75)"
            new_expr = f"({price_change} - {quantile_50}) / ({quantile_75} - {quantile_25} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略63: 价格与均线的相对位置变化（标准化）
    for ma_window in [20, 30, 60]:
        new_name = f"Price_MA_Relative_Position_{ma_window}_LongTerm"
        if new_name not in tried_factors:
            ratio_current = f"$close / Mean($close, {ma_window})"
            ratio_min = f"Min({ratio_current}, {ma_window})"
            ratio_max = f"Max({ratio_current}, {ma_window})"
            new_expr = f"({ratio_current} - {ratio_min}) / ({ratio_max} - {ratio_min} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略64: 价格变化的稳定性指标（低波动时的信号强度）
    for window in [20, 30]:
        new_name = f"Price_Stability_Signal_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            volatility = f"Std({price_change}, {window})"
            smoothness = f"1 / ({volatility} + 0.0001)"
            ma_ratio = f"$close / Mean($close, {window}) - 1"
            new_expr = f"{smoothness} * {ma_ratio}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略65: 价格与均线的距离的符号一致性
    for ma_window in [20, 30]:
        new_name = f"Price_MA_Distance_Sign_{ma_window}_LongTerm"
        if new_name not in tried_factors:
            distance = f"($close - Mean($close, {ma_window})) / Mean($close, {ma_window})"
            above_count = f"Sum(If({distance} > 0, 1, 0), {ma_window // 2})"
            below_count = f"Sum(If({distance} < 0, 1, 0), {ma_window // 2})"
            new_expr = f"({above_count} - {below_count}) / ({ma_window // 2} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略66: 价格变化的相对强度（相对于历史极值）
    for window in [20, 30]:
        new_name = f"Price_Change_Relative_Extreme_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            max_change = f"Max({price_change}, {window})"
            min_change = f"Min({price_change}, {window})"
            new_expr = f"({price_change} - {min_change}) / ({max_change} - {min_change} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略67: 价格与均线的距离的累积变化
    for ma_window in [20, 30]:
        new_name = f"MA_Distance_Cumulative_Change_{ma_window}_LongTerm"
        if new_name not in tried_factors:
            distance_current = f"($close - Mean($close, {ma_window})) / Mean($close, {ma_window})"
            distance_prev = f"(Ref($close, {ma_window // 2}) - Mean(Ref($close, {ma_window // 2}), {ma_window})) / Mean(Ref($close, {ma_window // 2}), {ma_window})"
            cum_current = f"Sum({distance_current}, {ma_window // 2})"
            cum_prev = f"Ref(Sum({distance_prev}, {ma_window // 2}), {ma_window // 2})"
            new_expr = f"{cum_current} - {cum_prev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略68: 价格变化的相对位置（相对于历史平均变化）
    for window in [20, 30]:
        new_name = f"Price_Change_Relative_Mean_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            mean_change = f"Mean({price_change}, {window})"
            std_change = f"Std({price_change}, {window})"
            new_expr = f"({price_change} - {mean_change}) / ({std_change} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略69: 价格与均线的相对位置（多分位数，累积）
    for ma_window in [20, 30]:
        new_name = f"Price_MA_Quantile_Cumulative_{ma_window}_LongTerm"
        if new_name not in tried_factors:
            ratio = f"$close / Mean($close, {ma_window})"
            quantile_50 = f"Quantile({ratio}, {ma_window}, 0.5)"
            ratio_diff = f"{ratio} - {quantile_50}"
            cum_diff = f"Sum({ratio_diff}, {ma_window // 2})"
            new_expr = f"{cum_diff} / ({ma_window // 2} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略70: 价格变化的相对强度（相对于历史分位数，累积）
    for window in [20, 30]:
        new_name = f"Price_Change_Quantile_Cumulative_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            quantile_50 = f"Quantile({price_change}, {window}, 0.5)"
            change_diff = f"{price_change} - {quantile_50}"
            cum_diff = f"Sum({change_diff}, {window // 2})"
            new_expr = f"{cum_diff} / ({window // 2} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略71: 基于好因子的扩展（如果存在好因子）
    if len(good_factors) > 0:
        # 取前5个好因子进行扩展
        top_good_factors = list(good_factors.items())[:5]
        for factor_name, factor_expr in top_good_factors:
            # 尝试添加不同的时间窗口组合
            for ma_window in [40, 50, 80, 100]:
                new_name = f"{factor_name}_MA{ma_window}_Extend"
                if new_name not in tried_factors:
                    ma_ratio = f"$close / Mean($close, {ma_window}) - 1"
                    new_expr = f"{factor_expr} * {ma_ratio}"
                    new_factors[new_name] = new_expr
                    if len(new_factors) >= max_new_factors:
                        return new_factors
            
            # 尝试添加斜率组合
            for slope_window in [15, 25, 40]:
                new_name = f"{factor_name}_Slope{slope_window}_Extend"
                if new_name not in tried_factors:
                    slope = f"Slope($close, {slope_window}) / $close"
                    new_expr = f"{factor_expr} * {slope}"
                    new_factors[new_name] = new_expr
                    if len(new_factors) >= max_new_factors:
                        return new_factors
    
    # 策略72: 价格变化的相对强度（相对于历史分位数，多分位数，累积变化）
    for window in [20, 30]:
        new_name = f"Price_Change_Quantile_CumChange_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            quantile_50 = f"Quantile({price_change}, {window}, 0.5)"
            change_diff = f"{price_change} - {quantile_50}"
            cum_current = f"Sum({change_diff}, {window // 2})"
            cum_prev = f"Ref(Sum({change_diff}, {window // 2}), {window // 2})"
            new_expr = f"{cum_current} - {cum_prev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略73: 价格与均线的相对位置（多分位数，累积变化率）
    for ma_window in [20, 30]:
        new_name = f"Price_MA_Quantile_CumChange_Rate_{ma_window}_LongTerm"
        if new_name not in tried_factors:
            ratio = f"$close / Mean($close, {ma_window})"
            quantile_50 = f"Quantile({ratio}, {ma_window}, 0.5)"
            ratio_diff = f"{ratio} - {quantile_50}"
            cum_current = f"Sum({ratio_diff}, {ma_window // 2})"
            cum_prev = f"Ref(Sum({ratio_diff}, {ma_window // 2}), {ma_window // 2})"
            new_expr = f"({cum_current} - {cum_prev}) / (Abs({cum_prev}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略74: 价格变化的相对强度（相对于历史分位数，多分位数，累积变化率）
    for window in [20, 30]:
        new_name = f"Price_Change_Quantile_CumChange_Rate_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            quantile_50 = f"Quantile({price_change}, {window}, 0.5)"
            change_diff = f"{price_change} - {quantile_50}"
            cum_current = f"Sum({change_diff}, {window // 2})"
            cum_prev = f"Ref(Sum({change_diff}, {window // 2}), {window // 2})"
            new_expr = f"({cum_current} - {cum_prev}) / (Abs({cum_prev}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略75: 价格与均线的距离的标准化累积（多时间框架）
    for short_ma, long_ma in [(10, 20), (20, 30)]:
        new_name = f"MA_Distance_Norm_Cum_{short_ma}_{long_ma}_LongTerm"
        if new_name not in tried_factors:
            dist_short = f"($close - Mean($close, {short_ma})) / (Std($close, {short_ma}) + 0.0001)"
            dist_long = f"($close - Mean($close, {long_ma})) / (Std($close, {long_ma}) + 0.0001)"
            cum_short = f"Sum({dist_short}, {short_ma // 2})"
            cum_long = f"Sum({dist_long}, {long_ma // 2})"
            new_expr = f"{cum_short} - {cum_long}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略76-80: 多周期价格变化的一致性（趋势确认）
    for short_period in [3, 5]:
        for long_period in [10, 20]:
            for consistency_window in [5, 10]:
                new_name = f"Price_Change_Consistency_{short_period}_{long_period}_{consistency_window}_LongTerm"
                if new_name not in tried_factors:
                    short_change = f"($close - Ref($close, {short_period})) / Ref($close, {short_period})"
                    long_change = f"($close - Ref($close, {long_period})) / Ref($close, {long_period})"
                    sign_consistency = f"Sign({short_change}) * Sign({long_change})"
                    consistency = f"Mean({sign_consistency}, {consistency_window})"
                    magnitude = f"Abs({short_change}) * Abs({long_change})"
                    new_expr = f"{consistency} * {magnitude}"
                    new_factors[new_name] = new_expr
                    if len(new_factors) >= max_new_factors:
                        return new_factors
    
    # 策略81-85: 价格加速度的标准化（二阶导数）
    for window in [5, 10, 20]:
        for norm_window in [10, 20]:
            new_name = f"Price_Acceleration_Norm_{window}_{norm_window}_LongTerm"
            if new_name not in tried_factors:
                velocity = f"($close - Ref($close, 1)) / Ref($close, 1)"
                prev_velocity = f"Ref(($close - Ref($close, 1)) / Ref($close, 1), {window})"
                acceleration = f"{velocity} - {prev_velocity}"
                std_accel = f"Std({acceleration}, {norm_window})"
                new_expr = f"{acceleration} / ({std_accel} + 0.0001)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略86-90: 价格与均线的角度变化（趋势强度变化率）
    for ma_window in [10, 20, 30]:
        for change_window in [5, 10]:
            new_name = f"Price_MA_Angle_Change_{ma_window}_{change_window}_LongTerm"
            if new_name not in tried_factors:
                current_dist = f"($close - Mean($close, {ma_window})) / Mean($close, {ma_window})"
                prev_dist = f"Ref(($close - Mean($close, {ma_window})) / Mean($close, {ma_window}), {change_window})"
                angle_change = f"({current_dist} - {prev_dist}) / (Abs({prev_dist}) + 0.0001)"
                new_expr = f"{angle_change} * {current_dist}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略91-95: 价格分位数位置的变化率（相对强度变化）
    for window in [20, 30, 60]:
        for change_window in [5, 10]:
            new_name = f"Price_Quantile_Position_Change_{window}_{change_window}_LongTerm"
            if new_name not in tried_factors:
                current_pos = f"($close - Quantile($close, {window}, 0.5)) / (Quantile($close, {window}, 0.75) - Quantile($close, {window}, 0.25) + 0.0001)"
                prev_pos = f"Ref(($close - Quantile($close, {window}, 0.5)) / (Quantile($close, {window}, 0.75) - Quantile($close, {window}, 0.25) + 0.0001), {change_window})"
                position_change = f"{current_pos} - {prev_pos}"
                new_expr = f"{position_change} * Sign({current_pos})"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略96-100: 多时间框架动量的一致性（趋势确认）
    for short_mom in [5, 10]:
        for long_mom in [20, 30]:
            new_name = f"Momentum_Consistency_{short_mom}_{long_mom}_LongTerm"
            if new_name not in tried_factors:
                short_momentum = f"($close - Ref($close, {short_mom})) / Ref($close, {short_mom})"
                long_momentum = f"($close - Ref($close, {long_mom})) / Ref($close, {long_mom})"
                sign_match = f"Sign({short_momentum}) * Sign({long_momentum})"
                magnitude_ratio = f"Abs({short_momentum}) / (Abs({long_momentum}) + 0.0001)"
                new_expr = f"{sign_match} * {magnitude_ratio} * {long_momentum}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略101-105: 价格波动率的相对位置（波动率分位数）
    for vol_window in [10, 20, 30]:
        for quantile_window in [20, 40]:
            new_name = f"Volatility_Quantile_Position_{vol_window}_{quantile_window}_LongTerm"
            if new_name not in tried_factors:
                volatility = f"Std(($close - Ref($close, 1)) / Ref($close, 1), {vol_window})"
                vol_quantile_50 = f"Quantile({volatility}, {quantile_window}, 0.5)"
                vol_quantile_75 = f"Quantile({volatility}, {quantile_window}, 0.75)"
                vol_quantile_25 = f"Quantile({volatility}, {quantile_window}, 0.25)"
                vol_position = f"({volatility} - {vol_quantile_50}) / ({vol_quantile_75} - {vol_quantile_25} + 0.0001)"
                price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
                new_expr = f"{vol_position} * {price_change}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略106-110: 价格与均线的距离的动量（距离变化率）
    for ma_window in [10, 20, 30]:
        for momentum_window in [5, 10]:
            new_name = f"MA_Distance_Momentum_{ma_window}_{momentum_window}_LongTerm"
            if new_name not in tried_factors:
                current_dist = f"($close - Mean($close, {ma_window})) / Mean($close, {ma_window})"
                prev_dist = f"Ref(($close - Mean($close, {ma_window})) / Mean($close, {ma_window}), {momentum_window})"
                dist_momentum = f"({current_dist} - {prev_dist}) / (Abs({prev_dist}) + 0.0001)"
                new_expr = f"{dist_momentum} * Sign({current_dist})"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略111-115: 价格突破强度的标准化（突破幅度相对波动率）
    for breakout_window in [10, 20, 30]:
        for vol_window in [10, 20]:
            new_name = f"Breakout_Strength_Norm_{breakout_window}_{vol_window}_LongTerm"
            if new_name not in tried_factors:
                high_breakout = f"($close - Max($close, {breakout_window})) / (Max($close, {breakout_window}) + 0.0001)"
                low_breakout = f"(Min($close, {breakout_window}) - $close) / (Min($close, {breakout_window}) + 0.0001)"
                breakout = f"{high_breakout} - {low_breakout}"
                volatility = f"Std(($close - Ref($close, 1)) / Ref($close, 1), {vol_window})"
                new_expr = f"{breakout} / ({volatility} + 0.0001)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略116-120: 价格变化的自相关性（趋势持续性）
    for window in [10, 20, 30]:
        for lag in [1, 3, 5]:
            new_name = f"Price_Change_Autocorr_{window}_{lag}_LongTerm"
            if new_name not in tried_factors:
                price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
                current_change = f"Mean({price_change}, {window})"
                lagged_change = f"Ref(Mean({price_change}, {window}), {lag})"
                autocorr = f"{current_change} * {lagged_change}"
                new_expr = f"{autocorr} / (Sqrt(Abs({current_change}) * Abs({lagged_change})) + 0.0001)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略121-125: 价格与均线的相对位置变化（趋势反转信号）
    for ma_window in [10, 20, 30]:
        for reversal_window in [3, 5]:
            new_name = f"Price_MA_Reversal_Signal_{ma_window}_{reversal_window}_LongTerm"
            if new_name not in tried_factors:
                current_ratio = f"$close / Mean($close, {ma_window})"
                prev_ratio = f"Ref($close / Mean($close, {ma_window}), {reversal_window})"
                ratio_change = f"{current_ratio} - {prev_ratio}"
                prev_sign = f"Sign({prev_ratio} - 1)"
                current_sign = f"Sign({current_ratio} - 1)"
                reversal = f"(1 - {prev_sign} * {current_sign}) / 2"  # 符号相反时为1
                new_expr = f"{reversal} * {ratio_change}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略126-130: 价格变化的偏度（不对称性）
    for window in [10, 20, 30]:
        new_name = f"Price_Change_Skewness_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            mean_change = f"Mean({price_change}, {window})"
            std_change = f"Std({price_change}, {window})"
            normalized = f"({price_change} - {mean_change}) / ({std_change} + 0.0001)"
            skewness = f"Mean(Power({normalized}, 3), {window})"
            new_expr = f"{skewness} * {mean_change}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略131-135: 价格与均线的距离的累积变化（趋势强度累积）
    for ma_window in [10, 20, 30]:
        for cum_window in [5, 10]:
            new_name = f"MA_Distance_CumChange_{ma_window}_{cum_window}_LongTerm"
            if new_name not in tried_factors:
                dist = f"($close - Mean($close, {ma_window})) / Mean($close, {ma_window})"
                dist_change = f"{dist} - Ref({dist}, 1)"
                cum_change = f"Sum({dist_change}, {cum_window})"
                new_expr = f"{cum_change} * Sign({dist})"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略136-140: 多周期均线交叉的强度（趋势确认）
    for short_ma in [5, 10]:
        for long_ma in [20, 30]:
            for strength_window in [3, 5]:
                new_name = f"MA_Cross_Strength_{short_ma}_{long_ma}_{strength_window}_LongTerm"
                if new_name not in tried_factors:
                    ma_diff = f"(Mean($close, {short_ma}) - Mean($close, {long_ma})) / Mean($close, {long_ma})"
                    cross_strength = f"Mean({ma_diff}, {strength_window})"
                    cross_direction = f"Sign({ma_diff})"
                    new_expr = f"{cross_strength} * {cross_direction}"
                    new_factors[new_name] = new_expr
                    if len(new_factors) >= max_new_factors:
                        return new_factors
    
    # 策略141-145: 价格变化的峰度（极端性）
    for window in [10, 20, 30]:
        new_name = f"Price_Change_Kurtosis_{window}_LongTerm"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            mean_change = f"Mean({price_change}, {window})"
            std_change = f"Std({price_change}, {window})"
            normalized = f"({price_change} - {mean_change}) / ({std_change} + 0.0001)"
            kurtosis = f"Mean(Power({normalized}, 4), {window})"
            new_expr = f"{kurtosis} * Abs({mean_change})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略146-150: 价格与均线的相对位置的多时间框架一致性
    for short_ma in [5, 10]:
        for long_ma in [20, 30]:
            new_name = f"Price_MA_Position_Consistency_{short_ma}_{long_ma}_LongTerm"
            if new_name not in tried_factors:
                short_pos = f"($close - Mean($close, {short_ma})) / (Max($close, {short_ma}) - Min($close, {short_ma}) + 0.0001)"
                long_pos = f"($close - Mean($close, {long_ma})) / (Max($close, {long_ma}) - Min($close, {long_ma}) + 0.0001)"
                consistency = f"Sign({short_pos} - 0.5) * Sign({long_pos} - 0.5)"
                avg_position = f"({short_pos} + {long_pos}) / 2"
                new_expr = f"{consistency} * {avg_position}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    return new_factors


def analyze_custom_factors_longterm(
    instruments="csi300",
    label_expr=None,  # 如果为None，自动生成未来20日平均价格相对变化表达式
    custom_factors=None,
    start_time=None,
    end_time=None,
    freq="day",
    limit_data_days=None,
    min_valid_days=10,
    output_dir=None,
):
    """
    分析自定义因子在沪深300上的相关性（长期目标：未来20日平均价格）
    
    Parameters
    ----------
    instruments : str or list
        标的列表，可以是"csi500"、"csi300"等预定义市场，或股票代码列表
    label_expr : str
        标签表达式，默认："Mean(Ref($close, -1), 20) / $close - 1"（未来20日平均价格相对变化）
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
        
        if len(valid_instruments) == 0:
            print("⚠️  没有有效的股票代码")
            return None
        
        instruments_list = valid_instruments
        print(f"使用标的数量: {len(instruments_list)}只（固定股票池）")
    
    # 统一使用instruments_param变量
    instruments_param = instruments_list if instruments_list is not None else instruments
    
    # 获取数据时间范围
    if start_time is None or end_time is None:
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
    
    # 如果没有提供label_expr，自动生成未来20日平均价格表达式
    if label_expr is None:
        # 手动列出所有Ref，因为Mean函数不支持负数窗口（未来数据）
        ref_list = " + ".join([f"Ref($close, -{i})" for i in range(1, 21)])
        label_expr = f"(({ref_list}) / 20) / $close - 1"
    
    print(f"\n使用时间范围: {start_time} 到 {end_time}")
    print(f"标签表达式: {label_expr}")
    print(f"  含义: 未来20日平均价格相对于当前价格的收益率")
    
    # 加载因子配置（长期目标专用）
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
        print("\n基于好因子生成新的组合因子（长期目标优化）...")
        new_factors = generate_new_factors_from_good_ones(
            good_factors=good_factors,
            bad_factors=bad_factors,
            max_new_factors=600  # 大幅增加生成因子数量，支持更多发散思维（已添加策略76-150）
        )
        
        print(f"✓ 生成了 {len(new_factors)} 个新因子（已排除重复因子）")
        
        if len(new_factors) > 0:
            custom_factors = new_factors
        else:
            custom_factors = good_factors.copy()
            print("⚠️  没有生成新因子，使用配置文件中的好因子")
        
        # 如果配置文件中没有好因子且没有生成新因子，使用默认的长期因子
        if len(custom_factors) == 0:
            print("⚠️  配置文件中没有好因子，使用默认长期因子")
            custom_factors = {
                # 长期趋势因子
                "MA_Ratio_20": "$close / Mean($close, 20) - 1",
                "MA_Ratio_30": "$close / Mean($close, 30) - 1",
                "MA_Ratio_60": "$close / Mean($close, 60) - 1",
                "Slope_Close_20": "Slope($close, 20) / $close",
                "Slope_Close_30": "Slope($close, 30) / $close",
                "Price_Change_20": "($close - Ref($close, 20)) / Ref($close, 20)",
                "Price_Change_30": "($close - Ref($close, 30)) / Ref($close, 30)",
                "CumSum_Return_20": "Sum(($close - Ref($close, 1)) / Ref($close, 1), 20)",
                "Price_Position_20": "($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20) + 0.0001)",
                "Price_Position_30": "($close - Min($close, 30)) / (Max($close, 30) - Min($close, 30) + 0.0001)",
            }
    
    # 过滤掉已测试的因子
    tried_factors = {**good_factors, **bad_factors}
    original_count = len(custom_factors)
    custom_factors = {k: v for k, v in custom_factors.items() if k not in tried_factors}
    filtered_count = original_count - len(custom_factors)
    if filtered_count > 0:
        print(f"\n⚠️  跳过了 {filtered_count} 个已测试的因子")
    
    if len(custom_factors) == 0:
        print("\n⚠️  所有因子都已测试过，没有新因子需要测试")
        return None
    
    print(f"\n正在测试 {len(custom_factors)} 个自定义因子...")
    
    # 获取标签数据
    print("\n正在获取标签数据（未来20日平均价格）...")
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
        
        if isinstance(label_data.columns, pd.MultiIndex):
            label_col = label_data.columns[0]
            labels = label_data[label_col]
        else:
            labels = label_data.iloc[:, 0]
        
        print(f"标签数据量: {len(labels)}")
        
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
            
            # 对齐因子和标签
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
            
            # 计算IC
            try:
                ic_series, ric_series = calc_ic(
                    pred=aligned_factor,
                    label=aligned_label,
                    date_col="datetime",
                    dropna=True
                )
                
                if len(ic_series) >= min_valid_days:
                    mean_ic = ic_series.mean()
                    mean_ric = ric_series.mean()
                    ic_std = ic_series.std()
                    
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
        results_df['IC_abs'] = results_df['IC'].abs().fillna(0)
        results_df = results_df.sort_values('IC_abs', ascending=False)
        results_df = results_df.drop('IC_abs', axis=1)
        
        print("\n" + "="*100)
        print("长期目标因子IC分析结果（未来20日平均价格）")
        print("="*100)
        
        valid_results = results_df[results_df['status'] == 'success']
        if len(valid_results) > 0:
            print(f"\n总体统计:")
            print(f"  成功测试因子数: {len(valid_results)} / {len(factor_results)}")
            print(f"  平均IC: {valid_results['IC'].mean():.6f}")
            print(f"  IC标准差: {valid_results['IC'].std():.6f}")
            print(f"  平均Rank IC: {valid_results['Rank_IC'].mean():.6f}")
            print(f"  平均ICIR: {valid_results['ICIR'].mean():.6f}")
            
            effective_factors = valid_results[
                (valid_results['IC'].abs() > min_ic_threshold) | (valid_results['ICIR'].abs() > min_icir_threshold)
            ]
            
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
            
            # 保存结果
            if output_dir is None:
                output_dir = Path.cwd()
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / "custom_factors_longterm_results.csv"
            results_df.to_csv(output_file, index=False)
            print(f"\n结果已保存到: {output_file}")
            
            if len(effective_factors) > 0:
                effective_file = output_dir / "effective_factors_longterm.csv"
                effective_factors.to_csv(effective_file, index=False)
                print(f"有效因子已保存到: {effective_file}")
            
            # 更新因子配置文件（长期目标专用）
            print("\n更新因子配置文件（长期目标）...")
            new_good_factors = {}
            new_bad_factors = {}
            
            for idx, row in effective_factors.iterrows():
                factor_name = row['factor']
                if factor_name not in good_factors:
                    new_good_factors[factor_name] = row['expression']
            
            for idx, row in bad_factors_new.iterrows():
                factor_name = row['factor']
                if factor_name not in bad_factors and factor_name not in good_factors:
                    new_bad_factors[factor_name] = row['expression']
            
            failed_factors = results_df[results_df['status'] != 'success']
            for idx, row in failed_factors.iterrows():
                factor_name = row['factor']
                if factor_name not in bad_factors and factor_name not in good_factors:
                    new_bad_factors[factor_name] = row['expression']
            
            effective_factor_names = set(effective_factors['factor'].values) if len(effective_factors) > 0 else set()
            for idx, row in valid_results.iterrows():
                factor_name = row['factor']
                if factor_name not in effective_factor_names and factor_name not in bad_factors and factor_name not in good_factors:
                    new_bad_factors[factor_name] = row['expression']
            
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
                all_good_factors = {**good_factors, **new_good_factors}
                
                if len(all_good_factors) > 1:
                    deduplicated_factors, removed_factors = deduplicate_good_factors_by_correlation(
                        good_factors=all_good_factors,
                        instruments_list=instruments_list,  # 使用固定股票池列表
                        start_time=start_time,
                        end_time=end_time,
                        freq=freq,
                        correlation_threshold=0.85,
                        ic_results=valid_results
                    )
                    
                    if len(deduplicated_factors) < len(all_good_factors):
                        config = load_factor_config()
                        config['good_factors'] = deduplicated_factors
                        
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
    print("沪深500自定义因子相关性分析（长期目标：未来20日平均价格）")
    print("="*80)
    
    # 配置参数
    instruments = "csi300"
    
    # 长期目标：未来20日平均价格相对变化
    # 注意：Mean函数不支持负数窗口（未来数据），需要手动列出所有Ref
    # 未来20日的平均价格 = (Ref($close, -1) + Ref($close, -2) + ... + Ref($close, -20)) / 20
    ref_list = " + ".join([f"Ref($close, -{i})" for i in range(1, 21)])
    label_expr = f"(({ref_list}) / 20) / $close - 1"
    
    # 自定义因子（可选）
    custom_factors = None
    
    # 时间范围
    start_time = "2015-01-01"
    end_time = "2025-01-01"
    
    # 数据频率
    freq = "day"
    
    # 限制数据天数（可选）
    limit_data_days = None
    
    # 输出目录
    output_dir = Path(__file__).parent / "factor_evaluation_results_longterm"
    
    # 执行分析
    results = analyze_custom_factors_longterm(
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

