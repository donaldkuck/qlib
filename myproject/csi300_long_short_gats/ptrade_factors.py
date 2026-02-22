# -*- coding: utf-8 -*-
"""
Ptrade 因子计算模块

实现 Qlib 风格的因子计算函数，完全基于 numpy 和 pandas
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from scipy import stats


class FactorCalculator:
    """因子计算器"""

    @staticmethod
    def Ref(series: pd.Series, periods: int) -> pd.Series:
        """引用历史值 (shift)"""
        return series.shift(periods)

    @staticmethod
    def Mean(series: pd.Series, periods: int) -> pd.Series:
        """简单移动平均"""
        return series.rolling(window=periods, min_periods=1).mean()

    @staticmethod
    def Std(series: pd.Series, periods: int) -> pd.Series:
        """滚动标准差"""
        return series.rolling(window=periods, min_periods=1).std()

    @staticmethod
    def Sum(series: pd.Series, periods: int) -> pd.Series:
        """滚动求和"""
        return series.rolling(window=periods, min_periods=1).sum()

    @staticmethod
    def Min(series: pd.Series, periods: int) -> pd.Series:
        """滚动最小值"""
        return series.rolling(window=periods, min_periods=1).min()

    @staticmethod
    def Max(series: pd.Series, periods: int) -> pd.Series:
        """滚动最大值"""
        return series.rolling(window=periods, min_periods=1).max()

    @staticmethod
    def Quantile(series: pd.Series, periods: int, q: float) -> pd.Series:
        """滚动分位数"""
        return series.rolling(window=periods, min_periods=1).quantile(q)

    @staticmethod
    def Slope(series: pd.Series, periods: int) -> pd.Series:
        """线性回归斜率"""
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(periods - 1, len(series)):
            y = series.iloc[i - periods + 1:i + 1].values
            x = np.arange(len(y))
            if len(y) == periods:
                slope, _ = stats.linregress(x, y)[:2]
                slopes.iloc[i] = slope
        return slopes

    @staticmethod
    def If(condition: pd.Series, true_val: Union[pd.Series, float], false_val: Union[pd.Series, float]) -> pd.Series:
        """条件判断"""
        if isinstance(true_val, pd.Series):
            true_val = true_val.reindex(condition.index)
        if isinstance(false_val, pd.Series):
            false_val = false_val.reindex(condition.index)
        return np.where(condition, true_val, false_val)

    @staticmethod
    def Sign(series: pd.Series) -> pd.Series:
        """符号函数"""
        return np.sign(series)

    @staticmethod
    def Abs(series: pd.Series) -> pd.Series:
        """绝对值"""
        return np.abs(series)

    @staticmethod
    def Resi(series: pd.Series, periods: int) -> pd.Series:
        """残差（价格对均值的偏离）"""
        return series - FactorCalculator.Mean(series, periods)

    @staticmethod
    def ChangeInstrument(series: pd.Series, other_series: pd.Series) -> pd.Series:
        """切换股票（简化版：直接返回其他序列）"""
        return other_series

    @staticmethod
    def evaluate_expression(expr: str, data: pd.DataFrame) -> pd.Series:
        """
        计算 Qlib 表达式

        Parameters
        ----------
        expr : str
            Qlib 表达式，如 "($close - Ref($close, 1)) / Ref($close, 1)"
        data : pd.DataFrame
            包含 OHLCV 数据的 DataFrame

        Returns
        -------
        pd.Series
            计算结果
        """
        # 提取基础字段
        close = data['close']
        high = data['high']
        low = data['low']
        open_price = data.get('open', close)
        volume = data.get('volume', pd.Series(index=data.index))

        # 将表达式中的字段名替换为实际变量
        # 需要按照从右到左的顺序替换，避免部分匹配
        calc = expr

        # 替换 ChangeInstrument - 特殊处理
        # 格式: ChangeInstrument('SH000300', expression)
        import re
        change_instr_pattern = r"ChangeInstrument\(['\"]([^'\"]+)['\"],\s*([^)]+)\)"
        calc = re.sub(change_instr_pattern, r"\2", calc)  # 简化：忽略切换，直接使用表达式

        # 替换字段名
        calc = calc.replace('$close', 'close')
        calc = calc.replace('$high', 'high')
        calc = calc.replace('$low', 'low')
        calc = calc.replace('$open', 'open_price')
        calc = calc.replace('$volume', 'volume')

        # 替换函数名
        calc = calc.replace('Ref(', 'FactorCalculator.Ref(')
        calc = calc.replace('Mean(', 'FactorCalculator.Mean(')
        calc = calc.replace('Std(', 'FactorCalculator.Std(')
        calc = calc.replace('Sum(', 'FactorCalculator.Sum(')
        calc = calc.replace('Min(', 'FactorCalculator.Min(')
        calc = calc.replace('Max(', 'FactorCalculator.Max(')
        calc = calc.replace('Quantile(', 'FactorCalculator.Quantile(')
        calc = calc.replace('Slope(', 'FactorCalculator.Slope(')
        calc = calc.replace('If(', 'FactorCalculator.If(')
        calc = calc.replace('Sign(', 'FactorCalculator.Sign(')
        calc = calc.replace('Abs(', 'FactorCalculator.Abs(')
        calc = calc.replace('Resi(', 'FactorCalculator.Resi(')

        try:
            result = eval(calc, {'__builtins__': {}}, {
                'close': close,
                'high': high,
                'low': low,
                'open_price': open_price,
                'volume': volume,
                'FactorCalculator': FactorCalculator,
                'np': np,
                'pd': pd
            })
        except Exception as e:
            print(f"表达式计算错误: {expr}")
            print(f"错误: {e}")
            return pd.Series(index=data.index, dtype=float)

        return result


class FactorLibrary:
    """因子库 - 包含所有短期和长期因子"""

    @staticmethod
    def calculate_all_factors(data: pd.DataFrame, factor_config: dict) -> pd.DataFrame:
        """
        计算所有因子

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV 数据，包含 close, high, low, open, volume 列
        factor_config : dict
            因子配置字典，格式：{factor_name: expression}

        Returns
        -------
        pd.DataFrame
            包含所有因子的 DataFrame
        """
        factors = pd.DataFrame(index=data.index)

        for factor_name, expr in factor_config.items():
            try:
                result = FactorCalculator.evaluate_expression(expr, data)
                factors[factor_name] = result
            except Exception as e:
                print(f"因子 {factor_name} 计算失败: {e}")
                factors[factor_name] = np.nan

        return factors


# 预定义的短期因子表达式（从 factor_config.yaml 提取）
SHORT_TERM_FACTORS = {
    'Price_Change_1_MA7_Slope7': '($close - Ref($close, 1)) / Ref($close, 1) * ($close / Mean($close, 7) - 1) * (Slope($close, 7) / $close)',
    'Price_Change_Relative_Volatility_20': '($close - Ref($close, 1)) / Ref($close, 1) / (Std(($close - Ref($close, 1)) / Ref($close, 1), 20) + 0.0001)',
    'Price_Change_Squared_MA5': '($close - Ref($close, 1)) / Ref($close, 1) * ($close - Ref($close, 1)) / Ref($close, 1) * $close / Mean($close, 5) - 1',
    'Cumulative_Return_Normalized_7': 'Sum(($close - Ref($close, 1)) / Ref($close, 1), 7) / (Std(($close - Ref($close, 1)) / Ref($close, 1), 7) + 0.0001)',
    'Price_Change_1_2_3_4': '($close - Ref($close, 1)) / Ref($close, 1) * ($close - Ref($close, 2)) / Ref($close, 2) * ($close - Ref($close, 3)) / Ref($close, 3) * ($close - Ref($close, 4)) / Ref($close, 4)',
    'Cumulative_Return_Normalized_3': 'Sum(($close - Ref($close, 1)) / Ref($close, 1), 3) / (Std(($close - Ref($close, 1)) / Ref($close, 1), 3) + 0.0001)',
    'Breakout_Low_5': '(Min($close, 5) - $close) / (Min($close, 5) + 0.0001)',
    'Momentum_Reversal_Interaction_3_2': '($close - Ref($close, 3)) / Ref($close, 3) * (Ref($close, 2) / $close - 1) * (($close - Ref($close, 3)) / Ref($close, 3) + (Ref($close, 2) / $close - 1))',
    'MA_Ratio_5_15': '($close / Mean($close, 5) - 1) * ($close / Mean($close, 15) - 1)',
    'Price_Change_Quantile_20': 'Quantile(($close - Ref($close, 1)) / Ref($close, 1), 20, 0.5)',
    'Momentum_Reversal_Interaction_3_1': '($close - Ref($close, 3)) / Ref($close, 3) * (Ref($close, 1) / $close - 1) * (($close - Ref($close, 3)) / Ref($close, 3) + (Ref($close, 1) / $close - 1))',
    'Price_Change_1_MA_Ratio_7_Resi_Close_5': '($close - Ref($close, 1)) / Ref($close, 1) * $close / Mean($close, 7) - 1 * Resi($close, 5) / $close',
    'Price_Smoothness_10': '1 / (Std(($close - Ref($close, 1)) / Ref($close, 1), 10) + 0.0001)',
    'Volatility_Norm_20': 'Std($close, 20) / (Mean($close, 20) + 0.0001)',
    'Price_Change_Quantile_10': 'Quantile(($close - Ref($close, 1)) / Ref($close, 1), 10, 0.5)',
    'Stock_Market_Return_CSI100_1': '($close - Ref($close, 1)) / Ref($close, 1) - ChangeInstrument("SH000903", ($close - Ref($close, 1)) / Ref($close, 1))',
    'Stock_Market_Return_CSI500_1': '($close - Ref($close, 1)) / Ref($close, 1) - ChangeInstrument("SH000905", ($close - Ref($close, 1)) / Ref($close, 1))',
    'Stock_Market_Return_CSI300_1': '($close - Ref($close, 1)) / Ref($close, 1) - ChangeInstrument("SH000300", ($close - Ref($close, 1)) / Ref($close, 1))',
}

# 从配置文件中加载更多因子
def load_factors_from_config(config_path: str, top_n: int = None) -> dict:
    """
    从配置文件加载因子表达式

    Parameters
    ----------
    config_path : str
        配置文件路径
    top_n : int, optional
        只加载前 N 个因子

    Returns
    -------
    dict
        {factor_name: expression}
    """
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    good_factors = config.get('good_factors', {})
    factors = {}

    for factor_name, factor_info in good_factors.items():
        if isinstance(factor_info, dict):
            expr = factor_info.get('expression', '')
        else:
            expr = str(factor_info)

        if expr:
            factors[factor_name] = expr

        if top_n and len(factors) >= top_n:
            break

    return factors


def calculate_factor_series(stock_data: pd.DataFrame, factor_config: dict) -> pd.DataFrame:
    """
    计算股票的因子序列

    Parameters
    ----------
    stock_data : pd.DataFrame
        股票 OHLCV 数据
    factor_config : dict
        因子配置字典

    Returns
    -------
    pd.DataFrame
        因子值，index 为日期
    """
    # 确保数据格式正确
    if isinstance(stock_data.index, pd.DatetimeIndex):
        stock_data = stock_data.copy()
    else:
        stock_data = stock_data.reset_index()
        if 'date' in stock_data.columns:
            stock_data = stock_data.set_index('date')
        elif 'datetime' in stock_data.columns:
            stock_data = stock_data.set_index('datetime')

    # 确保有必要的列
    required_columns = ['close', 'high', 'low']
    for col in required_columns:
        if col not in stock_data.columns:
            raise ValueError(f"缺少必要的列: {col}")

    if 'open' not in stock_data.columns:
        stock_data['open'] = stock_data['close'].shift(1)
        stock_data['open'].fillna(stock_data['close'].iloc[0], inplace=True)

    if 'volume' not in stock_data.columns:
        stock_data['volume'] = pd.Series(index=stock_data.index, dtype=float)

    return FactorLibrary.calculate_all_factors(stock_data, factor_config)


def prepare_time_series_data(stock_data: pd.DataFrame, step_len: int, d_feat: int, factor_names: list) -> np.ndarray:
    """
    准备时间序列数据用于 GATs TS 模型

    Parameters
    ----------
    stock_data : pd.DataFrame
        因子数据
    step_len : int
        时间序列长度
    d_feat : int
        特征维度
    factor_names : list
        要使用的因子名称列表

    Returns
    -------
    np.ndarray
        时间序列数据，形状为 (T, F)，T 为时间步数，F 为特征数
    """
    # 确保因子顺序一致
    data = stock_data[factor_names].copy()

    # 只取最后 step_len 行
    if len(data) < step_len:
        # 数据不足，用 NaN 填充
        padded_data = np.full((step_len, d_feat), np.nan)
        padded_data[-len(data):, :min(d_feat, len(factor_names))] = data.values
        return padded_data
    else:
        return data.iloc[-step_len:, :d_feat].values


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    归一化特征（RobustZScoreNorm）

    Parameters
    ----------
    features : np.ndarray
        特征数据，形状为 (T, F)

    Returns
    -------
    np.ndarray
        归一化后的特征
    """
    result = features.copy()

    for i in range(features.shape[1]):
        col = features[:, i]
        # 使用中位数和 MAD 进行归一化
        median = np.nanmedian(col)
        mad = np.nanmedian(np.abs(col - median))

        if mad > 0:
            result[:, i] = (col - median) / mad
        else:
            result[:, i] = 0

    return result


if __name__ == "__main__":
    # 测试代码
    print("Ptrade 因子计算模块测试")

    # 创建测试数据
    dates = pd.date_range('2020-01-01', '2020-12-31')
    np.random.seed(42)
    test_data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) + np.random.rand(len(dates)) * 2,
        'low': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) - np.random.rand(len(dates)) * 2,
        'open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'volume': np.random.randint(100000, 1000000, len(dates))
    }, index=dates)

    # 计算测试因子
    factors = calculate_factor_series(test_data, SHORT_TERM_FACTORS)
    print(f"计算了 {len(factors.columns)} 个因子")
    print(factors.tail())
