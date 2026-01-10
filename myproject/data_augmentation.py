"""
数据增强处理器
用于在训练时增强时间序列数据，提高模型泛化能力
"""
import numpy as np
import pandas as pd
from typing import Union, Text, Optional
from qlib.data.dataset.processor import Processor, get_group_columns
from qlib.log import get_module_logger

logger = get_module_logger("DataAugmentation")


class GaussianNoiseAugment(Processor):
    """
    添加高斯噪声的数据增强
    
    在训练时向特征添加小幅度的高斯噪声，提高模型对噪声的鲁棒性
    """
    
    def __init__(self, fields_group="feature", noise_std=0.01, prob=0.5):
        """
        Parameters
        ----------
        fields_group : str
            要增强的字段组（通常是"feature"）
        noise_std : float
            噪声标准差（相对于特征的标准差）
        prob : float
            应用增强的概率（0-1之间）
        """
        self.fields_group = fields_group
        self.noise_std = noise_std
        self.prob = prob
    
    def is_for_infer(self) -> bool:
        """推理时不使用数据增强"""
        return False
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加高斯噪声
        
        Parameters
        ----------
        df : pd.DataFrame
            输入数据
            
        Returns
        -------
        pd.DataFrame
            增强后的数据
        """
        if np.random.random() > self.prob:
            # 按概率决定是否应用增强
            return df
        
        cols = get_group_columns(df, self.fields_group)
        if len(cols) == 0:
            return df
        
        df = df.copy()
        
        # 计算每个特征的标准差
        for col in cols:
            if col in df.columns:
                col_std = df[col].std()
                if col_std > 0:
                    # 添加高斯噪声
                    noise = np.random.normal(0, col_std * self.noise_std, size=len(df))
                    df[col] = df[col] + noise
        
        return df


class TimeWarpAugment(Processor):
    """
    时间扭曲增强（适用于时间序列数据）
    
    通过轻微调整时间序列的采样点来增强数据
    """
    
    def __init__(self, fields_group="feature", warp_factor=0.1, prob=0.3):
        """
        Parameters
        ----------
        fields_group : str
            要增强的字段组
        warp_factor : float
            扭曲因子（0-1之间，越大扭曲越明显）
        prob : float
            应用增强的概率
        """
        self.fields_group = fields_group
        self.warp_factor = warp_factor
        self.prob = prob
    
    def is_for_infer(self) -> bool:
        """推理时不使用数据增强"""
        return False
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用时间扭曲
        
        Parameters
        ----------
        df : pd.DataFrame
            输入数据
            
        Returns
        -------
        pd.DataFrame
            增强后的数据
        """
        if np.random.random() > self.prob:
            return df
        
        cols = get_group_columns(df, self.fields_group)
        if len(cols) == 0:
            return df
        
        df = df.copy()
        
        # 对于时间序列数据，按日期分组进行扭曲
        if isinstance(df.index, pd.MultiIndex) and 'datetime' in df.index.names:
            # 按日期分组
            for date, group in df.groupby(level='datetime'):
                if len(group) < 3:  # 至少需要3个样本才能扭曲
                    continue
                
                # 对每个特征列应用轻微的时间扭曲
                for col in cols:
                    if col in group.columns:
                        values = group[col].values
                        if len(values) > 2:
                            # 使用线性插值进行轻微扭曲
                            n = len(values)
                            # 生成扭曲后的索引
                            warp_indices = np.linspace(0, n-1, n) + np.random.normal(0, self.warp_factor, n)
                            warp_indices = np.clip(warp_indices, 0, n-1)
                            # 使用插值
                            warped_values = np.interp(warp_indices, np.arange(n), values)
                            df.loc[group.index, col] = warped_values
        else:
            # 对于非时间序列数据，直接对列进行轻微扭曲
            for col in cols:
                if col in df.columns:
                    values = df[col].values
                    if len(values) > 2:
                        n = len(values)
                        warp_indices = np.linspace(0, n-1, n) + np.random.normal(0, self.warp_factor, n)
                        warp_indices = np.clip(warp_indices, 0, n-1)
                        warped_values = np.interp(warp_indices, np.arange(n), values)
                        df[col] = warped_values
        
        return df


class FeatureScalingAugment(Processor):
    """
    特征缩放增强
    
    通过随机缩放特征值来增强数据
    """
    
    def __init__(self, fields_group="feature", scale_range=(0.95, 1.05), prob=0.3):
        """
        Parameters
        ----------
        fields_group : str
            要增强的字段组
        scale_range : tuple
            缩放范围（min, max）
        prob : float
            应用增强的概率
        """
        self.fields_group = fields_group
        self.scale_range = scale_range
        self.prob = prob
    
    def is_for_infer(self) -> bool:
        """推理时不使用数据增强"""
        return False
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用特征缩放
        
        Parameters
        ----------
        df : pd.DataFrame
            输入数据
            
        Returns
        -------
        pd.DataFrame
            增强后的数据
        """
        if np.random.random() > self.prob:
            return df
        
        cols = get_group_columns(df, self.fields_group)
        if len(cols) == 0:
            return df
        
        df = df.copy()
        
        # 对每个特征列应用随机缩放
        for col in cols:
            if col in df.columns:
                scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
                df[col] = df[col] * scale
        
        return df


class MixupAugment(Processor):
    """
    Mixup数据增强
    
    通过混合两个样本的特征和标签来创建新样本
    注意：这个增强器需要同时处理特征和标签
    """
    
    def __init__(self, fields_group="feature", alpha=0.2, prob=0.3):
        """
        Parameters
        ----------
        fields_group : str
            要增强的字段组
        alpha : float
            Beta分布的参数，控制混合强度
        prob : float
            应用增强的概率
        """
        self.fields_group = fields_group
        self.alpha = alpha
        self.prob = prob
    
    def is_for_infer(self) -> bool:
        """推理时不使用数据增强"""
        return False
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用Mixup增强
        
        Parameters
        ----------
        df : pd.DataFrame
            输入数据（包含feature和label）
            
        Returns
        -------
        pd.DataFrame
            增强后的数据
        """
        if np.random.random() > self.prob:
            return df
        
        if len(df) < 2:
            return df
        
        df = df.copy()
        
        # 获取特征列和标签列
        feature_cols = get_group_columns(df, self.fields_group)
        label_cols = get_group_columns(df, "label")
        
        if len(feature_cols) == 0:
            return df
        
        # 随机选择两个样本进行混合
        n_samples = len(df)
        idx1 = np.random.randint(0, n_samples)
        idx2 = np.random.randint(0, n_samples)
        while idx2 == idx1:
            idx2 = np.random.randint(0, n_samples)
        
        # 生成混合系数（Beta分布）
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 混合特征
        for col in feature_cols:
            if col in df.columns:
                df.iloc[idx1, df.columns.get_loc(col)] = (
                    lam * df.iloc[idx1, df.columns.get_loc(col)] +
                    (1 - lam) * df.iloc[idx2, df.columns.get_loc(col)]
                )
        
        # 混合标签（如果存在）
        for col in label_cols:
            if col in df.columns:
                df.iloc[idx1, df.columns.get_loc(col)] = (
                    lam * df.iloc[idx1, df.columns.get_loc(col)] +
                    (1 - lam) * df.iloc[idx2, df.columns.get_loc(col)]
                )
        
        return df


class CutoutAugment(Processor):
    """
    Cutout数据增强
    
    随机遮挡部分特征值（设置为0或均值）
    """
    
    def __init__(self, fields_group="feature", cutout_ratio=0.1, prob=0.3, fill_value="zero"):
        """
        Parameters
        ----------
        fields_group : str
            要增强的字段组
        cutout_ratio : float
            遮挡比例（0-1之间）
        prob : float
            应用增强的概率
        fill_value : str
            填充值类型："zero"（填充0）或"mean"（填充均值）
        """
        self.fields_group = fields_group
        self.cutout_ratio = cutout_ratio
        self.prob = prob
        self.fill_value = fill_value
    
    def is_for_infer(self) -> bool:
        """推理时不使用数据增强"""
        return False
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用Cutout增强
        
        Parameters
        ----------
        df : pd.DataFrame
            输入数据
            
        Returns
        -------
        pd.DataFrame
            增强后的数据
        """
        if np.random.random() > self.prob:
            return df
        
        cols = get_group_columns(df, self.fields_group)
        if len(cols) == 0:
            return df
        
        df = df.copy()
        
        # 随机选择要遮挡的特征列
        n_cutout = max(1, int(len(cols) * self.cutout_ratio))
        cutout_cols = np.random.choice(cols, size=n_cutout, replace=False)
        
        # 对选中的列进行遮挡
        for col in cutout_cols:
            if col in df.columns:
                if self.fill_value == "zero":
                    fill_val = 0
                elif self.fill_value == "mean":
                    fill_val = df[col].mean()
                else:
                    fill_val = 0
                
                # 随机遮挡部分值
                mask = np.random.random(len(df)) < self.cutout_ratio
                df.loc[mask, col] = fill_val
        
        return df

