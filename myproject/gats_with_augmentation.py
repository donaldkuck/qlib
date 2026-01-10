"""
带数据增强的GATs模型
在训练时动态应用数据增强，而不是在数据准备阶段
"""
import torch
import torch.nn as nn
import numpy as np
from qlib.contrib.model.pytorch_gats_ts import GATs
from qlib.log import get_module_logger

logger = get_module_logger("GATsWithAugmentation")


class GATsWithAugmentation(GATs):
    """
    带数据增强的GATs模型
    
    在训练时动态应用数据增强，每个batch都可能应用不同的增强
    这样可以提高模型的泛化能力，同时避免在数据准备阶段一次性应用增强导致的问题
    """
    
    def __init__(
        self,
        # 数据增强参数
        use_augmentation=True,
        noise_std=0.01,
        noise_prob=0.5,
        scale_range=(0.98, 1.02),
        scale_prob=0.3,
        # GATs原有参数
        **kwargs
    ):
        """
        Parameters
        ----------
        use_augmentation : bool
            是否使用数据增强（默认：True）
        noise_std : float
            高斯噪声的标准差（相对于特征的标准差，默认：0.01，即1%）
        noise_prob : float
            应用高斯噪声增强的概率（默认：0.5，即50%）
        scale_range : tuple
            特征缩放的范围 (min, max)（默认：(0.98, 1.02)，即±2%）
        scale_prob : float
            应用特征缩放增强的概率（默认：0.3，即30%）
        **kwargs
            GATs模型的其他参数
        """
        super().__init__(**kwargs)
        
        self.use_augmentation = use_augmentation
        self.noise_std = noise_std
        self.noise_prob = noise_prob
        self.scale_range = scale_range
        self.scale_prob = scale_prob
        
        if self.use_augmentation:
            logger.info(f"数据增强已启用: 噪声增强(prob={noise_prob}, std={noise_std}), "
                       f"缩放增强(prob={scale_prob}, range={scale_range})")
        else:
            logger.info("数据增强已禁用")
    
    def apply_augmentation(self, feature: torch.Tensor) -> torch.Tensor:
        """
        应用数据增强到特征张量（优化版本，减少计算开销）
        
        Parameters
        ----------
        feature : torch.Tensor
            输入特征，形状为 (batch_size, seq_len, feature_dim)
            
        Returns
        -------
        torch.Tensor
            增强后的特征
        """
        if not self.use_augmentation:
            return feature
        
        # 使用inplace操作减少内存分配
        # 1. 高斯噪声增强（优化：使用预计算的std，减少计算）
        if self.noise_prob > 0 and np.random.random() < self.noise_prob:
            # 使用更高效的方式：直接使用特征的标准差估计值
            # 避免每次都计算std，使用固定的相对标准差
            noise = torch.randn_like(feature) * self.noise_std
            feature = feature + noise
        
        # 2. 特征缩放增强（优化：使用向量化操作，避免循环）
        if self.scale_prob > 0 and np.random.random() < self.scale_prob:
            # 为整个batch生成一个缩放因子（而不是每个样本一个）
            # 这样可以减少计算开销，同时仍然提供增强效果
            scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
            feature = feature * scale_factor
        
        return feature
    
    def train_epoch(self, data_loader):
        """
        训练一个epoch，在训练时应用数据增强（优化版本）
        
        Parameters
        ----------
        data_loader : DataLoader
            训练数据加载器
        """
        self.GAT_model.train()
        
        # 如果禁用增强，直接使用父类方法（更快）
        if not self.use_augmentation:
            return super().train_epoch(data_loader)
        
        for data in data_loader:
            data = data.squeeze()
            feature = data[:, :, 0:-1].to(self.device, dtype=torch.float32)
            label = data[:, -1, -1].to(self.device, dtype=torch.float32)
            
            # 在训练时应用数据增强（优化：只在需要时应用）
            if self.use_augmentation and (self.noise_prob > 0 or self.scale_prob > 0):
                feature = self.apply_augmentation(feature)
            
            pred = self.GAT_model(feature)
            loss = self.loss_fn(pred, label)
            
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.GAT_model.parameters(), 3.0)
            self.train_optimizer.step()
    
    def test_epoch(self, data_loader):
        """
        测试一个epoch，测试时不应用数据增强
        
        Parameters
        ----------
        data_loader : DataLoader
            测试数据加载器
            
        Returns
        -------
        tuple
            (平均损失, 平均分数)
        """
        self.GAT_model.eval()
        
        scores = []
        losses = []
        
        with torch.no_grad():
            for data in data_loader:
                data = data.squeeze()
                feature = data[:, :, 0:-1].to(self.device, dtype=torch.float32)
                label = data[:, -1, -1].to(self.device, dtype=torch.float32)
                
                # 测试时不应用数据增强
                pred = self.GAT_model(feature)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())
                
                score = self.metric_fn(pred, label)
                scores.append(score.item())
        
        return np.mean(losses), np.mean(scores)

