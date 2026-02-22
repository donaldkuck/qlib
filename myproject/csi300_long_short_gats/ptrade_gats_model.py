# -*- coding: utf-8 -*-
"""
Ptrade GATs 模型加载和预测模块

完全独立于 Qlib，使用 PyTorch 加载预训练的 GATs TS 模型并进行预测
"""

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union, Dict, List


class GATModel(nn.Module):
    """
    GATs TS 模型（时间序列版本）

    基于图注意力机制和 RNN（LSTM/GRU）的混合模型
    """

    def __init__(self, d_feat: int = 20, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.0, base_model: str = "GRU"):
        super().__init__()

        # RNN 层
        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError(f"unknown base model name `{base_model}`")

        self.hidden_size = hidden_size
        self.d_feat = d_feat

        # 注意力机制
        self.transformation = nn.Linear(self.hidden_size, self.hidden_size)
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
        self.a.requires_grad = True

        # 全连接层
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)

        # 激活函数
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def cal_attention(self, x, y):
        """计算注意力权重"""
        x = self.transformation(x)
        y = self.transformation(y)

        sample_num = x.shape[0]
        dim = x.shape[1]
        e_x = x.expand(sample_num, sample_num, dim)
        e_y = torch.transpose(e_x, 0, 1)
        attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)
        self.a_t = torch.t(self.a)
        attention_out = self.a_t.mm(torch.t(attention_in)).view(sample_num, sample_num)
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight

    def forward(self, x):
        """
        前向传播

        Parameters
        ----------
        x : torch.Tensor
            输入特征，形状为 (N, T, F)，N 为样本数，T 为时间步数，F 为特征数

        Returns
        -------
        torch.Tensor
            预测结果，形状为 (N,)
        """
        out, _ = self.rnn(x)  # out: [N, T, hidden_size]
        hidden = out[:, -1, :]  # 取最后一个时间步的隐藏状态

        # 计算注意力权重
        att_weight = self.cal_attention(hidden, hidden)
        hidden = att_weight.mm(hidden) + hidden

        # 全连接层
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)

        return self.fc_out(hidden).squeeze()


class PtradeGATsPredictor:
    """
    Ptrade GATs 预测器

    负责加载预训练模型并进行预测
    """

    def __init__(
        self,
        model_path: str,
        d_feat: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        base_model: str = "GRU",
        device: str = "cpu",
    ):
        """
        初始化预测器

        Parameters
        ----------
        model_path : str
            预训练模型文件路径
        d_feat : int
            输入特征维度
        hidden_size : int
            隐藏层维度
        num_layers : int
            RNN 层数
        dropout : float
            Dropout 比例
        base_model : str
            基础模型类型（LSTM/GRU）
        device : str
            设备（cpu/cuda/mps）
        """
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.base_model = base_model
        self.device = torch.device(device)

        # 创建模型
        self.model = GATModel(
            d_feat=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            base_model=base_model
        )

        # 加载预训练权重
        self.load_model(model_path)

        self.model.eval()

    def load_model(self, model_path: str):
        """加载预训练模型"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"✓ 已加载模型: {model_path}")

    def predict_single(self, features: np.ndarray) -> float:
        """
        对单个样本进行预测

        Parameters
        ----------
        features : np.ndarray
            输入特征，形状为 (T, F)

        Returns
        -------
        float
            预测值
        """
        # 处理 NaN
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # 转换为张量
        x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)  # [1, T, F]

        # 预测
        with torch.no_grad():
            pred = self.model(x).cpu().numpy()

        return float(pred[0])

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """
        批量预测

        Parameters
        ----------
        features : np.ndarray
            输入特征，形状为 (N, T, F)

        Returns
        -------
        np.ndarray
            预测结果，形状为 (N,)
        """
        # 处理 NaN
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # 转换为张量
        x = torch.from_numpy(features).float().to(self.device)

        # 预测
        with torch.no_grad():
            preds = self.model(x).cpu().numpy()

        return preds


class ModelEnsemble:
    """
    模型集成（支持 n-fold 交叉验证集成）
    """

    def __init__(self, model_paths: List[str], d_feat: int, ensemble_method: str = "mean",
                 hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.0,
                 base_model: str = "GRU", device: str = "cpu"):
        """
        初始化集成模型

        Parameters
        ----------
        model_paths : List[str]
            模型文件路径列表（n-fold 的 n 个模型）
        d_feat : int
            输入特征维度
        ensemble_method : str
            集成方法（mean/median）
        device : str
            设备
        """
        self.models = []
        self.ensemble_method = ensemble_method

        for model_path in model_paths:
            predictor = PtradeGATsPredictor(
                model_path=model_path,
                d_feat=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                base_model=base_model,
                device=device
            )
            self.models.append(predictor)

        print(f"✓ 已加载 {len(self.models)} 个模型进行集成")

    def predict_single(self, features: np.ndarray) -> float:
        """集成预测单个样本"""
        preds = [model.predict_single(features) for model in self.models]

        if self.ensemble_method == "median":
            return np.median(preds)
        else:
            return np.mean(preds)


def load_model_from_qlib_run(
    run_id: str,
    experiment_name: str = "long_short_factor_strategy",
    model_type: str = "short_term",
    device: str = "cpu"
) -> PtradeGATsPredictor:
    """
    从 Qlib 的 mlruns 目录加载模型

    Parameters
    ----------
    run_id : str
        Run ID（32 位十六进制字符串）
    experiment_name : str
        实验名称
    model_type : str
        模型类型（short_term/long_term）
    device : str
        设备

    Returns
    -------
    PtradeGATsPredictor
        预测器实例

    注意：此函数需要访问 mlruns 目录，可能需要在同一台机器上运行
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    try:
        run = client.get_run(run_id)
        run_dir = Path(run.info.artifact_uri.replace("file://", ""))
    except Exception as e:
        raise RuntimeError(f"无法获取 run 信息: {e}")

    # 查找模型文件
    if model_type == "short_term":
        model_files = [
            "short_term_all_models.pkl",
            "short_term_params.pkl"
        ]
    else:
        model_files = [
            "long_term_all_models.pkl",
            "long_term_params.pkl"
        ]

    model_path = None
    for fname in model_files:
        candidate = run_dir / fname
        if candidate.exists():
            model_path = str(candidate)
            break

    if model_path is None:
        raise FileNotFoundError(f"未找到 {model_type} 模型文件")

    # 加载模型权重
    import pickle
    with open(model_path, 'rb') as f:
        params = pickle.load(f)

    # 确定模型参数
    if isinstance(params, dict):
        state_dict = params
        # 从 run 的参数中推断模型超参数
        d_feat = run.data.params.get('d_feat', 143 if model_type == 'short_term' else 150)
        hidden_size = int(run.data.params.get('hidden_size', 64))
        num_layers = int(run.data.params.get('num_layers', 2))
        base_model = run.data.params.get('base_model', 'GRU')
        dropout = float(run.data.params.get('dropout', 0.5))
    else:
        # 如果是模型对象，直接使用
        raise ValueError("不支持的模型格式")

    # 创建预测器
    temp_path = run_dir / f"{model_type}_temp_model.pt"
    torch.save(state_dict, temp_path)

    predictor = PtradeGATsPredictor(
        model_path=str(temp_path),
        d_feat=int(d_feat),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        base_model=base_model,
        device=device
    )

    return predictor


def load_model_from_pickle(
    pickle_path: str,
    device: str = "cpu"
) -> PtradeGATsPredictor:
    """
    从 pickle 文件加载模型

    Parameters
    ----------
    pickle_path : str
        pickle 文件路径
    device : str
        设备

    Returns
    -------
    PtradeGATsPredictor
        预测器实例
    """
    import pickle

    with open(pickle_path, 'rb') as f:
        obj = pickle.load(f)

    # 如果是模型对象，提取参数
    if hasattr(obj, 'GAT_model'):
        state_dict = obj.GAT_model.state_dict()
        d_feat = obj.d_feat
        hidden_size = obj.hidden_size
        num_layers = obj.num_layers
        dropout = obj.dropout
        base_model = obj.base_model
    elif isinstance(obj, dict):
        state_dict = obj
        # 使用默认参数（实际应该从配置读取）
        d_feat = 143
        hidden_size = 64
        num_layers = 2
        dropout = 0.5
        base_model = "GRU"
    else:
        raise ValueError("不支持的 pickle 格式")

    # 临时保存权重
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        torch.save(state_dict, tmp.name)
        temp_path = tmp.name

    predictor = PtradeGATsPredictor(
        model_path=temp_path,
        d_feat=d_feat,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        base_model=base_model,
        device=device
    )

    return predictor


if __name__ == "__main__":
    print("Ptrade GATs 模型模块测试")

    # 创建测试模型
    test_model = GATModel(d_feat=10, hidden_size=32, num_layers=1)

    # 测试预测
    test_input = torch.randn(5, 20, 10)  # [N=5, T=20, F=10]
    output = test_model(test_input)
    print(f"模型输出形状: {output.shape}")
    print(f"模型输出值: {output}")
