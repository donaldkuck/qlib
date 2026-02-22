#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行长期和短期因子结合的交易策略

优化版本：
- 添加性能监控和时间统计
- 增强错误处理和验证
- 优化代码结构和模块化
- 添加预测信号质量检查
- 增强回测结果分析
"""
import sys
import time
import warnings
import gc  # 用于内存管理
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np

# 过滤回测过程中的警告（空数组平均值警告等）
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Mean of empty slice.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')

# 过滤回测过程中的数据缺失警告（这些警告不影响回测逻辑，只是提示数据缺失）
import logging
logging.getLogger('qlib.online operator').setLevel(logging.ERROR)  # 过滤 $close: nan!!! 警告
logging.getLogger('qlib.BaseExecutor').setLevel(logging.ERROR)  # 过滤 common_infra 警告

# 确保使用安装的qlib包，而不是本地源码
# 策略：先导入qlib（从安装的包），然后添加项目根目录用于导入myproject
current_dir = Path(__file__).parent.absolute()  # myproject目录
project_root = current_dir.parent  # qlib项目根目录
qlib_local_path = project_root / 'qlib'

# 从sys.path中移除所有可能包含本地qlib源码的路径
paths_to_remove = []
for p in sys.path:
    p_str = str(p)
    # 移除包含本地qlib源码的路径
    if str(qlib_local_path) in p_str or (str(project_root) in p_str and 'qlib' in p_str.lower()):
        paths_to_remove.append(p_str)
    # 移除包含myqlib的路径（可能是另一个本地安装）
    if 'myqlib' in p_str.lower() and 'qlib' in p_str.lower():
        paths_to_remove.append(p_str)

for p in paths_to_remove:
    if p in sys.path:
        sys.path.remove(p)

# 先导入qlib（从安装的包）
import qlib

# 然后添加项目根目录到路径（用于导入myproject模块，如 custom_factor_handler）
# 此时qlib已经导入，即使项目根目录在sys.path中，也不会重新导入qlib
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# 添加当前目录（hongli_long_short_gats）到路径，使 workflow 中的 module_path: long_short_strategy 能正确导入
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
from qlib.workflow import R
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord
from qlib.contrib.evaluate import risk_analysis
from qlib.utils.time import Freq
import qlib.contrib.report.analysis_position as qcr_analysis_position
import qlib.contrib.report.analysis_model as qcr_analysis_model
import yaml


# 红利股列表 (qlib格式)
RED_CHIP_STOCKS = [
    # 银行
    "SH600036", "SH600015", "SH600016", "SH600000",
    "SH601166", "SH601169", "SH601288", "SH601398",
    "SH601939", "SH601988", "SH601998", "SH601229",
    "SZ000001", "SZ002142",
    # 保险
    "SH601318", "SH601336", "SH601601", "SH601628", "SH601319",
    # 公用事业
    "SH600900", "SH600886", "SH600795", "SH600905",
    "SH601985", "SH600021", "SH600726", "SH600011",
    # 能源
    "SH600028", "SH601857", "SH601872", "SH601868",
    "SH601088", "SH601898", "SH600188", "SH601699",
    # 材料
    "SH600019", "SH600585",
    # 消费
    "SH600519", "SH600887", "SH600690", "SH600809",
    "SH600600", "SH600132", "SZ000858", "SZ000568",
    "SZ000651", "SZ000333", "SZ002304", "SZ000596",
    # 医药
    "SH600276", "SH600436", "SH600521",
    "SZ000538", "SZ000661", "SH600196",
    # 交运
    "SH600018", "SH600029", "SH600115", "SH601111",
    "SH600350", "SH600377", "SH601006", "SH601919",
    # 电信
    "SH600941", "SH601728", "SH600050",
    # 地产
    "SH600048", "SZ000002", "SZ001979",
    # 汽车
    "SH600104", "SH601238", "SH601633", "SZ000625",
    # 机械
    "SH600031", "SZ000425",
    # 基建
    "SH601186", "SH601390", "SH601668",
    "SH601800", "SH601618", "SH601669",
    # 化工
    "SH600309", "SH600346", "SH600426",
    # 有色
    "SH600362", "SH601899", "SH601168", "SH600489",
    # 券商
    "SH600030", "SH600837", "SH600999", "SH601688",
]


def validate_config(config: Dict[str, Any]) -> None:
    """验证配置文件的有效性"""
    required_keys = ['task', 'port_analysis_config']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"配置文件缺少必需的键: {key}")
    
    task = config['task']
    required_task_keys = ['short_term_model', 'short_term_dataset', 'long_term_model', 'long_term_dataset']
    for key in required_task_keys:
        if key not in task:
            raise ValueError(f"任务配置缺少必需的键: {key}")
    
    port_config = config.get('port_analysis_config', {})
    if 'strategy' not in port_config:
        raise ValueError("回测配置缺少策略配置")
    
    if 'backtest' not in port_config:
        raise ValueError("回测配置缺少回测参数")


def check_prediction_quality(pred: pd.DataFrame, name: str) -> Dict[str, Any]:
    """检查预测信号的质量"""
    stats = {}
    
    if pred is None or len(pred) == 0:
        stats['valid'] = False
        stats['message'] = f"{name}预测信号为空"
        return stats
    
    try:
        # 转换为Series格式（如果是DataFrame）
        if isinstance(pred, pd.DataFrame):
            pred_series = pred.iloc[:, 0] if len(pred.columns) > 0 else pred.squeeze()
        else:
            pred_series = pred
        
        # 基本统计信息
        stats['valid'] = True
        stats['count'] = len(pred_series)
        stats['mean'] = float(pred_series.mean())
        stats['std'] = float(pred_series.std())
        stats['min'] = float(pred_series.min())
        stats['max'] = float(pred_series.max())
        stats['nan_count'] = int(pred_series.isna().sum())
        stats['nan_ratio'] = float(pred_series.isna().sum() / len(pred_series)) if len(pred_series) > 0 else 0.0
        
        # 检查异常值
        if stats['std'] > 0:
            z_scores = (pred_series - stats['mean']) / stats['std']
            outliers = (z_scores.abs() > 3).sum()
            stats['outliers'] = int(outliers)
            stats['outlier_ratio'] = float(outliers / len(pred_series)) if len(pred_series) > 0 else 0.0
        else:
            stats['outliers'] = 0
            stats['outlier_ratio'] = 0.0
        
        # 检查是否有无效值
        if stats['nan_ratio'] > 0.1:
            stats['warning'] = f"NaN比例过高（{stats['nan_ratio']*100:.2f}%），可能影响预测质量"
        
        if stats.get('outlier_ratio', 0) > 0.1:
            stats['warning'] = f"异常值比例过高（{stats['outlier_ratio']*100:.2f}%），可能影响预测质量"
            
    except Exception as e:
        stats['valid'] = False
        stats['message'] = f"检查{name}预测信号质量时出错: {e}"
    
    return stats


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.2f}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}小时{minutes}分{secs:.2f}秒"


def print_metrics(metrics: Dict[str, Any], title: str) -> None:
    """打印性能指标"""
    print(f"\n【{title}】")
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'ratio' in key or 'return' in key or 'drawdown' in key:
                print(f"  {key}: {value*100:.2f}%")
            elif 'sharpe' in key.lower() or 'ratio' in key.lower():
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def load_port_analysis(recorder, par: Optional[PortAnaRecord] = None) -> Optional[Dict[str, Any]]:
    """加载回测分析结果，尝试多个可能的路径"""
    port_analysis = None
    
    # 尝试多个可能的路径
    possible_paths = [
        "portfolio_analysis/port_analysis_1day.pkl",
        "port_analysis_1day.pkl",
        "portfolio_analysis/port_analysis.pkl",
        "port_analysis.pkl"
    ]
    
    for path in possible_paths:
        try:
            port_analysis = recorder.load_object(path)
            if port_analysis is not None:
                return port_analysis
        except Exception:
            continue
    
    # 如果直接加载失败，尝试从par对象加载
    if par is not None:
        for path in possible_paths:
            try:
                port_analysis = par.load(path)
                if port_analysis is not None:
                    return port_analysis
            except Exception:
                continue
    
    return None


def split_time_series_folds(train_start: str, train_end: str, n_folds: int = 5) -> List[Tuple[str, str, str, str]]:
    """
    将时间序列数据分割为 n 个 fold（用于时间序列交叉验证）
    
    对于时间序列数据，使用时间顺序分割，每个 fold 的训练集是前面的数据，验证集是后面的数据。
    例如，5-fold 分割：
    - Fold 1: train=[0-20%], valid=[20-40%]
    - Fold 2: train=[0-40%], valid=[40-60%]
    - Fold 3: train=[0-60%], valid=[60-80%]
    - Fold 4: train=[0-80%], valid=[80-100%]
    - Fold 5: train=[0-100%], valid=None (使用原始valid集)
    
    Parameters
    ----------
    train_start : str
        训练集开始时间
    train_end : str
        训练集结束时间
    n_folds : int
        fold 数量（默认5）
        
    Returns
    -------
    List[Tuple[str, str, str, str]]
        每个 fold 的 (fold_train_start, fold_train_end, fold_valid_start, fold_valid_end)
        最后一个 fold 的 valid 为 None，表示使用原始 valid 集
    """
    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    total_days = (train_end_ts - train_start_ts).days
    
    folds = []
    for i in range(1, n_folds + 1):
        if i < n_folds:
            # 前 n-1 个 fold：训练集是前 i/n 的数据，验证集是接下来的 1/n 数据
            fold_train_end_ratio = i / n_folds
            fold_valid_end_ratio = (i + 1) / n_folds
            
            fold_train_end_days = int(total_days * fold_train_end_ratio)
            fold_valid_end_days = int(total_days * fold_valid_end_ratio)
            
            fold_train_end = (train_start_ts + pd.Timedelta(days=fold_train_end_days)).strftime('%Y-%m-%d')
            fold_valid_start = (train_start_ts + pd.Timedelta(days=fold_train_end_days + 1)).strftime('%Y-%m-%d')
            fold_valid_end = (train_start_ts + pd.Timedelta(days=fold_valid_end_days)).strftime('%Y-%m-%d')
            
            folds.append((train_start, fold_train_end, fold_valid_start, fold_valid_end))
        else:
            # 最后一个 fold：使用全部训练数据，验证集使用原始 valid 集
            folds.append((train_start, train_end, None, None))
    
    return folds


def train_model_with_nfold(
    model, 
    dataset, 
    model_name: str,
    n_folds: int = 5,
    recorder=None,
    use_nfold: bool = True
) -> List[Any]:
    """
    使用 n-fold 交叉验证训练模型
    
    Parameters
    ----------
    model : BaseModel
        模型实例
    dataset : Dataset
        数据集
    model_name : str
        模型名称（用于保存）
    n_folds : int
        fold 数量（默认5）
    recorder : Recorder
        记录器（用于保存模型）
    use_nfold : bool
        是否使用 n-fold（如果 False，则只训练一次）
        
    Returns
    -------
    List[BaseModel]
        所有 fold 的模型列表
    """
    from qlib.data.dataset import TSDatasetH
    from qlib.utils import init_instance_by_config
    import copy
    
    if not use_nfold or n_folds <= 1:
        # 不使用 n-fold，直接训练
        print(f"  训练 {model_name}（不使用 n-fold）...")
        model.fit(dataset)
        if recorder:
            recorder.save_objects(**{f"{model_name}_params.pkl": model})
        return [model]
    
    # 使用 n-fold 交叉验证
    print(f"  训练 {model_name}（使用 {n_folds}-fold 交叉验证）...")
    
    # 获取训练集时间段
    train_segment = dataset.segments.get('train')
    if isinstance(train_segment, (list, tuple)):
        train_start, train_end = train_segment[0], train_segment[1]
    else:
        # 如果 train_segment 是 slice，需要从 dataset 获取
        print(f"    ⚠️  无法从 dataset 获取训练集时间段，使用单次训练")
        model.fit(dataset)
        if recorder:
            recorder.save_objects(**{f"{model_name}_params.pkl": model})
        return [model]
    
    # 分割 fold
    folds = split_time_series_folds(train_start, train_end, n_folds)
    
    models = []
    for fold_idx, (fold_train_start, fold_train_end, fold_valid_start, fold_valid_end) in enumerate(folds, 1):
        print(f"\n    Fold {fold_idx}/{n_folds}:")
        print(f"      训练集: {fold_train_start} 到 {fold_train_end}")
        if fold_valid_start and fold_valid_end:
            print(f"      验证集: {fold_valid_start} 到 {fold_valid_end}")
        else:
            print(f"      验证集: 使用原始验证集")
        
        # 创建新的 dataset 配置（修改 segments）
        dataset_config = {
            'class': dataset.__class__.__name__,
            'module_path': dataset.__class__.__module__,
            'kwargs': {}
        }
        
        # 复制 handler 配置
        if hasattr(dataset, 'handler'):
            handler = dataset.handler
            handler_config = {
                'class': handler.__class__.__name__,
                'module_path': handler.__class__.__module__,
                'kwargs': {}
            }
            
            # 复制 handler 的初始化参数
            if hasattr(handler, 'instruments'):
                handler_config['kwargs']['instruments'] = handler.instruments
            if hasattr(handler, 'start_time'):
                handler_config['kwargs']['start_time'] = handler.start_time
            if hasattr(handler, 'end_time'):
                handler_config['kwargs']['end_time'] = handler.end_time
            if hasattr(handler, 'fit_start_time'):
                handler_config['kwargs']['fit_start_time'] = handler.fit_start_time
            if hasattr(handler, 'fit_end_time'):
                handler_config['kwargs']['fit_end_time'] = handler.fit_end_time
            if hasattr(handler, 'freq'):
                handler_config['kwargs']['freq'] = handler.freq
            if hasattr(handler, 'infer_processors'):
                handler_config['kwargs']['infer_processors'] = handler.infer_processors
            if hasattr(handler, 'learn_processors'):
                handler_config['kwargs']['learn_processors'] = handler.learn_processors
            
            # 添加自定义因子相关参数
            if hasattr(handler, 'custom_factors'):
                handler_config['kwargs']['custom_factors'] = handler.custom_factors
            if hasattr(handler, 'custom_factors_file'):
                handler_config['kwargs']['custom_factors_file'] = handler.custom_factors_file
            if hasattr(handler, 'label_expr'):
                handler_config['kwargs']['label_expr'] = handler.label_expr
            
            dataset_config['kwargs']['handler'] = handler_config
        
        # 设置 segments
        if fold_valid_start and fold_valid_end:
            dataset_config['kwargs']['segments'] = {
                'train': [fold_train_start, fold_train_end],
                'valid': [fold_valid_start, fold_valid_end],
                'test': dataset.segments.get('test', dataset.segments.get('valid'))
            }
        else:
            # 最后一个 fold：使用原始 segments
            dataset_config['kwargs']['segments'] = dataset.segments.copy()
            dataset_config['kwargs']['segments']['train'] = [fold_train_start, fold_train_end]
        
        # 复制其他 dataset 参数
        if isinstance(dataset, TSDatasetH):
            dataset_config['kwargs']['step_len'] = dataset.step_len
        
        # 创建新的 dataset
        fold_dataset = init_instance_by_config(dataset_config)
        
        # 确保 fold_dataset 的数据已准备好（setup_data）
        if not hasattr(fold_dataset.handler, '_learn') or fold_dataset.handler._learn is None:
            fold_dataset.setup_data()
        
        # 从 fold_dataset 中获取实际特征维度（最准确的方法）
        actual_d_feat = None
        try:
            if hasattr(fold_dataset, 'handler') and hasattr(fold_dataset.handler, '_learn'):
                if fold_dataset.handler._learn is not None:
                    # 获取特征维度
                    if isinstance(fold_dataset.handler._learn.columns, pd.MultiIndex):
                        feature_cols = fold_dataset.handler._learn.columns.get_level_values(0).unique()
                        if 'feature' in feature_cols:
                            feature_data = fold_dataset.handler._learn['feature']
                            if hasattr(feature_data, 'shape') and len(feature_data.shape) > 1:
                                actual_d_feat = feature_data.shape[1]
                                print(f"      - 从数据中检测到特征维度: {actual_d_feat}")
        except Exception as e:
            print(f"      ⚠️  无法从数据中获取特征维度: {e}")
        
        # 创建新的模型（复制配置）
        model_config = {
            'class': model.__class__.__name__,
            'module_path': model.__class__.__module__,
            'kwargs': {}
        }
        
        # 复制模型参数（但优先使用从数据中检测到的 d_feat）
        for attr in ['d_feat', 'fea_dim', 'd_model', 'hidden_size', 'num_layers', 'dropout', 'n_epochs', 
                     'lr', 'early_stop', 'metric', 'loss', 'base_model', 'optimizer', 
                     'GPU', 'n_jobs', 'batch_size', 'use_augmentation', 'noise_std', 
                     'noise_prob', 'scale_range', 'scale_prob', 'nhead', 'reg', 'seed',
                     'cnn_dim', 'cnn_kernel_size', 'rnn_dim', 'rnn_dups', 'rnn_layers']:
            if hasattr(model, attr):
                model_config['kwargs'][attr] = getattr(model, attr)
        
        # 优先使用从数据中检测到的特征维度（d_feat 或 fea_dim）
        if actual_d_feat is not None:
            model_config['kwargs']['d_feat'] = actual_d_feat
            if model.__class__.__name__ == 'KRNN':
                model_config['kwargs']['fea_dim'] = actual_d_feat
            print(f"      - 使用从数据中检测到的特征维度: {actual_d_feat}")
        elif 'd_feat' not in model_config['kwargs'] or model_config['kwargs']['d_feat'] is None:
            # 如果无法从数据中获取，使用模型实例中的值或配置中的 fea_dim（KRNN）
            if hasattr(model, 'd_feat') and model.d_feat is not None:
                model_config['kwargs']['d_feat'] = model.d_feat
                print(f"      - 使用模型实例中的 d_feat: {model.d_feat}")
            elif model.__class__.__name__ == 'KRNN' and hasattr(model, 'fea_dim'):
                model_config['kwargs']['fea_dim'] = model.fea_dim
                print(f"      - 使用模型实例中的 fea_dim: {model.fea_dim}")
            else:
                # 使用默认值（根据模型类型）
                if 'TransformerModel' in model.__class__.__name__:
                    default_d_feat = 20  # TransformerModel 的默认值
                elif model.__class__.__name__ == 'KRNN':
                    default_d_feat = model_config['kwargs'].get('fea_dim', 6)
                else:
                    default_d_feat = 360  # 其他模型的默认值
                model_config['kwargs']['d_feat'] = default_d_feat
                if model.__class__.__name__ == 'KRNN':
                    model_config['kwargs']['fea_dim'] = default_d_feat
                print(f"      ⚠️  使用默认特征维度: {default_d_feat}")
        
        fold_model = init_instance_by_config(model_config)
        
        # 训练 fold 模型
        try:
            fold_model.fit(fold_dataset)
            models.append(fold_model)
            
            # 保存 fold 模型
            if recorder:
                recorder.save_objects(**{f"{model_name}_fold{fold_idx}_params.pkl": fold_model})
            
            print(f"      ✓ Fold {fold_idx} 训练完成")
            
            # 清理内存
            del fold_dataset
            gc.collect()
            
        except Exception as e:
            print(f"      ❌ Fold {fold_idx} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            # 如果某个 fold 失败，继续训练其他 fold
    
    if len(models) == 0:
        print(f"    ⚠️  所有 fold 训练失败，使用单次训练")
        model.fit(dataset)
        if recorder:
            recorder.save_objects(**{f"{model_name}_params.pkl": model})
        return [model]
    
    # 保存所有模型的列表
    if recorder:
        recorder.save_objects(**{f"{model_name}_all_folds.pkl": models})
    
    print(f"\n    ✓ {n_folds}-fold 交叉验证完成，共训练 {len(models)} 个模型")
    
    return models


def predict_with_nfold_ensemble(
    models: List[Any],
    dataset,
    ensemble_method: str = 'mean'
) -> pd.Series:
    """
    使用 n-fold 模型进行集成预测
    
    Parameters
    ----------
    models : List[BaseModel]
        所有 fold 的模型列表
    dataset : Dataset
        数据集
    ensemble_method : str
        集成方法：'mean'（平均）或 'median'（中位数）
        
    Returns
    -------
    pd.Series
        集成预测结果
    """
    if len(models) == 0:
        raise ValueError("模型列表为空")
    
    if len(models) == 1:
        # 只有一个模型，直接预测
        return models[0].predict(dataset)
    
    # 使用多个模型预测并集成
    predictions = []
    for i, model in enumerate(models):
        try:
            pred = model.predict(dataset)
            if isinstance(pred, pd.DataFrame):
                pred = pred.iloc[:, 0] if len(pred.columns) > 0 else pred.squeeze()
            predictions.append(pred)
        except Exception as e:
            print(f"    ⚠️  Fold {i+1} 模型预测失败: {e}")
            continue
    
    if len(predictions) == 0:
        raise ValueError("所有模型预测失败")
    
    # 对齐所有预测的索引
    if len(predictions) > 1:
        common_idx = predictions[0].index
        for pred in predictions[1:]:
            common_idx = common_idx.intersection(pred.index)
        
        aligned_predictions = [pred.loc[common_idx] for pred in predictions]
    else:
        aligned_predictions = predictions
        common_idx = predictions[0].index
    
    # 集成预测
    if ensemble_method == 'mean':
        ensemble_pred = pd.concat(aligned_predictions, axis=1).mean(axis=1)
    elif ensemble_method == 'median':
        ensemble_pred = pd.concat(aligned_predictions, axis=1).median(axis=1)
    else:
        raise ValueError(f"不支持的集成方法: {ensemble_method}")
    
    return ensemble_pred


def calculate_prediction_win_rate(recorder, long_term_dataset, short_term_dataset) -> Tuple[Optional[List], Optional[List]]:
    """
    计算并打印长期和短期预测在每只股票上的胜率
    
    Parameters
    ----------
    recorder : Recorder
        记录器对象
    long_term_dataset : Dataset
        长期数据集
    short_term_dataset : Dataset
        短期数据集
    """
    try:
        print("\n" + "=" * 80)
        print("预测胜率分析")
        print("=" * 80)
        
        # 加载长期预测和标签
        long_term_pred = recorder.load_object("long_term_pred.pkl")
        if long_term_pred is None:
            print("  ⚠️  无法加载长期预测数据")
            return
        
        # 加载短期预测
        short_term_pred = recorder.load_object("short_term_pred.pkl")
        if short_term_pred is None:
            print("  ⚠️  无法加载短期预测数据")
            return
        
        # 获取长期标签数据
        # 使用 handler.fetch() 方法获取标签数据
        try:
            # 获取test段的slice
            if 'test' in long_term_dataset.segments:
                test_slice = long_term_dataset.segments['test']
                if isinstance(test_slice, (tuple, list)):
                    test_selector = slice(*test_slice)
                else:
                    test_selector = test_slice
            else:
                raise ValueError("长期数据集中没有test段")
            
            # 确保数据集已经setup
            if not hasattr(long_term_dataset.handler, '_learn'):
                long_term_dataset.setup_data()
            
            # 使用handler.fetch获取标签数据
            long_term_label_data = long_term_dataset.handler.fetch(
                selector=test_selector,
                col_set=["label"],
                data_key=long_term_dataset.handler.DK_L
            )
            
            if isinstance(long_term_label_data, pd.DataFrame) and len(long_term_label_data.columns) > 0:
                long_term_label = long_term_label_data.iloc[:, 0]
            else:
                long_term_label = long_term_label_data.squeeze() if hasattr(long_term_label_data, 'squeeze') else long_term_label_data
        except Exception as e:
            print(f"  ⚠️  无法加载长期标签数据: {e}")
            long_term_label = None
        
        # 获取短期标签数据
        try:
            # 获取test段的slice
            if 'test' in short_term_dataset.segments:
                test_slice = short_term_dataset.segments['test']
                if isinstance(test_slice, (tuple, list)):
                    test_selector = slice(*test_slice)
                else:
                    test_selector = test_slice
            else:
                raise ValueError("短期数据集中没有test段")
            
            # 确保数据集已经setup
            if not hasattr(short_term_dataset.handler, '_learn'):
                short_term_dataset.setup_data()
            
            # 使用handler.fetch获取标签数据
            short_term_label_data = short_term_dataset.handler.fetch(
                selector=test_selector,
                col_set=["label"],
                data_key=short_term_dataset.handler.DK_L
            )
            
            if isinstance(short_term_label_data, pd.DataFrame) and len(short_term_label_data.columns) > 0:
                short_term_label = short_term_label_data.iloc[:, 0]
            else:
                short_term_label = short_term_label_data.squeeze() if hasattr(short_term_label_data, 'squeeze') else short_term_label_data
        except Exception as e:
            print(f"  ⚠️  无法加载短期标签数据: {e}")
            short_term_label = None
        
        # 处理DataFrame格式
        if isinstance(long_term_pred, pd.DataFrame):
            long_term_pred = long_term_pred.iloc[:, 0] if len(long_term_pred.columns) > 0 else long_term_pred.squeeze()
        if isinstance(short_term_pred, pd.DataFrame):
            short_term_pred = short_term_pred.iloc[:, 0] if len(short_term_pred.columns) > 0 else short_term_pred.squeeze()
        if isinstance(long_term_label, pd.DataFrame):
            long_term_label = long_term_label.iloc[:, 0] if len(long_term_label.columns) > 0 else long_term_label.squeeze()
        if isinstance(short_term_label, pd.DataFrame):
            short_term_label = short_term_label.iloc[:, 0] if len(short_term_label.columns) > 0 else short_term_label.squeeze()
        
        # 对齐索引
        if long_term_label is not None:
            common_idx_long = long_term_pred.index.intersection(long_term_label.index)
            if len(common_idx_long) > 0:
                long_term_pred_aligned = long_term_pred.loc[common_idx_long]
                long_term_label_aligned = long_term_label.loc[common_idx_long]
            else:
                long_term_pred_aligned = None
                long_term_label_aligned = None
        else:
            long_term_pred_aligned = None
            long_term_label_aligned = None
        
        if short_term_label is not None:
            common_idx_short = short_term_pred.index.intersection(short_term_label.index)
            if len(common_idx_short) > 0:
                short_term_pred_aligned = short_term_pred.loc[common_idx_short]
                short_term_label_aligned = short_term_label.loc[common_idx_short]
            else:
                short_term_pred_aligned = None
                short_term_label_aligned = None
        else:
            short_term_pred_aligned = None
            short_term_label_aligned = None
        
        # 计算长期预测胜率（按股票分组）
        if long_term_pred_aligned is not None and long_term_label_aligned is not None:
            print("\n【长期预测胜率（按股票）】")
            long_term_results = []
            
            # 按股票分组
            if isinstance(long_term_pred_aligned.index, pd.MultiIndex):
                # MultiIndex格式 (datetime, instrument)
                for stock_id in long_term_pred_aligned.index.get_level_values(1).unique():
                    stock_mask = long_term_pred_aligned.index.get_level_values(1) == stock_id
                    stock_pred = long_term_pred_aligned[stock_mask]
                    stock_label = long_term_label_aligned[stock_mask]
                    
                    if len(stock_pred) > 0:
                        # 计算胜率：预测方向正确（预测>0且实际>0，或预测<0且实际<0）
                        correct = ((stock_pred > 0) & (stock_label > 0)) | ((stock_pred < 0) & (stock_label < 0))
                        win_rate = correct.sum() / len(correct) if len(correct) > 0 else 0.0
                        long_term_results.append({
                            'stock_id': stock_id,
                            'win_rate': win_rate,
                            'total': len(correct),
                            'correct': correct.sum()
                        })
            else:
                # 单索引格式，无法按股票分组
                correct = ((long_term_pred_aligned > 0) & (long_term_label_aligned > 0)) | ((long_term_pred_aligned < 0) & (long_term_label_aligned < 0))
                win_rate = correct.sum() / len(correct) if len(correct) > 0 else 0.0
                print(f"  整体胜率: {win_rate*100:.2f}% ({correct.sum()}/{len(correct)})")
            
            if len(long_term_results) > 0:
                # 按胜率排序
                long_term_results.sort(key=lambda x: x['win_rate'], reverse=True)
                
                # 打印Top10和Bottom10
                print(f"  统计股票数: {len(long_term_results)}")
                print(f"\n  Top 10 胜率最高的股票:")
                for i, result in enumerate(long_term_results[:10], 1):
                    print(f"    {i:2d}. {result['stock_id']:10s} | 胜率: {result['win_rate']*100:6.2f}% | 正确/总数: {result['correct']:3d}/{result['total']:3d}")
                
                if len(long_term_results) > 10:
                    print(f"\n  Bottom 10 胜率最低的股票:")
                    for i, result in enumerate(long_term_results[-10:], 1):
                        print(f"    {i:2d}. {result['stock_id']:10s} | 胜率: {result['win_rate']*100:6.2f}% | 正确/总数: {result['correct']:3d}/{result['total']:3d}")
                
                # 统计信息
                avg_win_rate = sum(r['win_rate'] for r in long_term_results) / len(long_term_results)
                print(f"\n  平均胜率: {avg_win_rate*100:.2f}%")
                print(f"  胜率中位数: {pd.Series([r['win_rate'] for r in long_term_results]).median()*100:.2f}%")
                print(f"  胜率>50%的股票数: {sum(1 for r in long_term_results if r['win_rate'] > 0.5)}/{len(long_term_results)}")
                print(f"  胜率>60%的股票数: {sum(1 for r in long_term_results if r['win_rate'] > 0.6)}/{len(long_term_results)}")
        
        # 计算短期预测胜率（按股票分组）
        if short_term_pred_aligned is not None and short_term_label_aligned is not None:
            print("\n【短期预测胜率（按股票）】")
            short_term_results = []
            
            # 按股票分组
            if isinstance(short_term_pred_aligned.index, pd.MultiIndex):
                # MultiIndex格式 (datetime, instrument)
                for stock_id in short_term_pred_aligned.index.get_level_values(1).unique():
                    stock_mask = short_term_pred_aligned.index.get_level_values(1) == stock_id
                    stock_pred = short_term_pred_aligned[stock_mask]
                    stock_label = short_term_label_aligned[stock_mask]
                    
                    if len(stock_pred) > 0:
                        # 计算胜率：预测方向正确
                        correct = ((stock_pred > 0) & (stock_label > 0)) | ((stock_pred < 0) & (stock_label < 0))
                        win_rate = correct.sum() / len(correct) if len(correct) > 0 else 0.0
                        short_term_results.append({
                            'stock_id': stock_id,
                            'win_rate': win_rate,
                            'total': len(correct),
                            'correct': correct.sum()
                        })
            else:
                # 单索引格式，无法按股票分组
                correct = ((short_term_pred_aligned > 0) & (short_term_label_aligned > 0)) | ((short_term_pred_aligned < 0) & (short_term_label_aligned < 0))
                win_rate = correct.sum() / len(correct) if len(correct) > 0 else 0.0
                print(f"  整体胜率: {win_rate*100:.2f}% ({correct.sum()}/{len(correct)})")
            
            if len(short_term_results) > 0:
                # 按胜率排序
                short_term_results.sort(key=lambda x: x['win_rate'], reverse=True)
                
                # 打印Top10和Bottom10
                print(f"  统计股票数: {len(short_term_results)}")
                print(f"\n  Top 10 胜率最高的股票:")
                for i, result in enumerate(short_term_results[:10], 1):
                    print(f"    {i:2d}. {result['stock_id']:10s} | 胜率: {result['win_rate']*100:6.2f}% | 正确/总数: {result['correct']:3d}/{result['total']:3d}")
                
                if len(short_term_results) > 10:
                    print(f"\n  Bottom 10 胜率最低的股票:")
                    for i, result in enumerate(short_term_results[-10:], 1):
                        print(f"    {i:2d}. {result['stock_id']:10s} | 胜率: {result['win_rate']*100:6.2f}% | 正确/总数: {result['correct']:3d}/{result['total']:3d}")
                
                # 统计信息
                avg_win_rate = sum(r['win_rate'] for r in short_term_results) / len(short_term_results)
                print(f"\n  平均胜率: {avg_win_rate*100:.2f}%")
                print(f"  胜率中位数: {pd.Series([r['win_rate'] for r in short_term_results]).median()*100:.2f}%")
                print(f"  胜率>50%的股票数: {sum(1 for r in short_term_results if r['win_rate'] > 0.5)}/{len(short_term_results)}")
                print(f"  胜率>60%的股票数: {sum(1 for r in short_term_results if r['win_rate'] > 0.6)}/{len(short_term_results)}")
                # 若短期胜率偏低，给出可操作建议
                avg_st = sum(r['win_rate'] for r in short_term_results) / len(short_term_results) if short_term_results else 0
                if avg_st < 0.52:
                    print(f"\n  [短期胜率偏低建议] 可尝试：")
                    print(f"    1. 在 workflow_config 中启用 enable_stock_pool_filter，仅交易长期+短期胜率≥阈值的股票")
                    print(f"    2. 短期 handler 开启 add_stock_id_onehot: true，或增加 hidden_size/n_epochs")
                    print(f"    3. 短期标签改为 2～3 日收益（label_expr）可能更易预测")
        
        # 返回结果
        return long_term_results if 'long_term_results' in locals() and len(long_term_results) > 0 else None, \
               short_term_results if 'short_term_results' in locals() and len(short_term_results) > 0 else None
    
    except Exception as e:
        print(f"  ⚠️  计算预测胜率时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def calculate_and_output_prediction_statistics(recorder, long_term_dataset, short_term_dataset):
    """
    计算并输出预测数据统计结果
    
    输出内容：
    1. 个股总体正确率（从高到低排序，只输出正确率>55%的）
    2. 分年度个股正确率（只输出正确率>55%的）
    
    Parameters
    ----------
    recorder : Recorder
        记录器对象
    long_term_dataset : Dataset
        长期数据集
    short_term_dataset : Dataset
        短期数据集
    """
    from pathlib import Path
    
    try:
        # 加载预测数据
        long_term_pred = recorder.load_object("long_term_pred.pkl")
        short_term_pred = recorder.load_object("short_term_pred.pkl")
        
        if long_term_pred is None or short_term_pred is None:
            print("  ⚠️  无法加载预测数据，跳过统计")
            return
        
        # 处理DataFrame格式
        if isinstance(long_term_pred, pd.DataFrame):
            long_term_pred = long_term_pred.iloc[:, 0] if len(long_term_pred.columns) > 0 else long_term_pred.squeeze()
        if isinstance(short_term_pred, pd.DataFrame):
            short_term_pred = short_term_pred.iloc[:, 0] if len(short_term_pred.columns) > 0 else short_term_pred.squeeze()
        
        # 获取label数据
        print("  - 获取标签数据...")
        try:
            # 获取test段的slice
            if 'test' in long_term_dataset.segments:
                test_slice = long_term_dataset.segments['test']
                if isinstance(test_slice, (tuple, list)):
                    test_selector = slice(*test_slice)
                else:
                    test_selector = test_slice
            else:
                print("  ⚠️  长期数据集中没有test段，跳过统计")
                return
            
            # 确保数据集已经setup
            if not hasattr(long_term_dataset.handler, '_learn') or long_term_dataset.handler._learn is None:
                long_term_dataset.setup_data()
            
            # 获取长期标签
            long_term_label_data = long_term_dataset.handler.fetch(
                selector=test_selector,
                col_set=["label"],
                data_key=long_term_dataset.handler.DK_L
            )
            
            if isinstance(long_term_label_data, pd.DataFrame) and len(long_term_label_data.columns) > 0:
                long_term_label = long_term_label_data.iloc[:, 0]
            else:
                long_term_label = long_term_label_data.squeeze() if hasattr(long_term_label_data, 'squeeze') else long_term_label_data
            
            # 获取短期标签
            if 'test' in short_term_dataset.segments:
                test_slice = short_term_dataset.segments['test']
                if isinstance(test_slice, (tuple, list)):
                    test_selector = slice(*test_slice)
                else:
                    test_selector = test_slice
            else:
                print("  ⚠️  短期数据集中没有test段，跳过统计")
                return
            
            if not hasattr(short_term_dataset.handler, '_learn') or short_term_dataset.handler._learn is None:
                short_term_dataset.setup_data()
            
            short_term_label_data = short_term_dataset.handler.fetch(
                selector=test_selector,
                col_set=["label"],
                data_key=short_term_dataset.handler.DK_L
            )
            
            if isinstance(short_term_label_data, pd.DataFrame) and len(short_term_label_data.columns) > 0:
                short_term_label = short_term_label_data.iloc[:, 0]
            else:
                short_term_label = short_term_label_data.squeeze() if hasattr(short_term_label_data, 'squeeze') else short_term_label_data
        except Exception as e:
            print(f"  ⚠️  获取标签数据失败: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 对齐索引
        if isinstance(long_term_pred.index, pd.MultiIndex) and isinstance(long_term_label.index, pd.MultiIndex):
            common_idx_long = long_term_pred.index.intersection(long_term_label.index)
            if len(common_idx_long) > 0:
                long_term_pred_aligned = long_term_pred.loc[common_idx_long]
                long_term_label_aligned = long_term_label.loc[common_idx_long]
            else:
                print("  ⚠️  长期预测和标签索引不匹配，跳过统计")
                return
        else:
            print("  ⚠️  预测或标签索引格式不正确，跳过统计")
            return
        
        if isinstance(short_term_pred.index, pd.MultiIndex) and isinstance(short_term_label.index, pd.MultiIndex):
            common_idx_short = short_term_pred.index.intersection(short_term_label.index)
            if len(common_idx_short) > 0:
                short_term_pred_aligned = short_term_pred.loc[common_idx_short]
                short_term_label_aligned = short_term_label.loc[common_idx_short]
            else:
                print("  ⚠️  短期预测和标签索引不匹配，跳过统计")
                return
        else:
            print("  ⚠️  预测或标签索引格式不正确，跳过统计")
            return
        
        # 准备输出目录
        output_dir = Path(f"mlruns/{recorder.experiment_id}/{recorder.id}/prediction_stats")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 将预测记录转换为DataFrame
        records_list = []
        for idx in common_idx_long:
            if idx in common_idx_short:
                date = idx[0] if isinstance(idx, tuple) else idx
                stock_id = idx[1] if isinstance(idx, tuple) and len(idx) > 1 else idx
                records_list.append({
                    'date': date,
                    'stock_id': stock_id,
                    'long_term_pred': long_term_pred_aligned.loc[idx],
                    'short_term_pred': short_term_pred_aligned.loc[idx] if idx in short_term_pred_aligned.index else None,
                    'long_term_label': long_term_label_aligned.loc[idx],
                    'short_term_label': short_term_label_aligned.loc[idx] if idx in short_term_label_aligned.index else None,
                })
        
        if len(records_list) == 0:
            print("  ⚠️  没有有效的预测记录，跳过统计")
            return
        
        df = pd.DataFrame(records_list)
        
        # 计算正确率（预测方向与实际收益方向一致）
        df['long_term_correct'] = (
            ((df['long_term_pred'] > 0) & (df['long_term_label'] > 0)) |
            ((df['long_term_pred'] < 0) & (df['long_term_label'] < 0))
        ) & df['long_term_label'].notna()
        
        df['short_term_correct'] = (
            ((df['short_term_pred'] > 0) & (df['short_term_label'] > 0)) |
            ((df['short_term_pred'] < 0) & (df['short_term_label'] < 0))
        ) & df['short_term_label'].notna()
        
        df['combined_correct'] = df['long_term_correct'] & df['short_term_correct']
        
        # 1. 计算个股总体正确率
        print("  - 计算个股总体正确率...")
        stock_stats = []
        
        for stock_id in df['stock_id'].unique():
            stock_df = df[df['stock_id'] == stock_id]
            
            # 长期因子正确率
            long_term_valid = stock_df['long_term_label'].notna()
            if long_term_valid.sum() > 0:
                long_term_accuracy = stock_df.loc[long_term_valid, 'long_term_correct'].sum() / long_term_valid.sum()
                long_term_total = long_term_valid.sum()
                long_term_correct = stock_df.loc[long_term_valid, 'long_term_correct'].sum()
            else:
                long_term_accuracy = 0.0
                long_term_total = 0
                long_term_correct = 0
            
            # 短期因子正确率
            short_term_valid = stock_df['short_term_label'].notna()
            if short_term_valid.sum() > 0:
                short_term_accuracy = stock_df.loc[short_term_valid, 'short_term_correct'].sum() / short_term_valid.sum()
                short_term_total = short_term_valid.sum()
                short_term_correct = stock_df.loc[short_term_valid, 'short_term_correct'].sum()
            else:
                short_term_accuracy = 0.0
                short_term_total = 0
                short_term_correct = 0
            
            # 综合正确率
            combined_valid = stock_df['long_term_label'].notna() & stock_df['short_term_label'].notna()
            if combined_valid.sum() > 0:
                combined_accuracy = stock_df.loc[combined_valid, 'combined_correct'].sum() / combined_valid.sum()
                combined_total = combined_valid.sum()
                combined_correct = stock_df.loc[combined_valid, 'combined_correct'].sum()
            else:
                combined_accuracy = 0.0
                combined_total = 0
                combined_correct = 0
            
            stock_stats.append({
                'stock_id': stock_id,
                'long_term_accuracy': long_term_accuracy,
                'long_term_total': long_term_total,
                'long_term_correct': long_term_correct,
                'short_term_accuracy': short_term_accuracy,
                'short_term_total': short_term_total,
                'short_term_correct': short_term_correct,
                'combined_accuracy': combined_accuracy,
                'combined_total': combined_total,
                'combined_correct': combined_correct,
            })
        
        stock_stats_df = pd.DataFrame(stock_stats)
        stock_stats_df = stock_stats_df.sort_values('combined_accuracy', ascending=False)
        
        # 只输出正确率>55%的
        filtered_stock_stats = stock_stats_df[stock_stats_df['combined_accuracy'] > 0.55].copy()
        
        # 输出到文件
        output_file = output_dir / "stock_prediction_accuracy_overall.csv"
        filtered_stock_stats.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"  ✓ 个股总体正确率统计已保存到: {output_file}")
        print(f"    共 {len(filtered_stock_stats)} 只股票正确率>55%（共 {len(stock_stats_df)} 只股票）")
        
        # 2. 分年度计算个股正确率
        print("  - 计算分年度个股正确率...")
        df['year'] = pd.to_datetime(df['date']).dt.year
        
        for year in sorted(df['year'].unique()):
            year_df = df[df['year'] == year]
            
            year_stock_stats = []
            for stock_id in year_df['stock_id'].unique():
                stock_year_df = year_df[year_df['stock_id'] == stock_id]
                
                # 长期因子正确率
                long_term_valid = stock_year_df['long_term_label'].notna()
                if long_term_valid.sum() > 0:
                    long_term_accuracy = stock_year_df.loc[long_term_valid, 'long_term_correct'].sum() / long_term_valid.sum()
                    long_term_total = long_term_valid.sum()
                    long_term_correct = stock_year_df.loc[long_term_valid, 'long_term_correct'].sum()
                else:
                    long_term_accuracy = 0.0
                    long_term_total = 0
                    long_term_correct = 0
                
                # 短期因子正确率
                short_term_valid = stock_year_df['short_term_label'].notna()
                if short_term_valid.sum() > 0:
                    short_term_accuracy = stock_year_df.loc[short_term_valid, 'short_term_correct'].sum() / short_term_valid.sum()
                    short_term_total = short_term_valid.sum()
                    short_term_correct = stock_year_df.loc[short_term_valid, 'short_term_correct'].sum()
                else:
                    short_term_accuracy = 0.0
                    short_term_total = 0
                    short_term_correct = 0
                
                # 综合正确率
                combined_valid = stock_year_df['long_term_label'].notna() & stock_year_df['short_term_label'].notna()
                if combined_valid.sum() > 0:
                    combined_accuracy = stock_year_df.loc[combined_valid, 'combined_correct'].sum() / combined_valid.sum()
                    combined_total = combined_valid.sum()
                    combined_correct = stock_year_df.loc[combined_valid, 'combined_correct'].sum()
                else:
                    combined_accuracy = 0.0
                    combined_total = 0
                    combined_correct = 0
                
                year_stock_stats.append({
                    'year': year,
                    'stock_id': stock_id,
                    'long_term_accuracy': long_term_accuracy,
                    'long_term_total': long_term_total,
                    'long_term_correct': long_term_correct,
                    'short_term_accuracy': short_term_accuracy,
                    'short_term_total': short_term_total,
                    'short_term_correct': short_term_correct,
                    'combined_accuracy': combined_accuracy,
                    'combined_total': combined_total,
                    'combined_correct': combined_correct,
                })
            
            if len(year_stock_stats) > 0:
                year_stats_df = pd.DataFrame(year_stock_stats)
                year_stats_df = year_stats_df.sort_values('combined_accuracy', ascending=False)
                
                # 只输出正确率>55%的
                filtered_year_stats = year_stats_df[year_stats_df['combined_accuracy'] > 0.55].copy()
                
                # 输出到文件
                output_file = output_dir / f"stock_prediction_accuracy_{year}.csv"
                filtered_year_stats.to_csv(output_file, index=False, encoding='utf-8-sig')
                print(f"  ✓ {year}年个股正确率统计已保存到: {output_file}")
                print(f"    {year}年共 {len(filtered_year_stats)} 只股票正确率>55%（共 {len(year_stats_df)} 只股票）")
        
        print(f"  ✓ 预测数据统计完成，结果保存在: {output_dir}")
        
    except Exception as e:
        print(f"  ⚠️  计算预测统计失败: {e}")
        import traceback
        traceback.print_exc()


def filter_stock_pool_by_win_rate(
    long_term_results: Optional[List[Dict]],
    short_term_results: Optional[List[Dict]],
    min_win_rate: float = 0.5,
    segment: str = 'valid'
) -> List[str]:
    """
    根据胜率筛选股票池
    
    Parameters
    ----------
    long_term_results : Optional[List[Dict]]
        长期预测胜率结果列表
    short_term_results : Optional[List[Dict]]
        短期预测胜率结果列表
    min_win_rate : float
        最小胜率阈值（默认0.5，即50%）
    segment : str
        评估的数据集分段（用于显示）
    
    Returns
    -------
    List[str]
        筛选后的股票池列表（长期和短期胜率都>=min_win_rate的股票）
    """
    if long_term_results is None or short_term_results is None:
        print(f"  ⚠️  无法筛选股票池：缺少胜率数据")
        return []
    
    # 转换为字典格式，方便查找
    long_term_dict = {r['stock_id']: r['win_rate'] for r in long_term_results}
    short_term_dict = {r['stock_id']: r['win_rate'] for r in short_term_results}
    
    # 找到两个模型都评估过的股票
    common_stocks = set(long_term_dict.keys()) & set(short_term_dict.keys())
    
    if len(common_stocks) == 0:
        print(f"  ⚠️  长期和短期模型没有共同的股票")
        return []
    
    # 筛选出两个模型胜率都>=阈值的股票
    filtered_stocks = []
    for stock_id in common_stocks:
        long_term_win_rate = long_term_dict[stock_id]
        short_term_win_rate = short_term_dict[stock_id]
        
        if long_term_win_rate >= min_win_rate and short_term_win_rate >= min_win_rate:
            filtered_stocks.append(stock_id)
    
    print(f"\n【股票池筛选（基于{segment}集胜率）】")
    print(f"  最小胜率阈值: {min_win_rate*100:.1f}%")
    print(f"  共同股票数: {len(common_stocks)}")
    print(f"  筛选后股票数: {len(filtered_stocks)} ({len(filtered_stocks)/len(common_stocks)*100:.1f}%)")
    
    if len(filtered_stocks) > 0:
        # 计算筛选后股票的平均胜率
        avg_long_term_wr = sum(long_term_dict[s] for s in filtered_stocks) / len(filtered_stocks)
        avg_short_term_wr = sum(short_term_dict[s] for s in filtered_stocks) / len(filtered_stocks)
        print(f"  筛选后长期平均胜率: {avg_long_term_wr*100:.2f}%")
        print(f"  筛选后短期平均胜率: {avg_short_term_wr*100:.2f}%")
    else:
        print(f"  ⚠️  警告: 没有股票满足筛选条件，将使用所有股票")
        filtered_stocks = list(common_stocks)
    
    return filtered_stocks


def log_daily_trades_and_positions(recorder) -> None:
    """
    打印每日买入卖出和持仓情况到日志文件
    
    Parameters
    ----------
    recorder : Recorder
        记录器对象
    """
    import logging
    from pathlib import Path
    
    # 创建日志文件路径
    log_dir = Path(f"mlruns/{recorder.experiment_id}/{recorder.id}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "daily_trades_and_positions.log"
    
    # 配置日志
    logger = logging.getLogger("DailyTrades")
    logger.setLevel(logging.INFO)
    
    # 如果已经有handler，先清除
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建文件handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建格式
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    try:
        # 加载持仓数据
        positions_path = "portfolio_analysis/positions_normal_1day.pkl"
        positions = recorder.load_object(positions_path)
        
        if positions is None:
            logger.warning("无法加载持仓数据")
            print(f"  ⚠️  无法加载持仓数据: {positions_path}")
            return
        
        # 加载报告数据（用于获取每日收益等信息）
        report_path = "portfolio_analysis/report_normal_1day.pkl"
        report = recorder.load_object(report_path)
        
        logger.info("=" * 100)
        logger.info("每日交易和持仓情况")
        logger.info("=" * 100)
        logger.info("")
        
        # 获取所有日期
        if isinstance(positions, dict):
            dates = sorted(positions.keys())
        elif isinstance(positions, pd.DataFrame):
            dates = sorted(positions.index.unique()) if hasattr(positions.index, 'unique') else sorted(set(positions.index))
        else:
            logger.warning(f"未知的持仓数据格式: {type(positions)}")
            print(f"  ⚠️  持仓数据格式: {type(positions)}")
            if positions is not None:
                print(f"  ⚠️  持仓数据内容示例: {str(positions)[:500]}")
            return
        
        logger.info(f"回测日期范围: {dates[0]} 到 {dates[-1]} (共 {len(dates)} 个交易日)")
        logger.info("")
        
        # 前一日持仓（用于计算变化）
        prev_positions = {}
        
        for i, date in enumerate(dates):
            logger.info("=" * 100)
            logger.info(f"日期: {date} (第 {i+1}/{len(dates)} 个交易日)")
            logger.info("-" * 100)
            
            # 获取当日持仓
            current_positions = {}
            cash = 0.0
            account_value = 0.0  # 账户总价值
            
            if isinstance(positions, dict):
                # positions是字典，键是日期，值是Position对象
                position_obj = positions.get(date)
                if position_obj is not None:
                    # Position对象有position属性，是一个字典
                    if hasattr(position_obj, 'position'):
                        pos_dict = position_obj.position
                        # 首先获取账户总价值（如果存在）
                        if 'now_account_value' in pos_dict:
                            account_value = pos_dict['now_account_value']
                            if not isinstance(account_value, (int, float)):
                                account_value = 0.0
                        
                        # 遍历持仓字典，排除'cash'和'now_account_value'键
                        for stock_id, stock_info in pos_dict.items():
                            if stock_id == 'cash':
                                cash = stock_info if isinstance(stock_info, (int, float)) else 0.0
                            elif stock_id == 'now_account_value':
                                # 已经处理过了
                                continue
                            else:
                                # stock_info可能是字典（包含amount, weight等）或直接是数量
                                if isinstance(stock_info, dict):
                                    amount = stock_info.get('amount', 0)
                                    weight = stock_info.get('weight', 0)
                                    # 优先使用weight，如果没有则使用amount
                                    value = weight if weight > 0 else amount
                                elif isinstance(stock_info, (int, float)):
                                    value = stock_info
                                else:
                                    value = 0
                                
                                if value > 0:
                                    current_positions[stock_id] = {
                                        'amount': stock_info.get('amount', 0) if isinstance(stock_info, dict) else stock_info,
                                        'weight': stock_info.get('weight', 0) if isinstance(stock_info, dict) else 0,
                                        'value': value
                                    }
                        
                        # 如果没有now_account_value，尝试从Position对象计算
                        if account_value == 0.0 and hasattr(position_obj, 'calculate_value'):
                            try:
                                account_value = position_obj.calculate_value()
                            except:
                                pass
                    elif isinstance(position_obj, dict):
                        # 如果直接是字典
                        if 'now_account_value' in position_obj:
                            account_value = position_obj['now_account_value']
                            if not isinstance(account_value, (int, float)):
                                account_value = 0.0
                        
                        for stock_id, stock_info in position_obj.items():
                            if stock_id == 'cash':
                                cash = stock_info if isinstance(stock_info, (int, float)) else 0.0
                            elif stock_id == 'now_account_value':
                                continue
                            else:
                                if isinstance(stock_info, dict):
                                    amount = stock_info.get('amount', 0)
                                    weight = stock_info.get('weight', 0)
                                    value = weight if weight > 0 else amount
                                elif isinstance(stock_info, (int, float)):
                                    value = stock_info
                                else:
                                    value = 0
                                
                                if value > 0:
                                    current_positions[stock_id] = {
                                        'amount': stock_info.get('amount', 0) if isinstance(stock_info, dict) else stock_info,
                                        'weight': stock_info.get('weight', 0) if isinstance(stock_info, dict) else 0,
                                        'value': value
                                    }
            elif isinstance(positions, pd.DataFrame):
                date_positions = positions.loc[date] if date in positions.index else pd.DataFrame()
                if len(date_positions) > 0:
                    for idx, row in date_positions.iterrows():
                        stock_id = idx if isinstance(idx, str) else row.get('instrument', idx)
                        if pd.isna(stock_id) or stock_id == 'cash':
                            continue
                        
                        if 'amount' in row:
                            amount = row['amount']
                        elif 'weight' in row:
                            amount = row['weight']
                        else:
                            amount = row.iloc[0] if len(row) > 0 else 0
                        
                        if amount > 0:
                            current_positions[stock_id] = {
                                'amount': amount,
                                'weight': row.get('weight', 0),
                                'value': amount
                            }
            
            # 计算买入和卖出
            buy_stocks = []
            sell_stocks = []
            hold_stocks = []
            
            # 提取当前持仓的股票ID和值（用于比较）
            def get_stock_value(stock_info):
                """从股票信息中提取数值用于比较"""
                if isinstance(stock_info, dict):
                    return stock_info.get('value', stock_info.get('amount', stock_info.get('weight', 0)))
                elif isinstance(stock_info, (int, float)):
                    return stock_info
                else:
                    return 0
            
            # 提取当前持仓的股票ID和值（用于比较）
            current_stocks = {k: get_stock_value(v) for k, v in current_positions.items()}
            prev_stocks = {k: get_stock_value(v) for k, v in prev_positions.items()}
            
            # 计算买入和卖出（基于数值比较）
            for stock_id in current_stocks:
                current_value = current_stocks[stock_id]
                prev_value = prev_stocks.get(stock_id, 0)
                
                if stock_id not in prev_stocks:
                    # 新买入
                    buy_stocks.append(stock_id)
                elif current_value > prev_value * 1.01:  # 允许1%的误差，视为加仓
                    buy_stocks.append(stock_id)
                else:
                    # 持仓不变或减少（减少会在下面处理）
                    hold_stocks.append(stock_id)
            
            for stock_id in prev_stocks:
                prev_value = prev_stocks[stock_id]
                current_value = current_stocks.get(stock_id, 0)
                
                if stock_id not in current_stocks:
                    # 完全卖出
                    sell_stocks.append(stock_id)
                elif prev_value > current_value * 1.01:  # 减仓超过1%
                    sell_stocks.append(stock_id)
            
            # 打印交易信息
            logger.info(f"【买入】 ({len(buy_stocks)} 只)")
            if buy_stocks:
                for stock_id in buy_stocks:
                    stock_info = current_positions.get(stock_id, {})
                    if isinstance(stock_info, dict):
                        amount = stock_info.get('amount', 0)
                        weight = stock_info.get('weight', 0)
                        logger.info(f"  + {stock_id:10s} | 数量: {amount:.2f} | 权重: {weight*100:.4f}%")
                    else:
                        logger.info(f"  + {stock_id:10s} | 数量/权重: {stock_info:.6f}")
            else:
                logger.info("  无")
            
            logger.info(f"\n【卖出】 ({len(sell_stocks)} 只)")
            if sell_stocks:
                for stock_id in sell_stocks:
                    prev_stock_info = prev_positions.get(stock_id, {})
                    if isinstance(prev_stock_info, dict):
                        prev_amount = prev_stock_info.get('amount', 0)
                        prev_weight = prev_stock_info.get('weight', 0)
                        logger.info(f"  - {stock_id:10s} | 原数量: {prev_amount:.2f} | 原权重: {prev_weight*100:.4f}%")
                    else:
                        logger.info(f"  - {stock_id:10s} | 原数量/权重: {prev_stock_info:.6f}")
            else:
                logger.info("  无")
            
            logger.info(f"\n【持仓】 (共 {len(current_positions)} 只)")
            if current_positions:
                # 按持仓权重或数量排序
                sorted_positions = sorted(
                    current_positions.items(), 
                    key=lambda x: x[1].get('weight', x[1].get('amount', 0)) if isinstance(x[1], dict) else x[1], 
                    reverse=True
                )
                for stock_id, stock_info in sorted_positions:
                    if isinstance(stock_info, dict):
                        amount = stock_info.get('amount', 0)
                        weight = stock_info.get('weight', 0)
                        logger.info(f"  {stock_id:10s} | 数量: {amount:.2f} | 权重: {weight*100:.4f}%")
                    else:
                        logger.info(f"  {stock_id:10s} | 数量/权重: {stock_info:.6f}")
            else:
                logger.info("  无持仓")
            
            # 打印账户信息
            logger.info(f"\n【账户信息】")
            
            # 优先使用从Position对象获取的账户总价值
            if account_value > 0:
                logger.info(f"  账户总价值: {account_value:,.2f}")
            elif report is not None:
                # 如果没有从Position获取到，尝试从report获取
                try:
                    if isinstance(report, pd.DataFrame) and date in report.index:
                        date_report = report.loc[date]
                        # 使用 'account' 字段（账户总价值），而不是 'value'（股票价值）
                        if 'account' in date_report:
                            account_value = date_report['account']
                            logger.info(f"  账户总价值: {account_value:,.2f}")
                        elif 'value' in date_report:
                            # 如果只有value，需要加上现金
                            stock_value = date_report['value']
                            account_value = stock_value + cash
                            logger.info(f"  账户总价值: {account_value:,.2f} (股票: {stock_value:,.2f} + 现金: {cash:,.2f})")
                except Exception as e:
                    pass
            
            # 打印现金信息
            if cash > 0:
                logger.info(f"  现金: {cash:,.2f}")
            
            # 打印当日收益率（从report获取）
            if report is not None:
                try:
                    if isinstance(report, pd.DataFrame) and date in report.index:
                        date_report = report.loc[date]
                        if 'return' in date_report:
                            daily_return = date_report['return']
                            logger.info(f"  当日收益率: {daily_return*100:.4f}%")
                            
                            # 验证收益率和账户价值的一致性
                            if i > 0 and account_value > 0:
                                # 获取前一日账户价值
                                prev_date = dates[i-1]
                                prev_account_value = 0.0
                                if isinstance(positions, dict):
                                    prev_position_obj = positions.get(prev_date)
                                    if prev_position_obj is not None:
                                        if hasattr(prev_position_obj, 'position'):
                                            prev_pos_dict = prev_position_obj.position
                                            if 'now_account_value' in prev_pos_dict:
                                                prev_account_value = prev_pos_dict['now_account_value']
                                        elif isinstance(prev_position_obj, dict) and 'now_account_value' in prev_position_obj:
                                            prev_account_value = prev_position_obj['now_account_value']
                                
                                # 如果无法从Position获取，尝试从report获取
                                if prev_account_value == 0.0:
                                    try:
                                        if prev_date in report.index:
                                            prev_date_report = report.loc[prev_date]
                                            if 'account' in prev_date_report:
                                                prev_account_value = prev_date_report['account']
                                    except:
                                        pass
                                
                                # 验证：当日账户价值应该等于前一日账户价值 * (1 + 收益率)
                                if prev_account_value > 0:
                                    expected_value = prev_account_value * (1 + daily_return)
                                    diff = abs(account_value - expected_value)
                                    diff_pct = (diff / prev_account_value) * 100 if prev_account_value > 0 else 0
                                    
                                    if diff_pct > 0.1:  # 差异超过0.1%时警告
                                        logger.warning(f"  ⚠️  数据不一致: 账户价值变化 {((account_value/prev_account_value - 1)*100):.4f}% vs 收益率 {daily_return*100:.4f}% (差异: {diff_pct:.4f}%)")
                except Exception as e:
                    pass
            
            logger.info("")
            
            # 更新前一日持仓
            prev_positions = current_positions.copy()
        
        logger.info("=" * 100)
        logger.info("日志生成完成")
        logger.info(f"日志文件路径: {log_file}")
        
        print(f"  ✓ 交易日志已保存到: {log_file}")
        
    except Exception as e:
        logger.error(f"生成交易日志时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def analyze_backtest_results(port_analysis: Dict[str, Any], recorder=None) -> None:
    """分析并打印回测结果"""
    print("\n" + "=" * 80)
    print("回测结果汇总")
    print("=" * 80)
    
    def calculate_sharpe(metrics: Dict[str, Any]) -> float:
        """计算夏普比率"""
        std = metrics.get('std', 0)
        ann_ret = metrics.get('annualized_return', 0)
        if std > 0:
            # 假设无风险利率为0，年化252个交易日
            return ann_ret / (std * (252 ** 0.5))
        return 0.0
    
    # 基准收益
    if 'benchmark' in port_analysis:
        bm = port_analysis['benchmark']
        bm_metrics = {
            '年化收益': bm.get('annualized_return', 0) * 100,
            '最大回撤': bm.get('max_drawdown', 0) * 100,
            '信息比率': bm.get('information_ratio', 0),
            '夏普比率': calculate_sharpe(bm),
            '年化波动率': bm.get('std', 0) * (252 ** 0.5) * 100
        }
        print_metrics(bm_metrics, "基准收益")
    
    # 超额收益(不含成本)
    if 'excess_return_without_cost' in port_analysis:
        ex = port_analysis['excess_return_without_cost']
        ex_metrics = {
            '年化收益': ex.get('annualized_return', 0) * 100,
            '最大回撤': ex.get('max_drawdown', 0) * 100,
            '信息比率': ex.get('information_ratio', 0),
            '夏普比率': calculate_sharpe(ex),
            '年化波动率': ex.get('std', 0) * (252 ** 0.5) * 100
        }
        print_metrics(ex_metrics, "超额收益(不含成本)")
    
    # 超额收益(含成本)
    if 'excess_return_with_cost' in port_analysis:
        ex = port_analysis['excess_return_with_cost']
        ex_metrics = {
            '年化收益': ex.get('annualized_return', 0) * 100,
            '最大回撤': ex.get('max_drawdown', 0) * 100,
            '信息比率': ex.get('information_ratio', 0),
            '夏普比率': calculate_sharpe(ex),
            '年化波动率': ex.get('std', 0) * (252 ** 0.5) * 100
        }
        print_metrics(ex_metrics, "超额收益(含成本)")
    
    # 计算相对基准的表现
    if 'benchmark' in port_analysis and 'excess_return_with_cost' in port_analysis:
        bm_ret = port_analysis['benchmark'].get('annualized_return', 0) * 100
        ex_ret = port_analysis['excess_return_with_cost'].get('annualized_return', 0) * 100
        print(f"\n【策略表现】")
        print(f"  策略年化收益: {ex_ret:.2f}%")
        print(f"  基准年化收益: {bm_ret:.2f}%")
        print(f"  超额年化收益: {ex_ret - bm_ret:.2f}%")
        if bm_ret != 0:
            print(f"  相对基准表现: {(ex_ret / bm_ret - 1) * 100:.2f}%")
    
    # 加载并显示信号分析结果（如果可用）
    if recorder is not None:
        try:
            ic = recorder.load_object("sig_analysis/ic.pkl")
            ric = recorder.load_object("sig_analysis/ric.pkl")
            if ic is not None and len(ic) > 0:
                # 使用IC绝对值来评估因子强度（IC有正有负，不能直接平均）
                ic_mean = ic.mean()  # 带符号的平均IC（用于判断方向）
                ic_mean_abs = ic.abs().mean()  # IC绝对值平均（用于判断强度）
                ic_std = ic.std()
                icir = ic_mean / ic_std if ic_std > 0 else 0
                icir_abs = ic_mean_abs / ic.std() if ic.std() > 0 else 0  # 使用IC绝对值计算的ICIR
                
                print(f"\n【信号质量分析】")
                print(f"  IC均值（带符号）: {ic_mean:.6f} {'✓' if abs(ic_mean) > 0.05 else '⚠️' if abs(ic_mean) > 0.02 else '❌'}")
                print(f"  IC均值（绝对值）: {ic_mean_abs:.6f} ← 这才是真正的因子强度")
                print(f"  IC标准差: {ic_std:.6f}")
                print(f"  ICIR（基于绝对值）: {icir_abs:.4f} {'✓' if icir_abs > 1.0 else '⚠️' if icir_abs > 0.5 else '❌'}")
                print(f"  ICIR（基于带符号）: {icir:.4f} {'✓' if abs(icir) > 1.0 else '⚠️' if abs(icir) > 0.5 else '❌'}")
                
                # 统计正IC和负IC
                positive_ic = ic[ic > 0]
                negative_ic = ic[ic < 0]
                if len(positive_ic) > 0:
                    print(f"  正IC数量: {len(positive_ic)}/{len(ic)}, 平均: {positive_ic.mean():.6f}")
                if len(negative_ic) > 0:
                    print(f"  负IC数量: {len(negative_ic)}/{len(ic)}, 平均: {negative_ic.mean():.6f}")
                
                if ric is not None and len(ric) > 0:
                    ric_mean = ric.mean()
                    ric_mean_abs = ric.abs().mean()
                    ric_std = ric.std()
                    ricir = ric_mean / ric_std if ric_std > 0 else 0
                    ricir_abs = ric_mean_abs / ric_std if ric_std > 0 else 0
                    print(f"  Rank IC均值（带符号）: {ric_mean:.6f} {'✓' if abs(ric_mean) > 0.05 else '⚠️' if abs(ric_mean) > 0.02 else '❌'}")
                    print(f"  Rank IC均值（绝对值）: {ric_mean_abs:.6f} ← 这才是真正的因子强度")
                    print(f"  Rank IC标准差: {ric_std:.6f}")
                    print(f"  Rank ICIR（基于绝对值）: {ricir_abs:.4f} {'✓' if ricir_abs > 1.0 else '⚠️' if ricir_abs > 0.5 else '❌'}")
                
                # 诊断建议（使用IC绝对值）
                print(f"\n【诊断建议】")
                if ic_mean_abs < 0.02:
                    print(f"  ⚠️  IC绝对值均值过低（{ic_mean_abs:.6f}），模型预测能力不足")
                    print(f"     - 检查模型训练是否正常")
                    print(f"     - 检查特征质量（因子IC是否足够高）")
                    print(f"     - 检查label_expr是否与因子挖掘时一致")
                elif ic_mean_abs < 0.05:
                    print(f"  ⚠️  IC绝对值均值偏低（{ic_mean_abs:.6f}），建议优化模型")
                else:
                    print(f"  ✓ IC绝对值均值正常（{ic_mean_abs:.6f}）")
                
                if icir_abs < 0.5:
                    print(f"  ⚠️  ICIR过低（{icir_abs:.4f}），信号不稳定")
                    print(f"     - 考虑使用更保守的策略参数")
                    print(f"     - 检查数据质量（是否有缺失值、异常值）")
                elif icir_abs < 1.0:
                    print(f"  ⚠️  ICIR偏低（{icir_abs:.4f}），信号稳定性一般")
                else:
                    print(f"  ✓ ICIR正常（{icir_abs:.4f}）")
                
                # 对比信号IC和回测表现
                if 'excess_return_without_cost' in port_analysis:
                    ex_ir = port_analysis['excess_return_without_cost'].get('information_ratio', 0)
                    if ic_mean_abs > 0.05 and ex_ir < 0.1:
                        print(f"  ⚠️  信号IC绝对值较高（{ic_mean_abs:.6f}）但回测信息比率较低（{ex_ir:.4f}）")
                        print(f"     - 可能是策略参数不当（topk、hold_thresh等）")
                        print(f"     - 可能是交易成本过高")
                        print(f"     - 可能是信号到交易的转换有问题")
        except Exception as e:
            print(f"  ⚠️  加载信号分析结果失败: {e}")
            # 不抛出异常，不影响主流程


def plot_backtest_results(port_analysis: Dict[str, Any], recorder=None) -> None:
    """使用 qlib 的绘图工具绘制回测结果图表"""
    if recorder is None:
        print("  ⚠️  无法绘图：缺少 recorder 对象")
        return
    
    try:
        print("\n" + "=" * 80)
        print("生成回测分析图表...")
        print("=" * 80)
        
        # 创建输出目录
        output_dir = Path(f"mlruns/{recorder.experiment_id}/{recorder.id}/charts")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载回测报告数据
        report_path = "portfolio_analysis/report_normal_1day.pkl"
        report_df = recorder.load_object(report_path)
        
        if report_df is None:
            print("  ⚠️  无法加载回测报告数据")
            return
        
        # 确保索引名为 date
        if report_df.index.name != 'date':
            report_df.index.name = 'date'
        
        # 检查必需的列
        required_cols = ['return', 'cost', 'bench', 'turnover']
        missing_cols = [col for col in required_cols if col not in report_df.columns]
        if missing_cols:
            print(f"  ⚠️  报告数据缺少必需的列: {missing_cols}")
            return
        
        # 1. 绘制回测报告图表（收益、成本、换手率等）
        print("  [1/3] 生成回测报告图表...")
        try:
            fig_list = qcr_analysis_position.report_graph(report_df, show_notebook=False)
            for i, fig in enumerate(fig_list):
                chart_path = output_dir / f"report_chart_{i+1}.html"
                fig.write_html(str(chart_path))
                print(f"    ✓ 已保存: {chart_path.name}")
        except Exception as e:
            print(f"    ⚠️  生成回测报告图表失败: {e}")
        
        # 2. 绘制风险分析图表
        print("  [2/3] 生成风险分析图表...")
        try:
            # 准备风险分析数据
            analysis_freq = "1day"  # 日频数据
            analysis = {}
            
            if 'benchmark' in port_analysis:
                # 基准风险分析
                analysis["benchmark"] = risk_analysis(
                    report_df["bench"], freq=analysis_freq
                )
            
            if 'excess_return_without_cost' in port_analysis:
                # 超额收益（不含成本）风险分析
                analysis["excess_return_without_cost"] = risk_analysis(
                    report_df["return"] - report_df["bench"], freq=analysis_freq
                )
            
            if 'excess_return_with_cost' in port_analysis:
                # 超额收益（含成本）风险分析
                analysis["excess_return_with_cost"] = risk_analysis(
                    report_df["return"] - report_df["bench"] - report_df["cost"], freq=analysis_freq
                )
            
            if analysis:
                analysis_df = pd.concat(analysis)  # type: pd.DataFrame
                fig_list = qcr_analysis_position.risk_analysis_graph(
                    analysis_df=analysis_df,
                    report_normal_df=report_df,
                    show_notebook=False
                )
                for i, fig in enumerate(fig_list):
                    chart_path = output_dir / f"risk_analysis_chart_{i+1}.html"
                    fig.write_html(str(chart_path))
                    print(f"    ✓ 已保存: {chart_path.name}")
        except Exception as e:
            print(f"    ⚠️  生成风险分析图表失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 3. 绘制累计收益曲线
        print("  [3/3] 生成累计收益曲线...")
        try:
            # 加载持仓数据
            positions_path = "portfolio_analysis/positions_normal_1day.pkl"
            positions = recorder.load_object(positions_path)
            
            if positions is None:
                print("    ⚠️  无法加载持仓数据，跳过累计收益曲线")
            else:
                # 准备标签数据（用于计算买入/卖出收益）
                # label_data 需要是 MultiIndex DataFrame，索引为 [instrument, datetime]
                # 从持仓数据中提取所有股票代码
                all_instruments = set()
                all_dates = []
                if isinstance(positions, dict):
                    for date, position_obj in positions.items():
                        all_dates.append(date)
                        if hasattr(position_obj, 'position'):
                            pos_dict = position_obj.position
                        elif isinstance(position_obj, dict):
                            pos_dict = position_obj
                        else:
                            continue
                        
                        for stock_id in pos_dict.keys():
                            if stock_id not in ['cash', 'now_account_value']:
                                all_instruments.add(stock_id)
                
                if not all_instruments or not all_dates:
                    print("    ⚠️  无法从持仓数据中提取股票代码，跳过累计收益曲线")
                else:
                    # 创建 MultiIndex DataFrame
                    # 使用基准收益作为所有股票的 label（因为基准收益对所有股票都一样）
                    all_dates = sorted(set(all_dates))
                    all_instruments = sorted(all_instruments)
                    
                    # 确保日期在 report_df 中存在
                    available_dates = [d for d in all_dates if d in report_df.index]
                    if not available_dates:
                        print("    ⚠️  持仓日期与报告日期不匹配，跳过累计收益曲线")
                    else:
                        # 创建 MultiIndex（只使用可用的日期）
                        index = pd.MultiIndex.from_product(
                            [all_instruments, available_dates],
                            names=['instrument', 'datetime']
                        )
                        
                        # 从 report_df 获取基准收益
                        label_values = []
                        for instrument in all_instruments:
                            for date in available_dates:
                                label_values.append(report_df.loc[date, 'bench'])
                        
                        label_data = pd.DataFrame({'label': label_values}, index=index)
                        
                        try:
                            fig_list = qcr_analysis_position.cumulative_return_graph(
                                position=positions,
                                report_normal=report_df,
                                label_data=label_data,
                                show_notebook=False
                            )
                            for i, fig in enumerate(fig_list):
                                chart_path = output_dir / f"cumulative_return_chart_{i+1}.html"
                                fig.write_html(str(chart_path))
                                print(f"    ✓ 已保存: {chart_path.name}")
                        except Exception as inner_e:
                            print(f"    ⚠️  调用 cumulative_return_graph 失败: {inner_e}")
                            import traceback
                            traceback.print_exc()
        except Exception as e:
            print(f"    ⚠️  生成累计收益曲线失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 4. 绘制模型性能分析图表（Score IC 和 Model Performance）
        print("  [4/5] 生成模型性能分析图表...")
        try:
            # 尝试加载长期和短期模型的预测数据
            models_to_analyze = []
            
            # 长期模型
            long_term_pred = recorder.load_object("long_term_pred.pkl")
            long_term_label = recorder.load_object("label.pkl")  # SignalRecord 保存的标签
            if long_term_pred is not None and long_term_label is not None:
                models_to_analyze.append(("长期模型", long_term_pred, long_term_label))
            
            # 短期模型
            short_term_pred = recorder.load_object("short_term_pred.pkl")
            # 短期模型的标签可能也在 label.pkl 中，或者需要从数据集加载
            if short_term_pred is not None:
                # 尝试加载短期标签
                short_term_label = None
                # 如果短期模型有单独的标签文件，可以在这里加载
                # 否则使用长期标签（如果可用）
                if long_term_label is not None:
                    short_term_label = long_term_label
                models_to_analyze.append(("短期模型", short_term_pred, short_term_label))
            
            # 如果没有找到长期和短期，尝试使用默认的 pred.pkl
            if not models_to_analyze:
                default_pred = recorder.load_object("pred.pkl")
                default_label = recorder.load_object("label.pkl")
                if default_pred is not None and default_label is not None:
                    models_to_analyze.append(("模型", default_pred, default_label))
            
            for model_name, pred_df, label_df in models_to_analyze:
                if pred_df is None or label_df is None:
                    continue
                
                try:
                    # 准备 pred_label（合并预测和标签）
                    # 确保 pred_df 有 'score' 列
                    if isinstance(pred_df, pd.Series):
                        pred_df = pred_df.to_frame("score")
                    elif "score" not in pred_df.columns:
                        # 使用第一列作为 score
                        pred_df = pred_df.iloc[:, [0]].copy()
                        pred_df.columns = ["score"]
                    
                    # 确保 label_df 有 'label' 列
                    if isinstance(label_df, pd.Series):
                        label_df = label_df.to_frame("label")
                    elif "label" not in label_df.columns:
                        # 使用第一列作为 label
                        label_df = label_df.iloc[:, [0]].copy()
                        label_df.columns = ["label"]
                    
                    # 合并预测和标签
                    pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
                    
                    # 4.1 Score IC 图表
                    print(f"    [{model_name}] 生成 Score IC 图表...")
                    try:
                        fig_list = qcr_analysis_position.score_ic_graph(pred_label, show_notebook=False)
                        for i, fig in enumerate(fig_list):
                            chart_path = output_dir / f"{model_name}_score_ic_chart_{i+1}.html"
                            chart_path = chart_path.with_name(chart_path.name.replace(" ", "_").replace("/", "_"))
                            fig.write_html(str(chart_path))
                            print(f"      ✓ 已保存: {chart_path.name}")
                    except Exception as e:
                        print(f"      ⚠️  生成 Score IC 图表失败: {e}")
                    
                    # 4.2 Model Performance 图表
                    print(f"    [{model_name}] 生成 Model Performance 图表...")
                    try:
                        fig_list = qcr_analysis_model.model_performance_graph(
                            pred_label=pred_label,
                            show_notebook=False,
                            graph_names=["group_return", "pred_ic", "pred_autocorr"]
                        )
                        for i, fig in enumerate(fig_list):
                            chart_path = output_dir / f"{model_name}_model_performance_chart_{i+1}.html"
                            chart_path = chart_path.with_name(chart_path.name.replace(" ", "_").replace("/", "_"))
                            fig.write_html(str(chart_path))
                            print(f"      ✓ 已保存: {chart_path.name}")
                    except Exception as e:
                        print(f"      ⚠️  生成 Model Performance 图表失败: {e}")
                        import traceback
                        traceback.print_exc()
                        
                except Exception as e:
                    print(f"    ⚠️  处理 {model_name} 失败: {e}")
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"    ⚠️  生成模型性能分析图表失败: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n✓ 所有图表已保存到: {output_dir}")
        print(f"  可以在浏览器中打开 HTML 文件查看交互式图表")
        
    except Exception as e:
        print(f"  ⚠️  绘图过程出错: {e}")
        import traceback
        traceback.print_exc()
            # 不抛出异常，不影响主流程


def main():
    """运行长期和短期因子结合的交易策略"""
    start_time = time.time()
    
    try:
        # 初始化 qlib
        print("=" * 80)
        print("初始化 Qlib...")
        qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn')
        print("✓ Qlib 初始化成功")
        
        # 加载配置文件
        config_path = Path(__file__).parent / 'workflow_config_long_short_strategy.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        print(f"\n加载配置文件: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 验证配置
        validate_config(config)
        print("✓ 配置文件验证通过")
        
        # 使用红利股固定股票池
        print("\n[检查股票池配置] 使用红利股固定股票池...")
        print(f"  红利股数量: {len(RED_CHIP_STOCKS)} 只")
        
        try:
            # 检查短期数据处理器配置
            # 路径: task -> short_term_dataset -> kwargs -> handler -> kwargs -> instruments
            try:
                short_handler_kwargs = config.get('task', {}).get('short_term_dataset', {}).get('kwargs', {}).get('handler', {}).get('kwargs', {})
                if short_handler_kwargs is not None:
                    short_handler_kwargs['instruments'] = RED_CHIP_STOCKS.copy()
                    print(f"  ✓ 短期数据处理器: 使用红利股股票池（{len(RED_CHIP_STOCKS)}只股票）")
            except Exception as e:
                print(f"  ⚠️  检查短期数据处理器配置失败: {e}")
            
            # 检查长期数据处理器配置
            # 路径: task -> long_term_dataset -> kwargs -> handler -> kwargs -> instruments
            try:
                long_handler_kwargs = config.get('task', {}).get('long_term_dataset', {}).get('kwargs', {}).get('handler', {}).get('kwargs', {})
                if long_handler_kwargs is not None:
                    long_handler_kwargs['instruments'] = RED_CHIP_STOCKS.copy()
                    print(f"  ✓ 长期数据处理器: 使用红利股股票池（{len(RED_CHIP_STOCKS)}只股票）")
            except Exception as e:
                print(f"  ⚠️  检查长期数据处理器配置失败: {e}")
                
        except Exception as e:
            print(f"  ⚠️  设置红利股股票池失败: {e}，将使用原始配置")
            import traceback
            traceback.print_exc()
        
        # 初始化短期模型和数据集
        print("\n" + "=" * 80)
        print("[初始化] 短期模型和数据集...")
        init_start = time.time()
        
        short_term_dataset = init_instance_by_config(config['task']['short_term_dataset'])
        
        # 获取实际的特征维度
        short_term_factor_count = None  # 初始化为None，表示未检测到
        try:
            # 方法1: 检查handler类型（适用于Alpha158、Alpha360等内置handler）
            handler_class_name = short_term_dataset.handler.__class__.__name__
            if handler_class_name == 'Alpha158':
                short_term_factor_count = 158
                print(f"  - 检测到Alpha158 handler，特征维度: 158")
            elif handler_class_name == 'Alpha360':
                short_term_factor_count = 360
                print(f"  - 检测到Alpha360 handler，特征维度: 360")
            
            # 方法2: 从handler的custom_factors获取（适用于CustomFactorHandler）
            if short_term_factor_count is None:
                if hasattr(short_term_dataset.handler, 'custom_factors') and short_term_dataset.handler.custom_factors:
                    short_term_factor_count = len(short_term_dataset.handler.custom_factors)
                    print(f"  - 从custom_factors检测到因子数量: {short_term_factor_count}")
                    
                    # 如果启用了股票ID one-hot编码，需要加上股票数量
                    if hasattr(short_term_dataset.handler, 'add_stock_id_onehot') and short_term_dataset.handler.add_stock_id_onehot:
                        if hasattr(short_term_dataset.handler, 'n_stocks') and short_term_dataset.handler.n_stocks > 0:
                            short_term_factor_count += short_term_dataset.handler.n_stocks
                            print(f"  - 添加股票ID one-hot编码: +{short_term_dataset.handler.n_stocks} 维")
                            print(f"  - 最终特征维度: {short_term_factor_count} (因子数 + 股票数)")
            
            # 方法3: 从实际数据中获取特征维度（最准确的方法）
            if short_term_factor_count is None:
                # 内存优化：只在必要时调用setup_data，避免重复加载
                if not hasattr(short_term_dataset.handler, '_learn') or short_term_dataset.handler._learn is None:
                    short_term_dataset.setup_data()
                # 尝试从handler的数据中获取特征维度
                if hasattr(short_term_dataset.handler, '_learn') and short_term_dataset.handler._learn is not None:
                    # 从学习数据中获取特征列数
                    feature_cols = short_term_dataset.handler._learn.columns.get_level_values(0).unique()
                    if 'feature' in feature_cols:
                        feature_data = short_term_dataset.handler._learn['feature']
                        if hasattr(feature_data, 'shape'):
                            actual_feature_dim = feature_data.shape[1] if len(feature_data.shape) > 1 else 1
                            if actual_feature_dim > 0:
                                short_term_factor_count = actual_feature_dim
                                print(f"  - 从实际数据检测到特征维度: {actual_feature_dim}")
                                # 如果数据已经包含了one-hot编码，这里已经包含了股票数量
                
                # 如果还是None，尝试从infer数据获取
                if short_term_factor_count is None:
                    if hasattr(short_term_dataset.handler, '_infer') and short_term_dataset.handler._infer is not None:
                        feature_cols = short_term_dataset.handler._infer.columns.get_level_values(0).unique()
                        if 'feature' in feature_cols:
                            feature_data = short_term_dataset.handler._infer['feature']
                            if hasattr(feature_data, 'shape'):
                                actual_feature_dim = feature_data.shape[1] if len(feature_data.shape) > 1 else 1
                                if actual_feature_dim > 0:
                                    short_term_factor_count = actual_feature_dim
                                    print(f"  - 从infer数据检测到特征维度: {actual_feature_dim}")
        except Exception as e:
            print(f"  ⚠️  检测特征维度时出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 在初始化模型之前，先修改配置中的d_feat并确保类型正确
        short_term_model_config = config['task']['short_term_model'].copy()
        if 'kwargs' in short_term_model_config:
            # 获取配置中的d_feat（作为默认值）
            config_d_feat = short_term_model_config['kwargs'].get('d_feat', None)
            
            # 确保特征维度正确：只有当检测到的特征维度 > 0 且与配置不同时才调整
            short_term_model_class = short_term_model_config.get('class', '')
            if short_term_factor_count is not None and short_term_factor_count > 0:
                if config_d_feat is None or config_d_feat != short_term_factor_count:
                    print(f"  - 自动调整配置中的特征维度: {config_d_feat} -> {short_term_factor_count}")
                    short_term_model_config['kwargs']['d_feat'] = short_term_factor_count
                    if short_term_model_class == 'KRNN':
                        short_term_model_config['kwargs']['fea_dim'] = short_term_factor_count
                else:
                    print(f"  - 配置中的特征维度={config_d_feat} 与检测到的匹配")
            else:
                # 如果无法检测到特征维度，使用配置中的值（不覆盖）
                if config_d_feat is not None:
                    print(f"  - 无法自动检测特征维度，使用配置中的值: {config_d_feat}")
                    short_term_factor_count = config_d_feat  # 用于后续验证
                else:
                    # 如果配置中也没有，使用默认值（KRNN 优先用配置中的 fea_dim）
                    if short_term_model_class == 'KRNN':
                        default_d_feat = short_term_model_config['kwargs'].get('fea_dim', 6)
                    else:
                        default_d_feat = 360
                    print(f"  - 无法检测特征维度且配置中未指定，使用默认值: {default_d_feat}")
                    short_term_model_config['kwargs']['d_feat'] = default_d_feat
                    if short_term_model_class == 'KRNN':
                        short_term_model_config['kwargs']['fea_dim'] = default_d_feat
                    short_term_factor_count = default_d_feat  # 用于后续验证
            
            # 确保数值类型参数是数字类型（YAML可能将科学计数法解析为字符串）
            for key in ['lr', 'dropout', 'reg', 'd_feat', 'fea_dim', 'd_model', 'hidden_size', 'batch_size', 'n_epochs', 'early_stop', 'cnn_dim', 'rnn_dim', 'rnn_dups', 'rnn_layers', 'cnn_kernel_size']:
                if key in short_term_model_config['kwargs']:
                    value = short_term_model_config['kwargs'][key]
                    if isinstance(value, str):
                        try:
                            # 尝试转换为浮点数（处理科学计数法）
                            short_term_model_config['kwargs'][key] = float(value)
                        except (ValueError, TypeError):
                            pass  # 如果转换失败，保持原值
        
        short_term_model = init_instance_by_config(short_term_model_config)
        
        # 验证特征维度是否正确设置（d_feat 或 fea_dim）
        if hasattr(short_term_model, 'fea_dim'):
            if short_term_model.fea_dim == short_term_factor_count:
                print(f"  ✓ 模型 fea_dim={short_term_model.fea_dim} 与特征维度匹配")
            else:
                print(f"  ⚠️  警告: 模型 fea_dim={short_term_model.fea_dim} 与特征维度={short_term_factor_count} 仍不匹配")
        elif hasattr(short_term_model, 'd_feat'):
            if short_term_model.d_feat == short_term_factor_count:
                print(f"  ✓ 模型 d_feat={short_term_model.d_feat} 与特征维度匹配")
            else:
                print(f"  ⚠️  警告: 模型 d_feat={short_term_model.d_feat} 与特征维度={short_term_factor_count} 仍不匹配")
        
        init_time = time.time() - init_start
        print(f"✓ 短期模型初始化成功 (耗时: {format_time(init_time)})")
        
        # 内存优化：如果短期数据集已经setup，释放不需要的临时数据
        if hasattr(short_term_dataset.handler, '_data') and hasattr(short_term_dataset.handler, 'drop_raw') and short_term_dataset.handler.drop_raw:
            # drop_raw=True时，_data应该已经被释放，但确保清理其他临时数据
            gc.collect()
        
        # 初始化长期模型和数据集
        print("\n[初始化] 长期模型和数据集...")
        init_start = time.time()
        
        long_term_dataset = init_instance_by_config(config['task']['long_term_dataset'])
        
        # 获取实际的特征维度
        long_term_factor_count = None  # 初始化为None，表示未检测到
        try:
            # 方法1: 从handler的custom_factors获取（适用于CustomFactorHandler）
            if hasattr(long_term_dataset.handler, 'custom_factors') and long_term_dataset.handler.custom_factors:
                long_term_factor_count = len(long_term_dataset.handler.custom_factors)
                print(f"  - 从custom_factors检测到因子数量: {long_term_factor_count}")
                
                # 如果启用了股票ID one-hot编码，需要加上股票数量
                if hasattr(long_term_dataset.handler, 'add_stock_id_onehot') and long_term_dataset.handler.add_stock_id_onehot:
                    if hasattr(long_term_dataset.handler, 'n_stocks') and long_term_dataset.handler.n_stocks > 0:
                        long_term_factor_count += long_term_dataset.handler.n_stocks
                        print(f"  - 添加股票ID one-hot编码: +{long_term_dataset.handler.n_stocks} 维")
                        print(f"  - 最终特征维度: {long_term_factor_count} (因子数 + 股票数)")
            
            # 方法2: 从实际数据中获取特征维度（更准确，适用于所有handler）
            # 内存优化：只在必要时调用setup_data，避免重复加载
            if not hasattr(long_term_dataset.handler, '_learn') or long_term_dataset.handler._learn is None:
                long_term_dataset.setup_data()
            # 尝试从handler的数据中获取特征维度
            if hasattr(long_term_dataset.handler, '_learn') and long_term_dataset.handler._learn is not None:
                # 从学习数据中获取特征列数
                feature_cols = long_term_dataset.handler._learn.columns.get_level_values(0).unique()
                if 'feature' in feature_cols:
                    feature_data = long_term_dataset.handler._learn['feature']
                    if hasattr(feature_data, 'shape'):
                        actual_feature_dim = feature_data.shape[1] if len(feature_data.shape) > 1 else 1
                        if actual_feature_dim > 0:
                            long_term_factor_count = actual_feature_dim
                            print(f"  - 从实际数据检测到特征维度: {actual_feature_dim}")
        except Exception as e:
            print(f"  ⚠️  检测特征维度时出错: {e}")
        
        # 在初始化模型之前，先修改配置中的特征维度并确保类型正确
        long_term_model_config = config['task']['long_term_model'].copy()
        if 'kwargs' in long_term_model_config:
            # 获取配置中的特征维度（作为默认值）
            config_d_feat = long_term_model_config['kwargs'].get('d_feat', None)
            long_term_model_class = long_term_model_config.get('class', '')
            
            # 确保特征维度正确：只有当检测到的特征维度 > 0 且与配置不同时才调整
            if long_term_factor_count is not None and long_term_factor_count > 0:
                if config_d_feat is None or config_d_feat != long_term_factor_count:
                    print(f"  - 自动调整配置中的特征维度: {config_d_feat} -> {long_term_factor_count}")
                    long_term_model_config['kwargs']['d_feat'] = long_term_factor_count
                    if long_term_model_class == 'KRNN':
                        long_term_model_config['kwargs']['fea_dim'] = long_term_factor_count
                else:
                    print(f"  - 配置中的特征维度={config_d_feat} 与检测到的匹配")
            else:
                # 如果无法检测到特征维度，使用配置中的值（不覆盖）
                if config_d_feat is not None:
                    print(f"  - 无法自动检测特征维度，使用配置中的值: {config_d_feat}")
                else:
                    if long_term_model_class == 'KRNN':
                        default_d_feat = long_term_model_config['kwargs'].get('fea_dim', 6)
                    else:
                        default_d_feat = 360
                    print(f"  - 无法检测特征维度且配置中未指定，使用默认值: {default_d_feat}")
                    long_term_model_config['kwargs']['d_feat'] = default_d_feat
                    if long_term_model_class == 'KRNN':
                        long_term_model_config['kwargs']['fea_dim'] = default_d_feat
            
            # 确保数值类型参数是数字类型（YAML可能将科学计数法解析为字符串）
            for key in ['lr', 'dropout', 'reg', 'd_feat', 'fea_dim', 'd_model', 'hidden_size', 'batch_size', 'n_epochs', 'early_stop', 'cnn_dim', 'rnn_dim', 'rnn_dups', 'rnn_layers', 'cnn_kernel_size']:
                if key in long_term_model_config['kwargs']:
                    value = long_term_model_config['kwargs'][key]
                    if isinstance(value, str):
                        try:
                            # 尝试转换为浮点数（处理科学计数法）
                            long_term_model_config['kwargs'][key] = float(value)
                        except (ValueError, TypeError):
                            pass  # 如果转换失败，保持原值
        
        long_term_model = init_instance_by_config(long_term_model_config)
        
        # 验证特征维度是否正确设置（fea_dim 或 d_feat）
        if hasattr(long_term_model, 'fea_dim'):
            final_fea_dim = long_term_model_config['kwargs'].get('fea_dim', long_term_model.fea_dim)
            if long_term_model.fea_dim == final_fea_dim:
                print(f"  ✓ 模型 fea_dim={long_term_model.fea_dim} 设置正确")
            else:
                print(f"  ⚠️  警告: 模型 fea_dim={long_term_model.fea_dim} 与配置值={final_fea_dim} 不匹配")
        elif hasattr(long_term_model, 'd_feat'):
            final_d_feat = long_term_model_config['kwargs'].get('d_feat', long_term_model.d_feat)
            if long_term_model.d_feat == final_d_feat:
                print(f"  ✓ 模型 d_feat={long_term_model.d_feat} 设置正确")
            else:
                print(f"  ⚠️  警告: 模型 d_feat={long_term_model.d_feat} 与配置值={final_d_feat} 不匹配")
        
        init_time = time.time() - init_start
        print(f"✓ 长期模型初始化成功 (耗时: {format_time(init_time)})")
        
        # 运行工作流
        experiment_name = 'long_short_factor_strategy'
        print("\n" + "=" * 80)
        print(f"开始运行实验: {experiment_name}")
        print("=" * 80)
        
        with R.start(experiment_name=experiment_name):
            recorder = R.get_recorder()
            
            # 记录参数
            task_config = config['task']
            R.log_params(**flatten_dict(task_config))
            
            # 训练短期模型
            print("\n[1/6] 训练短期模型...")
            train_start = time.time()
            try:
                # 诊断：检查训练数据质量
                print("  [诊断] 检查训练数据...")
                try:
                    # 检查 dataset 类型：TSDatasetH 返回 TSDataSampler，DatasetH 返回 DataFrame
                    from qlib.data.dataset import TSDatasetH
                    is_ts_dataset = isinstance(short_term_dataset, TSDatasetH)
                    
                    if is_ts_dataset:
                        # TSDatasetH: 使用 handler.fetch() 直接获取原始数据
                        train_slice = short_term_dataset.segments.get("train")
                        if train_slice:
                            train_data = short_term_dataset.handler.fetch(
                                slice(*train_slice) if isinstance(train_slice, (list, tuple)) else train_slice,
                                col_set=["feature", "label"],
                                data_key=short_term_dataset.handler.DK_L
                            )
                        else:
                            train_data = None
                    else:
                        # DatasetH: 使用 prepare() 获取处理后的数据
                        train_data = short_term_dataset.prepare("train", col_set=["feature", "label"], data_key=short_term_dataset.handler.DK_L)
                    
                    if train_data is not None and hasattr(train_data, 'empty') and not train_data.empty:
                        feature_cols = train_data["feature"].columns
                        print(f"    - 特征数量: {len(feature_cols)}")
                        print(f"    - 特征名称示例: {list(feature_cols[:5])}")
                        print(f"    - 训练样本数: {len(train_data)}")
                        
                        # 检查特征名是否丢失（如果都是Column_X格式，说明特征名丢失）
                        if all(str(col).startswith('Column_') for col in feature_cols[:10]):
                            print(f"    ⚠️  警告: 特征名显示为Column_X格式，可能特征名在数据处理中丢失")
                            print(f"       这可能导致模型无法正确学习因子信息")
                        
                        # 检查训练数据的IC（快速验证）
                        try:
                            train_features = train_data["feature"]
                            train_labels = train_data["label"].iloc[:, 0] if train_data["label"].ndim > 1 else train_data["label"]
                            
                            # 随机选择几个特征检查IC
                            sample_features = train_features.iloc[:, :min(5, len(train_features.columns))]
                            for col in sample_features.columns:
                                common_idx = sample_features[col].index.intersection(train_labels.index)
                                if len(common_idx) > 100:
                                    aligned_feat = sample_features[col].loc[common_idx]
                                    aligned_label = train_labels.loc[common_idx]
                                    valid_mask = aligned_feat.notna() & aligned_label.notna()
                                    if valid_mask.sum() > 100:
                                        try:
                                            feat_ic, _ = calc_ic(
                                                aligned_feat[valid_mask], 
                                                aligned_label[valid_mask], 
                                                dropna=True
                                            )
                                            if len(feat_ic) > 0:
                                                feat_ic_mean = feat_ic.abs().mean()
                                                print(f"    - 训练集特征 {col} 的IC强度: {feat_ic_mean:.6f}")
                                        except:
                                            pass
                        except Exception as e:
                            print(f"    ⚠️  无法检查训练数据IC: {e}")
                    elif is_ts_dataset:
                        # TSDatasetH 的数据检查：显示基本信息
                        print(f"    - 数据集类型: TSDatasetH (时间序列数据集)")
                        print(f"    - 时间序列长度: {short_term_dataset.step_len}")
                        print(f"    - 训练集时间段: {short_term_dataset.segments.get('train', 'N/A')}")
                        # 尝试获取特征数量
                        try:
                            if hasattr(short_term_dataset.handler, '_learn') and 'feature' in short_term_dataset.handler._learn:
                                feature_count = len(short_term_dataset.handler._learn['feature'])
                                print(f"    - 特征数量: {feature_count}")
                        except:
                            pass
                except Exception as e:
                    print(f"    ⚠️  无法检查训练数据: {e}")
                
                # 检查是否使用 n-fold 交叉验证
                use_nfold = config.get('task', {}).get('use_nfold', False)
                n_folds = config.get('task', {}).get('n_folds', 5)
                
                if use_nfold:
                    short_term_models = train_model_with_nfold(
                        short_term_model, 
                        short_term_dataset, 
                        "short_term",
                        n_folds=n_folds,
                        recorder=recorder,
                        use_nfold=True
                    )
                    # 保存主模型（使用最后一个 fold 的模型，或所有模型的集成）
                    if len(short_term_models) > 0:
                        short_term_model = short_term_models[-1]  # 使用最后一个模型作为主模型
                        R.save_objects(**{"short_term_params.pkl": short_term_model})
                        # 保存所有模型用于集成预测
                        R.save_objects(**{"short_term_all_models.pkl": short_term_models})
                else:
                    short_term_model.fit(short_term_dataset)
                    R.save_objects(**{"short_term_params.pkl": short_term_model})
                
                train_time = time.time() - train_start
                print(f"✓ 短期模型训练完成 (耗时: {format_time(train_time)})")
                
                # 内存优化：训练完成后，如果不需要原始数据，可以释放handler中的_learn数据
                # 注意：drop_raw=True时，_data已经被释放，但_learn可能还在
                # 这里不主动释放_learn，因为后续IC分析可能需要用到
                gc.collect()  # 触发垃圾回收，释放训练过程中的临时数据
            except Exception as e:
                print(f"❌ 短期模型训练失败: {e}")
                raise
            
            # 训练长期模型
            print("\n[2/6] 训练长期模型...")
            train_start = time.time()
            try:
                # 检查是否使用 n-fold 交叉验证
                use_nfold = config.get('task', {}).get('use_nfold', False)
                n_folds = config.get('task', {}).get('n_folds', 5)
                
                if use_nfold:
                    long_term_models = train_model_with_nfold(
                        long_term_model, 
                        long_term_dataset, 
                        "long_term",
                        n_folds=n_folds,
                        recorder=recorder,
                        use_nfold=True
                    )
                    # 保存主模型（使用最后一个 fold 的模型，或所有模型的集成）
                    if len(long_term_models) > 0:
                        long_term_model = long_term_models[-1]  # 使用最后一个模型作为主模型
                        R.save_objects(**{"long_term_params.pkl": long_term_model})
                        # 保存所有模型用于集成预测
                        R.save_objects(**{"long_term_all_models.pkl": long_term_models})
                else:
                    long_term_model.fit(long_term_dataset)
                    R.save_objects(**{"long_term_params.pkl": long_term_model})
                
                train_time = time.time() - train_start
                print(f"✓ 长期模型训练完成 (耗时: {format_time(train_time)})")
                
                # 内存优化：训练完成后触发垃圾回收
                gc.collect()
            except Exception as e:
                print(f"❌ 长期模型训练失败: {e}")
                raise
            
            # 生成短期预测信号
            print("\n[3/6] 生成短期预测信号...")
            pred_start = time.time()
            try:
                # 检查是否使用 n-fold 集成预测
                use_nfold = config.get('task', {}).get('use_nfold', False)
                ensemble_method = config.get('task', {}).get('ensemble_method', 'mean')
                
                # 初始化 short_term_sr 变量（用于后续IC分析）
                short_term_sr = None
                
                if use_nfold:
                    # 尝试加载所有模型
                    try:
                        short_term_all_models = recorder.load_object("short_term_all_models.pkl")
                        if short_term_all_models and len(short_term_all_models) > 1:
                            print(f"  使用 {len(short_term_all_models)} 个 fold 模型进行集成预测（方法: {ensemble_method}）...")
                            short_term_pred = predict_with_nfold_ensemble(
                                short_term_all_models,
                                short_term_dataset,
                                ensemble_method=ensemble_method
                            )
                            # 保存集成预测结果
                            recorder.save_objects(**{"short_term_pred.pkl": short_term_pred})
                            # 创建 SignalRecord 对象用于保存标签数据（用于后续IC分析）
                            short_term_sr = SignalRecord(short_term_model, short_term_dataset, recorder)
                            # 只生成标签，不生成预测（预测已经通过集成得到）
                            try:
                                short_term_sr.generate()  # 这会保存 label.pkl
                            except:
                                pass  # 如果生成失败，后续会从 dataset 获取标签
                        else:
                            # 如果只有一个模型，直接使用
                            short_term_sr = SignalRecord(short_term_model, short_term_dataset, recorder)
                            short_term_sr.generate()
                            short_term_pred = recorder.load_object("pred.pkl")
                    except Exception as e:
                        print(f"  ⚠️  加载 n-fold 模型失败，使用单模型预测: {e}")
                        short_term_sr = SignalRecord(short_term_model, short_term_dataset, recorder)
                        short_term_sr.generate()
                        short_term_pred = recorder.load_object("pred.pkl")
                else:
                    short_term_sr = SignalRecord(short_term_model, short_term_dataset, recorder)
                    short_term_sr.generate()
                    short_term_pred = recorder.load_object("pred.pkl")
                
                if short_term_pred is None or len(short_term_pred) == 0:
                    raise ValueError("短期预测信号为空")
                
                # 检查预测信号质量
                short_term_stats = check_prediction_quality(short_term_pred, "短期")
                if short_term_stats['valid']:
                    print(f"  - 预测信号统计: 数量={short_term_stats['count']}, "
                          f"均值={short_term_stats['mean']:.4f}, "
                          f"标准差={short_term_stats['std']:.4f}, "
                          f"NaN比例={short_term_stats['nan_ratio']*100:.2f}%")
                    if short_term_stats.get('outliers', 0) > 0:
                        print(f"  - 异常值: {short_term_stats['outliers']}个 ({short_term_stats.get('outlier_ratio', 0)*100:.2f}%)")
                    if 'warning' in short_term_stats:
                        print(f"  ⚠️  {short_term_stats['warning']}")
                else:
                    print(f"  ⚠️  警告: {short_term_stats.get('message', '短期预测信号无效')}")
                
                recorder.save_objects(**{"short_term_pred.pkl": short_term_pred})
                pred_time = time.time() - pred_start
                print(f"✓ 短期预测信号生成完成 (耗时: {format_time(pred_time)})")
            except Exception as e:
                print(f"❌ 短期预测信号生成失败: {e}")
                raise
            
            # 生成长期预测信号
            print("\n[4/6] 生成长期预测信号...")
            pred_start = time.time()
            try:
                # 检查是否使用 n-fold 集成预测
                use_nfold = config.get('task', {}).get('use_nfold', False)
                ensemble_method = config.get('task', {}).get('ensemble_method', 'mean')
                
                # 初始化 long_term_sr 变量（用于后续IC分析）
                long_term_sr = None
                
                if use_nfold:
                    # 尝试加载所有模型
                    try:
                        long_term_all_models = recorder.load_object("long_term_all_models.pkl")
                        if long_term_all_models and len(long_term_all_models) > 1:
                            print(f"  使用 {len(long_term_all_models)} 个 fold 模型进行集成预测（方法: {ensemble_method}）...")
                            long_term_pred = predict_with_nfold_ensemble(
                                long_term_all_models,
                                long_term_dataset,
                                ensemble_method=ensemble_method
                            )
                            # 保存集成预测结果
                            recorder.save_objects(**{"long_term_pred.pkl": long_term_pred})
                            # 创建 SignalRecord 对象用于保存标签数据（用于后续IC分析）
                            long_term_sr = SignalRecord(long_term_model, long_term_dataset, recorder)
                            # 只生成标签，不生成预测（预测已经通过集成得到）
                            try:
                                long_term_sr.generate()  # 这会保存 label.pkl
                            except:
                                pass  # 如果生成失败，后续会从 dataset 获取标签
                        else:
                            # 如果只有一个模型，直接使用
                            long_term_sr = SignalRecord(long_term_model, long_term_dataset, recorder)
                            long_term_sr.generate()
                            long_term_pred = recorder.load_object("pred.pkl")
                    except Exception as e:
                        print(f"  ⚠️  加载 n-fold 模型失败，使用单模型预测: {e}")
                        long_term_sr = SignalRecord(long_term_model, long_term_dataset, recorder)
                        long_term_sr.generate()
                        long_term_pred = recorder.load_object("pred.pkl")
                else:
                    long_term_sr = SignalRecord(long_term_model, long_term_dataset, recorder)
                    long_term_sr.generate()
                    long_term_pred = recorder.load_object("pred.pkl")  # 注意：这会覆盖短期预测的pred.pkl
                
                if long_term_pred is None or len(long_term_pred) == 0:
                    raise ValueError("长期预测信号为空")
                
                # 检查预测信号质量
                long_term_stats = check_prediction_quality(long_term_pred, "长期")
                if long_term_stats['valid']:
                    print(f"  - 预测信号统计: 数量={long_term_stats['count']}, "
                          f"均值={long_term_stats['mean']:.4f}, "
                          f"标准差={long_term_stats['std']:.4f}, "
                          f"NaN比例={long_term_stats['nan_ratio']*100:.2f}%")
                    if long_term_stats.get('outliers', 0) > 0:
                        print(f"  - 异常值: {long_term_stats['outliers']}个 ({long_term_stats.get('outlier_ratio', 0)*100:.2f}%)")
                    if 'warning' in long_term_stats:
                        print(f"  ⚠️  {long_term_stats['warning']}")
                else:
                    print(f"  ⚠️  警告: {long_term_stats.get('message', '长期预测信号无效')}")
                
                recorder.save_objects(**{"long_term_pred.pkl": long_term_pred})
                pred_time = time.time() - pred_start
                print(f"✓ 长期预测信号生成完成 (耗时: {format_time(pred_time)})")
            except Exception as e:
                print(f"❌ 长期预测信号生成失败: {e}")
                raise
            
            # 信号分析（分别分析长期和短期模型的预测信号）
            # 信号分析用于评估预测信号的质量，计算IC、ICIR等指标
            # IC (Information Coefficient): 预测值与真实收益的相关系数，衡量预测能力
            # ICIR (IC Information Ratio): IC的稳定性指标，ICIR越高说明信号越稳定
            # Rank IC: 使用排名计算的IC，更适合排序策略
            print("\n[5/6] 信号分析（分析长期和短期模型预测信号）...")
            print("  - 分析预测信号与真实收益的相关性")
            print("  - 计算IC、ICIR、Rank IC等指标评估信号质量")
            sig_start = time.time()
            
            # 导入calc_ic函数
            from qlib.contrib.eva.alpha import calc_ic
            
            try:
                # 分析长期模型IC
                print("\n  【长期模型IC分析】")
                try:
                    # 加载长期预测
                    long_term_pred = recorder.load_object("long_term_pred.pkl")
                    if long_term_pred is None:
                        long_term_pred = recorder.load_object("pred.pkl")  # 如果long_term_pred不存在，使用pred.pkl
                    
                    # 获取长期标签（从SignalRecord保存的label.pkl）
                    long_term_label = None
                    if long_term_sr is not None and hasattr(long_term_sr, 'load'):
                        try:
                            long_term_label = long_term_sr.load("label.pkl")
                        except:
                            pass
                    if long_term_label is None:
                        # 尝试从dataset获取标签
                        try:
                            test_slice = long_term_dataset.segments['test']
                            if isinstance(test_slice, (tuple, list)):
                                test_selector = slice(*test_slice)
                            else:
                                test_selector = test_slice
                            
                            # 内存优化：避免重复setup_data，复用已加载的数据
                            if not hasattr(long_term_dataset.handler, '_learn') or long_term_dataset.handler._learn is None:
                                long_term_dataset.setup_data()
                            # 内存优化：复用已获取的标签数据，避免重复fetch
                            if 'long_term_label_data' not in locals() or long_term_label_data is None:
                                long_term_label_data = long_term_dataset.handler.fetch(
                                    selector=test_selector,
                                    col_set=["label"],
                                    data_key=long_term_dataset.handler.DK_R
                                )
                            if isinstance(long_term_label_data, pd.DataFrame) and len(long_term_label_data.columns) > 0:
                                long_term_label = long_term_label_data.iloc[:, 0]
                            else:
                                long_term_label = long_term_label_data.squeeze() if hasattr(long_term_label_data, 'squeeze') else long_term_label_data
                        except Exception as e:
                            print(f"    ⚠️  获取长期标签失败: {e}")
                            long_term_label = None
                    
                    if long_term_pred is not None and long_term_label is not None:
                        # 处理DataFrame格式
                        if isinstance(long_term_pred, pd.DataFrame):
                            long_term_pred = long_term_pred.iloc[:, 0] if len(long_term_pred.columns) > 0 else long_term_pred.squeeze()
                        if isinstance(long_term_label, pd.DataFrame):
                            long_term_label = long_term_label.iloc[:, 0] if len(long_term_label.columns) > 0 else long_term_label.squeeze()
                        
                        # 对齐索引
                        common_idx = long_term_pred.index.intersection(long_term_label.index)
                        if len(common_idx) > 0:
                            long_term_pred_aligned = long_term_pred.loc[common_idx]
                            long_term_label_aligned = long_term_label.loc[common_idx]
                            
                            # 计算IC
                            long_term_ic, long_term_ric = calc_ic(long_term_pred_aligned, long_term_label_aligned, dropna=True)
                            
                            if long_term_ic is not None and len(long_term_ic) > 0:
                                ic_mean = long_term_ic.mean()
                                ic_std = long_term_ic.std()
                                icir = ic_mean / ic_std if ic_std > 0 else 0
                                
                                print(f"    IC均值: {ic_mean:.6f}")
                                print(f"    IC标准差: {ic_std:.6f}")
                                print(f"    ICIR: {icir:.4f}")
                            
                            if long_term_ric is not None and len(long_term_ric) > 0:
                                ric_mean = long_term_ric.mean()
                                ric_std = long_term_ric.std()
                                ricir = ric_mean / ric_std if ric_std > 0 else 0
                                
                                print(f"    Rank IC均值: {ric_mean:.6f}")
                                print(f"    Rank IC标准差: {ric_std:.6f}")
                                print(f"    Rank ICIR: {ricir:.4f}")
                        else:
                            print(f"    ⚠️  长期预测和标签索引不匹配")
                    else:
                        print(f"    ⚠️  无法获取长期预测或标签数据")
                except Exception as e:
                    print(f"    ⚠️  分析长期模型IC失败: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 分析短期模型IC
                print("\n  【短期模型IC分析】")
                try:
                    # 加载短期预测
                    short_term_pred = recorder.load_object("short_term_pred.pkl")
                    
                    # 获取短期标签（从SignalRecord保存的label.pkl）
                    short_term_label = None
                    if short_term_sr is not None and hasattr(short_term_sr, 'load'):
                        try:
                            short_term_label = short_term_sr.load("label.pkl")
                        except:
                            pass
                    if short_term_label is None:
                        # 尝试从dataset获取标签
                        try:
                            test_slice = short_term_dataset.segments['test']
                            if isinstance(test_slice, (tuple, list)):
                                test_selector = slice(*test_slice)
                            else:
                                test_selector = test_slice
                            
                            # 内存优化：避免重复setup_data，复用已加载的数据
                            if not hasattr(short_term_dataset.handler, '_learn') or short_term_dataset.handler._learn is None:
                                short_term_dataset.setup_data()
                            # 内存优化：复用已获取的标签数据，避免重复fetch
                            if 'short_term_label_data' not in locals() or short_term_label_data is None:
                                short_term_label_data = short_term_dataset.handler.fetch(
                                    selector=test_selector,
                                    col_set=["label"],
                                    data_key=short_term_dataset.handler.DK_R
                                )
                            if isinstance(short_term_label_data, pd.DataFrame) and len(short_term_label_data.columns) > 0:
                                short_term_label = short_term_label_data.iloc[:, 0]
                            else:
                                short_term_label = short_term_label_data.squeeze() if hasattr(short_term_label_data, 'squeeze') else short_term_label_data
                        except Exception as e:
                            print(f"    ⚠️  获取短期标签失败: {e}")
                            short_term_label = None
                    
                    if short_term_pred is not None and short_term_label is not None:
                        # 处理DataFrame格式
                        if isinstance(short_term_pred, pd.DataFrame):
                            short_term_pred = short_term_pred.iloc[:, 0] if len(short_term_pred.columns) > 0 else short_term_pred.squeeze()
                        if isinstance(short_term_label, pd.DataFrame):
                            short_term_label = short_term_label.iloc[:, 0] if len(short_term_label.columns) > 0 else short_term_label.squeeze()
                        
                        # 对齐索引
                        common_idx = short_term_pred.index.intersection(short_term_label.index)
                        if len(common_idx) > 0:
                            short_term_pred_aligned = short_term_pred.loc[common_idx]
                            short_term_label_aligned = short_term_label.loc[common_idx]
                            
                            # 计算IC
                            short_term_ic, short_term_ric = calc_ic(short_term_pred_aligned, short_term_label_aligned, dropna=True)
                            
                            if short_term_ic is not None and len(short_term_ic) > 0:
                                ic_mean = short_term_ic.mean()
                                ic_std = short_term_ic.std()
                                icir = ic_mean / ic_std if ic_std > 0 else 0
                                
                                print(f"    IC均值: {ic_mean:.6f}")
                                print(f"    IC标准差: {ic_std:.6f}")
                                print(f"    ICIR: {icir:.4f}")
                                
                                # 🔍 诊断：检查模型是否真的在学习
                                print(f"\n    【模型学习效果诊断】")
                                ic_abs_mean = short_term_ic.abs().mean()
                                print(f"    IC强度（绝对值均值）: {ic_abs_mean:.6f}")
                                
                                # 判断模型是否有效
                                if ic_abs_mean < 0.01:
                                    print(f"    ❌ 严重警告：模型IC强度 < 0.01，模型可能没有学习到有效信息！")
                                    print(f"       可能原因：")
                                    print(f"       1. 模型欠拟合（dropout过高、模型容量不足）")
                                    print(f"       2. 数据处理导致信息损失（归一化、填充等）")
                                    print(f"       3. 时间序列模型与单点数据不匹配")
                                    print(f"       4. 标签定义不一致")
                                    print(f"       5. 模型训练不充分（epochs太少、学习率不合适）")
                                elif ic_abs_mean < 0.02:
                                    print(f"    ⚠️  警告：模型IC强度 < 0.02，模型学习效果较差")
                                    print(f"       建议：检查模型配置、数据处理、训练参数")
                                elif ic_abs_mean < 0.05:
                                    print(f"    ⚠️  模型IC强度 < 0.05，学习效果一般，有改进空间")
                                else:
                                    print(f"    ✓ 模型IC强度 >= 0.05，学习效果较好")
                                
                                # 检查预测值分布
                                pred_values = short_term_pred_aligned.values
                                pred_std = np.std(pred_values)
                                pred_mean = np.mean(pred_values)
                                print(f"    预测值统计: 均值={pred_mean:.6f}, 标准差={pred_std:.6f}")
                                
                                if pred_std < 0.001:
                                    print(f"    ⚠️  警告：预测值标准差极小，模型可能输出常数（未学习）")
                                elif pred_std < 0.01:
                                    print(f"    ⚠️  警告：预测值变化很小，模型可能学习不充分")
                                
                                # 检查IC的稳定性
                                positive_ic_ratio = (short_term_ic > 0).sum() / len(short_term_ic)
                                print(f"    正IC比例: {positive_ic_ratio:.2%}")
                                
                                if 0.3 < positive_ic_ratio < 0.7:
                                    print(f"    ⚠️  警告：正IC比例接近50%，模型预测可能接近随机")
                                elif positive_ic_ratio < 0.3 or positive_ic_ratio > 0.7:
                                    print(f"    ✓ 正IC比例偏离50%，说明模型有方向性预测能力")
                            
                            if short_term_ric is not None and len(short_term_ric) > 0:
                                ric_mean = short_term_ric.mean()
                                ric_std = short_term_ric.std()
                                ricir = ric_mean / ric_std if ric_std > 0 else 0
                                
                                print(f"    Rank IC均值: {ric_mean:.6f}")
                                print(f"    Rank IC标准差: {ric_std:.6f}")
                                print(f"    Rank ICIR: {ricir:.4f}")
                            
                            # 🔍 诊断：检查训练集IC vs 测试集IC（判断是否过拟合）
                            print(f"\n    【过拟合/欠拟合诊断】")
                            try:
                                # 获取训练集预测
                                train_slice = short_term_dataset.segments.get('train')
                                if train_slice:
                                    # 临时设置test segment为train segment
                                    original_test = short_term_dataset.segments.get('test')
                                    short_term_dataset.segments['test'] = train_slice
                                    try:
                                        train_pred = short_term_model.predict(short_term_dataset)
                                        if train_pred is not None and len(train_pred) > 0:
                                            # 获取训练集标签
                                            train_selector = slice(*train_slice) if isinstance(train_slice, (list, tuple)) else train_slice
                                            # 内存优化：避免重复setup_data，复用已加载的数据
                                            if not hasattr(short_term_dataset.handler, '_learn') or short_term_dataset.handler._learn is None:
                                                short_term_dataset.setup_data()
                                            train_label_data = short_term_dataset.handler.fetch(
                                                selector=train_selector,
                                                col_set=["label"],
                                                data_key=short_term_dataset.handler.DK_L  # 使用DK_L，与训练时一致
                                            )
                                            if isinstance(train_label_data, pd.DataFrame) and len(train_label_data.columns) > 0:
                                                train_label = train_label_data.iloc[:, 0]
                                            else:
                                                train_label = train_label_data.squeeze() if hasattr(train_label_data, 'squeeze') else train_label_data
                                            
                                            # 对齐并计算训练集IC
                                            if isinstance(train_pred, pd.DataFrame):
                                                train_pred = train_pred.iloc[:, 0] if len(train_pred.columns) > 0 else train_pred.squeeze()
                                            
                                            train_common_idx = train_pred.index.intersection(train_label.index)
                                            if len(train_common_idx) > 0:
                                                train_pred_aligned = train_pred.loc[train_common_idx]
                                                train_label_aligned = train_label.loc[train_common_idx]
                                                train_ic, _ = calc_ic(train_pred_aligned, train_label_aligned, dropna=True)
                                                
                                                if train_ic is not None and len(train_ic) > 0:
                                                    train_ic_mean = train_ic.mean()
                                                    train_ic_abs = train_ic.abs().mean()
                                                    test_ic_abs = short_term_ic.abs().mean()
                                                    
                                                    print(f"    训练集IC强度: {train_ic_abs:.6f}")
                                                    print(f"    测试集IC强度: {test_ic_abs:.6f}")
                                                    print(f"    差异: {train_ic_abs - test_ic_abs:.6f}")
                                                    
                                                    if train_ic_abs > test_ic_abs * 2:
                                                        print(f"    ❌ 严重过拟合：训练集IC强度是测试集的 {train_ic_abs/test_ic_abs:.2f} 倍")
                                                        print(f"       说明：模型在训练集上学到了噪声，泛化能力差")
                                                        print(f"       建议：增加正则化（dropout、L1/L2）、减少模型复杂度")
                                                    elif train_ic_abs > test_ic_abs * 1.5:
                                                        print(f"    ⚠️  过拟合：训练集IC强度明显高于测试集")
                                                        print(f"       建议：适当增加正则化")
                                                    elif train_ic_abs < test_ic_abs * 0.8:
                                                        print(f"    ⚠️  欠拟合：训练集IC强度低于测试集")
                                                        print(f"       说明：模型学习不充分，可能原因：")
                                                        print(f"       1. dropout过高（当前: {config['task']['short_term_model']['kwargs'].get('dropout', 'N/A')}）")
                                                        print(f"       2. 模型容量不足（hidden_size: {config['task']['short_term_model']['kwargs'].get('hidden_size', 'N/A')}）")
                                                        print(f"       3. 训练轮数不足（n_epochs: {config['task']['short_term_model']['kwargs'].get('n_epochs', 'N/A')}）")
                                                        print(f"       建议：降低dropout、增加模型容量、增加训练轮数")
                                                    else:
                                                        print(f"    ✓ 训练集和测试集IC强度接近，模型泛化能力正常")
                                    finally:
                                        # 恢复test segment
                                        if original_test is not None:
                                            short_term_dataset.segments['test'] = original_test
                                        elif 'test' in short_term_dataset.segments and train_slice == short_term_dataset.segments.get('test'):
                                            del short_term_dataset.segments['test']
                            except Exception as e:
                                print(f"    ⚠️  无法计算训练集IC: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            # 诊断：分析模型IC与因子IC的差异
                            print(f"\n    【模型IC vs 因子IC 对比分析】")
                            
                            # 尝试计算单个因子的IC（用于对比）
                            try:
                                # 获取因子数据
                                test_slice = short_term_dataset.segments['test']
                                if isinstance(test_slice, (tuple, list)):
                                    test_selector = slice(*test_slice)
                                else:
                                    test_selector = test_slice
                                
                                # 内存优化：复用之前已加载的标签数据，避免重复fetch
                                # 如果之前已经获取过标签数据，直接使用
                                if 'short_term_label_data' in locals() and short_term_label_data is not None:
                                    label_data = short_term_label_data
                                else:
                                    # 内存优化：避免重复setup_data
                                    if not hasattr(short_term_dataset.handler, '_learn') or short_term_dataset.handler._learn is None:
                                        short_term_dataset.setup_data()
                                    
                                    # 获取标签数据（复用之前获取的）
                                    label_data = short_term_label_data if 'short_term_label_data' in locals() else short_term_dataset.handler.fetch(
                                        selector=test_selector,
                                        col_set=["label"],
                                        data_key=short_term_dataset.handler.DK_R
                                    )
                                
                                # 获取特征数据
                                # 内存优化：避免重复fetch，如果可能的话复用已加载的数据
                                if not hasattr(short_term_dataset.handler, '_learn') or short_term_dataset.handler._learn is None:
                                    short_term_dataset.setup_data()
                                
                                feature_data = short_term_dataset.handler.fetch(
                                    selector=test_selector,
                                    col_set=["feature"],
                                    data_key=short_term_dataset.handler.DK_R
                                )
                                
                                # 诊断：检查特征名
                                if isinstance(feature_data, pd.DataFrame):
                                    print(f"    [诊断] 测试集特征名: {list(feature_data.columns[:10])}")
                                    if all(str(col).startswith('Column_') for col in feature_data.columns[:10]):
                                        print(f"    ⚠️  警告: 特征名显示为Column_X格式，特征名可能在processor处理中丢失")
                                        print(f"       这会导致模型无法正确学习因子信息")
                                        print(f"       建议：检查processor是否正确保留了特征名")
                                        
                                        # 尝试从handler获取原始特征名
                                        try:
                                            if hasattr(short_term_dataset.handler, 'custom_factors'):
                                                original_factor_names = list(short_term_dataset.handler.custom_factors.keys())
                                                print(f"    [诊断] 原始因子名（前10个）: {original_factor_names[:10]}")
                                                print(f"    [诊断] 原始因子数量: {len(original_factor_names)}")
                                                print(f"    [诊断] 当前特征数量: {len(feature_data.columns)}")
                                                if len(original_factor_names) != len(feature_data.columns):
                                                    print(f"    ⚠️  因子数量不匹配！原始因子数={len(original_factor_names)}, 特征数={len(feature_data.columns)}")
                                        except Exception as e:
                                            print(f"    ⚠️  无法获取原始因子名: {e}")
                                
                                if isinstance(label_data, pd.DataFrame) and len(label_data.columns) > 0:
                                    labels = label_data.iloc[:, 0]
                                else:
                                    labels = label_data.squeeze() if hasattr(label_data, 'squeeze') else label_data
                                
                                # 计算每个因子的IC
                                if isinstance(feature_data, pd.DataFrame) and len(feature_data) > 0:
                                    factor_ics = []
                                    factor_names = []
                                    
                                    for col in feature_data.columns:
                                        factor_values = feature_data[col]
                                        # 对齐数据
                                        common_idx = factor_values.index.intersection(labels.index)
                                        if len(common_idx) > 0:
                                            aligned_factor = factor_values.loc[common_idx]
                                            aligned_label = labels.loc[common_idx]
                                            valid_mask = aligned_factor.notna() & aligned_label.notna()
                                            aligned_factor = aligned_factor[valid_mask]
                                            aligned_label = aligned_label[valid_mask]
                                            
                                            if len(aligned_factor) > 10:
                                                try:
                                                    factor_ic, factor_ric = calc_ic(aligned_factor, aligned_label, dropna=True)
                                                    if len(factor_ic) > 0:
                                                        # 计算IC的均值（带符号）和绝对值均值（强度）
                                                        factor_ic_mean = factor_ic.mean()  # 带符号的IC均值
                                                        factor_ic_abs_mean = factor_ic.abs().mean()  # IC强度（绝对值均值）
                                                        factor_ic_std = factor_ic.std()
                                                        factor_icir = factor_ic_mean / factor_ic_std if factor_ic_std > 0 else 0
                                                        
                                                        factor_ric_mean = factor_ric.mean() if len(factor_ric) > 0 else 0
                                                        factor_ric_abs_mean = factor_ric.abs().mean() if len(factor_ric) > 0 else 0
                                                        
                                                        factor_ics.append({
                                                            'mean': factor_ic_mean,
                                                            'abs_mean': factor_ic_abs_mean,
                                                            'std': factor_ic_std,
                                                            'icir': factor_icir,
                                                            'ric_mean': factor_ric_mean,
                                                            'ric_abs_mean': factor_ric_abs_mean,
                                                            'name': str(col)
                                                        })
                                                        factor_names.append(str(col))
                                                except Exception as e:
                                                    logger.debug(f"计算因子 {col} 的IC失败: {e}")
                                                    pass
                                    
                                    if len(factor_ics) > 0:
                                        # 提取带符号的IC和绝对值IC
                                        factor_ic_means = [f['mean'] for f in factor_ics]
                                        factor_ic_abs_means = [f['abs_mean'] for f in factor_ics]
                                        
                                        avg_factor_ic = np.mean(factor_ic_means)  # 带符号的平均IC
                                        avg_factor_ic_abs = np.mean(factor_ic_abs_means)  # 平均IC强度
                                        max_factor_ic = np.max(factor_ic_abs_means)  # 最大IC强度
                                        min_factor_ic = np.min(factor_ic_abs_means)  # 最小IC强度
                                        
                                        print(f"    单个因子IC统计:")
                                        print(f"      - 平均IC（带符号）: {avg_factor_ic:.6f}")
                                        print(f"      - 平均IC强度: {avg_factor_ic_abs:.6f}")
                                        print(f"      - 最大IC强度: {max_factor_ic:.6f}")
                                        print(f"      - 最小IC强度: {min_factor_ic:.6f}")
                                        print(f"      - 因子数量: {len(factor_ics)}")
                                        
                                        # 显示Top 10和Bottom 10因子的详细IC信息
                                        print(f"\n    Top 10 因子（按IC强度排序）:")
                                        sorted_factors = sorted(factor_ics, key=lambda x: x['abs_mean'], reverse=True)
                                        for i, f in enumerate(sorted_factors[:10], 1):
                                            factor_name = f.get('name', 'Unknown')
                                            print(f"      {i:2d}. {factor_name[:40]:40s} | IC: {f['mean']:8.6f} | IC强度: {f['abs_mean']:8.6f} | ICIR: {f.get('icir', 0):7.4f}")
                                        
                                        print(f"\n    Bottom 10 因子（按IC强度排序）:")
                                        for i, f in enumerate(sorted_factors[-10:], 1):
                                            factor_name = f.get('name', 'Unknown')
                                            print(f"      {i:2d}. {factor_name[:40]:40s} | IC: {f['mean']:8.6f} | IC强度: {f['abs_mean']:8.6f} | ICIR: {f.get('icir', 0):7.4f}")
                                        
                                        print(f"\n    模型IC vs 因子IC:")
                                        print(f"      - 模型IC: {ic_mean:.6f}")
                                        print(f"      - 平均因子IC（带符号）: {avg_factor_ic:.6f}")
                                        print(f"      - 平均因子IC强度: {avg_factor_ic_abs:.6f}")
                                        
                                        # 使用IC强度（绝对值）来比较，因为IC的正负只是方向问题
                                        model_ic_strength = abs(ic_mean)
                                        if model_ic_strength < avg_factor_ic_abs * 0.8:
                                            print(f"      ⚠️  模型IC强度（{model_ic_strength:.6f}）明显低于平均因子IC强度（{avg_factor_ic_abs:.6f}），可能原因：")
                                            print(f"        1. 特征标准化（ZScoreNorm）可能改变了因子分布")
                                            print(f"        2. 模型复杂度不够（max_depth={config['task']['short_term_model']['kwargs'].get('max_depth', 'N/A')}）")
                                            print(f"        3. 因子之间存在多重共线性，模型未充分利用")
                                            print(f"        4. 模型过拟合或欠拟合")
                                            print(f"        5. 特征重要性分布不均，模型只利用了部分因子")
                                        elif model_ic_strength > avg_factor_ic_abs:
                                            print(f"      ✓ 模型IC强度（{model_ic_strength:.6f}）高于平均因子IC强度（{avg_factor_ic_abs:.6f}），说明模型很好地利用了因子组合")
                                        else:
                                            print(f"      ⚠️  模型IC强度（{model_ic_strength:.6f}）略低于平均因子IC强度（{avg_factor_ic_abs:.6f}），建议优化模型参数")
                            except Exception as e:
                                print(f"    ⚠️  无法计算因子IC对比: {e}")
                            
                            # 内存优化：IC分析完成后，释放临时数据
                            # 注意：保留short_term_label_data，因为可能在其他地方用到
                            if 'feature_data' in locals():
                                del feature_data
                            if 'label_data' in locals() and 'short_term_label_data' in locals():
                                # 如果label_data是short_term_label_data的引用，不要删除
                                pass
                            gc.collect()
                            
                            # 尝试获取特征重要性（如果是LGBM模型）
                            try:
                                if hasattr(short_term_model, 'get_feature_importance'):
                                    feature_importance = short_term_model.get_feature_importance()
                                    if feature_importance is not None and len(feature_importance) > 0:
                                        print(f"\n    【特征重要性分析】")
                                        print(f"    Top 10 重要特征:")
                                        top_features = feature_importance.head(10)
                                        for i, (feat_name, importance) in enumerate(top_features.items(), 1):
                                            print(f"      {i:2d}. {feat_name:30s} | 重要性: {importance:.2f}")
                                        
                                        # 检查特征重要性分布
                                        importance_ratio = feature_importance.max() / feature_importance.mean() if feature_importance.mean() > 0 else 0
                                        if importance_ratio > 10:
                                            print(f"    ⚠️  特征重要性分布不均（最大/平均={importance_ratio:.2f}），可能存在主导特征")
                                            print(f"       建议：检查是否有异常重要的特征，考虑特征选择或正则化")
                                        elif importance_ratio < 2:
                                            print(f"    ⚠️  特征重要性分布过于均匀（最大/平均={importance_ratio:.2f}），模型可能未充分学习")
                                            print(f"       建议：增加模型复杂度（max_depth、num_leaves）或调整学习率")
                                        else:
                                            print(f"    ✓ 特征重要性分布正常（最大/平均={importance_ratio:.2f}）")
                                        
                                        # 检查是否有大量特征重要性为0
                                        zero_importance_count = (feature_importance == 0).sum()
                                        if zero_importance_count > len(feature_importance) * 0.5:
                                            print(f"    ⚠️  有 {zero_importance_count}/{len(feature_importance)} 个特征重要性为0，模型可能未充分利用所有因子")
                                            print(f"       建议：检查特征质量，考虑特征选择或增加模型复杂度")
                            except Exception as e:
                                print(f"    ⚠️  无法获取特征重要性: {e}")
                            
                            # 优化建议
                            print(f"\n    【优化建议】")
                            if abs(ic_mean) < 0.05:
                                print(f"    1. 增加模型复杂度：")
                                print(f"       - 增加 max_depth（当前: {config['task']['short_term_model']['kwargs'].get('max_depth', 'N/A')}）")
                                print(f"       - 增加 num_leaves（当前: {config['task']['short_term_model']['kwargs'].get('num_leaves', 'N/A')}）")
                                print(f"    2. 调整学习率：")
                                print(f"       - 降低 learning_rate（当前: {config['task']['short_term_model']['kwargs'].get('learning_rate', 'N/A')}）并增加 num_boost_round")
                                print(f"    3. 检查特征处理：")
                                print(f"       - RobustZScoreNorm 可能改变了因子分布，考虑使用 ZScoreNorm 或 MinMaxNorm")
                                print(f"       - 检查 clip_outlier 参数是否过于激进")
                                print(f"    4. 检查数据质量：")
                                print(f"       - 确认 label_expr 与因子挖掘时一致（已修复为: Ref($close, -1)/$close - 1）")
                                print(f"       - 检查是否有大量缺失值或异常值")
                                print(f"    5. 考虑使用更复杂的模型：")
                                print(f"       - 尝试 GATs TS 或其他深度学习模型")
                                print(f"       - 或使用集成方法（多个LGBM模型）")
                        else:
                            print(f"    ⚠️  短期预测和标签索引不匹配")
                    else:
                        print(f"    ⚠️  无法获取短期预测或标签数据")
                except Exception as e:
                    print(f"    ⚠️  分析短期模型IC失败: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 同时使用SigAnaRecord分析长期模型（用于兼容性，因为pred.pkl是长期预测）
                try:
                    sar = SigAnaRecord(recorder)
                    sar.generate()
                except Exception as e:
                    print(f"  ⚠️  SigAnaRecord分析失败: {e}（不影响主流程）")
                
                sig_time = time.time() - sig_start
                print(f"\n✓ 信号分析完成 (耗时: {format_time(sig_time)})")
            except Exception as e:
                print(f"  ⚠️  信号分析失败: {e}")
                print(f"     - 这不会影响回测，但无法评估信号质量")
                import traceback
                traceback.print_exc()
                # 不抛出异常，继续执行回测
            
            # 股票池筛选（基于验证集胜率）
            filtered_stock_pool = None
            stock_pool_filter_config = config.get('stock_pool_filter_config', {})
            enable_filter = stock_pool_filter_config.get('enable_stock_pool_filter', True)
            min_win_rate = stock_pool_filter_config.get('min_win_rate', 0.5)
            filter_segment = stock_pool_filter_config.get('filter_segment', 'test')  # 已合并valid和test
            
            if enable_filter:
                print(f"\n[筛选股票池] 基于{filter_segment}集胜率筛选股票池...")
                filter_start = time.time()
                try:
                    # 在验证集上生成预测（用于筛选）
                    print(f"  - 在{filter_segment}集上生成预测用于筛选...")
                    
                    # 临时修改dataset的segments，使predict使用验证集
                    # 保存原始的test segment
                    original_long_term_test = long_term_dataset.segments.get('test', None)
                    original_short_term_test = short_term_dataset.segments.get('test', None)
                    
                    # 将filter_segment设置为test（因为predict方法使用test segment）
                    if filter_segment in long_term_dataset.segments:
                        long_term_dataset.segments['test'] = long_term_dataset.segments[filter_segment]
                    if filter_segment in short_term_dataset.segments:
                        short_term_dataset.segments['test'] = short_term_dataset.segments[filter_segment]
                    
                    try:
                        # 生成长期预测（验证集）
                        long_term_pred_valid = long_term_model.predict(long_term_dataset)
                        # 生成短期预测（验证集）
                        short_term_pred_valid = short_term_model.predict(short_term_dataset)
                    finally:
                        # 恢复原始的test segment
                        if original_long_term_test is not None:
                            long_term_dataset.segments['test'] = original_long_term_test
                        elif 'test' in long_term_dataset.segments and filter_segment != 'test':
                            del long_term_dataset.segments['test']
                        
                        if original_short_term_test is not None:
                            short_term_dataset.segments['test'] = original_short_term_test
                        elif 'test' in short_term_dataset.segments and filter_segment != 'test':
                            del short_term_dataset.segments['test']
                    
                    # 获取标签数据
                    try:
                        valid_slice = long_term_dataset.segments[filter_segment]
                        if isinstance(valid_slice, (tuple, list)):
                            valid_selector = slice(*valid_slice)
                        else:
                            valid_selector = valid_slice
                        
                        # 内存优化：避免重复setup_data
                        if not hasattr(long_term_dataset.handler, '_learn') or long_term_dataset.handler._learn is None:
                            long_term_dataset.setup_data()
                        long_term_label_data = long_term_dataset.handler.fetch(
                            selector=valid_selector,
                            col_set=["label"],
                            data_key=long_term_dataset.handler.DK_L
                        )
                        if not hasattr(short_term_dataset.handler, '_learn') or short_term_dataset.handler._learn is None:
                            short_term_dataset.setup_data()
                        short_term_label_data = short_term_dataset.handler.fetch(
                            selector=valid_selector,
                            col_set=["label"],
                            data_key=short_term_dataset.handler.DK_L
                        )
                        
                        if isinstance(long_term_label_data, pd.DataFrame) and len(long_term_label_data.columns) > 0:
                            long_term_label_valid = long_term_label_data.iloc[:, 0]
                        else:
                            long_term_label_valid = long_term_label_data.squeeze() if hasattr(long_term_label_data, 'squeeze') else long_term_label_data
                        
                        if isinstance(short_term_label_data, pd.DataFrame) and len(short_term_label_data.columns) > 0:
                            short_term_label_valid = short_term_label_data.iloc[:, 0]
                        else:
                            short_term_label_valid = short_term_label_data.squeeze() if hasattr(short_term_label_data, 'squeeze') else short_term_label_data
                        
                        # 处理DataFrame格式的预测
                        if isinstance(long_term_pred_valid, pd.DataFrame):
                            long_term_pred_valid = long_term_pred_valid.iloc[:, 0] if len(long_term_pred_valid.columns) > 0 else long_term_pred_valid.squeeze()
                        if isinstance(short_term_pred_valid, pd.DataFrame):
                            short_term_pred_valid = short_term_pred_valid.iloc[:, 0] if len(short_term_pred_valid.columns) > 0 else short_term_pred_valid.squeeze()
                        
                        # 对齐索引
                        common_idx_long = long_term_pred_valid.index.intersection(long_term_label_valid.index)
                        common_idx_short = short_term_pred_valid.index.intersection(short_term_label_valid.index)
                        
                        if len(common_idx_long) > 0 and len(common_idx_short) > 0:
                            long_term_pred_aligned = long_term_pred_valid.loc[common_idx_long]
                            long_term_label_aligned = long_term_label_valid.loc[common_idx_long]
                            short_term_pred_aligned = short_term_pred_valid.loc[common_idx_short]
                            short_term_label_aligned = short_term_label_valid.loc[common_idx_short]
                            
                            # 计算胜率（按股票分组）
                            long_term_results = []
                            short_term_results = []
                            
                            # 长期胜率
                            if isinstance(long_term_pred_aligned.index, pd.MultiIndex):
                                for stock_id in long_term_pred_aligned.index.get_level_values(1).unique():
                                    stock_mask = long_term_pred_aligned.index.get_level_values(1) == stock_id
                                    stock_pred = long_term_pred_aligned[stock_mask]
                                    stock_label = long_term_label_aligned[stock_mask]
                                    if len(stock_pred) > 0:
                                        correct = ((stock_pred > 0) & (stock_label > 0)) | ((stock_pred < 0) & (stock_label < 0))
                                        win_rate = correct.sum() / len(correct) if len(correct) > 0 else 0.0
                                        long_term_results.append({
                                            'stock_id': stock_id,
                                            'win_rate': win_rate,
                                            'total': len(correct),
                                            'correct': correct.sum()
                                        })
                            
                            # 短期胜率
                            if isinstance(short_term_pred_aligned.index, pd.MultiIndex):
                                for stock_id in short_term_pred_aligned.index.get_level_values(1).unique():
                                    stock_mask = short_term_pred_aligned.index.get_level_values(1) == stock_id
                                    stock_pred = short_term_pred_aligned[stock_mask]
                                    stock_label = short_term_label_aligned[stock_mask]
                                    if len(stock_pred) > 0:
                                        correct = ((stock_pred > 0) & (stock_label > 0)) | ((stock_pred < 0) & (stock_label < 0))
                                        win_rate = correct.sum() / len(correct) if len(correct) > 0 else 0.0
                                        short_term_results.append({
                                            'stock_id': stock_id,
                                            'win_rate': win_rate,
                                            'total': len(correct),
                                            'correct': correct.sum()
                                        })
                            
                            # 筛选股票池
                            if len(long_term_results) > 0 and len(short_term_results) > 0:
                                filtered_stock_pool = filter_stock_pool_by_win_rate(
                                    long_term_results,
                                    short_term_results,
                                    min_win_rate=min_win_rate,
                                    segment=filter_segment
                                )
                            else:
                                print(f"  ⚠️  无法计算胜率，跳过股票池筛选")
                        else:
                            print(f"  ⚠️  预测和标签索引不匹配，跳过股票池筛选")
                    except Exception as e:
                        print(f"  ⚠️  获取{filter_segment}集标签数据失败: {e}")
                        print(f"     - 跳过股票池筛选")
                    
                    filter_time = time.time() - filter_start
                    if filtered_stock_pool is not None:
                        print(f"✓ 股票池筛选完成 (耗时: {format_time(filter_time)})")
                        print(f"  筛选后股票池大小: {len(filtered_stock_pool)} 只股票")
                except Exception as e:
                    print(f"  ⚠️  股票池筛选失败: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"     - 将使用所有股票进行回测")
            else:
                print(f"\n[跳过股票池筛选] enable_stock_pool_filter=false")
            
            # 回测分析
            print("\n[6/6] 回测分析...")
            backtest_start = time.time()
            try:
                port_analysis_config = config['port_analysis_config'].copy()
                
                # 设置策略参数
                strategy_config = port_analysis_config['strategy'].copy()
                strategy_config['kwargs']['signal'] = "<PRED>"  # 使用长期模型的预测（pred.pkl）
                strategy_config['kwargs']['short_term_signal'] = "short_term_pred.pkl"
                strategy_config['kwargs']['recorder'] = recorder
                # 传递dataset引用，用于统计
                strategy_config['kwargs']['long_term_dataset'] = long_term_dataset
                strategy_config['kwargs']['short_term_dataset'] = short_term_dataset
                # 设置统计输出目录
                stats_output_dir = Path(f"mlruns/{recorder.experiment_id}/{recorder.id}/prediction_stats")
                strategy_config['kwargs']['stats_output_dir'] = str(stats_output_dir)
                
                # 如果筛选了股票池，传递给策略
                if filtered_stock_pool is not None and len(filtered_stock_pool) > 0:
                    strategy_config['kwargs']['fixed_stock_pool'] = filtered_stock_pool
                    print(f"  - 使用筛选后的股票池（{len(filtered_stock_pool)}只股票）")
                else:
                    print(f"  - 未使用股票池筛选，使用所有股票")
                
                port_analysis_config['strategy'] = strategy_config
                
                par = PortAnaRecord(recorder, port_analysis_config, "day")
                par.generate()
                backtest_time = time.time() - backtest_start
                print(f"✓ 回测分析完成 (耗时: {format_time(backtest_time)})")
            except Exception as e:
                print(f"❌ 回测分析失败: {e}")
                raise
            
            # 保存配置
            config_to_save = config.copy()
            if 'port_analysis_config' in config_to_save:
                port_config = config_to_save['port_analysis_config'].copy()
                if 'strategy' in port_config and 'kwargs' in port_config['strategy']:
                    port_config['strategy']['kwargs'] = {
                        k: v for k, v in port_config['strategy']['kwargs'].items() 
                        if k != 'recorder'
                    }
                config_to_save['port_analysis_config'] = port_config
            recorder.save_objects(config=config_to_save)
            
            # 加载并分析回测结果
            print("\n加载回测结果...")
            port_analysis = load_port_analysis(recorder, par)
            
            if port_analysis is None:
                print("⚠️  警告: 无法加载回测结果，请检查回测是否成功完成")
            else:
                analyze_backtest_results(port_analysis, recorder=recorder)
                # 绘制回测分析图表
                plot_backtest_results(port_analysis, recorder=recorder)
            
            # 计算并打印预测胜率
            print("\n计算预测胜率...")
            try:
                calculate_prediction_win_rate(recorder, long_term_dataset, short_term_dataset)
            except Exception as e:
                print(f"  ⚠️  计算预测胜率失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 打印每日交易和持仓情况到日志文件
            print("\n生成每日交易和持仓日志...")
            try:
                log_daily_trades_and_positions(recorder)
            except Exception as e:
                print(f"  ⚠️  生成交易日志失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 计算并输出预测数据统计
            print("\n计算并输出预测数据统计...")
            try:
                calculate_and_output_prediction_statistics(
                    recorder, long_term_dataset, short_term_dataset
                )
            except Exception as e:
                print(f"  ⚠️  计算预测统计失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 总耗时统计
            total_time = time.time() - start_time
            print("\n" + "=" * 80)
            print("实验完成！")
            print(f"  - 实验ID: {recorder.experiment_id}")
            print(f"  - 记录ID: {recorder.id}")
            print(f"  - 结果路径: mlruns/{recorder.experiment_id}/{recorder.id}/")
            print(f"  - 总耗时: {format_time(total_time)}")
            print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断执行")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
