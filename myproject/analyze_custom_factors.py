"""
分析自定义因子在沪深500上的相关性（目标：未来5日平均收益）
支持自定义label表达式
"""

import os
import sys
import json
import hashlib
import re
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


def _extract_factor_info(good_factors):
    """
    从因子配置中提取表达式和性能参数，支持新旧两种格式
    
    Parameters
    ----------
    good_factors : dict
        因子配置，可能是旧格式 {factor_name: expression} 或新格式 {factor_name: {expression: ..., ic: ..., ...}}
        
    Returns
    -------
    dict
        表达式字典 {factor_name: expression}
    dict
        性能参数字典 {factor_name: {ic: ..., icir: ..., stability_score: ..., ...}}
    """
    expression_dict = {}
    metadata_dict = {}
    
    for factor_name, factor_value in good_factors.items():
        if isinstance(factor_value, str):
            # 旧格式：{factor_name: expression}
            expression_dict[factor_name] = factor_value
            metadata_dict[factor_name] = {}
        elif isinstance(factor_value, dict):
            # 新格式：{factor_name: {expression: ..., ic: ..., ...}}
            if 'expression' in factor_value:
                expression_dict[factor_name] = factor_value['expression']
                # 提取性能参数（排除expression）
                metadata = {k: v for k, v in factor_value.items() if k != 'expression'}
                metadata_dict[factor_name] = metadata
            else:
                # 格式错误，尝试作为表达式使用
                expression_dict[factor_name] = str(factor_value)
                metadata_dict[factor_name] = {}
        else:
            # 其他格式，转换为字符串
            expression_dict[factor_name] = str(factor_value)
            metadata_dict[factor_name] = {}
    
    return expression_dict, metadata_dict


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
            'label_expr': None,
            'label_horizon': None,
            'horizon_bucket': None,
            'factor_window_profile': None,
            'scoring_profile': None,
            'min_ic_threshold': 0.03,
            'min_icir_threshold': 0.5
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
    if 'label_expr' not in config:
        config['label_expr'] = None
    if 'label_horizon' not in config:
        config['label_horizon'] = None
    if 'horizon_bucket' not in config:
        config['horizon_bucket'] = None
    if 'factor_window_profile' not in config:
        config['factor_window_profile'] = None
    if 'scoring_profile' not in config:
        config['scoring_profile'] = None
    if 'min_ic_threshold' not in config or config.get('min_ic_threshold') is None:
        config['min_ic_threshold'] = 0.03
    if 'min_icir_threshold' not in config or config.get('min_icir_threshold') is None:
        config['min_icir_threshold'] = 0.5

    # 用户要求 IC/ICIR 阈值不要降低：即使配置文件里填得更低，也在加载时做下限保护
    try:
        config['min_ic_threshold'] = max(float(config.get('min_ic_threshold', 0.03)), 0.03)
    except Exception:
        config['min_ic_threshold'] = 0.03
    try:
        config['min_icir_threshold'] = max(float(config.get('min_icir_threshold', 0.5)), 0.5)
    except Exception:
        config['min_icir_threshold'] = 0.5
    
    # 确保good_factors和bad_factors是字典类型
    if not isinstance(config['good_factors'], dict):
        config['good_factors'] = {}
    if not isinstance(config['bad_factors'], dict):
        config['bad_factors'] = {}
    
    # 提取因子表达式和性能参数（用于内部处理）
    factor_expressions, factor_metadata = _extract_factor_info(config['good_factors'])
    config['_factor_expressions'] = factor_expressions
    config['_factor_metadata'] = factor_metadata
    
    return config


def save_factor_config(config, config_path=None):
    """保存因子配置文件"""
    if config_path is None:
        config_path = Path(__file__).parent / "factor_config.yaml"
    else:
        config_path = Path(config_path)
    
    # 创建副本，移除内部字段（不保存到文件）
    config_to_save = {k: v for k, v in config.items() if not k.startswith('_')}
    
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_to_save, f)
    print(f"✓ 因子配置已保存到: {config_path}")


def update_factor_config(new_good_factors=None, new_bad_factors=None, config_path=None):
    """更新因子配置文件"""
    config = load_factor_config(config_path)
    
    if new_good_factors:
        # 新因子可能已经是新格式（包含性能参数），直接更新
        config['good_factors'].update(new_good_factors)
        print(f"✓ 添加了 {len(new_good_factors)} 个好因子到配置（包含性能参数）")
    
    if new_bad_factors:
        # 坏因子保持旧格式（只有表达式）
        config['bad_factors'].update(new_bad_factors)
        print(f"✓ 添加了 {len(new_bad_factors)} 个坏因子到配置")
    
    save_factor_config(config, config_path)
    return config


def _build_experiment_signature(metadata):
    """为当前实验生成稳定签名，用于缓存和追踪。"""
    payload = json.dumps(metadata, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


def _load_cached_factor_results(cache_file):
    """读取缓存结果。"""
    if cache_file is None or not cache_file.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(cache_file)
    except Exception as e:
        print(f"⚠️  读取缓存失败，将忽略旧缓存: {e}")
        return pd.DataFrame()


def _append_factor_result_to_cache(cache_file, factor_result):
    """追加单个因子结果到缓存文件。"""
    cache_df = pd.DataFrame([factor_result])
    write_header = not cache_file.exists()
    cache_df.to_csv(cache_file, mode="a", header=write_header, index=False)


def make_future_avg_price_label_expr(window):
    """生成未来 N 日平均价格相对当前价格的目标公式。"""
    if window <= 0:
        raise ValueError("window must be positive")
    ref_list = " + ".join([f"Ref($close, -{i})" for i in range(1, window + 1)])
    return f"(({ref_list}) / {window}) / $close - 1"


def make_future_avg_return_label_expr(window):
    """生成未来 N 日相对当前价格的平均收益率目标公式。"""
    if window <= 0:
        raise ValueError("window must be positive")
    return_terms = [f"(Ref($close, -{i})/$close - 1)" for i in range(1, window + 1)]
    return f"({' + '.join(return_terms)}) / {window}"


def infer_label_horizon(label_expr):
    """从 label 表达式中推断目标周期。"""
    if not label_expr:
        return 5
    matches = [int(m) for m in re.findall(r"Ref\(\$close,\s*-(\d+)\)", label_expr)]
    if not matches:
        return 5
    return max(matches)


def build_factor_horizon_profile(label_expr):
    """根据 label 周期构建短/中/长期因子参数模板。"""
    horizon = infer_label_horizon(label_expr)
    if horizon <= 7:
        bucket = "short"
        fast_windows = [2, 3, 5]
        core_windows = [3, 5, 10]
        slow_windows = [10, 20, 30]
        long_windows = [20, 30, 60]
        ma_pairs = [(3, 8), (5, 13), (8, 21)]
    elif horizon <= 20:
        bucket = "medium"
        fast_windows = [3, 5, 10]
        core_windows = [5, 10, 20]
        slow_windows = [10, 20, 40]
        long_windows = [20, 40, 60]
        ma_pairs = [(5, 20), (10, 30), (20, 60)]
    else:
        bucket = "long"
        fast_windows = [5, 10, 20]
        core_windows = [10, 20, 40]
        slow_windows = [20, 40, 60]
        long_windows = [40, 60, 120]
        ma_pairs = [(10, 30), (20, 60), (40, 120)]

    quantile_windows = sorted(set(core_windows + slow_windows))
    profile = {
        "label_horizon": horizon,
        "horizon_bucket": bucket,
        "fast_windows": fast_windows,
        "core_windows": core_windows,
        "slow_windows": slow_windows,
        "long_windows": long_windows,
        "quantile_windows": quantile_windows,
        "ma_pairs": ma_pairs,
        "suggested_min_valid_days": max(10, horizon),
        "suggested_segment_months": 6 if horizon <= 10 else 9 if horizon <= 20 else 12,
        "suggested_rolling_valid_years": 1 if horizon <= 10 else 2,
    }
    return profile


def build_horizon_scoring_profile(horizon_profile):
    """根据目标周期返回筛选阈值和打分权重建议。"""
    bucket = horizon_profile.get("horizon_bucket", "medium")
    if bucket == "short":
        return {
            "effective_ic_scale": 0.90,
            "effective_icir_scale": 0.90,
            "promising_ic_scale": 0.75,
            "promising_icir_scale": 0.75,
            "stability_weight": 0.60,
            "tradeability_weight": 0.25,
            "novelty_weight": 0.15,
            "master_weights": {
                "ic": 0.38,
                "stability": 0.14,
                "tradeability": 0.24,
                "monotonicity": 0.14,
                "novelty": 0.10,
            },
        }
    if bucket == "long":
        return {
            "effective_ic_scale": 1.00,
            "effective_icir_scale": 1.00,
            "promising_ic_scale": 0.80,
            "promising_icir_scale": 0.80,
            "stability_weight": 0.82,
            "tradeability_weight": 0.08,
            "novelty_weight": 0.10,
            "master_weights": {
                "ic": 0.40,
                "stability": 0.24,
                "tradeability": 0.12,
                "monotonicity": 0.09,
                "novelty": 0.15,
            },
        }
    return {
        "effective_ic_scale": 1.00,
        "effective_icir_scale": 1.00,
        "promising_ic_scale": 0.75,
        "promising_icir_scale": 0.75,
        "stability_weight": 0.75,
        "tradeability_weight": 0.15,
        "novelty_weight": 0.10,
        "master_weights": {
            "ic": 0.40,
            "stability": 0.18,
            "tradeability": 0.20,
            "monotonicity": 0.12,
            "novelty": 0.10,
        },
    }


def _evaluate_tradeability(pred, label, quantile=0.2, min_group_size=5):
    """基于横截面分组的未来收益差，评估因子的可交易性。"""
    if not isinstance(pred.index, pd.MultiIndex) or not isinstance(label.index, pd.MultiIndex):
        return {
            'ls_mean': np.nan,
            'ls_win_rate': np.nan,
            'top_mean': np.nan,
            'bottom_mean': np.nan,
            'coverage_days': 0,
            'tradeability_score': 0.0,
            'direction_match': False,
            'bucket_slope_mean': np.nan,
            'bucket_corr_mean': np.nan,
            'monotonicity_win_rate': np.nan,
            'monotonicity_score': 0.0,
            'bucket_profile': None,
            'bucket_profile_coverage_days': 0,
        }

    trade_records = []
    pred_df = pred.rename("pred").to_frame()
    label_df = label.rename("label").to_frame()
    merged = pred_df.join(label_df, how="inner").dropna()
    if len(merged) == 0:
        return {
            'ls_mean': np.nan,
            'ls_win_rate': np.nan,
            'top_mean': np.nan,
            'bottom_mean': np.nan,
            'coverage_days': 0,
            'tradeability_score': 0.0,
            'direction_match': False,
            'bucket_slope_mean': np.nan,
            'bucket_corr_mean': np.nan,
            'monotonicity_win_rate': np.nan,
            'monotonicity_score': 0.0,
            'bucket_profile': None,
            'bucket_profile_coverage_days': 0,
        }

    bucket_profiles = []
    for _, daily_df in merged.groupby(level="datetime"):
        daily_df = daily_df.reset_index(drop=False)
        if len(daily_df) < max(min_group_size * 2, 10):
            continue

        daily_df = daily_df.sort_values("pred")
        group_size = max(int(len(daily_df) * quantile), min_group_size)
        if group_size * 2 > len(daily_df):
            continue

        bottom = daily_df.head(group_size)
        top = daily_df.tail(group_size)
        top_mean = top["label"].mean()
        bottom_mean = bottom["label"].mean()
        bucket_count = min(5, max(3, len(daily_df) // min_group_size))
        monotonic_slope = np.nan
        monotonic_corr = np.nan
        if bucket_count >= 3:
            daily_df["bucket"] = pd.qcut(
                daily_df["pred"].rank(method="first"),
                q=bucket_count,
                labels=False,
                duplicates="drop",
            )
            bucket_returns = daily_df.groupby("bucket")["label"].mean()
            if len(bucket_returns) >= 3:
                x = np.arange(len(bucket_returns), dtype=float)
                y = bucket_returns.values.astype(float)
                if np.nanstd(y) > 1e-12:
                    monotonic_slope = float(np.polyfit(x, y, 1)[0])
                    monotonic_corr = float(np.corrcoef(x, y)[0, 1])
        if len(daily_df) >= max(min_group_size * 5, 25):
            daily_df["profile_bucket"] = pd.qcut(
                daily_df["pred"].rank(method="first"),
                q=5,
                labels=False,
                duplicates="drop",
            )
            profile_returns = daily_df.groupby("profile_bucket")["label"].mean()
            if len(profile_returns) == 5:
                bucket_profiles.append(profile_returns.values.astype(float))
        trade_records.append({
            'top_mean': top_mean,
            'bottom_mean': bottom_mean,
            'ls_return': top_mean - bottom_mean,
            'bucket_slope': monotonic_slope,
            'bucket_corr': monotonic_corr,
        })

    if len(trade_records) == 0:
        return {
            'ls_mean': np.nan,
            'ls_win_rate': np.nan,
            'top_mean': np.nan,
            'bottom_mean': np.nan,
            'coverage_days': 0,
            'tradeability_score': 0.0,
            'direction_match': False,
            'bucket_slope_mean': np.nan,
            'bucket_corr_mean': np.nan,
            'monotonicity_win_rate': np.nan,
            'monotonicity_score': 0.0,
            'bucket_profile': None,
            'bucket_profile_coverage_days': 0,
        }

    trade_df = pd.DataFrame(trade_records)
    ls_mean = float(trade_df["ls_return"].mean())
    ls_win_rate = float((trade_df["ls_return"] > 0).mean())
    top_mean = float(trade_df["top_mean"].mean())
    bottom_mean = float(trade_df["bottom_mean"].mean())
    coverage_days = int(len(trade_df))
    direction_match = bool(ls_mean > 0)
    bucket_slope_mean = float(trade_df["bucket_slope"].mean()) if trade_df["bucket_slope"].notna().any() else np.nan
    bucket_corr_mean = float(trade_df["bucket_corr"].mean()) if trade_df["bucket_corr"].notna().any() else np.nan
    monotonicity_win_rate = float((trade_df["bucket_slope"] > 0).mean()) if trade_df["bucket_slope"].notna().any() else np.nan
    bucket_profile = None
    bucket_profile_coverage_days = len(bucket_profiles)
    if bucket_profiles:
        bucket_profile = json.dumps(np.nanmean(np.vstack(bucket_profiles), axis=0).round(6).tolist(), ensure_ascii=True)

    score = 0.0
    if ls_mean > 0:
        score += 0.4
    if ls_win_rate >= 0.55:
        score += 0.3
    elif ls_win_rate >= 0.52:
        score += 0.15
    if coverage_days >= 60:
        score += 0.2
    elif coverage_days >= 20:
        score += 0.1
    if top_mean > bottom_mean:
        score += 0.1
    monotonicity_score = 0.0
    if not np.isnan(bucket_slope_mean) and bucket_slope_mean > 0:
        monotonicity_score += 0.4
    if not np.isnan(bucket_corr_mean) and bucket_corr_mean >= 0.5:
        monotonicity_score += 0.3
    elif not np.isnan(bucket_corr_mean) and bucket_corr_mean >= 0.3:
        monotonicity_score += 0.15
    if not np.isnan(monotonicity_win_rate) and monotonicity_win_rate >= 0.55:
        monotonicity_score += 0.3
    score = 0.7 * score + 0.3 * monotonicity_score

    return {
        'ls_mean': ls_mean,
        'ls_win_rate': ls_win_rate,
        'top_mean': top_mean,
        'bottom_mean': bottom_mean,
        'coverage_days': coverage_days,
        'tradeability_score': score,
        'direction_match': direction_match,
        'bucket_slope_mean': bucket_slope_mean,
        'bucket_corr_mean': bucket_corr_mean,
        'monotonicity_win_rate': monotonicity_win_rate,
        'monotonicity_score': monotonicity_score,
        'bucket_profile': bucket_profile,
        'bucket_profile_coverage_days': bucket_profile_coverage_days,
    }


def _evaluate_novelty(candidate_factor, reference_factor_map, min_common_points=200):
    """评估候选因子相对已有好因子的差异度。"""
    if not reference_factor_map:
        return {
            'max_ref_corr': np.nan,
            'mean_ref_corr': np.nan,
            'closest_ref_factor': None,
            'novelty_score': 1.0,
        }

    corr_records = []
    for ref_name, ref_series in reference_factor_map.items():
        common_index = candidate_factor.index.intersection(ref_series.index)
        if len(common_index) < min_common_points:
            continue
        pair_df = pd.DataFrame({
            'candidate': candidate_factor.loc[common_index],
            'reference': ref_series.loc[common_index],
        }).dropna()
        if len(pair_df) < min_common_points:
            continue
        corr_val = pair_df['candidate'].corr(pair_df['reference'])
        if pd.notna(corr_val):
            corr_records.append((ref_name, abs(float(corr_val))))

    if len(corr_records) == 0:
        return {
            'max_ref_corr': np.nan,
            'mean_ref_corr': np.nan,
            'closest_ref_factor': None,
            'novelty_score': 1.0,
        }

    corr_records.sort(key=lambda x: x[1], reverse=True)
    max_ref_name, max_ref_corr = corr_records[0]
    mean_ref_corr = float(np.mean([x[1] for x in corr_records]))
    novelty_score = max(0.0, 1.0 - max_ref_corr)
    return {
        'max_ref_corr': max_ref_corr,
        'mean_ref_corr': mean_ref_corr,
        'closest_ref_factor': max_ref_name,
        'novelty_score': novelty_score,
    }


def _build_rolling_windows(start_time, end_time, train_years=5, valid_years=1, step_months=12):
    """构建滚动训练/验证时间窗口。"""
    start_ts = pd.Timestamp(start_time)
    end_ts = pd.Timestamp(end_time)
    
    if end_ts <= start_ts:
        return []
    
    windows = []
    cursor = start_ts
    train_offset = pd.DateOffset(years=train_years)
    valid_offset = pd.DateOffset(years=valid_years)
    step_offset = pd.DateOffset(months=step_months)
    
    while True:
        train_start = cursor
        train_end = cursor + train_offset - pd.Timedelta(days=1)
        valid_start = train_end + pd.Timedelta(days=1)
        valid_end = valid_start + valid_offset - pd.Timedelta(days=1)
        
        if valid_end > end_ts:
            break
        
        windows.append((train_start, train_end, valid_start, valid_end))
        cursor = cursor + step_offset
        
        if cursor > end_ts:
            break
    
    return windows


def _evaluate_rolling_stability(
    ic_series,
    min_valid_days,
    min_ic_train,
    min_ic_valid,
    max_ic_decay,
    require_same_sign,
    rolling_windows,
    min_rolling_windows=3,
):
    """基于滚动窗口评估IC稳定性。"""
    if isinstance(ic_series.index, pd.MultiIndex):
        ic_dates = pd.to_datetime(ic_series.index.get_level_values('datetime'))
    else:
        ic_dates = pd.to_datetime(ic_series.index)
    
    if len(rolling_windows) == 0:
        return {
            'rolling_window_count': 0,
            'rolling_valid_windows': 0,
            'rolling_stable_windows': 0,
            'rolling_stable_ratio': 0.0,
            'rolling_same_sign_ratio': 0.0,
            'rolling_ic_train_mean': np.nan,
            'rolling_ic_valid_mean': np.nan,
            'rolling_ic_decay_mean': np.nan,
            'rolling_ic_valid_abs_mean': np.nan,
            'rolling_ic_valid_abs_min': np.nan,
            'rolling_stability_score': 0.0,
            'rolling_is_stable': False,
            'rolling_reason': 'no_rolling_windows',
        }
    
    window_stats = []
    for train_start, train_end, valid_start, valid_end in rolling_windows:
        train_mask = (ic_dates >= train_start) & (ic_dates <= train_end)
        valid_mask = (ic_dates >= valid_start) & (ic_dates <= valid_end)
        train_ic = ic_series[train_mask]
        valid_ic = ic_series[valid_mask]
        
        if len(train_ic) < min_valid_days or len(valid_ic) < min_valid_days:
            continue
        
        mean_train = train_ic.mean()
        mean_valid = valid_ic.mean()
        
        if not np.isnan(mean_train) and abs(mean_train) > 1e-10:
            ic_decay = (mean_train - mean_valid) / abs(mean_train)
        else:
            ic_decay = np.nan
        
        same_sign = (mean_train * mean_valid >= 0) if (not np.isnan(mean_train) and not np.isnan(mean_valid)) else False
        
        cond_train = (not np.isnan(mean_train)) and (abs(mean_train) >= min_ic_train)
        cond_valid = (not np.isnan(mean_valid)) and (abs(mean_valid) >= min_ic_valid)
        cond_decay = (not np.isnan(ic_decay)) and (abs(ic_decay) <= max_ic_decay)
        cond_sign = (same_sign if require_same_sign else True)
        
        window_stats.append({
            'mean_train': mean_train,
            'mean_valid': mean_valid,
            'ic_decay': ic_decay,
            'same_sign': same_sign,
            'is_stable': bool(cond_train and cond_valid and cond_decay and cond_sign),
        })
    
    if len(window_stats) == 0:
        return {
            'rolling_window_count': len(rolling_windows),
            'rolling_valid_windows': 0,
            'rolling_stable_windows': 0,
            'rolling_stable_ratio': 0.0,
            'rolling_same_sign_ratio': 0.0,
            'rolling_ic_train_mean': np.nan,
            'rolling_ic_valid_mean': np.nan,
            'rolling_ic_decay_mean': np.nan,
            'rolling_ic_valid_abs_mean': np.nan,
            'rolling_ic_valid_abs_min': np.nan,
            'rolling_stability_score': 0.0,
            'rolling_is_stable': False,
            'rolling_reason': 'insufficient_rolling_data',
        }
    
    train_vals = np.array([w['mean_train'] for w in window_stats], dtype=float)
    valid_vals = np.array([w['mean_valid'] for w in window_stats], dtype=float)
    decay_vals = np.array([w['ic_decay'] for w in window_stats], dtype=float)
    same_sign_vals = np.array([1.0 if w['same_sign'] else 0.0 for w in window_stats], dtype=float)
    stable_vals = np.array([1.0 if w['is_stable'] else 0.0 for w in window_stats], dtype=float)
    
    rolling_valid_windows = len(window_stats)
    rolling_stable_windows = int(stable_vals.sum())
    rolling_stable_ratio = float(stable_vals.mean()) if rolling_valid_windows > 0 else 0.0
    rolling_same_sign_ratio = float(same_sign_vals.mean()) if rolling_valid_windows > 0 else 0.0
    rolling_ic_train_mean = float(np.nanmean(train_vals))
    rolling_ic_valid_mean = float(np.nanmean(valid_vals))
    rolling_ic_decay_mean = float(np.nanmean(decay_vals))
    rolling_ic_valid_abs_mean = float(np.nanmean(np.abs(valid_vals)))
    rolling_ic_valid_abs_min = float(np.nanmin(np.abs(valid_vals)))
    
    # 滚动稳定性评分（0~1）
    rolling_score = 0.0
    if rolling_stable_ratio >= 0.7:
        rolling_score += 0.5
    elif rolling_stable_ratio >= 0.5:
        rolling_score += 0.3
    if rolling_same_sign_ratio >= 0.7:
        rolling_score += 0.2
    if rolling_ic_valid_abs_mean >= min_ic_valid:
        rolling_score += 0.2
    if rolling_ic_valid_abs_min >= min_ic_valid * 0.7:
        rolling_score += 0.1
    
    rolling_is_stable = (
        (rolling_valid_windows >= min_rolling_windows) and
        (rolling_stable_ratio >= 0.5) and
        (rolling_same_sign_ratio >= (0.7 if require_same_sign else 0.5))
    )
    
    reasons = []
    if rolling_valid_windows < min_rolling_windows:
        reasons.append(f"rolling_windows不足({rolling_valid_windows}<{min_rolling_windows})")
    if rolling_stable_ratio < 0.5:
        reasons.append(f"rolling稳定比例低({rolling_stable_ratio:.2f}<0.50)")
    if require_same_sign and rolling_same_sign_ratio < 0.7:
        reasons.append(f"rolling同号比例低({rolling_same_sign_ratio:.2f}<0.70)")
    
    return {
        'rolling_window_count': len(rolling_windows),
        'rolling_valid_windows': rolling_valid_windows,
        'rolling_stable_windows': rolling_stable_windows,
        'rolling_stable_ratio': rolling_stable_ratio,
        'rolling_same_sign_ratio': rolling_same_sign_ratio,
        'rolling_ic_train_mean': rolling_ic_train_mean,
        'rolling_ic_valid_mean': rolling_ic_valid_mean,
        'rolling_ic_decay_mean': rolling_ic_decay_mean,
        'rolling_ic_valid_abs_mean': rolling_ic_valid_abs_mean,
        'rolling_ic_valid_abs_min': rolling_ic_valid_abs_min,
        'rolling_stability_score': rolling_score,
        'rolling_is_stable': rolling_is_stable,
        'rolling_reason': '; '.join(reasons) if reasons else 'stable',
    }


def _evaluate_segment_stability(
    ic_series,
    min_valid_days,
    segment_months=6,
    min_segment_count=4,
    min_ic_abs=0.01,
    require_same_sign=True,
):
    """基于固定时间段切分评估IC稳定性（弱化单一验证期过短的问题）。"""
    if ic_series is None or len(ic_series) == 0:
        return {
            'segment_count': 0,
            'segment_valid_segments': 0,
            'segment_stable_segments': 0,
            'segment_stable_ratio': 0.0,
            'segment_same_sign_ratio': 0.0,
            'segment_ic_abs_mean': np.nan,
            'segment_ic_abs_min': np.nan,
            'segment_stability_score': 0.0,
            'segment_is_stable': False,
            'segment_reason': 'no_ic_series',
        }

    if isinstance(ic_series.index, pd.MultiIndex):
        ic_dates = pd.to_datetime(ic_series.index.get_level_values('datetime'))
    else:
        ic_dates = pd.to_datetime(ic_series.index)

    if len(ic_dates) == 0:
        return {
            'segment_count': 0,
            'segment_valid_segments': 0,
            'segment_stable_segments': 0,
            'segment_stable_ratio': 0.0,
            'segment_same_sign_ratio': 0.0,
            'segment_ic_abs_mean': np.nan,
            'segment_ic_abs_min': np.nan,
            'segment_stability_score': 0.0,
            'segment_is_stable': False,
            'segment_reason': 'no_ic_dates',
        }

    start_ts = ic_dates.min()
    end_ts = ic_dates.max()
    if pd.isna(start_ts) or pd.isna(end_ts) or end_ts <= start_ts:
        return {
            'segment_count': 0,
            'segment_valid_segments': 0,
            'segment_stable_segments': 0,
            'segment_stable_ratio': 0.0,
            'segment_same_sign_ratio': 0.0,
            'segment_ic_abs_mean': np.nan,
            'segment_ic_abs_min': np.nan,
            'segment_stability_score': 0.0,
            'segment_is_stable': False,
            'segment_reason': 'invalid_dates',
        }

    segments = []
    cursor = start_ts
    step = pd.DateOffset(months=segment_months)
    while cursor < end_ts:
        seg_start = cursor
        seg_end = min(cursor + step - pd.Timedelta(days=1), end_ts)
        segments.append((seg_start, seg_end))
        cursor = cursor + step

    if len(segments) == 0:
        return {
            'segment_count': 0,
            'segment_valid_segments': 0,
            'segment_stable_segments': 0,
            'segment_stable_ratio': 0.0,
            'segment_same_sign_ratio': 0.0,
            'segment_ic_abs_mean': np.nan,
            'segment_ic_abs_min': np.nan,
            'segment_stability_score': 0.0,
            'segment_is_stable': False,
            'segment_reason': 'no_segments',
        }

    overall_mean = ic_series.mean() if len(ic_series) > 0 else np.nan
    segment_stats = []
    for seg_start, seg_end in segments:
        seg_mask = (ic_dates >= seg_start) & (ic_dates <= seg_end)
        seg_ic = ic_series[seg_mask]
        if len(seg_ic) < min_valid_days:
            continue
        seg_mean = seg_ic.mean()
        seg_abs = abs(seg_mean) if not np.isnan(seg_mean) else np.nan
        seg_same_sign = False
        if not np.isnan(seg_mean) and not np.isnan(overall_mean):
            seg_same_sign = (seg_mean * overall_mean >= 0)
        seg_stable = (not np.isnan(seg_abs)) and (seg_abs >= min_ic_abs)
        segment_stats.append({
            'mean': seg_mean,
            'abs': seg_abs,
            'same_sign': seg_same_sign,
            'is_stable': seg_stable,
        })

    if len(segment_stats) == 0:
        return {
            'segment_count': len(segments),
            'segment_valid_segments': 0,
            'segment_stable_segments': 0,
            'segment_stable_ratio': 0.0,
            'segment_same_sign_ratio': 0.0,
            'segment_ic_abs_mean': np.nan,
            'segment_ic_abs_min': np.nan,
            'segment_stability_score': 0.0,
            'segment_is_stable': False,
            'segment_reason': 'insufficient_segment_data',
        }

    abs_vals = np.array([s['abs'] for s in segment_stats], dtype=float)
    same_sign_vals = np.array([1.0 if s['same_sign'] else 0.0 for s in segment_stats], dtype=float)
    stable_vals = np.array([1.0 if s['is_stable'] else 0.0 for s in segment_stats], dtype=float)

    valid_segments = len(segment_stats)
    stable_segments = int(stable_vals.sum())
    stable_ratio = float(stable_vals.mean()) if valid_segments > 0 else 0.0
    same_sign_ratio = float(same_sign_vals.mean()) if valid_segments > 0 else 0.0
    abs_mean = float(np.nanmean(abs_vals))
    abs_min = float(np.nanmin(abs_vals))

    score = 0.0
    if stable_ratio >= 0.7:
        score += 0.5
    elif stable_ratio >= 0.5:
        score += 0.3
    if same_sign_ratio >= 0.7:
        score += 0.2
    if abs_mean >= min_ic_abs:
        score += 0.2
    if abs_min >= min_ic_abs * 0.7:
        score += 0.1

    is_stable = (
        (valid_segments >= min_segment_count) and
        (stable_ratio >= 0.5) and
        (same_sign_ratio >= (0.7 if require_same_sign else 0.5))
    )

    reasons = []
    if valid_segments < min_segment_count:
        reasons.append(f"segment不足({valid_segments}<{min_segment_count})")
    if stable_ratio < 0.5:
        reasons.append(f"segment稳定比例低({stable_ratio:.2f}<0.50)")
    if require_same_sign and same_sign_ratio < 0.7:
        reasons.append(f"segment同号比例低({same_sign_ratio:.2f}<0.70)")

    return {
        'segment_count': len(segments),
        'segment_valid_segments': valid_segments,
        'segment_stable_segments': stable_segments,
        'segment_stable_ratio': stable_ratio,
        'segment_same_sign_ratio': same_sign_ratio,
        'segment_ic_abs_mean': abs_mean,
        'segment_ic_abs_min': abs_min,
        'segment_stability_score': score,
        'segment_is_stable': is_stable,
        'segment_reason': '; '.join(reasons) if reasons else 'stable',
    }


def deduplicate_good_factors_by_correlation(
    good_factors,
    instruments_list,
    start_time="2015-01-01",
    end_time="2025-12-31",
    freq="day",
    correlation_threshold=0.85,
    ic_results=None,
    factor_config=None
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
    # 提取因子表达式（支持新旧格式）
    factor_expressions, _ = _extract_factor_info(good_factors)
    
    if len(factor_expressions) <= 1:
        # 返回表达式字典（向后兼容）
        return factor_expressions, {}
    
    print(f"\n开始分析好因子相关性（阈值: {correlation_threshold}）...")
    print(f"原始好因子数量: {len(factor_expressions)}")
    
    # 初始化评分和IC映射
    factor_score_map = {}  # 综合评分字典
    factor_ic_map = {}  # IC值字典
    
    # 如果没有提供IC结果，尝试从配置文件读取性能参数
    if ic_results is None:
        config_metadata = {}
        if factor_config is not None:
            # 首先尝试从_factor_metadata读取（这是load_factor_config保存的）
            config_metadata = factor_config.get('_factor_metadata', {})
            if not config_metadata:
                # 如果_factor_metadata为空，尝试从good_factors中提取
                _, config_metadata = _extract_factor_info(factor_config.get('good_factors', {}))
        
        print("从配置文件读取因子性能参数...")
        for factor_name in factor_expressions.keys():
            if factor_name in config_metadata and config_metadata[factor_name]:
                metadata = config_metadata[factor_name]
                ic_abs = abs(metadata.get('ic', 0.0))
                icir_abs = abs(metadata.get('icir', 0.0))
                stability_score = metadata.get('stability_score', 0.0)
                if ic_abs > 0 or icir_abs > 0 or stability_score > 0:
                    # 计算综合评分
                    score = ic_abs * icir_abs + stability_score * 0.1
                    factor_score_map[factor_name] = score
                    factor_ic_map[factor_name] = ic_abs
                    print(f"  ✓ {factor_name}: IC={ic_abs:.6f}, ICIR={icir_abs:.6f}, 稳定性={stability_score:.4f}, 评分={score:.6f}")
        
        # 如果从配置文件读取到性能参数，使用它们；否则重新计算IC
        if len(factor_score_map) > 0:
            print(f"从配置文件读取到 {len(factor_score_map)} 个因子的性能参数，使用这些参数进行去重")
            # 从配置文件构建factor_ic_map用于显示
            factor_ic_map = {}
            for factor_name in factor_expressions.keys():
                if factor_name in config_metadata and config_metadata[factor_name]:
                    factor_ic_map[factor_name] = abs(config_metadata[factor_name].get('ic', 0.0))
                else:
                    factor_ic_map[factor_name] = 0.0
        else:
            print("正在计算因子IC值用于比较（配置文件无性能参数）...")
            factor_ic_map = {}
            factor_names = list(factor_expressions.keys())
            
            # 批量获取因子数据
            try:
                # 支持字典格式和列表格式（方法1：固定股票池）
                all_factor_data = D.features(
                    instruments_list,  # 可以是字典或列表
                    list(factor_expressions.values()),
                    start_time=start_time,
                    end_time=end_time,
                    freq=freq
                )
                
                # 获取标签数据用于计算IC
                # 确保instruments_list是列表格式
                if isinstance(instruments_list, dict):
                    print("⚠️  instruments_list是字典格式，无法计算IC，跳过IC计算")
                    factor_ic_map = {name: 0.03 for name in factor_names}
                else:
                    label_data = D.features(
                        instruments_list,  # 必须是列表
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
        # 使用提供的IC结果，计算综合评分
        factor_score_map = {}  # 综合评分：IC * ICIR + 稳定性得分
        factor_ic_map = {}  # IC值（用于显示）
        for _, row in ic_results.iterrows():
            factor_name = row['factor']
            if factor_name in factor_expressions:
                ic_abs = abs(row.get('IC', 0.0))
                icir_abs = abs(row.get('ICIR', 0.0))
                stability_score = row.get('stability_score', 0.0)
                # 综合评分：IC绝对值 * ICIR绝对值 + 稳定性得分
                # 权重可以根据需要调整，这里IC和ICIR占主要权重
                score = ic_abs * icir_abs + stability_score * 0.1
                factor_score_map[factor_name] = score
                factor_ic_map[factor_name] = ic_abs
    
    # 获取因子数据计算相关性
    print("正在计算因子相关性矩阵...")
    factor_names = list(factor_expressions.keys())
    
    # 构建因子值矩阵（按日期和股票）
    factor_matrix_dict = {}
    
    try:
        # 批量获取所有因子数据（方法1：固定股票池）
        all_factor_data = D.features(
            instruments_list,
            list(factor_expressions.values()),
            start_time=start_time,
            end_time=end_time,
            freq=freq
        )
        
        if len(all_factor_data) == 0:
            print("⚠️  无法获取因子数据，跳过相关性分析")
            return factor_expressions, {}
        
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
        return factor_expressions, {}
    
    # 计算相关性矩阵
    if len(factor_matrix_dict) < 2:
        print("因子数据不足，跳过相关性分析")
        return factor_expressions, {}
    
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
        return factor_expressions, {}
    
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
        return factor_expressions, {}
    
    # 找出高相关性的因子对
    factors_to_remove = set()
    factor_pairs_checked = set()
    
    print(f"\n发现高相关性因子对（相关性 > {correlation_threshold}）:")
    for i, factor1 in enumerate(correlation_matrix.index):
        for j, factor2 in enumerate(correlation_matrix.columns):
            if i >= j:  # 只检查上三角矩阵
                continue
            
            if factor1 not in factor_expressions or factor2 not in factor_expressions:
                continue
            
            pair_key = tuple(sorted([factor1, factor2]))
            if pair_key in factor_pairs_checked:
                continue
            factor_pairs_checked.add(pair_key)
            
            corr_value = correlation_matrix.loc[factor1, factor2]
            if abs(corr_value) > correlation_threshold:
                # 使用综合评分来选择最优因子（IC、ICIR、稳定性得分）
                # 优先使用综合评分，如果没有则使用IC
                if ic_results is not None and len(factor_score_map) > 0:
                    score1 = factor_score_map.get(factor1, 0.0)
                    score2 = factor_score_map.get(factor2, 0.0)
                    ic1 = factor_ic_map.get(factor1, 0.0)
                    ic2 = factor_ic_map.get(factor2, 0.0)
                    
                    if score1 > score2:
                        factors_to_remove.add(factor2)
                        print(f"  {factor1:40s} (评分={score1:.6f}, IC={ic1:.6f}) vs {factor2:40s} (评分={score2:.6f}, IC={ic2:.6f}) | 相关性: {corr_value:.4f} -> 移除 {factor2}（保留综合评分更高的）")
                    elif score2 > score1:
                        factors_to_remove.add(factor1)
                        print(f"  {factor1:40s} (评分={score1:.6f}, IC={ic1:.6f}) vs {factor2:40s} (评分={score2:.6f}, IC={ic2:.6f}) | 相关性: {corr_value:.4f} -> 移除 {factor1}（保留综合评分更高的）")
                    else:
                        # 评分相同，保留IC更大的
                        if ic1 > ic2:
                            factors_to_remove.add(factor2)
                            print(f"  {factor1:40s} (评分={score1:.6f}, IC={ic1:.6f}) vs {factor2:40s} (评分={score2:.6f}, IC={ic2:.6f}) | 相关性: {corr_value:.4f} | 评分相同 -> 移除 {factor2}（保留IC更大的）")
                        elif ic2 > ic1:
                            factors_to_remove.add(factor1)
                            print(f"  {factor1:40s} (评分={score1:.6f}, IC={ic1:.6f}) vs {factor2:40s} (评分={score2:.6f}, IC={ic2:.6f}) | 相关性: {corr_value:.4f} | 评分相同 -> 移除 {factor1}（保留IC更大的）")
                        else:
                            # IC也相同，保留名称更短的（通常更基础）
                            if len(factor1) <= len(factor2):
                                factors_to_remove.add(factor2)
                                print(f"  {factor1:40s} vs {factor2:40s} | 相关性: {corr_value:.4f} | 评分和IC都相同 -> 移除 {factor2}（保留名称更短的）")
                            else:
                                factors_to_remove.add(factor1)
                                print(f"  {factor1:40s} vs {factor2:40s} | 相关性: {corr_value:.4f} | 评分和IC都相同 -> 移除 {factor1}（保留名称更短的）")
                else:
                    # 没有IC结果，使用IC值比较
                    ic1_abs = factor_ic_map.get(factor1, 0.0)  # IC绝对值
                    ic2_abs = factor_ic_map.get(factor2, 0.0)  # IC绝对值
                    
                    if ic1_abs > ic2_abs:
                        factors_to_remove.add(factor2)
                        print(f"  {factor1:40s} (|IC|={ic1_abs:.6f}) vs {factor2:40s} (|IC|={ic2_abs:.6f}) | 相关性: {corr_value:.4f} -> 移除 {factor2}（保留IC更大的）")
                    elif ic2_abs > ic1_abs:
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
    
    # 创建去重后的因子字典（使用factor_expressions，确保是表达式字符串）
    deduplicated_factors = {
        name: expr for name, expr in factor_expressions.items()
        if name not in factors_to_remove
    }
    
    # 创建被去除的因子字典（用于加入到bad_factors，使用表达式字符串）
    removed_factors = {
        name: expr for name, expr in factor_expressions.items()
        if name in factors_to_remove
    }
    
    removed_count = len(factor_expressions) - len(deduplicated_factors)
    print(f"\n✓ 相关性去重完成:")
    print(f"  - 移除因子数: {removed_count}")
    print(f"  - 保留因子数: {len(deduplicated_factors)}")
    
    return deduplicated_factors, removed_factors


def generate_new_factors_from_good_ones(
    good_factors,
    bad_factors,
    max_new_factors=200,
    ignore_tried_factors=False,
    horizon_profile=None,
):
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
    if max_new_factors is None or max_new_factors <= 0:
        max_new_factors = 10**9  # 视为无限制，确保一次生成全部候选因子
    tried_count = len(good_factors) + len(bad_factors)
    if ignore_tried_factors:
        tried_factors = new_factors  # 动态视图，避免覆盖已生成因子
        print(f"  已尝试因子数(历史): {tried_count}（生成时忽略历史过滤）")
    else:
        tried_factors = {**good_factors, **bad_factors}  # 所有已尝试的因子
    print(f"  已尝试因子数: {len(tried_factors)}")
    print(f"  好因子数: {len(good_factors)}")
    print(f"  坏因子数: {len(bad_factors)}")
    if horizon_profile is None:
        horizon_profile = build_factor_horizon_profile(None)
    fast_windows = horizon_profile.get("fast_windows", [3, 5, 10])
    core_windows = horizon_profile.get("core_windows", [5, 10, 20])
    slow_windows = horizon_profile.get("slow_windows", [10, 20, 40])
    long_windows = horizon_profile.get("long_windows", [20, 40, 60])
    quantile_windows = horizon_profile.get("quantile_windows", sorted(set(core_windows + slow_windows)))
    ma_pairs = horizon_profile.get("ma_pairs", [(5, 20), (10, 30), (20, 60)])
    print(
        f"  目标周期模板: {horizon_profile.get('horizon_bucket', 'medium')} "
        f"(label={horizon_profile.get('label_horizon', 5)}日)"
    )
    
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
            for ma_period in long_windows:
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
    market_return_windows = sorted(set([1] + fast_windows + core_windows))
    market_sync_windows = core_windows
    market_beta_windows = sorted(set(slow_windows))
    market_ma_windows = core_windows
    market_trend_pairs = [(fast_w, slow_w) for fast_w, slow_w in ma_pairs if fast_w < slow_w]
    market_breakout_pairs = [(window, window) for window in core_windows]
    market_vol_quantile_pairs = list(zip(core_windows, slow_windows))
    market_reversal_pairs = [
        (ma_window, max(1, fast_windows[min(idx, len(fast_windows) - 1)]))
        for idx, ma_window in enumerate(core_windows)
    ]
    kline_windows = sorted(set(fast_windows + core_windows[:1]))
    short_kline_windows = fast_windows[:2]
    vwap_windows = sorted(set(fast_windows + core_windows[:1]))
    amount_quantile_windows = sorted(set(quantile_windows))
    breakout_windows = core_windows
    momentum_pairs = [(fast_w, slow_w) for fast_w, slow_w in ma_pairs if fast_w < slow_w]
    
    for market_name, market_code in market_indices:
        for window in market_return_windows:
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
        for window in market_sync_windows:
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
        for window in market_beta_windows:
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
        for window in sorted(set(fast_windows + core_windows)):
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
        for window in sorted(set(fast_windows + core_windows[:2])):
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
        for window in market_sync_windows:
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
        for ma_window in market_ma_windows:
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
        for window in market_sync_windows:
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
    # 注意：由于 ChangeInstrument 可能导致形状不匹配问题，改用更安全的表达式
    for market_name, market_code in market_indices:
        for window in market_sync_windows:
            # 个股创新高但市场未创新高（相对强势）
            # 先获取市场指数收盘价，然后分别计算突破幅度
            new_name = f"Stock_Market_Breakout_{market_name}_{window}"
            if new_name not in tried_factors:
                # 获取市场指数收盘价
                market_close = f"ChangeInstrument('{market_code}', $close)"
                # 个股突破幅度
                stock_breakout = f"($close - Max($close, {window})) / (Max($close, {window}) + 0.0001)"
                # 市场突破幅度（对市场指数收盘价计算 Max）
                market_max = f"Max({market_close}, {window})"
                market_breakout = f"({market_close} - {market_max}) / ({market_max} + 0.0001)"
                # 个股创新高条件
                stock_new_high = f"$close > Max($close, {window})"
                # 如果个股创新高，返回个股突破幅度减去市场突破幅度；否则返回0
                # 这样个股突破幅度大于市场突破幅度时，因子值为正（相对强势）
                new_expr = f"If({stock_new_high}, {stock_breakout} - {market_breakout}, 0)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略115: 个股与市场指数的背离（价格背离）
    for market_name, market_code in market_indices:
        for window in market_sync_windows:
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
        for window in fast_windows[:2]:
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
        for window in market_sync_windows:
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
        for window in market_sync_windows:
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
        for ma_window in fast_windows[:2]:
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
        for window in market_sync_windows:
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
    
    # 策略121: 个股与市场指数的多周期趋势一致性（借鉴长期策略的一致性思路）
    for market_name, market_code in market_indices:
        for short_period, long_period in market_trend_pairs:
            new_name = f"Stock_Market_Trend_Consistency_{market_name}_{short_period}_{long_period}"
            if new_name not in tried_factors:
                stock_short = f"($close - Ref($close, {short_period})) / Ref($close, {short_period})"
                stock_long = f"($close - Ref($close, {long_period})) / Ref($close, {long_period})"
                market_short = f"ChangeInstrument('{market_code}', ($close - Ref($close, {short_period})) / Ref($close, {short_period}))"
                market_long = f"ChangeInstrument('{market_code}', ($close - Ref($close, {long_period})) / Ref($close, {long_period}))"
                stock_consistency = f"Sign({stock_short}) * Sign({stock_long})"
                market_consistency = f"Sign({market_short}) * Sign({market_long})"
                new_expr = f"({stock_consistency} - {market_consistency}) * ({stock_long} - {market_long})"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略122: 个股相对市场动量的加速度（借鉴长期策略的趋势加速度）
    for market_name, market_code in market_indices:
        for window in market_sync_windows:
            new_name = f"Stock_Market_Momentum_Acceleration_{market_name}_{window}"
            if new_name not in tried_factors:
                stock_mom = f"($close - Ref($close, {window})) / Ref($close, {window})"
                market_mom = f"ChangeInstrument('{market_code}', ($close - Ref($close, {window})) / Ref($close, {window}))"
                rel_mom = f"{stock_mom} - {market_mom}"
                rel_mom_prev = f"Ref({rel_mom}, {max(1, window // 2)})"
                new_expr = f"({rel_mom} - {rel_mom_prev}) / (Abs({rel_mom_prev}) + 0.0001)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略123: 个股相对市场突破强度的标准化（借鉴长期策略突破强度归一化）
    for market_name, market_code in market_indices:
        for breakout_window, vol_window in market_breakout_pairs:
            new_name = f"Stock_Market_Breakout_Strength_Norm_{market_name}_{breakout_window}_{vol_window}"
            if new_name not in tried_factors:
                stock_breakout = f"($close - Max($close, {breakout_window})) / (Max($close, {breakout_window}) + 0.0001)"
                market_close = f"ChangeInstrument('{market_code}', $close)"
                market_breakout = f"({market_close} - Max({market_close}, {breakout_window})) / (Max({market_close}, {breakout_window}) + 0.0001)"
                stock_vol = f"Std(($close - Ref($close, 1)) / Ref($close, 1), {vol_window})"
                market_vol = f"Std(ChangeInstrument('{market_code}', ($close - Ref($close, 1)) / Ref($close, 1)), {vol_window})"
                rel_breakout = f"{stock_breakout} - {market_breakout}"
                new_expr = f"{rel_breakout} / ({stock_vol} + {market_vol} + 0.0001)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略124: 市场波动分位数下的个股超额收益（借鉴长期策略波动分位位置）
    for market_name, market_code in market_indices:
        for vol_window, quantile_window in market_vol_quantile_pairs:
            new_name = f"Stock_Market_Volatility_Quantile_Excess_{market_name}_{vol_window}_{quantile_window}"
            if new_name not in tried_factors:
                market_ret = f"ChangeInstrument('{market_code}', ($close - Ref($close, 1)) / Ref($close, 1))"
                market_vol = f"Std({market_ret}, {vol_window})"
                vol_q50 = f"Quantile({market_vol}, {quantile_window}, 0.5)"
                vol_q75 = f"Quantile({market_vol}, {quantile_window}, 0.75)"
                vol_q25 = f"Quantile({market_vol}, {quantile_window}, 0.25)"
                vol_position = f"({market_vol} - {vol_q50}) / ({vol_q75} - {vol_q25} + 0.0001)"
                stock_ret = f"($close - Ref($close, 1)) / Ref($close, 1)"
                excess_ret = f"{stock_ret} - {market_ret}"
                new_expr = f"{vol_position} * {excess_ret}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略125: 个股相对市场MA位置的反转信号（借鉴长期策略反转信号）
    for market_name, market_code in market_indices:
        for ma_window, reversal_window in market_reversal_pairs:
            new_name = f"Stock_Market_MA_Reversal_{market_name}_{ma_window}_{reversal_window}"
            if new_name not in tried_factors:
                stock_ratio = f"$close / Mean($close, {ma_window})"
                market_ratio = f"ChangeInstrument('{market_code}', $close / Mean($close, {ma_window}))"
                rel_ratio = f"{stock_ratio} - {market_ratio}"
                prev_rel_ratio = f"Ref({rel_ratio}, {reversal_window})"
                reversal = f"(1 - Sign({prev_rel_ratio}) * Sign({rel_ratio})) / 2"
                new_expr = f"{reversal} * ({rel_ratio} - {prev_rel_ratio})"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors

    # === 进一步发散：价量与K线结构因子（126-135）===
    # 注意：以下部分使用 $open/$high/$low/$volume 字段，需要数据中包含这些字段

    # 策略126: 日内振幅强度（区间波动均值）
    for window in kline_windows:
        new_name = f"Intraday_Range_{window}"
        if new_name not in tried_factors:
            intraday_range = f"($high - $low) / ($close + 0.0001)"
            new_expr = f"Mean({intraday_range}, {window})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略127: K线实体强度（实体占比）
    for window in kline_windows:
        new_name = f"Candle_Body_Strength_{window}"
        if new_name not in tried_factors:
            body = f"($close - $open) / ($high - $low + 0.0001)"
            new_expr = f"Mean({body}, {window})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略128: 跳空缺口与当日收益联动
    for window in short_kline_windows:
        new_name = f"Gap_Return_Interaction_{window}"
        if new_name not in tried_factors:
            gap = f"($open - Ref($close, 1)) / Ref($close, 1)"
            daily_ret = f"($close - Ref($close, 1)) / Ref($close, 1)"
            new_expr = f"Mean({gap} * {daily_ret}, {window})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略129: 收盘在日内区间的位置（CLV简化）
    for window in kline_windows:
        new_name = f"Close_Location_Value_{window}"
        if new_name not in tried_factors:
            clv = f"(2 * $close - $high - $low) / ($high - $low + 0.0001)"
            new_expr = f"Mean({clv}, {window})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略130: 量能放大与方向一致性
    for window in kline_windows:
        new_name = f"Volume_Direction_Confirm_{window}"
        if new_name not in tried_factors:
            vol_spike = f"$volume / Mean($volume, {window}) - 1"
            daily_ret = f"($close - Ref($close, 1)) / Ref($close, 1)"
            new_expr = f"{vol_spike} * Sign({daily_ret})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略131: 量能趋势与短期动量耦合
    for vol_short, vol_long in momentum_pairs:
        new_name = f"Volume_Trend_Momentum_{vol_short}_{vol_long}"
        if new_name not in tried_factors:
            vol_trend = f"Mean($volume, {vol_short}) / Mean($volume, {vol_long}) - 1"
            momentum = f"($close - Ref($close, {vol_long // 2})) / Ref($close, {vol_long // 2})"
            new_expr = f"{vol_trend} * {momentum}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略132: 价量协同（收益率与量能变化相关）
    for window in core_windows:
        new_name = f"Price_Volume_Corr_{window}"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            volume_change = f"($volume - Ref($volume, 1)) / Ref($volume, 1)"
            new_expr = f"Corr({price_change}, {volume_change}, {window})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略133: 量能加权动量
    for window in core_windows[:2]:
        new_name = f"Volume_Weighted_Momentum_{window}"
        if new_name not in tried_factors:
            momentum = f"($close - Ref($close, {window})) / Ref($close, {window})"
            vol_norm = f"Mean($volume, {window}) / (Mean($volume, {window * 2}) + 0.0001)"
            new_expr = f"{momentum} * {vol_norm}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略134: 区间突破与量能共振
    for window in core_windows[:2]:
        new_name = f"Breakout_Volume_Confirm_{window}"
        if new_name not in tried_factors:
            breakout = f"($close - Max($high, {window})) / (Max($high, {window}) + 0.0001)"
            vol_norm = f"$volume / (Mean($volume, {window}) + 0.0001)"
            new_expr = f"{breakout} * {vol_norm}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略135: 影线强度（上影线占比）
    for window in kline_windows:
        new_name = f"Upper_Shadow_Strength_{window}"
        if new_name not in tried_factors:
            upper_shadow = f"($high - If($close > $open, $close, $open)) / ($high - $low + 0.0001)"
            new_expr = f"Mean({upper_shadow}, {window})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # === VWAP/成交额/复权价因子（136-150）===
    # 注意：使用 $vwap/$amount/$adjclose/$factor 字段

    # 策略136: VWAP偏离（价格相对VWAP）
    for window in vwap_windows:
        new_name = f"VWAP_Deviation_{window}"
        if new_name not in tried_factors:
            vwap_dev = f"$close / ($vwap + 0.0001) - 1"
            new_expr = f"Mean({vwap_dev}, {window})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略137: VWAP动量（VWAP变化率）
    for window in vwap_windows:
        new_name = f"VWAP_Momentum_{window}"
        if new_name not in tried_factors:
            new_expr = f"($vwap - Ref($vwap, {window})) / Ref($vwap, {window})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略138: 价格- VWAP差的均值（均值回归信号）
    for window in vwap_windows:
        new_name = f"Price_VWAP_Spread_{window}"
        if new_name not in tried_factors:
            spread = f"($close - $vwap) / ($vwap + 0.0001)"
            new_expr = f"Mean({spread}, {window})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略139: 成交额动量
    for window in vwap_windows:
        new_name = f"Amount_Momentum_{window}"
        if new_name not in tried_factors:
            new_expr = f"($amount - Ref($amount, {window})) / Ref($amount, {window})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略140: 成交额相对均值
    for window in vwap_windows:
        new_name = f"Amount_Ratio_{window}"
        if new_name not in tried_factors:
            new_expr = f"$amount / (Mean($amount, {window}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略141: 成交额与收益率相关
    for window in core_windows[:2]:
        new_name = f"Amount_Price_Corr_{window}"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, 1)) / Ref($close, 1)"
            amount_change = f"($amount - Ref($amount, 1)) / Ref($amount, 1)"
            new_expr = f"Corr({price_change}, {amount_change}, {window})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略142: 价量效率（收益/成交额）
    for window in vwap_windows:
        new_name = f"Price_Amount_Efficiency_{window}"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, {window})) / Ref($close, {window})"
            amount_mean = f"Mean($amount, {window})"
            new_expr = f"{price_change} / ({amount_mean} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略143: 复权价动量
    for window in vwap_windows:
        new_name = f"AdjClose_Momentum_{window}"
        if new_name not in tried_factors:
            new_expr = f"($adjclose - Ref($adjclose, {window})) / Ref($adjclose, {window})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略144: 复权价与原价差（复权影响）
    for window in vwap_windows:
        new_name = f"AdjClose_Price_Spread_{window}"
        if new_name not in tried_factors:
            spread = f"($adjclose - $close) / ($close + 0.0001)"
            new_expr = f"Mean({spread}, {window})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略145: 复权因子变化率（分红送转影响）
    for window in core_windows[:2]:
        new_name = f"AdjFactor_Change_{window}"
        if new_name not in tried_factors:
            new_expr = f"($factor - Ref($factor, {window})) / Ref($factor, {window})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略146: VWAP与成交额联动
    for window in vwap_windows:
        new_name = f"VWAP_Amount_Interaction_{window}"
        if new_name not in tried_factors:
            vwap_change = f"($vwap - Ref($vwap, {window})) / Ref($vwap, {window})"
            amount_change = f"($amount - Ref($amount, {window})) / Ref($amount, {window})"
            new_expr = f"{vwap_change} * {amount_change}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略147: 成交额标准化波动率
    for window in core_windows[:2]:
        new_name = f"Amount_Volatility_{window}"
        if new_name not in tried_factors:
            amount_change = f"($amount - Ref($amount, 1)) / Ref($amount, 1)"
            new_expr = f"Std({amount_change}, {window})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略148: VWAP偏离的均值回归强度
    for window in core_windows[:2]:
        new_name = f"VWAP_Deviation_Reversion_{window}"
        if new_name not in tried_factors:
            vwap_dev = f"$close / ($vwap + 0.0001) - 1"
            vwap_dev_prev = f"Ref({vwap_dev}, {window})"
            new_expr = f"{vwap_dev_prev} - {vwap_dev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略149: 成交额分位位置
    for window in amount_quantile_windows:
        new_name = f"Amount_Quantile_Position_{window}"
        if new_name not in tried_factors:
            q50 = f"Quantile($amount, {window}, 0.5)"
            q25 = f"Quantile($amount, {window}, 0.25)"
            q75 = f"Quantile($amount, {window}, 0.75)"
            new_expr = f"($amount - {q50}) / ({q75} - {q25} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略150: 价格- VWAP偏离与收益协同
    for window in vwap_windows:
        new_name = f"VWAP_Deviation_Momentum_{window}"
        if new_name not in tried_factors:
            vwap_dev = f"$close / ($vwap + 0.0001) - 1"
            momentum = f"($close - Ref($close, {window})) / Ref($close, {window})"
            new_expr = f"{vwap_dev} * {momentum}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # === 资金流/分歧与复权价多周期因子（151-170）===

    # 策略151: 简化资金流（成交额方向）
    for window in vwap_windows:
        new_name = f"Money_Flow_{window}"
        if new_name not in tried_factors:
            direction = f"Sign($close - Ref($close, 1))"
            flow = f"{direction} * $amount"
            new_expr = f"Sum({flow}, {window}) / (Sum($amount, {window}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略152: 量价背离（价格动量 vs 成交额动量）
    for window in vwap_windows:
        new_name = f"Price_Amount_Divergence_{window}"
        if new_name not in tried_factors:
            price_mom = f"($close - Ref($close, {window})) / Ref($close, {window})"
            amount_mom = f"($amount - Ref($amount, {window})) / Ref($amount, {window})"
            new_expr = f"{price_mom} - {amount_mom}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略153: 量能背离强度（价格涨但量缩/价格跌但量增）
    for window in short_kline_windows:
        new_name = f"Price_Amount_Divergence_Score_{window}"
        if new_name not in tried_factors:
            price_change = f"($close - Ref($close, {window})) / Ref($close, {window})"
            amount_change = f"($amount - Ref($amount, {window})) / Ref($amount, {window})"
            new_expr = f"{price_change} * (-Sign({amount_change}))"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略154: 放量突破强度（突破幅度 × 成交额分位）
    for window in core_windows[:2]:
        new_name = f"Breakout_Amount_Quantile_{window}"
        if new_name not in tried_factors:
            breakout = f"($close - Max($high, {window})) / (Max($high, {window}) + 0.0001)"
            q50 = f"Quantile($amount, {window}, 0.5)"
            q25 = f"Quantile($amount, {window}, 0.25)"
            q75 = f"Quantile($amount, {window}, 0.75)"
            amount_pos = f"($amount - {q50}) / ({q75} - {q25} + 0.0001)"
            new_expr = f"{breakout} * {amount_pos}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略155: 成交额脉冲（短期均值/长期均值）
    for short_w, long_w in momentum_pairs:
        new_name = f"Amount_Pulse_{short_w}_{long_w}"
        if new_name not in tried_factors:
            new_expr = f"Mean($amount, {short_w}) / (Mean($amount, {long_w}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略156: 成交额动量的变化率（加速度）
    for window in core_windows[:2]:
        new_name = f"Amount_Momentum_Acceleration_{window}"
        if new_name not in tried_factors:
            mom = f"($amount - Ref($amount, {window})) / Ref($amount, {window})"
            mom_prev = f"Ref({mom}, {window})"
            new_expr = f"{mom} - {mom_prev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略157: 复权价多周期动量一致性
    for short_w, long_w in momentum_pairs:
        new_name = f"AdjClose_Momentum_Consistency_{short_w}_{long_w}"
        if new_name not in tried_factors:
            short_mom = f"($adjclose - Ref($adjclose, {short_w})) / Ref($adjclose, {short_w})"
            long_mom = f"($adjclose - Ref($adjclose, {long_w})) / Ref($adjclose, {long_w})"
            new_expr = f"Sign({short_mom}) * Sign({long_mom}) * {long_mom}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略158: 复权价动量衰减（短-长）
    for short_w, long_w in momentum_pairs:
        new_name = f"AdjClose_Momentum_Decay_{short_w}_{long_w}"
        if new_name not in tried_factors:
            short_mom = f"($adjclose - Ref($adjclose, {short_w})) / Ref($adjclose, {short_w})"
            long_mom = f"($adjclose - Ref($adjclose, {long_w})) / Ref($adjclose, {long_w})"
            new_expr = f"{short_mom} - {long_mom}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略159: 复权价均线偏离（均值回归）
    for window in core_windows:
        new_name = f"AdjClose_MA_Deviation_{window}"
        if new_name not in tried_factors:
            new_expr = f"$adjclose / Mean($adjclose, {window}) - 1"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略160: 复权价趋势强度（斜率）
    for window in core_windows:
        new_name = f"AdjClose_Slope_{window}"
        if new_name not in tried_factors:
            new_expr = f"Slope($adjclose, {window}) / ($adjclose + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略161: 复权价突破强度
    for window in core_windows:
        new_name = f"AdjClose_Breakout_{window}"
        if new_name not in tried_factors:
            new_expr = f"($adjclose - Max($adjclose, {window})) / (Max($adjclose, {window}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略162: 复权价分位位置
    for window in amount_quantile_windows:
        new_name = f"AdjClose_Quantile_Position_{window}"
        if new_name not in tried_factors:
            q50 = f"Quantile($adjclose, {window}, 0.5)"
            q25 = f"Quantile($adjclose, {window}, 0.25)"
            q75 = f"Quantile($adjclose, {window}, 0.75)"
            new_expr = f"($adjclose - {q50}) / ({q75} - {q25} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略163: 复权价动量与成交额协同
    for window in core_windows[:2]:
        new_name = f"AdjClose_Momentum_Amount_{window}"
        if new_name not in tried_factors:
            mom = f"($adjclose - Ref($adjclose, {window})) / Ref($adjclose, {window})"
            amt = f"($amount - Ref($amount, {window})) / Ref($amount, {window})"
            new_expr = f"{mom} * {amt}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # === 风险调整/极值/回撤类（171-190）===

    # 策略171: 风险调整动量（收益/波动）
    for window in core_windows:
        new_name = f"RiskAdj_Momentum_{window}"
        if new_name not in tried_factors:
            ret = f"($close - Ref($close, {window})) / Ref($close, {window})"
            vol = f"Std(($close - Ref($close, 1)) / Ref($close, 1), {window})"
            new_expr = f"{ret} / ({vol} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略172: VWAP波动率
    for window in core_windows[:2]:
        new_name = f"VWAP_Volatility_{window}"
        if new_name not in tried_factors:
            vwap_change = f"($vwap - Ref($vwap, 1)) / Ref($vwap, 1)"
            new_expr = f"Std({vwap_change}, {window})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略173: 成交额偏度（分位不对称）
    for window in amount_quantile_windows:
        new_name = f"Amount_Skewness_{window}"
        if new_name not in tried_factors:
            q75 = f"Quantile($amount, {window}, 0.75)"
            q50 = f"Quantile($amount, {window}, 0.5)"
            q25 = f"Quantile($amount, {window}, 0.25)"
            new_expr = f"({q75} + {q25} - 2*{q50}) / ({q75} - {q25} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略174: 复权价波动率
    for window in core_windows:
        new_name = f"AdjClose_Volatility_{window}"
        if new_name not in tried_factors:
            ret = f"($adjclose - Ref($adjclose, 1)) / Ref($adjclose, 1)"
            new_expr = f"Std({ret}, {window})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略175: 最大回撤近似（窗口内高点回撤）
    for window in core_windows:
        new_name = f"MaxDrawdown_Approx_{window}"
        if new_name not in tried_factors:
            peak = f"Max($close, {window})"
            new_expr = f"($close - {peak}) / ({peak} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略176: 回撤恢复速度（跌幅 vs 反弹）
    for window in core_windows[:2]:
        new_name = f"Drawdown_Recovery_{window}"
        if new_name not in tried_factors:
            peak = f"Max($close, {window})"
            drawdown = f"({peak} - $close) / ({peak} + 0.0001)"
            rebound = f"($close - Min($close, {window})) / (Min($close, {window}) + 0.0001)"
            new_expr = f"{rebound} - {drawdown}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略177: VWAP偏离的标准化强度
    for window in core_windows:
        new_name = f"VWAP_Deviation_Z_{window}"
        if new_name not in tried_factors:
            dev = f"$close / ($vwap + 0.0001) - 1"
            dev_std = f"Std({dev}, {window})"
            new_expr = f"{dev} / ({dev_std} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略178: 量能极值位置（相对历史极值）
    for window in amount_quantile_windows:
        new_name = f"Amount_Extreme_Position_{window}"
        if new_name not in tried_factors:
            amax = f"Max($amount, {window})"
            amin = f"Min($amount, {window})"
            new_expr = f"($amount - {amin}) / ({amax} - {amin} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略179: VWAP动量的变化率
    for window in core_windows[:2]:
        new_name = f"VWAP_Momentum_Acceleration_{window}"
        if new_name not in tried_factors:
            mom = f"($vwap - Ref($vwap, {window})) / Ref($vwap, {window})"
            mom_prev = f"Ref({mom}, {window})"
            new_expr = f"{mom} - {mom_prev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略180: 价格回撤后放量反弹
    for window in core_windows[:2]:
        new_name = f"Rebound_With_Amount_{window}"
        if new_name not in tried_factors:
            drawdown = f"(Max($close, {window}) - $close) / (Max($close, {window}) + 0.0001)"
            amount_ratio = f"$amount / (Mean($amount, {window}) + 0.0001)"
            new_expr = f"(1 - {drawdown}) * {amount_ratio}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # === 分位/反转/波动-动量耦合（191-210）===

    # 策略191: 价格分位反转（高分位做空/低分位做多）
    for window in amount_quantile_windows:
        new_name = f"Price_Quantile_Reversal_{window}"
        if new_name not in tried_factors:
            q50 = f"Quantile($close, {window}, 0.5)"
            q25 = f"Quantile($close, {window}, 0.25)"
            q75 = f"Quantile($close, {window}, 0.75)"
            pos = f"($close - {q50}) / ({q75} - {q25} + 0.0001)"
            new_expr = f"-{pos}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略192: 成交额分位反转
    for window in amount_quantile_windows:
        new_name = f"Amount_Quantile_Reversal_{window}"
        if new_name not in tried_factors:
            q50 = f"Quantile($amount, {window}, 0.5)"
            q25 = f"Quantile($amount, {window}, 0.25)"
            q75 = f"Quantile($amount, {window}, 0.75)"
            pos = f"($amount - {q50}) / ({q75} - {q25} + 0.0001)"
            new_expr = f"-{pos}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略193: VWAP分位反转
    for window in amount_quantile_windows:
        new_name = f"VWAP_Quantile_Reversal_{window}"
        if new_name not in tried_factors:
            q50 = f"Quantile($vwap, {window}, 0.5)"
            q25 = f"Quantile($vwap, {window}, 0.25)"
            q75 = f"Quantile($vwap, {window}, 0.75)"
            pos = f"($vwap - {q50}) / ({q75} - {q25} + 0.0001)"
            new_expr = f"-{pos}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略194: 波动率-动量耦合（高波动下动量）
    for window in core_windows:
        new_name = f"Volatility_Momentum_{window}"
        if new_name not in tried_factors:
            vol = f"Std(($close - Ref($close, 1)) / Ref($close, 1), {window})"
            mom = f"($close - Ref($close, {window})) / Ref($close, {window})"
            new_expr = f"{vol} * {mom}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略195: 波动率分位动量（动量被波动分位加权）
    for window in amount_quantile_windows:
        new_name = f"Volatility_Quantile_Momentum_{window}"
        if new_name not in tried_factors:
            ret = f"($close - Ref($close, 1)) / Ref($close, 1)"
            vol = f"Std({ret}, {window})"
            q50 = f"Quantile({vol}, {window}, 0.5)"
            q25 = f"Quantile({vol}, {window}, 0.25)"
            q75 = f"Quantile({vol}, {window}, 0.75)"
            vol_pos = f"({vol} - {q50}) / ({q75} - {q25} + 0.0001)"
            mom = f"($close - Ref($close, {window})) / Ref($close, {window})"
            new_expr = f"{mom} * {vol_pos}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略196: 复权价分位反转
    for window in amount_quantile_windows:
        new_name = f"AdjClose_Quantile_Reversal_{window}"
        if new_name not in tried_factors:
            q50 = f"Quantile($adjclose, {window}, 0.5)"
            q25 = f"Quantile($adjclose, {window}, 0.25)"
            q75 = f"Quantile($adjclose, {window}, 0.75)"
            pos = f"($adjclose - {q50}) / ({q75} - {q25} + 0.0001)"
            new_expr = f"-{pos}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略197: 量能分位与收益协同
    for window in amount_quantile_windows:
        new_name = f"Amount_Quantile_Return_{window}"
        if new_name not in tried_factors:
            q50 = f"Quantile($amount, {window}, 0.5)"
            q25 = f"Quantile($amount, {window}, 0.25)"
            q75 = f"Quantile($amount, {window}, 0.75)"
            pos = f"($amount - {q50}) / ({q75} - {q25} + 0.0001)"
            ret = f"($close - Ref($close, {window})) / Ref($close, {window})"
            new_expr = f"{pos} * {ret}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略198: VWAP偏离的分位位置
    for window in amount_quantile_windows:
        new_name = f"VWAP_Deviation_Quantile_{window}"
        if new_name not in tried_factors:
            dev = f"$close / ($vwap + 0.0001) - 1"
            q50 = f"Quantile({dev}, {window}, 0.5)"
            q25 = f"Quantile({dev}, {window}, 0.25)"
            q75 = f"Quantile({dev}, {window}, 0.75)"
            new_expr = f"({dev} - {q50}) / ({q75} - {q25} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略199: 波动率与成交额协同
    for window in amount_quantile_windows:
        new_name = f"Volatility_Amount_Interaction_{window}"
        if new_name not in tried_factors:
            vol = f"Std(($close - Ref($close, 1)) / Ref($close, 1), {window})"
            amt = f"$amount / (Mean($amount, {window}) + 0.0001)"
            new_expr = f"{vol} * {amt}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略200: 成交额反转动量（高量后收益衰减）
    for window in core_windows[:2]:
        new_name = f"Amount_Reversal_Momentum_{window}"
        if new_name not in tried_factors:
            amt_pos = f"$amount / (Mean($amount, {window}) + 0.0001)"
            ret = f"($close - Ref($close, {window})) / Ref($close, {window})"
            new_expr = f"-{amt_pos} * {ret}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # === 收敛/错配/持续性扩展（201-220）===

    # 策略201: 波动收敛后的趋势突破（低波动后更看突破）
    for window in core_windows:
        new_name = f"Vol_Contraction_Breakout_{window}"
        if new_name not in tried_factors:
            breakout = f"($close - Max($close, {window})) / (Max($close, {window}) + 0.0001)"
            short_vol = f"Std(($close - Ref($close, 1)) / Ref($close, 1), {window})"
            long_vol = f"Std(($close - Ref($close, 1)) / Ref($close, 1), {window * 2})"
            new_expr = f"{breakout} * ({long_vol} / ({short_vol} + 0.0001))"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略202: 下跌集中度（下跌收益是否集中主导）
    for window in core_windows:
        new_name = f"Downside_Concentration_{window}"
        if new_name not in tried_factors:
            ret1 = "($close - Ref($close, 1)) / Ref($close, 1)"
            down_moves = f"Sum(If({ret1} < 0, Abs({ret1}), 0), {window})"
            up_moves = f"Sum(If({ret1} > 0, {ret1}, 0), {window})"
            new_expr = f"({down_moves} - {up_moves}) / ({down_moves} + {up_moves} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略203: 价格趋势稳定度（斜率相对波动越大越稳）
    for window in core_windows:
        new_name = f"Trend_Stability_{window}"
        if new_name not in tried_factors:
            slope = f"Slope($close, {window}) / ($close + 0.0001)"
            vol = f"Std(($close - Ref($close, 1)) / Ref($close, 1), {window})"
            new_expr = f"{slope} / ({vol} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略204: 均线压缩后的偏离（短长均线压缩后看价格偏离）
    for short_w, long_w in ma_pairs:
        new_name = f"MA_Compression_Dislocation_{short_w}_{long_w}"
        if new_name not in tried_factors:
            ma_spread = f"Abs(Mean($close, {short_w}) / (Mean($close, {long_w}) + 0.0001) - 1)"
            price_dev = f"$close / (Mean($close, {short_w}) + 0.0001) - 1"
            new_expr = f"{price_dev} / ({ma_spread} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略205: VWAP与收盘价趋势错配
    for window in core_windows:
        new_name = f"VWAP_Close_Trend_Dislocation_{window}"
        if new_name not in tried_factors:
            close_slope = f"Slope($close, {window}) / ($close + 0.0001)"
            vwap_slope = f"Slope($vwap, {window}) / ($vwap + 0.0001)"
            new_expr = f"{close_slope} - {vwap_slope}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略206: 成交额放大下的VWAP偏离反转
    for window in core_windows:
        new_name = f"VWAP_Deviation_Amount_Reversal_{window}"
        if new_name not in tried_factors:
            dev = f"$close / ($vwap + 0.0001) - 1"
            amt_ratio = f"$amount / (Mean($amount, {window}) + 0.0001)"
            new_expr = f"-{dev} * {amt_ratio}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略207: 复权价回撤恢复不对称
    for window in amount_quantile_windows:
        new_name = f"AdjClose_Drawdown_Rebound_{window}"
        if new_name not in tried_factors:
            peak = f"Max($adjclose, {window})"
            trough = f"Min($adjclose, {window})"
            drawdown = f"({peak} - $adjclose) / ({peak} + 0.0001)"
            rebound = f"($adjclose - {trough}) / ({trough} + 0.0001)"
            new_expr = f"{rebound} - {drawdown}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略208: 量价相关性的变化率
    for window in core_windows:
        new_name = f"Price_Amount_Corr_Acceleration_{window}"
        if new_name not in tried_factors:
            price_change = "($close - Ref($close, 1)) / Ref($close, 1)"
            amount_change = "($amount - Ref($amount, 1)) / Ref($amount, 1)"
            corr_now = f"Corr({price_change}, {amount_change}, {window})"
            corr_prev = f"Ref(Corr({price_change}, {amount_change}, {window}), {window})"
            new_expr = f"{corr_now} - {corr_prev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略209: 价格位置与成交额位置错配
    for window in amount_quantile_windows:
        new_name = f"Price_Amount_Position_Divergence_{window}"
        if new_name not in tried_factors:
            price_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            amount_pos = f"($amount - Min($amount, {window})) / (Max($amount, {window}) - Min($amount, {window}) + 0.0001)"
            new_expr = f"{price_pos} - {amount_pos}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略210: 盘中实体与量能共振
    for window in core_windows[:2]:
        new_name = f"Intraday_Body_Amount_Resonance_{window}"
        if new_name not in tried_factors:
            body = "($close - $open) / ($high - $low + 0.0001)"
            amount_ratio = f"$amount / (Mean($amount, {window}) + 0.0001)"
            new_expr = f"{body} * {amount_ratio}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略211: 缺口后的回补压力
    for window in kline_windows:
        new_name = f"Gap_Fill_Pressure_{window}"
        if new_name not in tried_factors:
            gap = "($open - Ref($close, 1)) / (Ref($close, 1) + 0.0001)"
            intraday_reversal = "($close - $open) / ($open + 0.0001)"
            new_expr = f"-{gap} * {intraday_reversal} * ($amount / (Mean($amount, {window}) + 0.0001))"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略212: 上下影线不对称
    for window in core_windows[:2]:
        new_name = f"Shadow_Asymmetry_{window}"
        if new_name not in tried_factors:
            upper_shadow = "($high - If($close > $open, $close, $open)) / ($high - $low + 0.0001)"
            lower_shadow = "(If($close < $open, $close, $open) - $low) / ($high - $low + 0.0001)"
            amount_ratio = f"$amount / (Mean($amount, {window}) + 0.0001)"
            new_expr = f"({lower_shadow} - {upper_shadow}) * {amount_ratio}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略213: 收盘位置与VWAP位置错配
    for window in core_windows:
        new_name = f"Close_VWAP_Position_Divergence_{window}"
        if new_name not in tried_factors:
            close_pos = f"($close - Min($close, {window})) / (Max($close, {window}) - Min($close, {window}) + 0.0001)"
            vwap_pos = f"($vwap - Min($vwap, {window})) / (Max($vwap, {window}) - Min($vwap, {window}) + 0.0001)"
            new_expr = f"{close_pos} - {vwap_pos}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略214: 价格残差与量能趋势交互
    for window in core_windows:
        new_name = f"Residual_Amount_Interaction_{window}"
        if new_name not in tried_factors:
            residual = f"Resi($close, {window}) / ($close + 0.0001)"
            amount_trend = f"Mean($amount, {window}) / (Mean($amount, {window * 2}) + 0.0001) - 1"
            new_expr = f"{residual} * {amount_trend}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略215: 收益方向持续性（连续涨跌的净强度）
    for window in core_windows:
        new_name = f"Return_Sign_Persistence_{window}"
        if new_name not in tried_factors:
            ret1 = "($close - Ref($close, 1)) / Ref($close, 1)"
            sign_sum = f"Sum(If({ret1} > 0, 1, If({ret1} < 0, -1, 0)), {window})"
            new_expr = f"{sign_sum} / {window}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略216: 方向持续性与量能确认
    for window in core_windows[:2]:
        new_name = f"Return_Persistence_Amount_Confirm_{window}"
        if new_name not in tried_factors:
            ret1 = "($close - Ref($close, 1)) / Ref($close, 1)"
            sign_sum = f"Sum(If({ret1} > 0, 1, If({ret1} < 0, -1, 0)), {window})"
            amount_ratio = f"$amount / (Mean($amount, {window}) + 0.0001)"
            new_expr = f"({sign_sum} / {window}) * {amount_ratio}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略217: 波动率与振幅错配
    for window in core_windows:
        new_name = f"Range_Volatility_Dislocation_{window}"
        if new_name not in tried_factors:
            intraday_range = f"Mean(($high - $low) / ($close + 0.0001), {window})"
            close_vol = f"Std(($close - Ref($close, 1)) / Ref($close, 1), {window})"
            new_expr = f"{intraday_range} / ({close_vol} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略218: 复权价趋势与现价偏离错配
    for window in core_windows:
        new_name = f"AdjClose_Trend_Spot_Dislocation_{window}"
        if new_name not in tried_factors:
            adj_trend = f"Slope($adjclose, {window}) / ($adjclose + 0.0001)"
            spot_dev = f"$close / ($adjclose + 0.0001) - 1"
            new_expr = f"{adj_trend} * {spot_dev}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略219: 突破但量能未确认
    for window in core_windows:
        new_name = f"Breakout_Without_Amount_Confirm_{window}"
        if new_name not in tried_factors:
            breakout = f"($close - Max($close, {window})) / (Max($close, {window}) + 0.0001)"
            amount_ratio = f"$amount / (Mean($amount, {window}) + 0.0001)"
            new_expr = f"{breakout} / ({amount_ratio} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略220: 价量双分位错位后的反转
    for window in amount_quantile_windows:
        new_name = f"Price_Amount_Quantile_Misalignment_{window}"
        if new_name not in tried_factors:
            price_q50 = f"Quantile($close, {window}, 0.5)"
            price_q25 = f"Quantile($close, {window}, 0.25)"
            price_q75 = f"Quantile($close, {window}, 0.75)"
            amount_q50 = f"Quantile($amount, {window}, 0.5)"
            amount_q25 = f"Quantile($amount, {window}, 0.25)"
            amount_q75 = f"Quantile($amount, {window}, 0.75)"
            price_pos = f"($close - {price_q50}) / ({price_q75} - {price_q25} + 0.0001)"
            amount_pos = f"($amount - {amount_q50}) / ({amount_q75} - {amount_q25} + 0.0001)"
            new_expr = f"-({price_pos} - {amount_pos})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # === 周期偏向模板（231-236）===

    if horizon_profile.get("horizon_bucket") == "short":
        for window in fast_windows:
            new_name = f"ShortBias_Gap_VWAP_Reversal_{window}"
            if new_name not in tried_factors:
                gap = "($open - Ref($close, 1)) / (Ref($close, 1) + 0.0001)"
                vwap_dev = "$close / ($vwap + 0.0001) - 1"
                vol_ratio = f"$volume / (Mean($volume, {window}) + 0.0001)"
                new_expr = f"-{gap} * {vwap_dev} * {vol_ratio}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors

        for window in fast_windows:
            new_name = f"ShortBias_Intraday_Amount_Pressure_{window}"
            if new_name not in tried_factors:
                body = "($close - $open) / ($high - $low + 0.0001)"
                range_pos = "($close - $low) / ($high - $low + 0.0001)"
                amount_ratio = f"$amount / (Mean($amount, {window}) + 0.0001)"
                new_expr = f"{body} * ({range_pos} - 0.5) * {amount_ratio}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors

        for window in fast_windows:
            new_name = f"ShortBias_VWAP_Snapback_{window}"
            if new_name not in tried_factors:
                dev = "$close / ($vwap + 0.0001) - 1"
                dev_prev = f"Ref({dev}, {window})"
                new_expr = f"({dev_prev} - {dev}) * ($amount / (Mean($amount, {window}) + 0.0001))"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors

    if horizon_profile.get("horizon_bucket") == "long":
        for window in slow_windows:
            new_name = f"LongBias_Trend_Drawdown_Recovery_{window}"
            if new_name not in tried_factors:
                trend = f"Slope($adjclose, {window}) / ($adjclose + 0.0001)"
                peak = f"Max($adjclose, {window})"
                drawdown = f"({peak} - $adjclose) / ({peak} + 0.0001)"
                new_expr = f"{trend} * (1 - {drawdown})"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors

        for short_w, long_w in ma_pairs:
            new_name = f"LongBias_Market_Relative_Trend_{short_w}_{long_w}"
            if new_name not in tried_factors:
                stock_short = f"($close - Ref($close, {short_w})) / Ref($close, {short_w})"
                stock_long = f"($close - Ref($close, {long_w})) / Ref($close, {long_w})"
                market_long = f"ChangeInstrument('SH000300', ($close - Ref($close, {long_w})) / Ref($close, {long_w}))"
                new_expr = f"(Sign({stock_short}) * Sign({stock_long})) * ({stock_long} - {market_long})"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors

        for window in slow_windows:
            new_name = f"LongBias_Amount_Trend_Confirm_{window}"
            if new_name not in tried_factors:
                price_trend = f"Slope($close, {window}) / ($close + 0.0001)"
                amount_trend = f"Mean($amount, {max(1, window // 2)}) / (Mean($amount, {window}) + 0.0001) - 1"
                new_expr = f"{price_trend} * {amount_trend}"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors

    # === 周期自适应模板（221-230）===

    # 策略221: 与目标周期一致的动态趋势稳定度
    for window in core_windows:
        new_name = f"Adaptive_Trend_Stability_{window}"
        if new_name not in tried_factors:
            slope = f"Slope($close, {window}) / ($close + 0.0001)"
            vol = f"Std(($close - Ref($close, 1)) / Ref($close, 1), {window})"
            new_expr = f"{slope} / ({vol} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略222: 自适应价量协同动量
    for window in core_windows:
        new_name = f"Adaptive_Price_Amount_Momentum_{window}"
        if new_name not in tried_factors:
            price_mom = f"($close - Ref($close, {window})) / Ref($close, {window})"
            amount_mom = f"($amount - Ref($amount, {window})) / Ref($amount, {window})"
            new_expr = f"{price_mom} * {amount_mom}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略223: 自适应均线压缩突破
    for short_w, long_w in ma_pairs:
        new_name = f"Adaptive_MA_Compression_Breakout_{short_w}_{long_w}"
        if new_name not in tried_factors:
            ma_spread = f"Abs(Mean($close, {short_w}) / (Mean($close, {long_w}) + 0.0001) - 1)"
            price_dev = f"$close / (Mean($close, {short_w}) + 0.0001) - 1"
            new_expr = f"{price_dev} / ({ma_spread} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略224: 自适应分位反转
    for window in quantile_windows:
        new_name = f"Adaptive_Price_Quantile_Reversal_{window}"
        if new_name not in tried_factors:
            q50 = f"Quantile($close, {window}, 0.5)"
            q25 = f"Quantile($close, {window}, 0.25)"
            q75 = f"Quantile($close, {window}, 0.75)"
            pos = f"($close - {q50}) / ({q75} - {q25} + 0.0001)"
            new_expr = f"-{pos}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略225: 自适应波动收敛突破
    for fast_w, slow_w in zip(core_windows, slow_windows):
        new_name = f"Adaptive_Vol_Contraction_Breakout_{fast_w}_{slow_w}"
        if new_name not in tried_factors:
            breakout = f"($close - Max($close, {fast_w})) / (Max($close, {fast_w}) + 0.0001)"
            short_vol = f"Std(($close - Ref($close, 1)) / Ref($close, 1), {fast_w})"
            long_vol = f"Std(($close - Ref($close, 1)) / Ref($close, 1), {slow_w})"
            new_expr = f"{breakout} * ({long_vol} / ({short_vol} + 0.0001))"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略226: 自适应VWAP偏离反转
    for window in core_windows:
        new_name = f"Adaptive_VWAP_Deviation_Reversal_{window}"
        if new_name not in tried_factors:
            dev = f"$close / ($vwap + 0.0001) - 1"
            dev_std = f"Std({dev}, {window})"
            new_expr = f"-{dev} / ({dev_std} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略227: 自适应复权趋势
    for window in slow_windows:
        new_name = f"Adaptive_AdjClose_Trend_{window}"
        if new_name not in tried_factors:
            new_expr = f"Slope($adjclose, {window}) / ($adjclose + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略228: 自适应量能确认突破
    for window in core_windows:
        new_name = f"Adaptive_Breakout_Amount_Confirm_{window}"
        if new_name not in tried_factors:
            breakout = f"($close - Max($high, {window})) / (Max($high, {window}) + 0.0001)"
            amount_ratio = f"$amount / (Mean($amount, {window}) + 0.0001)"
            new_expr = f"{breakout} * {amount_ratio}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略229: 自适应方向持续性
    for window in core_windows:
        new_name = f"Adaptive_Return_Persistence_{window}"
        if new_name not in tried_factors:
            ret1 = "($close - Ref($close, 1)) / Ref($close, 1)"
            sign_sum = f"Sum(If({ret1} > 0, 1, If({ret1} < 0, -1, 0)), {window})"
            new_expr = f"{sign_sum} / {window}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors

    # 策略230: 自适应长期均线相对强弱
    for window in long_windows:
        new_name = f"Adaptive_Long_MA_Ratio_{window}"
        if new_name not in tried_factors:
            new_expr = f"$close / (Mean($close, {window}) + 0.0001) - 1"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # === 策略231-240: 新因子挖掘思路（2024年新增）===
    
    # 策略231: 资金流向强度因子（基于价量关系）
    for window in [5, 10, 20]:
        new_name = f"Money_Flow_Strength_{window}"
        if new_name not in tried_factors:
            # 收盘价在高低点之间的位置 × 成交量变化
            price_position = f"($close - $low) / ($high - $low + 0.0001)"
            vol_change = f"($volume - Ref($volume, 1)) / (Ref($volume, 1) + 0.0001)"
            new_expr = f"{price_position} * {vol_change}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略232: 筹码集中度变化因子
    for window in [10, 20]:
        new_name = f"Chip_Concentration_{window}"
        if new_name not in tried_factors:
            # 使用价格分布的标准差来衡量筹码集中度
            new_expr = f"1 / (Std($close, {window}) / Mean($close, {window}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略233: 量价趋势一致性因子
    for window in [5, 10]:
        new_name = f"Price_Volume_Trend_{window}"
        if new_name not in tried_factors:
            # 价格趋势与成交量趋势的一致性
            price_slope = f"Slope($close, {window})"
            vol_slope = f"Slope($volume, {window})"
            new_expr = f"({price_slope} * {vol_slope}) / (Abs({price_slope}) + Abs({vol_slope}) + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略234: 波动率扩张/收缩因子
    for window in [5, 10, 20]:
        new_name = f"Volatility_Expansion_{window}"
        if new_name not in tried_factors:
            # 当前波动率相对于历史波动率的变化
            current_vol = f"Std($close, 5) / Mean($close, 5)"
            hist_vol = f"Std(Ref($close, 5), {window}) / Mean(Ref($close, 5), {window})"
            new_expr = f"({current_vol} - {hist_vol}) / ({hist_vol} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略235: 价格惯性因子（非线性动量）
    for window in [5, 10]:
        new_name = f"Price_Inertia_{window}"
        if new_name not in tried_factors:
            # 使用平方项增强大幅波动的权重
            ret = f"($close - Ref($close, {window})) / Ref($close, {window})"
            new_expr = f"Sign({ret}) * Pow(Abs({ret}), 1.5)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略236: 均线系统发散度因子
    for short_win in [5, 10]:
        for long_win in [20, 60]:
            new_name = f"MA_Divergence_{short_win}_{long_win}"
            if new_name not in tried_factors:
                # 短期均线与长期均线的发散程度
                short_ma = f"Mean($close, {short_win})"
                long_ma = f"Mean($close, {long_win})"
                new_expr = f"({short_ma} - {long_ma}) / ({long_ma} + 0.0001) / (Std($close, {long_win}) / Mean($close, {long_win}) + 0.0001)"
                new_factors[new_name] = new_expr
                if len(new_factors) >= max_new_factors:
                    return new_factors
    
    # 策略237: 日内动量持续性因子
    for window in [3, 5, 10]:
        new_name = f"Intraday_Momentum_{window}"
        if new_name not in tried_factors:
            # 日内上涨天数占比减去下跌天数占比
            up_days = f"Mean(($close > $open), {window})"
            down_days = f"Mean(($close < $open), {window})"
            new_expr = f"{up_days} - {down_days}"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略238: 价格-均线修复速度因子
    for window in [5, 10, 20]:
        new_name = f"MA_Reversion_Speed_{window}"
        if new_name not in tried_factors:
            # 价格偏离均线的程度除以修复所需时间
            deviation = f"($close - Mean($close, {window})) / Mean($close, {window})"
            new_expr = f"{deviation} / ({window} + 0.0001)"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    # 策略239: 多周期动量共振因子
    for short_win in [3, 5]:
        for mid_win in [10]:
            for long_win in [20]:
                new_name = f"Multi_Cycle_Resonance_{short_win}_{mid_win}_{long_win}"
                if new_name not in tried_factors:
                    short_ret = f"($close - Ref($close, {short_win})) / Ref($close, {short_win})"
                    mid_ret = f"($close - Ref($close, {mid_win})) / Ref($close, {mid_win})"
                    long_ret = f"($close - Ref($close, {long_win})) / Ref($close, {long_win})"
                    # 三个周期同向时信号最强
                    new_expr = f"({short_ret} + {mid_ret} + {long_ret}) / 3 * Sign({short_ret} * {mid_ret} * {long_ret})"
                    new_factors[new_name] = new_expr
                    if len(new_factors) >= max_new_factors:
                        return new_factors
    
    # 策略240: 成交量加权价格动量因子
    for window in [5, 10, 20]:
        new_name = f"Volume_Weighted_Momentum_{window}"
        if new_name not in tried_factors:
            # 用成交量加权的收益率
            price_ret = f"($close - Ref($close, 1)) / Ref($close, 1)"
            vol_weight = f"$volume / Mean($volume, {window})"
            new_expr = f"Mean({price_ret} * {vol_weight}, {window})"
            new_factors[new_name] = new_expr
            if len(new_factors) >= max_new_factors:
                return new_factors
    
    return new_factors


def analyze_custom_factors(
    instruments="csi300",
    label_expr=None,  # 如果为None，自动生成未来5日平均价格相对变化表达式
    custom_factors=None,
    config_path=None,
    result_prefix="custom_factors",
    start_time=None,
    end_time=None,
    freq="day",
    limit_data_days=None,
    min_valid_days=10,
    output_dir=None,
    train_start_time=None,
    train_end_time=None,
    valid_start_time=None,
    valid_end_time=None,
    min_ic_train=0.02,
    min_ic_valid=0.02,
    max_ic_decay=0.5,
    require_same_sign=True,
    use_rolling_evaluation=True,
    rolling_train_years=5,
    rolling_valid_years=1,
    rolling_step_months=12,
    min_rolling_windows=3,
    use_segment_evaluation=True,
    segment_months=6,
    min_segment_count=4,
    segment_min_ic_abs=None,
    force_full_scan=False,
    enforce_novelty_filter=False,
    min_novelty_score=0.2,
    max_ref_corr_for_good=0.8,
    use_result_cache=True,
    resume_from_cache=True,
):
    """
    分析自定义因子在指定市场（默认沪深300）上的相关性（目标：未来5日平均价格）
    
    Parameters
    ----------
    instruments : str or list
        标的列表，可以是"csi500"、"csi300"、"csi1000"等预定义市场，或股票代码列表
    label_expr : str
        标签表达式，默认："(Ref($close, -1) + ... + Ref($close, -5)) / 5 / $close - 1"（未来5日平均价格相对变化）
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
        
        # 修复概率性卡死：将 joblib backend 从 multiprocessing 改为 threading
        # multiprocessing backend 在 macOS 上容易导致死锁/卡死
        from qlib.config import C
        C.joblib_backend = "threading"
        C.maxtasksperchild = None  # 禁用任务数限制，避免进程重启问题
        print(f"  - joblib backend: {C.joblib_backend} (已切换为 threading 避免卡死)")
    except Exception as e:
        print(f"初始化qlib失败: {e}")
        print("请确保已安装qlib数据，运行: python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data --region cn")
        return None
    
    # 使用固定股票池（方法1：获取当前成分股列表，确保股票池一致）
    # 显式获取当前成分股列表，固定使用这个列表
    if isinstance(instruments, str):
        if instruments.lower() in ["csi500", "csi300", "csi100", "csi1000", "all"]:
            try:
                # 获取当前成分股（使用最近的交易日，比如2024-12-31）
                # 使用相同的日期作为start_time和end_time，只获取该日期的成分股
                # 注意：使用一个固定的最近交易日，确保获取的是当前成分股（约300只）
                current_date = "2025-12-31"  # 使用最近的交易日
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
                
                # 验证：检查股票数量是否合理（可能获取了历史成分股）
                expected_count = {"csi300": 300, "csi500": 500, "csi100": 100, "csi1000": 1000}.get(instruments.lower(), None)
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
                current_date = "2025-12-31"  # 使用最近的交易日
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
                print(f"⚠️  从字典配置获取当前成分股失败: {e}，尝试使用字典格式")
                # 如果无法获取成分股列表，尝试使用字典格式，但去重时会跳过
                instruments_list = None
        else:
            instruments_list = None  # 使用字典格式
            print(f"⚠️  字典配置中没有market字段，无法获取成分股列表，去重功能将被跳过")
    else:
        # 列表格式，直接使用
        instruments_list = list(instruments)
        print(f"✓ 使用股票列表: {len(instruments_list)}只股票")
    
    # 验证并过滤股票代码
    if instruments_list is not None:
        # 确保instruments_list是列表格式
        if not isinstance(instruments_list, list):
            instruments_list = list(instruments_list)
        
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
    else:
        # instruments_list为None，说明无法获取成分股列表
        print("⚠️  无法获取成分股列表，相关性去重功能将被跳过")
    
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
                start_time="2015-01-01",
                end_time="2025-12-31",
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
        start_time = "2015-01-01"
    if end_time is None:
        end_time = "2025-12-31"
    
    # 设置训练集和验证集时间范围（用于时间一致性检查）
    if train_start_time is None:
        train_start_time = start_time
    if train_end_time is None:
        # 默认训练集占70%的时间
        start_ts = pd.Timestamp(start_time)
        end_ts = pd.Timestamp(end_time)
        total_days = (end_ts - start_ts).days
        train_end_ts = start_ts + pd.Timedelta(days=int(total_days * 0.7))
        train_end_time = train_end_ts.strftime('%Y-%m-%d')
    if valid_start_time is None:
        valid_start_time = train_end_time
    if valid_end_time is None:
        valid_end_time = end_time
    
    print(f"\n时间范围设置:")
    print(f"  训练集: {train_start_time} 到 {train_end_time}")
    print(f"  验证集: {valid_start_time} 到 {valid_end_time}")
    print(f"  时间一致性检查参数:")
    print(f"    - 训练集最小IC: {min_ic_train}")
    print(f"    - 验证集最小IC: {min_ic_valid}")
    print(f"    - 最大IC衰减率: {max_ic_decay}")
    print(f"    - 要求IC同号: {require_same_sign}")
    
    rolling_windows = []
    if use_rolling_evaluation:
        rolling_windows = _build_rolling_windows(
            start_time=start_time,
            end_time=end_time,
            train_years=rolling_train_years,
            valid_years=rolling_valid_years,
            step_months=rolling_step_months,
        )
        print(f"  滚动评估: 开启")
        print(f"    - 训练窗口: {rolling_train_years}年")
        print(f"    - 验证窗口: {rolling_valid_years}年")
        print(f"    - 滚动步长: {rolling_step_months}个月")
        print(f"    - 窗口数量: {len(rolling_windows)}")
    else:
        print(f"  滚动评估: 关闭")

    if segment_min_ic_abs is None:
        segment_min_ic_abs = min_ic_valid
    if use_segment_evaluation:
        print(f"  分段评估: 开启")
        print(f"    - 分段长度: {segment_months}个月")
        print(f"    - 最少分段数: {min_segment_count}")
        print(f"    - 分段IC阈值: {segment_min_ic_abs}")
    else:
        print(f"  分段评估: 关闭")
    print(f"  新颖度筛选: {'开启' if enforce_novelty_filter else '关闭'}")
    print(f"    - 最低新颖度: {min_novelty_score}")
    print(f"    - 最大参考相关性: {max_ref_corr_for_good}")
    
    # 如果指定了limit_data_days，限制数据范围
    if limit_data_days is not None:
        start_ts = pd.Timestamp(start_time)
        end_ts = pd.Timestamp(end_time)
        limited_start_ts = end_ts - pd.Timedelta(days=limit_data_days)
        if limited_start_ts > start_ts:
            start_time = limited_start_ts.strftime('%Y-%m-%d')
            print(f"⚠️  限制数据范围: 使用最近{limit_data_days}天的数据")
    
    # 如果没有提供label_expr，自动生成未来5日平均价格表达式
    if label_expr is None:
        label_expr = make_future_avg_price_label_expr(5)

    horizon_profile = build_factor_horizon_profile(label_expr)
    scoring_profile = build_horizon_scoring_profile(horizon_profile)
    if min_valid_days == 10:
        min_valid_days = horizon_profile["suggested_min_valid_days"]
    if use_segment_evaluation and segment_months == 6:
        segment_months = horizon_profile["suggested_segment_months"]
    if use_rolling_evaluation and rolling_valid_years == 1:
        rolling_valid_years = horizon_profile["suggested_rolling_valid_years"]
    
    print(f"\n使用时间范围: {start_time} 到 {end_time}")
    print(f"标签表达式: {label_expr}")
    print(
        f"  目标周期识别: {horizon_profile['label_horizon']}日 "
        f"({horizon_profile['horizon_bucket']})"
    )
    print(f"  因子窗口模板:")
    print(f"    - 快速窗口: {horizon_profile['fast_windows']}")
    print(f"    - 核心窗口: {horizon_profile['core_windows']}")
    print(f"    - 慢速窗口: {horizon_profile['slow_windows']}")
    print(f"    - 长周期窗口: {horizon_profile['long_windows']}")
    print(f"  评估参数建议已应用:")
    print(f"    - min_valid_days: {min_valid_days}")
    print(f"    - segment_months: {segment_months}")
    print(f"    - rolling_valid_years: {rolling_valid_years}")
    print(f"  周期评分模板:")
    print(f"    - effective_ic_scale: {scoring_profile['effective_ic_scale']}")
    print(f"    - effective_icir_scale: {scoring_profile['effective_icir_scale']}")
    print(
        f"    - stability/tradeability/novelty: "
        f"{scoring_profile['stability_weight']:.2f}/"
        f"{scoring_profile['tradeability_weight']:.2f}/"
        f"{scoring_profile['novelty_weight']:.2f}"
    )

    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载因子配置
    factor_config = load_factor_config(config_path)
    good_factors = factor_config.get('good_factors', {})
    bad_factors = factor_config.get('bad_factors', {})
    min_ic_threshold = factor_config.get('min_ic_threshold', 0.03)
    min_icir_threshold = factor_config.get('min_icir_threshold', 0.5)
    
    # 将当前挖掘目标和周期模板写回配置，便于后续追踪
    config_updated = False
    desired_config_fields = {
        'label_expr': label_expr,
        'label_horizon': horizon_profile['label_horizon'],
        'horizon_bucket': horizon_profile['horizon_bucket'],
        'factor_window_profile': {
            'fast_windows': horizon_profile['fast_windows'],
            'core_windows': horizon_profile['core_windows'],
            'slow_windows': horizon_profile['slow_windows'],
            'long_windows': horizon_profile['long_windows'],
            'quantile_windows': horizon_profile['quantile_windows'],
            'ma_pairs': horizon_profile['ma_pairs'],
        },
        'scoring_profile': scoring_profile,
    }
    for key, value in desired_config_fields.items():
        if factor_config.get(key) != value:
            factor_config[key] = value
            config_updated = True
    if config_updated:
        save_factor_config(factor_config, config_path)
        print("  - 已将当前标签表达式和周期模板保存到配置文件")
    
    print(f"\n从配置文件加载:")
    print(f"  - 好因子数量: {len(good_factors)}")
    print(f"  - 坏因子数量: {len(bad_factors)}")
    print(f"  - 标签表达式: {factor_config.get('label_expr')}")
    print(f"  - 目标周期: {factor_config.get('label_horizon')} ({factor_config.get('horizon_bucket')})")
    print(f"  - IC阈值: {min_ic_threshold}")
    print(f"  - ICIR阈值: {min_icir_threshold}")

    experiment_metadata = {
        'script': Path(__file__).name,
        'label_expr': label_expr,
        'label_horizon': horizon_profile['label_horizon'],
        'horizon_bucket': horizon_profile['horizon_bucket'],
        'factor_window_profile': {
            'fast_windows': horizon_profile['fast_windows'],
            'core_windows': horizon_profile['core_windows'],
            'slow_windows': horizon_profile['slow_windows'],
            'long_windows': horizon_profile['long_windows'],
            'quantile_windows': horizon_profile['quantile_windows'],
            'ma_pairs': horizon_profile['ma_pairs'],
        },
        'scoring_profile': scoring_profile,
        'instruments': instruments if isinstance(instruments, str) else 'custom_list',
        'instrument_count': len(instruments_list) if instruments_list is not None else None,
        'start_time': start_time,
        'end_time': end_time,
        'freq': freq,
        'limit_data_days': limit_data_days,
        'train_start_time': train_start_time,
        'train_end_time': train_end_time,
        'valid_start_time': valid_start_time,
        'valid_end_time': valid_end_time,
        'min_ic_threshold': min_ic_threshold,
        'min_icir_threshold': min_icir_threshold,
        'min_ic_train': min_ic_train,
        'min_ic_valid': min_ic_valid,
        'max_ic_decay': max_ic_decay,
        'require_same_sign': require_same_sign,
        'use_rolling_evaluation': use_rolling_evaluation,
        'rolling_train_years': rolling_train_years,
        'rolling_valid_years': rolling_valid_years,
        'rolling_step_months': rolling_step_months,
        'min_rolling_windows': min_rolling_windows,
        'use_segment_evaluation': use_segment_evaluation,
        'segment_months': segment_months,
        'min_segment_count': min_segment_count,
        'segment_min_ic_abs': segment_min_ic_abs,
        'force_full_scan': force_full_scan,
        'enforce_novelty_filter': enforce_novelty_filter,
        'min_novelty_score': min_novelty_score,
        'max_ref_corr_for_good': max_ref_corr_for_good,
    }
    experiment_id = _build_experiment_signature(experiment_metadata)
    cache_file = output_dir / f"{result_prefix}_cache_{experiment_id}.csv"
    metadata_file = output_dir / f"{result_prefix}_experiment_{experiment_id}.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(experiment_metadata, f, ensure_ascii=False, indent=2)
    print(f"  - 实验ID: {experiment_id}")
    print(f"  - 实验配置已保存: {metadata_file}")

    evaluate_all_candidates = False
    # 如果没有提供自定义因子，基于配置文件生成新因子
    if custom_factors is None:
        print("\n基于好因子生成新的组合因子（长期目标优化）...")
        all_candidates = generate_new_factors_from_good_ones(
            good_factors=good_factors,
            bad_factors=bad_factors,
            max_new_factors=None,  # 不限制数量，单次执行遍历全部候选因子
            ignore_tried_factors=True,
            horizon_profile=horizon_profile,
        )

        tried_names = set(good_factors.keys()) | set(bad_factors.keys())
        duplicate_count = sum(1 for name in all_candidates.keys() if name in tried_names)
        new_factors = {name: expr for name, expr in all_candidates.items() if name not in tried_names}

        print(f"✓ 候选因子总数: {len(all_candidates)}")
        print(f"  - 历史已测试: {len(tried_names)}")
        print(f"  - 本次新增: {len(new_factors)}")
        print(f"  - 与历史重复: {duplicate_count}")

        # 为了单次遍历全部候选，直接评估全部候选（包含与历史重复的因子）
        if len(all_candidates) > 0:
            custom_factors = all_candidates
            if duplicate_count > 0 and force_full_scan:
                print("⚠️  本次评估包含历史重复因子（force_full_scan=True）")
            evaluate_all_candidates = True
        else:
            # 如果没有生成候选因子，使用配置文件中的好因子
            custom_factors = good_factors.copy()
            print("⚠️  没有生成候选因子，使用配置文件中的好因子")
        
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
    
    # 过滤掉已测试的因子（使用表达式字典）
    good_expressions, _ = _extract_factor_info(good_factors)
    bad_expressions, _ = _extract_factor_info(bad_factors)
    tried_factors = {**good_expressions, **bad_expressions}
    if evaluate_all_candidates and force_full_scan:
        tried_factors = {}
    original_count = len(custom_factors)
    if evaluate_all_candidates and not force_full_scan:
        # 默认仍跳过已测试因子，避免重复运行；需要全量扫描时设置 force_full_scan=True
        custom_factors = {k: v for k, v in custom_factors.items() if k not in tried_factors}
        filtered_count = original_count - len(custom_factors)
        if filtered_count > 0:
            print(f"\n⚠️  跳过了 {filtered_count} 个已测试的因子")
    elif evaluate_all_candidates and force_full_scan:
        filtered_count = 0
    else:
        custom_factors = {k: v for k, v in custom_factors.items() if k not in tried_factors}
        filtered_count = original_count - len(custom_factors)
        if filtered_count > 0:
            print(f"\n⚠️  跳过了 {filtered_count} 个已测试的因子")
    
    if len(custom_factors) == 0:
        print("\n⚠️  所有因子都已测试过，没有新因子需要测试")
        return None

    cached_results_df = pd.DataFrame()
    cached_factor_names = set()
    if use_result_cache and resume_from_cache:
        cached_results_df = _load_cached_factor_results(cache_file)
        if len(cached_results_df) > 0 and 'factor' in cached_results_df.columns:
            cached_results_df = cached_results_df.drop_duplicates(subset=['factor'], keep='last')
            cached_factor_names = set(cached_results_df['factor'].astype(str).tolist())
            custom_factors = {k: v for k, v in custom_factors.items() if k not in cached_factor_names}
            print(f"\n检测到缓存结果: {len(cached_factor_names)} 个因子")
            print(f"  - 缓存文件: {cache_file}")
            print(f"  - 本次待计算: {len(custom_factors)} 个因子")
    
    print(f"\n正在测试 {len(custom_factors)} 个自定义因子...")
    
    # 获取标签数据
    print("\n正在获取标签数据（未来5日平均价格）...")
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

    # 预加载一小组参考好因子，用于评估候选因子的边际增益/新颖度
    reference_factor_map = {}
    config_metadata = factor_config.get('_factor_metadata', {})
    reference_good_items = list(good_expressions.items())
    if len(reference_good_items) > 0:
        scored_reference_items = []
        for factor_name, factor_expr in reference_good_items:
            metadata = config_metadata.get(factor_name, {}) if isinstance(config_metadata, dict) else {}
            score = abs(float(metadata.get('ic', 0.0))) + abs(float(metadata.get('icir', 0.0)))
            scored_reference_items.append((factor_name, factor_expr, score))
        scored_reference_items.sort(key=lambda x: x[2], reverse=True)
        selected_reference_items = scored_reference_items[:8]
        if len(selected_reference_items) > 0:
            try:
                ref_names = [item[0] for item in selected_reference_items]
                ref_exprs = [item[1] for item in selected_reference_items]
                ref_data = D.features(
                    instruments_list,
                    ref_exprs,
                    start_time=start_time,
                    end_time=end_time,
                    freq=freq
                )
                for idx, ref_name in enumerate(ref_names):
                    if isinstance(ref_data.columns, pd.MultiIndex):
                        reference_factor_map[ref_name] = ref_data.iloc[:, idx]
                    else:
                        reference_factor_map[ref_name] = ref_data.iloc[:, idx]
                print(f"已加载 {len(reference_factor_map)} 个参考好因子用于新颖度评估")
            except Exception as e:
                print(f"⚠️  参考好因子加载失败，跳过新颖度评估: {e}")
                reference_factor_map = {}
    
    # 测试每个因子
    print("\n开始计算因子IC...")
    factor_results = cached_results_df.to_dict('records') if len(cached_results_df) > 0 else []
    
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
                result_row = {
                    'factor': factor_name,
                    'expression': factor_expr,
                    'IC': np.nan,
                    'Rank_IC': np.nan,
                    'ICIR': np.nan,
                    'IC_std': np.nan,
                    'valid_days': 0,
                    'status': 'no_data'
                }
                factor_results.append(result_row)
                if use_result_cache:
                    _append_factor_result_to_cache(cache_file, result_row)
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
                result_row = {
                    'factor': factor_name,
                    'expression': factor_expr,
                    'IC': np.nan,
                    'Rank_IC': np.nan,
                    'ICIR': np.nan,
                    'IC_std': np.nan,
                    'valid_days': 0,
                    'status': 'no_common_index'
                }
                factor_results.append(result_row)
                if use_result_cache:
                    _append_factor_result_to_cache(cache_file, result_row)
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
                result_row = {
                    'factor': factor_name,
                    'expression': factor_expr,
                    'IC': np.nan,
                    'Rank_IC': np.nan,
                    'ICIR': np.nan,
                    'IC_std': np.nan,
                    'valid_days': len(aligned_factor),
                    'status': 'insufficient_data'
                }
                factor_results.append(result_row)
                if use_result_cache:
                    _append_factor_result_to_cache(cache_file, result_row)
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
                    # 分离训练集和验证集的IC
                    train_start_ts = pd.Timestamp(train_start_time)
                    train_end_ts = pd.Timestamp(train_end_time)
                    valid_start_ts = pd.Timestamp(valid_start_time)
                    valid_end_ts = pd.Timestamp(valid_end_time)
                    
                    # 获取IC的日期索引
                    if isinstance(ic_series.index, pd.MultiIndex):
                        ic_dates = pd.to_datetime(ic_series.index.get_level_values('datetime'))
                    else:
                        ic_dates = pd.to_datetime(ic_series.index)
                    
                    # 分离训练集和验证集
                    train_mask = (ic_dates >= train_start_ts) & (ic_dates <= train_end_ts)
                    valid_mask = (ic_dates >= valid_start_ts) & (ic_dates <= valid_end_ts)
                    
                    train_ic = ic_series[train_mask]
                    valid_ic = ic_series[valid_mask]
                    train_ric = ric_series[train_mask]
                    valid_ric = ric_series[valid_mask]
                    
                    # 计算训练集和验证集的IC统计量
                    mean_ic_train = train_ic.mean() if len(train_ic) > 0 else np.nan
                    mean_ic_valid = valid_ic.mean() if len(valid_ic) > 0 else np.nan
                    mean_ric_train = train_ric.mean() if len(train_ric) > 0 else np.nan
                    mean_ric_valid = valid_ric.mean() if len(valid_ric) > 0 else np.nan
                    
                    ic_std_train = train_ic.std() if len(train_ic) > 0 else np.nan
                    ic_std_valid = valid_ic.std() if len(valid_ic) > 0 else np.nan
                    
                    # 计算ICIR
                    if ic_std_train > 1e-10 and not np.isnan(mean_ic_train):
                        icir_train = mean_ic_train / ic_std_train
                    else:
                        icir_train = np.nan
                    
                    if ic_std_valid > 1e-10 and not np.isnan(mean_ic_valid):
                        icir_valid = mean_ic_valid / ic_std_valid
                    else:
                        icir_valid = np.nan
                    
                    # 计算时间一致性指标
                    # 1. IC衰减率 = (训练集IC - 验证集IC) / |训练集IC|
                    if not np.isnan(mean_ic_train) and abs(mean_ic_train) > 1e-10:
                        ic_decay = (mean_ic_train - mean_ic_valid) / abs(mean_ic_train)
                    else:
                        ic_decay = np.nan
                    
                    # 2. IC符号一致性
                    same_sign = (mean_ic_train * mean_ic_valid >= 0) if (not np.isnan(mean_ic_train) and not np.isnan(mean_ic_valid)) else False
                    
                    # 3. IC稳定性得分（基于IC衰减率和符号一致性）
                    stability_score = 0.0
                    if not np.isnan(ic_decay):
                        # 衰减率越小，稳定性越高
                        if abs(ic_decay) <= max_ic_decay:
                            stability_score += 0.5
                        if same_sign:
                            stability_score += 0.3
                        if not np.isnan(mean_ic_valid) and abs(mean_ic_valid) >= min_ic_valid:
                            stability_score += 0.2
                    
                    # 4. 计算整体IC（用于排序）
                    mean_ic = ic_series.mean()
                    mean_ric = ric_series.mean()
                    ic_std = ic_series.std()
                    if ic_std > 1e-10 and not np.isnan(mean_ic):
                        icir = mean_ic / ic_std
                    else:
                        icir = np.nan
                    trade_eval = _evaluate_tradeability(
                        pred=aligned_factor,
                        label=aligned_label,
                    )
                    novelty_eval = _evaluate_novelty(
                        candidate_factor=aligned_factor,
                        reference_factor_map=reference_factor_map,
                    )
                    
                    # 判断因子是否稳定（单切分 + 滚动评估）
                    stability_reason = []
                    is_stable_single_split = True
                    if np.isnan(mean_ic_train) or abs(mean_ic_train) < min_ic_train:
                        is_stable_single_split = False
                        stability_reason.append(f"训练集IC不足({mean_ic_train:.6f}<{min_ic_train})")
                    if np.isnan(mean_ic_valid) or abs(mean_ic_valid) < min_ic_valid:
                        is_stable_single_split = False
                        stability_reason.append(f"验证集IC不足({mean_ic_valid:.6f}<{min_ic_valid})")
                    if np.isnan(ic_decay) or abs(ic_decay) > max_ic_decay:
                        is_stable_single_split = False
                        stability_reason.append(f"IC衰减过大({ic_decay:.2%}>{max_ic_decay:.2%})")
                    if require_same_sign and not same_sign:
                        is_stable_single_split = False
                        stability_reason.append("IC符号不一致")
                    
                    rolling_eval = {
                        'rolling_window_count': 0,
                        'rolling_valid_windows': 0,
                        'rolling_stable_windows': 0,
                        'rolling_stable_ratio': np.nan,
                        'rolling_same_sign_ratio': np.nan,
                        'rolling_ic_train_mean': np.nan,
                        'rolling_ic_valid_mean': np.nan,
                        'rolling_ic_decay_mean': np.nan,
                        'rolling_ic_valid_abs_mean': np.nan,
                        'rolling_ic_valid_abs_min': np.nan,
                        'rolling_stability_score': 0.0,
                        'rolling_is_stable': is_stable_single_split,
                        'rolling_reason': 'not_enabled',
                    }
                    segment_eval = {
                        'segment_count': 0,
                        'segment_valid_segments': 0,
                        'segment_stable_segments': 0,
                        'segment_stable_ratio': np.nan,
                        'segment_same_sign_ratio': np.nan,
                        'segment_ic_abs_mean': np.nan,
                        'segment_ic_abs_min': np.nan,
                        'segment_stability_score': 0.0,
                        'segment_is_stable': is_stable_single_split,
                        'segment_reason': 'not_enabled',
                    }
                    if use_rolling_evaluation:
                        rolling_eval = _evaluate_rolling_stability(
                            ic_series=ic_series,
                            min_valid_days=min_valid_days,
                            min_ic_train=min_ic_train,
                            min_ic_valid=min_ic_valid,
                            max_ic_decay=max_ic_decay,
                            require_same_sign=require_same_sign,
                            rolling_windows=rolling_windows,
                            min_rolling_windows=min_rolling_windows,
                        )
                    if use_segment_evaluation:
                        segment_eval = _evaluate_segment_stability(
                            ic_series=ic_series,
                            min_valid_days=min_valid_days,
                            segment_months=segment_months,
                            min_segment_count=min_segment_count,
                            min_ic_abs=segment_min_ic_abs,
                            require_same_sign=require_same_sign,
                        )

                    rolling_ok = None
                    rolling_fail_reason = None
                    if use_rolling_evaluation and rolling_eval['rolling_valid_windows'] >= min_rolling_windows:
                        rolling_ok = rolling_eval['rolling_is_stable']
                        if (not rolling_ok) and rolling_eval['rolling_reason'] not in ('', 'stable'):
                            rolling_fail_reason = rolling_eval['rolling_reason']

                    segment_ok = None
                    segment_fail_reason = None
                    if use_segment_evaluation and segment_eval['segment_valid_segments'] >= min_segment_count:
                        segment_ok = segment_eval['segment_is_stable']
                        if (not segment_ok) and segment_eval['segment_reason'] not in ('', 'stable'):
                            segment_fail_reason = segment_eval['segment_reason']

                    if segment_ok is not None and rolling_ok is not None:
                        # 实践中滚动和分段容易出现“一个严格一个宽松”的分歧；用 OR 更贴近实盘筛选效果
                        is_stable = segment_ok or rolling_ok
                    elif segment_ok is not None:
                        is_stable = segment_ok
                    elif rolling_ok is not None:
                        is_stable = rolling_ok
                    else:
                        is_stable = False
                        stability_reason.append("evidence_insufficient")

                    # 仅在最终不稳定时，补充导致“不稳定判定”的关键原因，避免误导（例如分段通过但滚动未过）
                    if not is_stable:
                        if rolling_ok is False and rolling_fail_reason:
                            stability_reason.append(f"滚动评估不通过: {rolling_fail_reason}")
                        if segment_ok is False and segment_fail_reason:
                            stability_reason.append(f"分段评估不通过: {segment_fail_reason}")

                    # 融合得分：分段评估与滚动评估优先
                    score_parts = [(stability_score, 0.2)]
                    weight_sum = 0.2
                    if rolling_ok is not None:
                        score_parts.append((rolling_eval['rolling_stability_score'], 0.4))
                        weight_sum += 0.4
                    if segment_ok is not None:
                        score_parts.append((segment_eval['segment_stability_score'], 0.4))
                        weight_sum += 0.4
                    if weight_sum > 0:
                        stability_score = sum(s * w for s, w in score_parts) / weight_sum
                    stability_score = (
                        scoring_profile['stability_weight'] * stability_score
                        + scoring_profile['tradeability_weight'] * trade_eval['tradeability_score']
                        + scoring_profile['novelty_weight'] * novelty_eval['novelty_score']
                    )
                    
                    result_row = {
                        'factor': factor_name,
                        'expression': factor_expr,
                        'horizon_bucket': horizon_profile['horizon_bucket'],
                        'label_horizon': horizon_profile['label_horizon'],
                        'IC': mean_ic,
                        'Rank_IC': mean_ric,
                        'ICIR': icir,
                        'IC_std': ic_std,
                        'IC_train': mean_ic_train,
                        'IC_valid': mean_ic_valid,
                        'ICIR_train': icir_train,
                        'ICIR_valid': icir_valid,
                        'IC_decay': ic_decay,
                        'same_sign': same_sign,
                        'stability_score': stability_score,
                        'ls_mean': trade_eval['ls_mean'],
                        'ls_win_rate': trade_eval['ls_win_rate'],
                        'top_bucket_return': trade_eval['top_mean'],
                        'bottom_bucket_return': trade_eval['bottom_mean'],
                        'trade_coverage_days': trade_eval['coverage_days'],
                        'tradeability_score': trade_eval['tradeability_score'],
                        'trade_direction_match': trade_eval['direction_match'],
                        'bucket_slope_mean': trade_eval['bucket_slope_mean'],
                        'bucket_corr_mean': trade_eval['bucket_corr_mean'],
                        'monotonicity_win_rate': trade_eval['monotonicity_win_rate'],
                        'monotonicity_score': trade_eval['monotonicity_score'],
                        'bucket_profile': trade_eval['bucket_profile'],
                        'bucket_profile_coverage_days': trade_eval['bucket_profile_coverage_days'],
                        'max_ref_corr': novelty_eval['max_ref_corr'],
                        'mean_ref_corr': novelty_eval['mean_ref_corr'],
                        'closest_ref_factor': novelty_eval['closest_ref_factor'],
                        'novelty_score': novelty_eval['novelty_score'],
                        'single_split_stable': is_stable_single_split,
                        'rolling_window_count': rolling_eval['rolling_window_count'],
                        'rolling_valid_windows': rolling_eval['rolling_valid_windows'],
                        'rolling_stable_windows': rolling_eval['rolling_stable_windows'],
                        'rolling_stable_ratio': rolling_eval['rolling_stable_ratio'],
                        'rolling_same_sign_ratio': rolling_eval['rolling_same_sign_ratio'],
                        'rolling_ic_train_mean': rolling_eval['rolling_ic_train_mean'],
                        'rolling_ic_valid_mean': rolling_eval['rolling_ic_valid_mean'],
                        'rolling_ic_decay_mean': rolling_eval['rolling_ic_decay_mean'],
                        'rolling_ic_valid_abs_mean': rolling_eval['rolling_ic_valid_abs_mean'],
                        'rolling_ic_valid_abs_min': rolling_eval['rolling_ic_valid_abs_min'],
                        'rolling_stability_score': rolling_eval['rolling_stability_score'],
                        'segment_count': segment_eval['segment_count'],
                        'segment_valid_segments': segment_eval['segment_valid_segments'],
                        'segment_stable_segments': segment_eval['segment_stable_segments'],
                        'segment_stable_ratio': segment_eval['segment_stable_ratio'],
                        'segment_same_sign_ratio': segment_eval['segment_same_sign_ratio'],
                        'segment_ic_abs_mean': segment_eval['segment_ic_abs_mean'],
                        'segment_ic_abs_min': segment_eval['segment_ic_abs_min'],
                        'segment_stability_score': segment_eval['segment_stability_score'],
                        'is_stable': is_stable,
                        'stability_reason': '; '.join(stability_reason) if stability_reason else 'stable',
                        'valid_days': len(ic_series),
                        'train_days': len(train_ic),
                        'valid_days_split': len(valid_ic),
                        'status': 'success' if is_stable else 'unstable'
                    }
                    factor_results.append(result_row)
                    if use_result_cache:
                        _append_factor_result_to_cache(cache_file, result_row)
                    
                    status_mark = "✓" if is_stable else "⚠"
                    if use_rolling_evaluation or use_segment_evaluation:
                        rolling_part = ""
                        if use_rolling_evaluation:
                            rolling_part = f" | rolling稳定: {rolling_eval['rolling_stable_windows']}/{rolling_eval['rolling_valid_windows']}"
                        segment_part = ""
                        if use_segment_evaluation:
                            segment_part = f" | segment稳定: {segment_eval['segment_stable_segments']}/{segment_eval['segment_valid_segments']}"
                        print(
                            f"  {status_mark} {factor_name:30s} | IC_train: {mean_ic_train:8.6f} | IC_valid: {mean_ic_valid:8.6f}"
                            f" | LS: {trade_eval['ls_mean']:8.6f} | Mono: {trade_eval['bucket_corr_mean']:6.3f} | Novel: {novelty_eval['novelty_score']:5.3f}{rolling_part}{segment_part} | 稳定: {is_stable}"
                        )
                    else:
                        print(
                            f"  {status_mark} {factor_name:30s} | IC_train: {mean_ic_train:8.6f} | IC_valid: {mean_ic_valid:8.6f} "
                            f"| LS: {trade_eval['ls_mean']:8.6f} | Mono: {trade_eval['bucket_corr_mean']:6.3f} | Novel: {novelty_eval['novelty_score']:5.3f} | 衰减: {ic_decay:7.2%} | 稳定: {is_stable}"
                        )
                    if not is_stable:
                        print(f"    {stability_reason}")
                else:
                    print(f"  ⚠️  {factor_name:30s} | IC天数不足 (有效天数: {len(ic_series)} < {min_valid_days})")
                    result_row = {
                        'factor': factor_name,
                        'expression': factor_expr,
                        'IC': np.nan,
                        'Rank_IC': np.nan,
                        'ICIR': np.nan,
                        'IC_std': np.nan,
                        'IC_train': np.nan,
                        'IC_valid': np.nan,
                        'ICIR_train': np.nan,
                        'ICIR_valid': np.nan,
                        'IC_decay': np.nan,
                        'same_sign': False,
                        'stability_score': 0.0,
                        'ls_mean': np.nan,
                        'ls_win_rate': np.nan,
                        'top_bucket_return': np.nan,
                        'bottom_bucket_return': np.nan,
                        'trade_coverage_days': 0,
                        'tradeability_score': 0.0,
                        'trade_direction_match': False,
                        'bucket_slope_mean': np.nan,
                        'bucket_corr_mean': np.nan,
                        'monotonicity_win_rate': np.nan,
                        'monotonicity_score': 0.0,
                        'bucket_profile': None,
                        'bucket_profile_coverage_days': 0,
                        'max_ref_corr': np.nan,
                        'mean_ref_corr': np.nan,
                        'closest_ref_factor': None,
                        'novelty_score': 0.0,
                        'single_split_stable': False,
                        'rolling_window_count': 0,
                        'rolling_valid_windows': 0,
                        'rolling_stable_windows': 0,
                        'rolling_stable_ratio': np.nan,
                        'rolling_same_sign_ratio': np.nan,
                        'rolling_ic_train_mean': np.nan,
                        'rolling_ic_valid_mean': np.nan,
                        'rolling_ic_decay_mean': np.nan,
                        'rolling_ic_valid_abs_mean': np.nan,
                        'rolling_ic_valid_abs_min': np.nan,
                        'rolling_stability_score': 0.0,
                        'is_stable': False,
                        'stability_reason': 'insufficient_days',
                        'valid_days': len(ic_series),
                        'train_days': 0,
                        'valid_days_split': 0,
                        'status': 'insufficient_days'
                    }
                    factor_results.append(result_row)
                    if use_result_cache:
                        _append_factor_result_to_cache(cache_file, result_row)
            except Exception as e:
                print(f"  ✗ {factor_name:30s} | 计算IC错误: {str(e)[:100]}")
                result_row = {
                    'factor': factor_name,
                    'expression': factor_expr,
                    'IC': np.nan,
                    'Rank_IC': np.nan,
                    'ICIR': np.nan,
                    'IC_std': np.nan,
                    'valid_days': 0,
                    'status': f'error: {str(e)[:50]}'
                }
                factor_results.append(result_row)
                if use_result_cache:
                    _append_factor_result_to_cache(cache_file, result_row)
            
        except Exception as e:
            print(f"  ✗ {factor_name:30s} | 错误: {str(e)[:50]}")
            result_row = {
                'factor': factor_name,
                'expression': factor_expr,
                'IC': np.nan,
                'Rank_IC': np.nan,
                'ICIR': np.nan,
                'IC_std': np.nan,
                'valid_days': 0,
                'status': f'error: {str(e)[:50]}'
            }
            factor_results.append(result_row)
            if use_result_cache:
                _append_factor_result_to_cache(cache_file, result_row)
            continue
    
    # 分析结果
    if factor_results:
        results_df = pd.DataFrame(factor_results)
        # 按IC绝对值排序
        results_df['IC_abs'] = results_df['IC'].abs().fillna(0)
        results_df = results_df.sort_values('IC_abs', ascending=False)
        results_df = results_df.drop('IC_abs', axis=1)
        
        print("\n" + "="*100)
        print("因子IC分析结果（未来5日平均价格）")
        print("="*100)
        
        # 统计信息
        valid_results = results_df[results_df['status'].isin(['success', 'unstable'])]
        if len(valid_results) > 0:
            effective_ic_threshold = min_ic_threshold * scoring_profile['effective_ic_scale']
            effective_icir_threshold = min_icir_threshold * scoring_profile['effective_icir_scale']
            promising_ic_threshold = min_ic_threshold * scoring_profile['promising_ic_scale']
            promising_icir_threshold = min_icir_threshold * scoring_profile['promising_icir_scale']
            print(f"\n总体统计:")
            print(f"  成功测试因子数: {len(valid_results)} / {len(factor_results)}")
            print(f"  平均IC: {valid_results['IC'].mean():.6f}")
            print(f"  IC标准差: {valid_results['IC'].std():.6f}")
            print(f"  平均Rank IC: {valid_results['Rank_IC'].mean():.6f}")
            print(f"  平均ICIR: {valid_results['ICIR'].mean():.6f}")
            if 'ls_mean' in valid_results.columns:
                print(f"  平均分组多空收益: {valid_results['ls_mean'].mean():.6f}")
                print(f"  平均交易性得分: {valid_results['tradeability_score'].mean():.4f}")
            if 'bucket_corr_mean' in valid_results.columns:
                print(f"  平均分层相关性: {valid_results['bucket_corr_mean'].mean():.4f}")
                print(f"  平均单调性得分: {valid_results['monotonicity_score'].mean():.4f}")
            if 'novelty_score' in valid_results.columns:
                print(f"  平均新颖度得分: {valid_results['novelty_score'].mean():.4f}")
            print(f"  周期化有效阈值: |IC|>{effective_ic_threshold:.4f} 或 |ICIR|>{effective_icir_threshold:.4f}")
            print(f"  周期化观察阈值: |IC|>{promising_ic_threshold:.4f} 或 |ICIR|>{promising_icir_threshold:.4f}")
            
            # 有效因子（根据配置文件中的阈值，且通过时间一致性检查）
            effective_mask = (
                (valid_results['is_stable'] == True) &
                ((valid_results['IC'].abs() > effective_ic_threshold) | (valid_results['ICIR'].abs() > effective_icir_threshold))
            )
            if enforce_novelty_filter and 'novelty_score' in valid_results.columns:
                novelty_mask = (
                    (valid_results['novelty_score'].fillna(0) >= min_novelty_score) &
                    (
                        valid_results['max_ref_corr'].isna() |
                        (valid_results['max_ref_corr'] <= max_ref_corr_for_good)
                    )
                )
                effective_mask = effective_mask & novelty_mask
            effective_factors = valid_results[effective_mask]
            
            # 不稳定因子（未通过时间一致性检查）
            unstable_factors = valid_results[valid_results['is_stable'] == False]
            
            # 坏因子（IC绝对值 < min_ic_threshold 且 ICIR绝对值 < min_icir_threshold）
            bad_factors_new = valid_results[
                (valid_results['IC'].abs() < effective_ic_threshold) & (valid_results['ICIR'].abs() < effective_icir_threshold)
            ]

            tradeability_series = valid_results['tradeability_score'] if 'tradeability_score' in valid_results.columns else pd.Series(0.0, index=valid_results.index)
            monotonicity_series = valid_results['monotonicity_score'] if 'monotonicity_score' in valid_results.columns else pd.Series(0.0, index=valid_results.index)
            novelty_series = valid_results['novelty_score'] if 'novelty_score' in valid_results.columns else pd.Series(1.0, index=valid_results.index)
            max_ref_corr_series = valid_results['max_ref_corr'] if 'max_ref_corr' in valid_results.columns else pd.Series(np.nan, index=valid_results.index)

            # 推荐分类：导出全量满足条件的因子，而不是只展示前几名
            high_quality_novel = valid_results[
                (valid_results['is_stable'] == True) &
                ((valid_results['IC'].abs() > effective_ic_threshold) | (valid_results['ICIR'].abs() > effective_icir_threshold)) &
                (tradeability_series.fillna(0) >= (0.40 if horizon_profile['horizon_bucket'] == 'short' else 0.35 if horizon_profile['horizon_bucket'] == 'long' else 0.45)) &
                (monotonicity_series.fillna(0) >= (0.25 if horizon_profile['horizon_bucket'] == 'short' else 0.20 if horizon_profile['horizon_bucket'] == 'long' else 0.30)) &
                (novelty_series.fillna(0) >= min_novelty_score) &
                (
                    max_ref_corr_series.isna() |
                    (max_ref_corr_series <= max_ref_corr_for_good)
                )
            ].copy()

            effective_but_redundant = valid_results[
                (valid_results['is_stable'] == True) &
                ((valid_results['IC'].abs() > effective_ic_threshold) | (valid_results['ICIR'].abs() > effective_icir_threshold)) &
                (
                    (novelty_series.fillna(0) < min_novelty_score) |
                    (
                        max_ref_corr_series.notna() &
                        (max_ref_corr_series > max_ref_corr_for_good)
                    )
                )
            ].copy()

            promising_but_unstable = valid_results[
                (valid_results['is_stable'] == False) &
                (
                    (valid_results['IC'].abs() > promising_ic_threshold) |
                    (valid_results['ICIR'].abs() > promising_icir_threshold) |
                    (tradeability_series.fillna(0) >= (0.40 if horizon_profile['horizon_bucket'] == 'short' else 0.35 if horizon_profile['horizon_bucket'] == 'long' else 0.45)) |
                    (monotonicity_series.fillna(0) >= (0.25 if horizon_profile['horizon_bucket'] == 'short' else 0.20 if horizon_profile['horizon_bucket'] == 'long' else 0.30))
                )
            ].copy()
            
            print(f"\n时间一致性统计:")
            print(f"  稳定因子数: {len(effective_factors)}")
            print(f"  不稳定因子数: {len(unstable_factors)}")
            print(f"  无效因子数 (|IC|<{effective_ic_threshold:.4f} 且 |ICIR|<{effective_icir_threshold:.4f}): {len(bad_factors_new)}")
            print(f"  高质量新因子: {len(high_quality_novel)}")
            print(f"  有效但重复度高: {len(effective_but_redundant)}")
            print(f"  有潜力但不稳定: {len(promising_but_unstable)}")
            if enforce_novelty_filter and 'novelty_score' in valid_results.columns:
                stable_base = valid_results[
                    (valid_results['is_stable'] == True) &
                    ((valid_results['IC'].abs() > effective_ic_threshold) | (valid_results['ICIR'].abs() > effective_icir_threshold))
                ]
                filtered_by_novelty = stable_base[
                    (stable_base['novelty_score'].fillna(0) < min_novelty_score) |
                    (
                        stable_base['max_ref_corr'].notna() &
                        (stable_base['max_ref_corr'] > max_ref_corr_for_good)
                    )
                ]
                print(f"  因新颖度被过滤: {len(filtered_by_novelty)}")
            if use_rolling_evaluation and 'rolling_valid_windows' in valid_results.columns:
                rolling_ready = valid_results[valid_results['rolling_valid_windows'] >= min_rolling_windows]
                if len(rolling_ready) > 0:
                    print(f"  滚动窗口可评估因子数: {len(rolling_ready)}")
                    print(f"  平均滚动稳定比例: {rolling_ready['rolling_stable_ratio'].mean():.2%}")
                    print(f"  平均滚动同号比例: {rolling_ready['rolling_same_sign_ratio'].mean():.2%}")
            if use_segment_evaluation and 'segment_valid_segments' in valid_results.columns:
                segment_ready = valid_results[valid_results['segment_valid_segments'] >= min_segment_count]
                if len(segment_ready) > 0:
                    print(f"  分段可评估因子数: {len(segment_ready)}")
                    print(f"  平均分段稳定比例: {segment_ready['segment_stable_ratio'].mean():.2%}")
                    print(f"  平均分段同号比例: {segment_ready['segment_same_sign_ratio'].mean():.2%}")

            # 分析不稳定的主要原因
            if len(unstable_factors) > 0:
                print(f"\n不稳定因子主要原因统计:")
                reasons_count = {}
                for idx, row in unstable_factors.iterrows():
                    reasons = row['stability_reason'].split('; ')
                    for reason in reasons:
                        # 提取主要原因（去掉具体数值）
                        main_reason = reason.split('(')[0] if '(' in reason else reason
                        reasons_count[main_reason] = reasons_count.get(main_reason, 0) + 1
                
                for reason, count in sorted(reasons_count.items(), key=lambda x: x[1], reverse=True):
                    print(f"    {reason}: {count} 个因子")
                
                print(f"\n不稳定因子详情（前10个，按IC绝对值排序）:")
                unstable_sorted = unstable_factors.copy()
                unstable_sorted['IC_abs'] = unstable_sorted['IC'].abs().fillna(0)
                unstable_sorted = unstable_sorted.sort_values('IC_abs', ascending=False)
                for idx, row in unstable_sorted.head(10).iterrows():
                    print(f"  {row['factor']:30s} | IC_train: {row['IC_train']:8.6f} | IC_valid: {row['IC_valid']:8.6f} | 衰减: {row['IC_decay']:7.2%} | 原因: {row['stability_reason']}")
            
            # 显示稳定因子中IC绝对值最大的前20个因子
            print(f"\n稳定因子中IC绝对值最大的前20个因子:")
            stable_results = valid_results[valid_results['is_stable'] == True].copy()
            if len(stable_results) > 0:
                stable_results['IC_abs'] = stable_results['IC'].abs().fillna(0)
                if 'tradeability_score' in stable_results.columns:
                    stable_results = stable_results.sort_values(['tradeability_score', 'monotonicity_score', 'novelty_score', 'IC_abs'], ascending=[False, False, False, False])
                else:
                    stable_results = stable_results.sort_values('IC_abs', ascending=False)
                top_factors = stable_results.head(20)
                for idx, row in top_factors.iterrows():
                    print(f"  {row['factor']:30s} | IC_train: {row['IC_train']:8.6f} | IC_valid: {row['IC_valid']:8.6f} | LS: {row.get('ls_mean', np.nan):8.6f} | Mono: {row.get('bucket_corr_mean', np.nan):6.3f} | Novel: {row.get('novelty_score', np.nan):5.3f} | ICIR: {row['ICIR']:8.4f}")
                    if pd.notna(row.get('closest_ref_factor')):
                        print(f"    closest_ref: {row.get('closest_ref_factor')} | corr={row.get('max_ref_corr', np.nan):.4f}")
                    if pd.notna(row.get('bucket_profile')):
                        print(f"    buckets: {row.get('bucket_profile')}")
                    print(f"    {row['expression']}")
            else:
                print("  没有稳定的因子")
                print("\n  建议:")
                print("  1. 降低阈值参数（min_ic_train, min_ic_valid, max_ic_decay）")
                print("  2. 设置 require_same_sign=False 允许IC符号反转")
                print("  3. 查看下面的'相对稳定因子'（即使不完全满足条件）")
                
                # 显示相对稳定的因子（即使不完全满足条件）
                print(f"\n  相对稳定的因子（按稳定性得分排序，前20个）:")
                # 计算相对稳定性：至少训练集或验证集IC达到阈值
                relative_stable = valid_results[
                    ((valid_results['IC_train'].abs() >= min_ic_train * 0.5) | 
                     (valid_results['IC_valid'].abs() >= min_ic_valid * 0.5)) &
                    (valid_results['IC'].abs() > 0.005)  # 至少整体IC>0.005
                ].copy()
                
                if len(relative_stable) > 0:
                    # 按稳定性得分和IC绝对值排序
                    relative_stable['IC_abs'] = relative_stable['IC'].abs().fillna(0)
                    relative_stable = relative_stable.sort_values(['stability_score', 'IC_abs'], ascending=[False, False])
                    top_relative = relative_stable.head(20)
                    for idx, row in top_relative.iterrows():
                        print(f"    {row['factor']:30s} | IC_train: {row['IC_train']:8.6f} | IC_valid: {row['IC_valid']:8.6f} | 衰减: {row['IC_decay']:7.2%} | 得分: {row['stability_score']:.2f}")
                else:
                    print("    没有相对稳定的因子")
            
            # 显示所有因子中IC绝对值最大的前20个因子（包括不稳定的）
            print(f"\n所有因子中IC绝对值最大的前20个因子（包括不稳定的）:")
            all_results = valid_results.copy()
            all_results['IC_abs'] = all_results['IC'].abs().fillna(0)
            if 'tradeability_score' in all_results.columns:
                all_results = all_results.sort_values(['IC_abs', 'tradeability_score', 'monotonicity_score', 'novelty_score'], ascending=[False, False, False, False])
            else:
                all_results = all_results.sort_values('IC_abs', ascending=False)
            top_factors = all_results.head(20)
            for idx, row in top_factors.iterrows():
                stable_mark = "✓" if row['is_stable'] else "⚠"
                print(f"  {stable_mark} {row['factor']:30s} | IC_train: {row['IC_train']:8.6f} | IC_valid: {row['IC_valid']:8.6f} | LS: {row.get('ls_mean', np.nan):8.6f} | Mono: {row.get('bucket_corr_mean', np.nan):6.3f} | Novel: {row.get('novelty_score', np.nan):5.3f} | ICIR: {row['ICIR']:8.4f}")
                if pd.notna(row.get('closest_ref_factor')):
                    print(f"    closest_ref: {row.get('closest_ref_factor')} | corr={row.get('max_ref_corr', np.nan):.4f}")
                if pd.notna(row.get('bucket_profile')):
                    print(f"    buckets: {row.get('bucket_profile')}")
                print(f"    {row['expression']}")
            
            # 显示有效因子
            if len(effective_factors) > 0:
                print(f"\n有效因子详情:")
                for idx, row in effective_factors.iterrows():
                    print(f"  {row['factor']:30s} | IC: {row['IC']:8.6f} | Rank_IC: {row['Rank_IC']:8.6f} | LS: {row.get('ls_mean', np.nan):8.6f} | Mono: {row.get('bucket_corr_mean', np.nan):6.3f} | Novel: {row.get('novelty_score', np.nan):5.3f} | ICIR: {row['ICIR']:8.4f} | 有效天数: {row['valid_days']}")
                    if pd.notna(row.get('closest_ref_factor')):
                        print(f"    closest_ref: {row.get('closest_ref_factor')} | corr={row.get('max_ref_corr', np.nan):.4f}")
                    if pd.notna(row.get('bucket_profile')):
                        print(f"    buckets: {row.get('bucket_profile')}")
                    print(f"    {row['expression']}")
            
            # 保存结果
            if output_dir is None:
                output_dir = Path.cwd()
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"{result_prefix}_results.csv"
            results_df.to_csv(output_file, index=False)
            print(f"\n结果已保存到: {output_file}")
            
            # 保存有效因子
            if len(effective_factors) > 0:
                effective_file = output_dir / f"effective_{result_prefix}.csv"
                effective_factors.to_csv(effective_file, index=False)
                print(f"有效因子已保存到: {effective_file}")

            # 保存全量推荐分类结果
            category_exports = [
                ("high_quality_novel", high_quality_novel),
                ("effective_but_redundant", effective_but_redundant),
                ("promising_but_unstable", promising_but_unstable),
            ]
            for category_name, category_df in category_exports:
                category_file = output_dir / f"{result_prefix}_{category_name}.csv"
                category_df.to_csv(category_file, index=False)
                print(f"{category_name} 已保存到: {category_file} (共 {len(category_df)} 个因子)")

            # 生成主清单：汇总三类推荐因子，便于统一排序和人工筛选
            shortlist_frames = []
            shortlist_specs = [
                ("high_quality_novel", high_quality_novel, 3.0),
                ("effective_but_redundant", effective_but_redundant, 2.0),
                ("promising_but_unstable", promising_but_unstable, 1.0),
            ]
            for category_name, category_df, category_weight in shortlist_specs:
                if len(category_df) == 0:
                    continue
                df_copy = category_df.copy()
                master_weights = scoring_profile["master_weights"]
                df_copy["recommendation_bucket"] = category_name
                df_copy["recommendation_weight"] = category_weight
                df_copy["horizon_bucket"] = horizon_profile["horizon_bucket"]
                df_copy["label_horizon"] = horizon_profile["label_horizon"]
                df_copy["master_score"] = (
                    category_weight +
                    df_copy.get("stability_score", 0).fillna(0) * master_weights["stability"] +
                    df_copy.get("tradeability_score", 0).fillna(0) * master_weights["tradeability"] +
                    df_copy.get("monotonicity_score", 0).fillna(0) * master_weights["monotonicity"] +
                    df_copy.get("novelty_score", 0).fillna(0) * master_weights["novelty"] +
                    df_copy.get("IC", 0).abs().fillna(0) * (3.0 * master_weights["ic"]) +
                    df_copy.get("ICIR", 0).abs().fillna(0) * 0.10
                )
                shortlist_frames.append(df_copy)

            if shortlist_frames:
                master_shortlist = pd.concat(shortlist_frames, ignore_index=True)
                master_shortlist = master_shortlist.sort_values(
                    ["master_score", "tradeability_score", "monotonicity_score", "novelty_score", "IC"],
                    ascending=[False, False, False, False, False],
                )
                master_shortlist = master_shortlist.drop_duplicates(subset=["factor"], keep="first")
                master_file = output_dir / f"{result_prefix}_master_shortlist.csv"
                master_shortlist.to_csv(master_file, index=False)
                print(f"master_shortlist 已保存到: {master_file} (共 {len(master_shortlist)} 个因子)")
            
            # 更新因子配置文件（长期目标专用）
            print("\n更新因子配置文件（长期目标）...")
            new_good_factors = {}
            new_bad_factors = {}
            factors_to_remove_from_good = []  # 从好因子中移除的因子（现在不稳定了）
            
            # 收集新发现的稳定好因子（不在配置文件中的，且通过时间一致性检查）
            # 同时保存性能参数
            # 提取good_factors的表达式字典用于检查
            good_expressions, _ = _extract_factor_info(good_factors)
            for idx, row in effective_factors.iterrows():
                factor_name = row['factor']
                if factor_name not in good_expressions:
                    # 保存因子表达式和性能参数
                    new_good_factors[factor_name] = {
                        'expression': row['expression'],
                        'ic': float(row.get('IC', 0.0)),
                        'icir': float(row.get('ICIR', 0.0)),
                        'ic_train': float(row.get('IC_train', 0.0)),
                        'ic_valid': float(row.get('IC_valid', 0.0)),
                        'icir_train': float(row.get('ICIR_train', 0.0)),
                        'icir_valid': float(row.get('ICIR_valid', 0.0)),
                        'ic_decay': float(row.get('IC_decay', 0.0)),
                        'stability_score': float(row.get('stability_score', 0.0)),
                        'ls_mean': float(row.get('ls_mean', 0.0)),
                        'ls_win_rate': float(row.get('ls_win_rate', 0.0)),
                        'tradeability_score': float(row.get('tradeability_score', 0.0)),
                        'bucket_corr_mean': float(row.get('bucket_corr_mean', 0.0)),
                        'monotonicity_score': float(row.get('monotonicity_score', 0.0)),
                        'bucket_profile': row.get('bucket_profile'),
                        'bucket_profile_coverage_days': int(row.get('bucket_profile_coverage_days', 0)),
                        'closest_ref_factor': row.get('closest_ref_factor'),
                        'max_ref_corr': float(row.get('max_ref_corr', 0.0)),
                        'mean_ref_corr': float(row.get('mean_ref_corr', 0.0)),
                        'novelty_score': float(row.get('novelty_score', 0.0)),
                        'is_stable': bool(row.get('is_stable', False)),
                    }
                    print(f"  ✓ 发现稳定因子: {factor_name} (IC_train: {row['IC_train']:.6f}, IC_valid: {row['IC_valid']:.6f}, 衰减: {row['IC_decay']:.2%})")
            
            # 收集不稳定的因子（未通过时间一致性检查）
            for idx, row in unstable_factors.iterrows():
                factor_name = row['factor']
                factor_expr = row['expression']
                
                # 如果因子之前是好因子，现在不稳定了，需要从好因子中移除
                if factor_name in good_expressions:
                    factors_to_remove_from_good.append(factor_name)
                    print(f"  ⚠ 因子 {factor_name} 之前是好因子，现在不稳定，将从好因子中移除 (原因: {row['stability_reason']})")
                
                # 如果因子不在坏因子列表中，加入坏因子
                if factor_name not in bad_expressions:
                    new_bad_factors[factor_name] = factor_expr
                    print(f"  ✗ 发现不稳定因子: {factor_name} (原因: {row['stability_reason']})")
            
            # 收集新发现的坏因子（不在配置文件中的）
            # 包括：1) IC/ICIR不达标的因子 2) 测试失败的因子（错误、数据不足等）
            for idx, row in bad_factors_new.iterrows():
                factor_name = row['factor']
                factor_expr = row['expression']
                
                # 如果因子之前是好因子，现在IC/ICIR不达标了，需要从好因子中移除
                if factor_name in good_expressions:
                    factors_to_remove_from_good.append(factor_name)
                    print(f"  ⚠ 因子 {factor_name} 之前是好因子，现在IC/ICIR不达标，将从好因子中移除")
                
                # 如果因子不在坏因子列表中，加入坏因子
                if factor_name not in bad_expressions:
                    new_bad_factors[factor_name] = factor_expr
            
            # 收集所有测试失败的因子（包括错误、数据不足等）
            failed_factors = results_df[results_df['status'].isin(['no_data', 'no_common_index', 'insufficient_data', 'insufficient_days', 'error'])]
            for idx, row in failed_factors.iterrows():
                factor_name = row['factor']
                factor_expr = row['expression']
                
                # 如果因子不在好因子和坏因子列表中，且测试失败，则加入坏因子
                if factor_name not in bad_expressions and factor_name not in good_expressions:
                    new_bad_factors[factor_name] = factor_expr
                    print(f"  ✗ 测试失败的因子: {factor_name} (状态: {row['status']})")
            
            # 确保所有测试过的因子都被记录（包括IC/ICIR不达标的）
            # 对于成功测试但IC/ICIR不达标的因子，也要加入到坏因子列表
            effective_factor_names = set(effective_factors['factor'].values) if len(effective_factors) > 0 else set()
            unstable_factor_names = set(unstable_factors['factor'].values) if len(unstable_factors) > 0 else set()
            bad_factors_new_names = set(bad_factors_new['factor'].values) if len(bad_factors_new) > 0 else set()
            failed_factor_names = set(failed_factors['factor'].values) if len(failed_factors) > 0 else set()
            
            # 收集所有未分类的因子（确保没有遗漏）
            all_processed_factor_names = effective_factor_names | unstable_factor_names | bad_factors_new_names | failed_factor_names
            
            for idx, row in valid_results.iterrows():
                factor_name = row['factor']
                factor_expr = row['expression']
                
                # 如果因子不在任何已处理的列表中，且不在配置文件中，则加入坏因子
                if (factor_name not in all_processed_factor_names and
                    factor_name not in bad_expressions and 
                    factor_name not in good_expressions):
                    new_bad_factors[factor_name] = factor_expr
                    print(f"  ✗ 未分类因子加入坏因子: {factor_name}")
            
            # 额外检查：确保所有测试过的因子（包括status不是success的）都被记录
            new_good_expressions, _ = _extract_factor_info(new_good_factors)
            new_bad_expressions, _ = _extract_factor_info(new_bad_factors)
            all_tested_factor_names = set(results_df['factor'].values)
            for factor_name in all_tested_factor_names:
                if (factor_name not in good_expressions and 
                    factor_name not in bad_expressions and
                    factor_name not in new_good_expressions and
                    factor_name not in new_bad_expressions):
                    # 从结果中获取因子表达式
                    factor_row = results_df[results_df['factor'] == factor_name]
                    if len(factor_row) > 0:
                        factor_expr = factor_row.iloc[0]['expression']
                        new_bad_factors[factor_name] = factor_expr
                        print(f"  ✗ 遗漏的因子加入坏因子: {factor_name}")
            
            # 从好因子中移除不稳定的因子
            if factors_to_remove_from_good:
                config = load_factor_config(config_path)
                for factor_name in factors_to_remove_from_good:
                    if factor_name in config.get('good_factors', {}):
                        factor_value = config['good_factors'].pop(factor_name)
                        # 提取表达式（支持新旧格式）
                        if isinstance(factor_value, str):
                            factor_expr = factor_value
                        elif isinstance(factor_value, dict) and 'expression' in factor_value:
                            factor_expr = factor_value['expression']
                        else:
                            factor_expr = str(factor_value)
                        
                        if 'bad_factors' not in config or config['bad_factors'] is None:
                            config['bad_factors'] = {}
                        config['bad_factors'][factor_name] = factor_expr
                save_factor_config(config, config_path)
                print(f"  ✓ 已从好因子中移除 {len(factors_to_remove_from_good)} 个现在不稳定的因子")
            
            # 统计信息
            print(f"\n因子分类统计:")
            print(f"  新发现的好因子: {len(new_good_factors)}")
            print(f"  新发现的坏因子: {len(new_bad_factors)}")
            print(f"  从好因子中移除: {len(factors_to_remove_from_good)}")
            
            # 更新配置文件
            if new_good_factors or new_bad_factors:
                update_factor_config(
                    new_good_factors=new_good_factors if new_good_factors else None,
                    new_bad_factors=new_bad_factors if new_bad_factors else None,
                    config_path=config_path,
                )
            else:
                print("✓ 没有新的因子需要更新到配置文件")
            
            # 对好因子进行相关性去重
            print("\n对好因子进行相关性去重...")
            try:
                # 合并原有的好因子和新发现的好因子
                all_good_factors = {**good_factors, **new_good_factors}
                
                if len(all_good_factors) > 1:
                    # 传入factor_config以便从配置文件读取性能参数
                    deduplicated_factors, removed_factors = deduplicate_good_factors_by_correlation(
                        good_factors=all_good_factors,
                        instruments_list=instruments_list,  # 使用固定股票池列表
                        start_time=start_time,
                        end_time=end_time,
                        freq=freq,
                        correlation_threshold=0.90,
                        ic_results=valid_results,
                        factor_config=factor_config  # 传入配置文件，用于读取性能参数
                    )
                    
                    if len(deduplicated_factors) < len(all_good_factors):
                        config = load_factor_config(config_path)
                        
                        # 保存去重后的因子，同时保留性能参数
                        # 从原始配置中提取性能参数
                        _, original_metadata = _extract_factor_info(config.get('good_factors', {}))
                        _, new_metadata = _extract_factor_info(new_good_factors)
                        
                        # 从valid_results中提取性能参数（如果有）
                        ic_results_metadata = {}
                        if valid_results is not None and len(valid_results) > 0:
                            for _, row in valid_results.iterrows():
                                factor_name = row['factor']
                                if factor_name in deduplicated_factors:
                                    ic_results_metadata[factor_name] = {
                                        'ic': float(row.get('IC', 0.0)),
                                        'icir': float(row.get('ICIR', 0.0)),
                                        'ic_train': float(row.get('IC_train', 0.0)),
                                        'ic_valid': float(row.get('IC_valid', 0.0)),
                                        'icir_train': float(row.get('ICIR_train', 0.0)),
                                        'icir_valid': float(row.get('ICIR_valid', 0.0)),
                                        'ic_decay': float(row.get('IC_decay', 0.0)),
                                        'stability_score': float(row.get('stability_score', 0.0)),
                                        'ls_mean': float(row.get('ls_mean', 0.0)),
                                        'ls_win_rate': float(row.get('ls_win_rate', 0.0)),
                                        'tradeability_score': float(row.get('tradeability_score', 0.0)),
                                        'bucket_corr_mean': float(row.get('bucket_corr_mean', 0.0)),
                                        'monotonicity_score': float(row.get('monotonicity_score', 0.0)),
                                        'bucket_profile': row.get('bucket_profile'),
                                        'bucket_profile_coverage_days': int(row.get('bucket_profile_coverage_days', 0)),
                                        'closest_ref_factor': row.get('closest_ref_factor'),
                                        'max_ref_corr': float(row.get('max_ref_corr', 0.0)),
                                        'mean_ref_corr': float(row.get('mean_ref_corr', 0.0)),
                                        'novelty_score': float(row.get('novelty_score', 0.0)),
                                        'is_stable': bool(row.get('is_stable', False)),
                                    }
                        
                        # 合并性能参数：优先使用valid_results中的最新数据，其次使用配置文件中的，最后使用new_metadata
                        merged_metadata = {}
                        for factor_name in deduplicated_factors.keys():
                            if factor_name in ic_results_metadata:
                                merged_metadata[factor_name] = ic_results_metadata[factor_name]
                            elif factor_name in original_metadata and original_metadata[factor_name]:
                                merged_metadata[factor_name] = original_metadata[factor_name]
                            elif factor_name in new_metadata and new_metadata[factor_name]:
                                merged_metadata[factor_name] = new_metadata[factor_name]
                        
                        # 构建新的good_factors，包含表达式和性能参数
                        updated_good_factors = {}
                        for factor_name, factor_expr in deduplicated_factors.items():
                            if factor_name in merged_metadata and merged_metadata[factor_name]:
                                # 如果有性能参数，保存为新格式
                                updated_good_factors[factor_name] = {
                                    'expression': factor_expr,
                                    **merged_metadata[factor_name]
                                }
                            else:
                                # 如果没有性能参数，保存为旧格式（向后兼容）
                                updated_good_factors[factor_name] = factor_expr
                        
                        config['good_factors'] = updated_good_factors
                        
                        # 将被去除的因子加入到坏因子列表
                        if removed_factors:
                            if 'bad_factors' not in config or config['bad_factors'] is None:
                                config['bad_factors'] = {}
                            config['bad_factors'].update(removed_factors)
                            print(f"✓ 已将 {len(removed_factors)} 个去重移除的因子加入到坏因子列表")
                        
                        save_factor_config(config, config_path)
                        print(f"✓ 已更新配置文件，保留 {len(deduplicated_factors)} 个去重后的好因子（包含性能参数）")
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
    print("自定义因子相关性分析（带时间一致性检查）- 沪深1000")
    print("="*80)
    
    # 配置参数
    instruments = "csi300"  # 使用沪深1000进行因子挖掘
    
    # 自定义 label 表达式（预测目标）
    # 常用模板：
    # - make_future_avg_return_label_expr(5): 未来1-5日相对当前价格的平均收益率
    # - make_future_avg_price_label_expr(5): 未来5日平均价格相对当前价格
    # - make_future_avg_price_label_expr(20): 未来20日平均价格相对当前价格
    label_expr = make_future_avg_return_label_expr(5)
    
    # 自定义因子（可选，如果为None则使用默认因子列表）
    custom_factors = None
    # 示例：
    # custom_factors = {
    #     "Momentum_5": "Ref($close, 5)/$close - 1",
    #     "MA_Ratio_20": "$close / Mean($close, 20) - 1",
    # }
    
    # 时间范围
    start_time = "2015-01-01"
    end_time = "2025-12-31"
    
    # 训练集和验证集时间范围（用于时间一致性检查）
    train_start_time = "2015-01-01"  # 训练集开始时间
    train_end_time = "2025-01-01"    # 训练集结束时间（也是验证集开始时间）
    valid_start_time = "2025-01-01"  # 验证集开始时间
    valid_end_time = "2025-12-31"    # 验证集结束时间
    
    # 时间一致性检查参数
    # 注意：如果找不到稳定因子，可以适当放宽这些阈值
    min_ic_train = 0.006     # 训练集最小IC阈值（绝对值），进一步降低以找到更多稳定因子
    min_ic_valid = 0.005     # 验证集最小IC阈值（绝对值），进一步降低
    max_ic_decay = 1.0       # 最大IC衰减率（1.0表示允许验证集IC比训练集IC低100%，即允许符号反转）
    require_same_sign = False # 是否要求训练集和验证集IC符号一致（设为False允许符号反转）
    
    # 滚动时段评估参数（本次先关闭，避免过严筛掉“分段稳定但滚动略抖”的因子）
    use_rolling_evaluation = False
    rolling_train_years = 5
    rolling_valid_years = 2
    rolling_step_months = 12
    min_rolling_windows = 3

    # 分段稳定性评估参数（缓解验证期过短的问题）
    use_segment_evaluation = True
    segment_months = 6
    min_segment_count = 4
    segment_min_ic_abs = None  # None 表示跟随 min_ic_valid
    force_full_scan = False
    enforce_novelty_filter = False
    min_novelty_score = 0.2
    max_ref_corr_for_good = 0.8
    use_result_cache = True
    resume_from_cache = True
    
    # 数据频率
    freq = "day"  # 日线数据
    
    # 限制数据天数（可选，用于加速测试）
    limit_data_days = None  # 例如：365 表示只使用最近1年的数据
    
    # 输出目录
    output_dir = Path(__file__).parent / "factor_evaluation_results"
    
    # 执行分析
    results = analyze_custom_factors(
        instruments=instruments,
        label_expr=label_expr,
        custom_factors=custom_factors,
        config_path=Path(__file__).parent / "factor_config.yaml",
        result_prefix="custom_factors",
        start_time=start_time,
        end_time=end_time,
        freq=freq,
        limit_data_days=limit_data_days,
        output_dir=output_dir,
        train_start_time=train_start_time,
        train_end_time=train_end_time,
        valid_start_time=valid_start_time,
        valid_end_time=valid_end_time,
        min_ic_train=min_ic_train,
        min_ic_valid=min_ic_valid,
        max_ic_decay=max_ic_decay,
        require_same_sign=require_same_sign,
        use_rolling_evaluation=use_rolling_evaluation,
        rolling_train_years=rolling_train_years,
        rolling_valid_years=rolling_valid_years,
        rolling_step_months=rolling_step_months,
        min_rolling_windows=min_rolling_windows,
        use_segment_evaluation=use_segment_evaluation,
        segment_months=segment_months,
        min_segment_count=min_segment_count,
        segment_min_ic_abs=segment_min_ic_abs,
        force_full_scan=force_full_scan,
        enforce_novelty_filter=enforce_novelty_filter,
        min_novelty_score=min_novelty_score,
        max_ref_corr_for_good=max_ref_corr_for_good,
        use_result_cache=use_result_cache,
        resume_from_cache=resume_from_cache,
    )
    
    if results is not None:
        print("\n" + "="*80)
        print("分析完成！")
        print("="*80)
    else:
        print("\n分析失败，请检查错误信息。")


if __name__ == "__main__":
    main()
