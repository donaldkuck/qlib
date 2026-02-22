#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
仅回测：从已有实验 run 加载模型，按当前回测周期重新生成预测后再回测（可调回测周期与参数）。

默认会加载模型并重新生成预测，这样修改回测周期（--start_time/--end_time 或 backtest_config）
时预测会覆盖该周期，结果正确。若仅想用源 run 已保存的预测快速回测且不改周期，可用 --use_saved_pred。

用法：
  # 加载模型，按回测配置的周期重新生成预测后回测（推荐）
  python run_backtest_only.py --run_id <run_id>

  # 指定回测周期（覆盖 workflow/backtest_config 中的 start_time、end_time）
  python run_backtest_only.py --run_id <run_id> --start_time 2025-01-01 --end_time 2025-06-30

  # 使用自定义回测参数（YAML 中只写要覆盖的项，含 backtest 周期）
  python run_backtest_only.py --run_id <run_id> --backtest_config backtest_config.yaml

  # 不重新生成预测，直接使用源 run 中已保存的预测（仅当回测周期与源 run 一致时适用）
  python run_backtest_only.py --run_id <run_id> --use_saved_pred

如何查看 run_id（必为 MLflow 的 run UUID，即 32 位十六进制目录名）：
  - 训练完成后会打印：记录ID: xxx（即 run_id）、结果路径: mlruns/<experiment_id>/<run_id>/
  - 或到 mlruns/ 下先找实验目录（如 673062759933473454 对应 long_short_factor_strategy），
    再进入该目录，其子目录名即为 run_id，例如：0b8e2395033a4264803061c0f76b63c8
  - 注意：不要用数字形式的 ID（如 105947515894513072），必须用上述十六进制目录名

依赖：
  - 指定 run 中已存在训练好的模型：long_term_params.pkl、short_term_params.pkl（或 n-fold 的 *_all_models.pkl）
  - 可选 backtest_config.yaml：只写需要覆盖的 strategy.kwargs 和 backtest 字段（含 start_time/end_time）
"""
import argparse
import copy
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")

import logging
logging.getLogger("qlib.online operator").setLevel(logging.ERROR)
logging.getLogger("qlib.BaseExecutor").setLevel(logging.ERROR)

current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
qlib_local_path = project_root / "qlib"
# 只移除明确指向本地 qlib 源码的路径，避免误删 venv 中已安装的 qlib
try:
    _qlib_local = str(qlib_local_path.resolve())
    _proj_root = str(project_root.resolve())
    for p in list(sys.path):
        try:
            p_resolved = str(Path(p).resolve()) if p else ""
        except Exception:
            p_resolved = str(p)
        if p not in sys.path:
            continue
        # 移除：项目根目录（会令 import qlib 找到 myproject/qlib）或本地 qlib 目录本身
        if p_resolved == _proj_root or p_resolved == _qlib_local or p_resolved.startswith(_qlib_local + "/") or p_resolved.startswith(_qlib_local + "\\"):
            sys.path.remove(p)
except Exception:
    pass

import qlib
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# 保证本目录在 path 中，以便加载 long_short_strategy
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import yaml
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord
from qlib.utils import init_instance_by_config
from qlib.utils.exceptions import LoadObjectError
from qlib.contrib.report.analysis_position import report_graph, risk_analysis_graph
from qlib.contrib.evaluate import risk_analysis


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """递归合并 override 到 base，override 优先。不修改 base，返回新 dict。"""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _predict_with_models(
    model_or_list: Union[Any, List[Any]],
    dataset: Any,
    ensemble_method: str = "mean",
) -> pd.DataFrame:
    """单模型或 n-fold 集成预测，返回与 strategy 兼容的 DataFrame/Series。"""
    if isinstance(model_or_list, list):
        if len(model_or_list) == 0:
            raise ValueError("模型列表为空")
        if len(model_or_list) == 1:
            pred = model_or_list[0].predict(dataset)
        else:
            preds = []
            for m in model_or_list:
                p = m.predict(dataset)
                if isinstance(p, pd.DataFrame):
                    p = p.iloc[:, 0] if len(p.columns) > 0 else p.squeeze()
                preds.append(p)
            common_idx = preds[0].index
            for p in preds[1:]:
                common_idx = common_idx.intersection(p.index)
            aligned = [p.loc[common_idx] for p in preds]
            if ensemble_method == "median":
                pred = pd.concat(aligned, axis=1).median(axis=1)
            else:
                pred = pd.concat(aligned, axis=1).mean(axis=1)
    else:
        pred = model_or_list.predict(dataset)
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    return pred


def _resolve_handler_paths(config: Dict, base_dir: Path) -> None:
    """将 dataset kwargs 中 handler.kwargs 的相对路径解析为 base_dir 下的绝对路径（就地修改）。"""
    task = config.get("task", {})
    for key in ("short_term_dataset", "long_term_dataset"):
        if key not in task:
            continue
        kwargs = task[key].get("kwargs", {})
        handler = kwargs.get("handler", {})
        if not isinstance(handler, dict):
            continue
        hkw = handler.get("kwargs", {})
        if not isinstance(hkw, dict):
            continue
        for path_key in ("custom_factors_file",):
            if path_key in hkw and hkw[path_key]:
                p = Path(hkw[path_key])
                if not p.is_absolute():
                    hkw[path_key] = str((base_dir / p).resolve())
    return None


def _build_datasets_for_period(
    base_config: Dict,
    start_time: str,
    end_time: str,
    base_dir: Path,
) -> tuple:
    """根据 base_config 的 task 构建短期/长期 dataset，test 段为 [start_time, end_time]。"""
    task = base_config.get("task", {})
    short_cfg = copy.deepcopy(task["short_term_dataset"])
    short_cfg["kwargs"] = copy.deepcopy(short_cfg["kwargs"])
    short_cfg["kwargs"]["segments"] = copy.deepcopy(short_cfg["kwargs"]["segments"])
    short_cfg["kwargs"]["segments"]["test"] = [start_time, end_time]
    long_cfg = copy.deepcopy(task["long_term_dataset"])
    long_cfg["kwargs"] = copy.deepcopy(long_cfg["kwargs"])
    long_cfg["kwargs"]["segments"] = copy.deepcopy(long_cfg["kwargs"]["segments"])
    long_cfg["kwargs"]["segments"]["test"] = [start_time, end_time]
    _resolve_handler_paths({"task": {"short_term_dataset": short_cfg, "long_term_dataset": long_cfg}}, base_dir)
    short_ds = init_instance_by_config(short_cfg)
    long_ds = init_instance_by_config(long_cfg)
    return short_ds, long_ds


def _load_models_and_regenerate_pred(
    source_recorder: Any,
    base_config: Dict,
    start_time: str,
    end_time: str,
    base_dir: Path,
) -> tuple:
    """从 source_recorder 加载模型，按 [start_time, end_time] 重新生成长期/短期预测。返回 (long_term_pred, short_term_pred, label_pkl)。"""
    use_nfold = base_config.get("task", {}).get("use_nfold", False)
    ensemble_method = base_config.get("task", {}).get("ensemble_method", "mean")
    short_ds, long_ds = _build_datasets_for_period(base_config, start_time, end_time, base_dir)
    short_ds.setup_data()
    long_ds.setup_data()
    try:
        long_models = source_recorder.load_object("long_term_all_models.pkl")
    except (LoadObjectError, Exception):
        long_models = source_recorder.load_object("long_term_params.pkl")
    try:
        short_models = source_recorder.load_object("short_term_all_models.pkl")
    except (LoadObjectError, Exception):
        short_models = source_recorder.load_object("short_term_params.pkl")
    if long_models is None:
        raise FileNotFoundError("源 run 中未找到长期模型（long_term_params.pkl 或 long_term_all_models.pkl）")
    if short_models is None:
        raise FileNotFoundError("源 run 中未找到短期模型（short_term_params.pkl 或 short_term_all_models.pkl）")
    long_term_pred = _predict_with_models(long_models, long_ds, ensemble_method)
    short_term_pred = _predict_with_models(short_models, short_ds, ensemble_method)
    try:
        
        from qlib.data.dataset.handler import DataHandlerLP
        label_data = long_ds.handler.fetch(selector=slice(start_time, end_time), col_set=["label"], data_key=DataHandlerLP.DK_R)
        if hasattr(label_data, "iloc"):
            label_pkl = label_data.iloc[:, 0] if label_data.ndim > 1 else label_data
        else:
            label_pkl = label_data.squeeze() if hasattr(label_data, "squeeze") else label_data
    except Exception:
        label_pkl = pd.DataFrame()
    return long_term_pred, short_term_pred, label_pkl


def load_port_analysis(recorder, par: Optional[PortAnaRecord] = None) -> Optional[Dict[str, Any]]:
    """从 recorder 加载回测结果。"""
    for path in [
        "portfolio_analysis/port_analysis_1day.pkl",
        "port_analysis_1day.pkl",
        "portfolio_analysis/port_analysis.pkl",
        "port_analysis.pkl",
    ]:
        try:
            obj = recorder.load_object(path)
            if obj is not None:
                return obj
        except Exception:
            continue
    if par is not None:
        for path in [
            "portfolio_analysis/port_analysis_1day.pkl",
            "port_analysis_1day.pkl",
            "portfolio_analysis/port_analysis.pkl",
            "port_analysis.pkl",
        ]:
            try:
                obj = par.load(path)
                if obj is not None:
                    return obj
            except Exception:
                continue
    return None


def load_report_and_positions(recorder: Any) -> tuple:
    """从 recorder 加载每日 report 和 positions（用于交易日志与单股统计）。返回 (report_df, positions_dict)。"""
    report_df = None
    positions_dict = None
    for freq_key in ("1day", "day"):
        for prefix in ("", "portfolio_analysis/"):
            try:
                r = recorder.load_object(f"{prefix}report_normal_{freq_key}.pkl")
                p = recorder.load_object(f"{prefix}positions_normal_{freq_key}.pkl")
                if r is not None and p is not None:
                    return (r, p)
            except Exception:
                continue
    return (report_df, positions_dict)


def _get_position_value_dict(position: Any) -> Dict[str, float]:
    """从 Position 对象得到 {stock_id: 市值}。"""
    out = {}
    try:
        for code in position.get_stock_list():
            amt = position.get_stock_amount(code)
            price = position.get_stock_price(code)
            if amt and price and amt > 0 and price > 0:
                out[code] = amt * price
    except Exception:
        pass
    return out


def _get_close_prices_for_dates(
    stock_ids: List[str],
    dates: List[pd.Timestamp],
) -> Dict[tuple, float]:
    """
    批量获取多只股票在多个日期的收盘价，返回 (date, stock_id) -> price。
    使用 qlib D.features，需在 qlib 已初始化后调用。qlib 返回的 DataFrame 通常为
    MultiIndex(instrument, datetime)，即 idx=(inst, dt)。
    """
    if not stock_ids or not dates:
        return {}
    try:
        from qlib.data import D
        start = min(dates).strftime("%Y-%m-%d")
        end = max(dates).strftime("%Y-%m-%d")
        df = D.features(stock_ids, ["$close"], start_time=start, end_time=end, freq="day")
        if df is None or df.empty:
            return {}
        out = {}
        col = "$close" if "$close" in df.columns else df.columns[0]
        for idx in df.index:
            if isinstance(idx, tuple) and len(idx) == 2:
                inst, dt = idx[0], idx[1]
                dt_n = pd.Timestamp(dt).normalize()
                if inst not in stock_ids:
                    continue
                try:
                    v = df.loc[idx, col]
                except Exception:
                    v = df.loc[idx].iloc[0]
                if pd.notna(v):
                    out[(dt_n, inst)] = float(v)
            else:
                dt_n = pd.Timestamp(idx).normalize()
                for sid in stock_ids:
                    if sid not in df.columns:
                        continue
                    try:
                        v = df.loc[idx, sid]
                    except Exception:
                        v = df.loc[idx].iloc[0] if hasattr(df.loc[idx], "iloc") else df.loc[idx]
                    if pd.notna(v):
                        out[(dt_n, sid)] = float(v)
        return out
    except Exception:
        return {}


def log_daily_trades_and_stock_stats(
    recorder: Any,
    log_dir: Optional[Union[str, Path]] = None,
) -> None:
    """
    将每日交易记录写入日志（含盈亏、盈亏率），并统计单只股票的交易次数与盈亏率，打印并可选写入文件。
    """
    report_df, positions_dict = load_report_and_positions(recorder)
    if report_df is None or report_df.empty:
        print("⚠️ 未找到 report_normal，跳过交易日志与单股统计")
        return
    log_dir = Path(log_dir) if log_dir else current_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = log_dir / f"backtest_trades_{ts}.log"
    file_handlers = []

    def _log(msg: str, to_file: bool = True) -> None:
        if to_file:
            for f in file_handlers:
                f.write(msg + "\n")
                f.flush()

    try:
        fh = open(log_file, "w", encoding="utf-8")
        file_handlers.append(fh)
    except Exception as e:
        print(f"⚠️ 无法创建日志文件 {log_file}: {e}")
        file_handlers.clear()

    # 每日交易记录（盈亏、盈亏率、当日买入/卖出/持仓股票列表）
    _log("=" * 80)
    _log("每日交易记录（含当日盈亏、盈亏率、买入/卖出/持仓股票列表）")
    _log("=" * 80)
    report_df = report_df.sort_index()
    # 用于按日期查找 position（positions 的 key 可能是 Timestamp，统一用 date 匹配）
    pos_dates = sorted(positions_dict.keys()) if positions_dict else []
    date_to_pos = {}
    if positions_dict:
        for d in pos_dates:
            k = pd.Timestamp(d).normalize()
            date_to_pos[k] = positions_dict[d]

    prev_account = None
    prev_date = None
    for date in report_df.index:
        row = report_df.loc[date]
        account = float(row.get("account", 0) or 0)
        ret = float(row.get("return", 0) or 0)
        cost = float(row.get("cost", 0) or 0)
        total_cost = float(row.get("total_cost", 0) or 0)
        if prev_account is not None and prev_account > 0:
            daily_pnl = account - prev_account
            daily_pnl_rate = daily_pnl / prev_account
        else:
            daily_pnl = 0.0
            daily_pnl_rate = 0.0
        date_ts = pd.Timestamp(date).normalize()
        pos_today = date_to_pos.get(date_ts)
        pos_prev = date_to_pos.get(prev_date) if prev_date is not None else None

        # 当日买入、卖出、持仓列表
        buy_list: Dict[str, float] = {}
        sell_list: Dict[str, float] = {}
        hold_list: List[str] = []
        if pos_today is not None:
            hold_list = sorted([c for c in (pos_today.get_stock_list() or []) if (pos_today.get_stock_amount(c) or 0) > 0])
            if pos_prev is not None:
                all_codes = set(pos_today.get_stock_list() or []) | set(pos_prev.get_stock_list() or [])
                for code in all_codes:
                    try:
                        amt_prev = pos_prev.get_stock_amount(code) or 0
                        amt_now = pos_today.get_stock_amount(code) or 0
                        price = pos_today.get_stock_price(code) or 0
                    except Exception:
                        amt_prev, amt_now, price = 0, 0, 0
                    delta = amt_now - amt_prev
                    if delta > 1e-6:
                        # 买入：增加的市值
                        buy_value = delta * price if price > 0 else 0
                        buy_list[code] = buy_value
                    elif delta < -1e-6:
                        # 卖出：减少的市值
                        sell_value = abs(delta) * price if price > 0 else 0
                        sell_list[code] = sell_value
                # 按金额排序
                buy_list = dict(sorted(buy_list.items(), key=lambda x: -x[1]))
                sell_list = dict(sorted(sell_list.items(), key=lambda x: -x[1]))
            else:
                # 首日：全部视为买入，使用当前市值
                for code in hold_list:
                    try:
                        amt = pos_today.get_stock_amount(code) or 0
                        price = pos_today.get_stock_price(code) or 0
                        buy_list[code] = amt * price
                    except Exception:
                        pass
                buy_list = dict(sorted(buy_list.items(), key=lambda x: -x[1]))

        date_str = date_ts.strftime("%Y-%m-%d")
        _log(
            f"  {date_str}  账户: {account:,.0f}  当日收益: {ret*100:.4f}%  成本: {total_cost:,.2f}  "
            f"当日盈亏: {daily_pnl:+,.2f}  当日盈亏率: {daily_pnl_rate*100:+.4f}%"
        )
        # 格式化买入列表（含金额）
        if buy_list:
            buy_str = ", ".join([f"{code}({v:,.0f})" for code, v in buy_list.items()])
            total_buy = sum(buy_list.values())
            _log(f"      买入: {buy_str}  合计: {total_buy:,.0f}")
        else:
            _log("      买入: []")
        # 格式化卖出列表（含金额）
        if sell_list:
            sell_str = ", ".join([f"{code}({v:,.0f})" for code, v in sell_list.items()])
            total_sell = sum(sell_list.values())
            _log(f"      卖出: {sell_str}  合计: {total_sell:,.0f}")
        else:
            _log("      卖出: []")
        # 持仓列表显示市值
        if hold_list and pos_today is not None:
            hold_values = {}
            for code in hold_list:
                try:
                    amt = pos_today.get_stock_amount(code) or 0
                    price = pos_today.get_stock_price(code) or 0
                    hold_values[code] = amt * price
                except Exception:
                    pass
            hold_str = ", ".join([f"{code}({v:,.0f})" for code, v in hold_values.items()])
            total_hold = sum(hold_values.values())
            _log(f"      持仓: {hold_str}  合计: {total_hold:,.0f}")
        else:
            _log("      持仓: []")
        prev_account = account
        prev_date = date_ts
    _log("")

    # 单只股票统计：交易次数、盈亏、盈亏率（买卖与期末市值均用当日收盘价，与组合总收益一致）
    if positions_dict and len(positions_dict) >= 2:
        dates = sorted(positions_dict.keys())
        date_list = [pd.Timestamp(d).normalize() for d in dates]
        all_stock_ids = []
        for d in dates:
            all_stock_ids.extend(positions_dict[d].get_stock_list() or [])
        all_stock_ids = list(set(all_stock_ids))
        close_cache = _get_close_prices_for_dates(all_stock_ids, date_list)

        def _close_at(date_ts, code: str) -> float:
            k = (pd.Timestamp(date_ts).normalize(), code)
            if k in close_cache:
                return close_cache[k]
            try:
                pos = date_to_pos.get(k[0]) or positions_dict.get(date_ts)
                if pos and code in (pos.get_stock_list() or []):
                    return float(pos.get_stock_price(code) or 0)
            except Exception:
                pass
            return 0.0

        stock_buy_value: Dict[str, float] = {}
        stock_sell_value: Dict[str, float] = {}
        stock_trade_count: Dict[str, int] = {}
        stock_sell_count: Dict[str, int] = {}       # 卖出次数（有减仓的交易日数）
        stock_sell_win_count: Dict[str, int] = {}   # 盈利的卖出次数（卖出价 > 当前持仓成本）
        # 当前持仓成本（仅针对尚未卖出的部分）：买入时增加，卖出时按卖出比例减少
        stock_cost_basis: Dict[str, float] = {}
        # 胜率定义：每次卖出时，若 卖出价 > 当时持仓的平均成本，则计为盈利一次（用持仓成本基础算平均成本）

        # 先计入首日持仓（视为买入），并初始化持仓成本基础
        first_date = pd.Timestamp(dates[0]).normalize()
        first_pos = positions_dict[dates[0]]
        for code in first_pos.get_stock_list() or []:
            try:
                amt = first_pos.get_stock_amount(code) or 0
                if amt <= 0:
                    continue
                price = _close_at(first_date, code)
                if price <= 0:
                    price = first_pos.get_stock_price(code) or 0
                v = amt * price
                if v > 0:
                    stock_buy_value[code] = stock_buy_value.get(code, 0) + v
                    stock_trade_count[code] = stock_trade_count.get(code, 0) + 1
                    stock_cost_basis[code] = v  # 持仓成本 = 首日建仓市值
            except Exception:
                pass

        for i in range(1, len(dates)):
            date = dates[i]
            date_ts = pd.Timestamp(date).normalize()
            pos = positions_dict[date]
            pos_prev = positions_dict[dates[i - 1]]
            for code in set(pos.get_stock_list() or []) | set(pos_prev.get_stock_list() or []):
                try:
                    amt_prev = pos_prev.get_stock_amount(code) or 0
                    amt_now = pos.get_stock_amount(code) or 0
                except Exception:
                    amt_prev, amt_now = 0, 0
                delta = amt_now - amt_prev
                if abs(delta) < 1e-6:
                    continue
                price = _close_at(date_ts, code)
                if price <= 0:
                    continue
                stock_trade_count[code] = stock_trade_count.get(code, 0) + 1
                if delta > 0:
                    stock_buy_value[code] = stock_buy_value.get(code, 0) + delta * price
                    # 加仓：持仓成本增加
                    stock_cost_basis[code] = stock_cost_basis.get(code, 0) + delta * price
                else:
                    stock_sell_value[code] = stock_sell_value.get(code, 0) + (-delta) * price
                    stock_sell_count[code] = stock_sell_count.get(code, 0) + 1
                    # 盈利的卖出：卖出价 > 当前持仓的平均成本（用持仓成本基础/前一日股数）
                    cost_basis = stock_cost_basis.get(code, 0)
                    if amt_prev > 0 and cost_basis > 0:
                        avg_cost = cost_basis / amt_prev
                        if price > avg_cost:
                            stock_sell_win_count[code] = stock_sell_win_count.get(code, 0) + 1
                    # 卖出后按比例减少持仓成本基础
                    if amt_prev > 0:
                        stock_cost_basis[code] = cost_basis * (amt_now / amt_prev)
                    if amt_now <= 0:
                        stock_cost_basis.pop(code, None)
        # 期末市值按最后一日收盘价
        last_date = pd.Timestamp(dates[-1]).normalize()
        last_pos = positions_dict[dates[-1]]
        last_value: Dict[str, float] = {}
        for code in last_pos.get_stock_list() or []:
            try:
                amt = last_pos.get_stock_amount(code) or 0
                if amt <= 0:
                    continue
                price = _close_at(last_date, code)
                if price <= 0:
                    price = last_pos.get_stock_price(code) or 0
                last_value[code] = amt * price
            except Exception:
                pass
        _log("=" * 80)
        _log("单只股票统计（交易次数、总盈亏、盈亏率、胜率，买卖与期末均按当日收盘价）")
        _log("=" * 80)
        all_stocks = set(stock_buy_value.keys()) | set(stock_sell_value.keys()) | set(last_value.keys())
        rows = []
        for code in sorted(all_stocks):
            buy_v = stock_buy_value.get(code, 0)
            sell_v = stock_sell_value.get(code, 0)
            end_v = last_value.get(code, 0)
            total_invested = buy_v
            total_received = sell_v + end_v
            if total_invested <= 0:
                pnl_rate = 0.0
                pnl = 0.0
            else:
                pnl = total_received - total_invested
                pnl_rate = pnl / total_invested
            cnt = stock_trade_count.get(code, 0)
            sell_cnt = stock_sell_count.get(code, 0)
            sell_win = stock_sell_win_count.get(code, 0)
            win_rate = (sell_win / sell_cnt * 100) if sell_cnt > 0 else None  # 胜率 = 盈利卖出次数/卖出次数
            rows.append((code, cnt, pnl, pnl_rate, total_invested, total_received, win_rate))
        rows.sort(key=lambda x: -x[1])
        for code, cnt, pnl, pnl_rate, inv, rec, win_rate in rows:
            win_str = f"{win_rate:.1f}%" if win_rate is not None else "-"
            _log(
                f"  {code}  交易次数: {cnt}  总盈亏: {pnl:+,.2f}  盈亏率: {pnl_rate*100:+.2f}%  胜率: {win_str}  "
                f"投入: {inv:,.0f}  收回+持仓: {rec:,.0f}"
            )
        _log("")
    else:
        _log("⚠️ 无有效 positions，跳过单只股票统计")

    _log(f"日志已写入: {log_file}")
    for f in file_handlers:
        try:
            f.close()
        except Exception:
            pass


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        return f"{int(seconds // 60)}分{seconds % 60:.2f}秒"
    else:
        return f"{int(seconds // 3600)}小时{int((seconds % 3600) // 60)}分{seconds % 60:.2f}秒"


def print_metrics(metrics: Dict[str, Any], title: str) -> None:
    print(f"\n【{title}】")
    for key, value in metrics.items():
        if isinstance(value, float):
            if "ratio" in key or "return" in key or "drawdown" in key:
                print(f"  {key}: {value*100:.2f}%")
            else:
                print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def analyze_backtest_results(port_analysis: Dict[str, Any]) -> None:
    """打印回测结果摘要。"""
    def _sharpe(metrics: Dict) -> float:
        std = metrics.get("std", 0)
        ann = metrics.get("annualized_return", 0)
        return (ann / (std * (252 ** 0.5))) if std > 0 else 0.0

    print("\n" + "=" * 80)
    print("回测结果汇总")
    print("=" * 80)

    if "benchmark" in port_analysis:
        bm = port_analysis["benchmark"]
        print_metrics({
            "年化收益": bm.get("annualized_return", 0) * 100,
            "最大回撤": bm.get("max_drawdown", 0) * 100,
            "信息比率": bm.get("information_ratio", 0),
            "夏普比率": _sharpe(bm),
            "年化波动率": bm.get("std", 0) * (252 ** 0.5) * 100,
        }, "基准收益")

    if "excess_return_without_cost" in port_analysis:
        ex = port_analysis["excess_return_without_cost"]
        print_metrics({
            "年化收益": ex.get("annualized_return", 0) * 100,
            "最大回撤": ex.get("max_drawdown", 0) * 100,
            "信息比率": ex.get("information_ratio", 0),
            "夏普比率": _sharpe(ex),
            "年化波动率": ex.get("std", 0) * (252 ** 0.5) * 100,
        }, "超额收益(不含成本)")

    if "excess_return_with_cost" in port_analysis:
        ex = port_analysis["excess_return_with_cost"]
        print_metrics({
            "年化收益": ex.get("annualized_return", 0) * 100,
            "最大回撤": ex.get("max_drawdown", 0) * 100,
            "信息比率": ex.get("information_ratio", 0),
            "夏普比率": _sharpe(ex),
            "年化波动率": ex.get("std", 0) * (252 ** 0.5) * 100,
        }, "超额收益(含成本)")

    if "benchmark" in port_analysis and "excess_return_with_cost" in port_analysis:
        bm_ret = port_analysis["benchmark"].get("annualized_return", 0) * 100
        ex_ret = port_analysis["excess_return_with_cost"].get("annualized_return", 0) * 100
        print(f"\n【策略表现】")
        print(f"  策略年化收益: {ex_ret:.2f}%")
        print(f"  基准年化收益: {bm_ret:.2f}%")
        print(f"  超额年化收益: {ex_ret - bm_ret:.2f}%")


def generate_backtest_charts(
    recorder: Any,
    output_dir: Optional[Union[str, Path]] = None,
) -> None:
    """
    使用qlib自带的绘图功能生成回测图表。

    Parameters
    ----------
    recorder : Any
        回测结果的recorder
    output_dir : Optional[Union[str, Path]], optional
        图表输出目录，默认为当前目录下的log目录
    """
    report_df, _ = load_report_and_positions(recorder)
    if report_df is None or report_df.empty:
        print("⚠️ 未找到 report_normal，跳过图表生成")
        return

    if output_dir is None:
        output_dir = current_dir / "log"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 使用qlib自带的report_graph生成回测图表
    print("\n生成回测图表...")
    try:
        figures = report_graph(report_df, show_notebook=False)
        for i, fig in enumerate(figures):
            output_path = output_dir / f"backtest_report_{i}.html"
            fig.write_html(str(output_path))
            print(f"  ✓ 图表已保存: {output_path}")
    except Exception as e:
        print(f"  ⚠️ 生成回测报告图表失败: {e}")

    # 使用qlib自带的risk_analysis_graph生成风险分析图表
    try:
        # 准备风险分析数据
        analysis = {}
        analysis["excess_return_without_cost"] = risk_analysis(
            report_df["return"] - report_df["bench"], freq="day"
        )
        analysis["excess_return_with_cost"] = risk_analysis(
            report_df["return"] - report_df["bench"] - report_df["cost"], freq="day"
        )
        analysis_df = pd.concat(analysis)

        figures = risk_analysis_graph(analysis_df, report_df, show_notebook=False)
        for i, fig in enumerate(figures):
            output_path = output_dir / f"risk_analysis_{i}.html"
            fig.write_html(str(output_path))
            print(f"  ✓ 风险分析图表已保存: {output_path}")
    except Exception as e:
        print(f"  ⚠️ 生成风险分析图表失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="加载已有 run 的预测结果，仅执行回测（可覆盖回测参数）")
    parser.add_argument(
        "run_id_pos",
        nargs="?",
        type=str,
        default=None,
        metavar="run_id",
        help="源 run 的 run_id（位置参数）",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="源 run 的 run_id：mlruns/<experiment_id>/ 下的子目录名（32 位十六进制，如 0b8e2395033a4264803061c0f76b63c8）",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="long_short_factor_strategy",
        help="源实验名称（默认 long_short_factor_strategy）",
    )
    parser.add_argument(
        "--backtest_config",
        type=str,
        default=None,
        help="回测参数覆盖 YAML，只写需要覆盖的 strategy.kwargs 和 backtest 等",
    )
    parser.add_argument(
        "--workflow_config",
        type=str,
        default=None,
        help="完整 workflow 配置 YAML（默认使用同目录下 workflow_config_long_short_strategy.yaml）",
    )
    parser.add_argument(
        "--use_saved_pred",
        action="store_true",
        help="不重新生成预测，直接使用源 run 中已保存的预测（仅当回测周期与源 run 一致时使用）",
    )
    parser.add_argument(
        "--start_time",
        type=str,
        default=None,
        help="回测开始日期（覆盖配置），如 2025-01-01；与 --end_time 一起指定可只对该周期重新生成预测",
    )
    parser.add_argument(
        "--end_time",
        type=str,
        default=None,
        help="回测结束日期（覆盖配置），如 2025-12-31",
    )
    args = parser.parse_args()
    run_id = args.run_id if args.run_id is not None else args.run_id_pos
    if not run_id:
        parser.error("请提供 run_id（位置参数或 --run_id），例如: python run_backtest_only.py 0b8e2395033a4264803061c0f76b63c8")
    args.run_id = run_id

    main_start_time = time.time()

    # 1. 初始化 qlib
    print("=" * 80)
    print("初始化 Qlib...")
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
    print("✓ Qlib 初始化成功")

    # 2. 加载基础回测配置
    workflow_path = Path(args.workflow_config) if args.workflow_config else current_dir / "workflow_config_long_short_strategy.yaml"
    if not workflow_path.exists():
        print(f"❌ 配置文件不存在: {workflow_path}")
        sys.exit(1)
    with open(workflow_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
    if "port_analysis_config" not in base_config:
        print("❌ 配置中缺少 port_analysis_config")
        sys.exit(1)
    port_analysis_config = base_config["port_analysis_config"].copy()

    # 3. 合并回测覆盖配置
    if args.backtest_config:
        override_path = Path(args.backtest_config)
        if not override_path.is_absolute():
            override_path = current_dir / override_path
        if not override_path.exists():
            print(f"❌ 回测覆盖配置不存在: {override_path}")
            sys.exit(1)
        with open(override_path, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f)
        if "port_analysis_config" in override:
            port_analysis_config = _deep_merge(port_analysis_config, override["port_analysis_config"])
        elif "strategy" in override or "backtest" in override:
            port_analysis_config = _deep_merge(port_analysis_config, override)
        print(f"✓ 已应用回测覆盖配置: {override_path}")

    backtest_cfg = port_analysis_config.get("backtest", {})
    start_time = args.start_time or backtest_cfg.get("start_time")
    end_time = args.end_time or backtest_cfg.get("end_time")
    if start_time is None or end_time is None:
        print("❌ 回测周期未设置：请在 workflow/backtest_config 中指定 backtest.start_time 与 backtest.end_time，或使用 --start_time / --end_time")
        sys.exit(1)
    if args.use_saved_pred:
        print(f"\n回测周期: {start_time} ~ {end_time}（使用源 run 已保存的预测，请确保周期与源 run 一致）")
    else:
        print(f"\n回测周期: {start_time} ~ {end_time}（将加载模型并重新生成该周期预测）")

    # 4. 获取源 recorder，并生成或加载预测
    print(f"\n加载源 run: experiment_name={args.experiment_name}, run_id={args.run_id}")
    try:
        source_recorder = R.get_recorder(
            recorder_id=args.run_id,
            experiment_name=args.experiment_name,
        )
    except Exception as e:
        print(f"❌ 无法获取源 recorder: {e}")
        print("  提示: run_id 应为 MLflow 的 32 位十六进制 UUID（如 0b8e2395033a4264803061c0f76b63c8），")
        print("        即 mlruns/<experiment_id>/ 下的子目录名，不是数字。可用 ls mlruns/673062759933473454/ 查看。")
        sys.exit(1)

    if args.use_saved_pred:
        long_term_pred = source_recorder.load_object("long_term_pred.pkl")
        short_term_pred = source_recorder.load_object("short_term_pred.pkl")
        if long_term_pred is None:
            print("❌ 源 run 中未找到 long_term_pred.pkl")
            sys.exit(1)
        if short_term_pred is None:
            print("❌ 源 run 中未找到 short_term_pred.pkl")
            sys.exit(1)
        print("✓ 已加载 long_term_pred.pkl 与 short_term_pred.pkl")
        label_pkl = source_recorder.load_object("label.pkl")
        if label_pkl is None:
            label_pkl = pd.DataFrame()
    else:
        print("加载模型并按回测周期重新生成预测...")
        try:
            long_term_pred, short_term_pred, label_pkl = _load_models_and_regenerate_pred(
                source_recorder, base_config, start_time, end_time, current_dir
            )
            print("✓ 长期/短期预测已生成")
        except Exception as e:
            print(f"❌ 重新生成预测失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    # PortAnaRecord 依赖 SignalRecord，会检查 pred.pkl 与 label.pkl 是否存在
    if label_pkl is None:
        label_pkl = pd.DataFrame()

    # 确保回测配置使用当前确定的周期（便于 PortAnaRecord 使用）
    if "backtest" not in port_analysis_config:
        port_analysis_config["backtest"] = {}
    port_analysis_config["backtest"]["start_time"] = start_time
    port_analysis_config["backtest"]["end_time"] = end_time

    # 5. 在新 run 中写入预测并执行回测
    experiment_name_backtest = "backtest_only"
    print(f"\n在新实验 run 中执行回测: {experiment_name_backtest}")
    with R.start(experiment_name=experiment_name_backtest):
        new_recorder = R.get_recorder()
        new_recorder.save_objects(**{
            "pred.pkl": long_term_pred,
            "label.pkl": label_pkl,
            "short_term_pred.pkl": short_term_pred,
        })
        strategy_config = port_analysis_config["strategy"].copy()
        if "kwargs" not in strategy_config:
            strategy_config["kwargs"] = {}
        strategy_config["kwargs"]["signal"] = "<PRED>"
        strategy_config["kwargs"]["short_term_signal"] = "short_term_pred.pkl"
        strategy_config["kwargs"]["recorder"] = new_recorder
        port_analysis_config["strategy"] = strategy_config

        par = PortAnaRecord(new_recorder, port_analysis_config, "day")
        par.generate()
        print("✓ 回测执行完成")

        port_analysis = load_port_analysis(new_recorder, par)
        if port_analysis is None:
            print("⚠️ 无法加载回测结果对象，请检查 PortAnaRecord 是否正常写入")
        else:
            analyze_backtest_results(port_analysis)

        # 每日交易记录与单只股票统计（写入日志文件并打印）
        log_daily_trades_and_stock_stats(new_recorder, log_dir=current_dir / "log")

        # 生成回测图表
        generate_backtest_charts(new_recorder, output_dir=current_dir / "log")

        print("\n" + "=" * 80)
        print("回测 run 信息")
        print(f"  - 实验名: {experiment_name_backtest}")
        print(f"  - 实验ID: {new_recorder.experiment_id}")
        print(f"  - 记录ID: {new_recorder.id}")
        print(f"  - 结果路径: mlruns/{new_recorder.experiment_id}/{new_recorder.id}/")
        print(f"  - 总耗时: {format_time(time.time() - main_start_time)}")
        print("=" * 80)


if __name__ == "__main__":
    main()
