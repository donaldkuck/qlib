# -*- coding: utf-8 -*-
"""
Ptrade 版本的多空 GATs 策略

基于 TopkDrop 思想的长期和短期因子结合策略

策略逻辑：
1. 买入和卖出基于长期预测值（使用 topkdrop 思想）
   - 持有 topk 只股票
   - 每天替换 n_drop 只股票
2. 买入条件：长期和短期预测都为正
3. 加仓/止盈：根据短期预测值对已持仓股票进行加仓或止盈
   - 短期预测为正：加仓
   - 短期预测为负且盈利：止盈（减仓）

用法：
    1. 准备预测数据文件（从 Qlib 回测中导出）：
       - long_term_pred.pkl: 长期预测（DataFrame，index 为 (stock_id, datetime)）
       - short_term_pred.pkl: 短期预测（DataFrame，index 为 (stock_id, datetime)）

    2. 运行回测：
       python ptrade_backtest_strategy.py --long_pred long_term_pred.pkl --short_pred short_term_pred.pkl
"""

import argparse
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class PtradeStrategy:
    """
    Ptrade 版本的多空 GATs 策略
    """

    def __init__(
        self,
        long_term_pred: pd.DataFrame,
        short_term_pred: pd.DataFrame,
        topk: int = 20,
        n_drop: int = 5,
        hold_thresh: int = 10,
        add_position_ratio: float = 0.03,
        profit_threshold: float = 0.05,
        reduce_position_ratio: float = 0.2,
        max_single_stock_ratio: float = 0.05,
        stop_loss_threshold: float = -0.05,
        max_drawdown: Optional[float] = None,
        drawdown_reduce_ratio: float = 0.3,
        risk_degree: float = 0.95,
    ):
        """
        初始化策略

        Parameters
        ----------
        long_term_pred : pd.DataFrame
            长期预测数据，index 为 (stock_id, datetime)，列为预测值
        short_term_pred : pd.DataFrame
            短期预测数据，index 为 (stock_id, datetime)，列为预测值
        topk : int
            持仓股票数量
        n_drop : int
            每天替换的股票数量
        hold_thresh : int
            最小持有天数
        add_position_ratio : float
            加仓比例（相对于单只股票仓位）
        profit_threshold : float
            止盈阈值
        reduce_position_ratio : float
            止盈时减仓比例
        max_single_stock_ratio : float
            单只股票最大仓位比例
        stop_loss_threshold : float
            止损阈值（负值）
        max_drawdown : float
            最大回撤阈值
        drawdown_reduce_ratio : float
            回撤时减仓比例
        risk_degree : float
            风险仓位比例
        """
        self.long_term_pred = long_term_pred
        self.short_term_pred = short_term_pred

        # 策略参数
        self.topk = topk
        self.n_drop = n_drop
        self.hold_thresh = hold_thresh
        self.add_position_ratio = add_position_ratio
        self.profit_threshold = profit_threshold
        self.reduce_position_ratio = reduce_position_ratio
        self.max_single_stock_ratio = max_single_stock_ratio
        self.stop_loss_threshold = stop_loss_threshold
        self.max_drawdown = max_drawdown
        self.drawdown_reduce_ratio = drawdown_reduce_ratio
        self.risk_degree = risk_degree

        # 状态变量
        self.entry_prices: Dict[str, float] = {}  # 入场价格
        self.peak_account_value: float = None  # 历史最高资产
        self.last_drawdown_reduce_date: Optional[datetime] = None  # 上次减仓日期
        self.drawdown_reduce_cooldown_days: int = 5  # 减仓冷却期
        self.stock_hold_days: Dict[str, int] = {}  # 持仓天数

    def is_prediction_positive(self, pred_value: float, all_pred_values: pd.Series = None) -> bool:
        """
        判断预测值是否为正

        Parameters
        ----------
        pred_value : float
            预测值
        all_pred_values : pd.Series, optional
            所有预测值（用于计算相对阈值）

        Returns
        -------
        bool
            是否为正
        """
        if pd.isna(pred_value):
            return False

        # 使用中位数作为阈值（相对排名）
        if all_pred_values is not None and len(all_pred_values) > 0:
            valid_values = all_pred_values.dropna()
            if len(valid_values) > 0:
                threshold = valid_values.quantile(0.5)
                return pred_value > threshold

        return pred_value > 0

    def get_predictions_for_date(self, date: datetime) -> Tuple[pd.Series, pd.Series]:
        """
        获取指定日期的预测值

        Parameters
        ----------
        date : datetime
            查询日期

        Returns
        -------
        Tuple[pd.Series, pd.Series]
            (长期预测, 短期预测)，Series 的 index 为股票代码
        """
        date_str = date.strftime("%Y-%m-%d")

        # 从 MultiIndex 中提取指定日期的数据
        if isinstance(self.long_term_pred.index, pd.MultiIndex):
            long_idx = self.long_term_pred.index.get_level_values(1) == date_str
            long_pred_date = self.long_term_pred[long_idx].copy()
            long_pred_date.index = long_pred_date.index.get_level_values(0)

            short_idx = self.short_term_pred.index.get_level_values(1) == date_str
            short_pred_date = self.short_term_pred[short_idx].copy()
            short_pred_date.index = short_pred_date.index.get_level_values(0)
        else:
            # 如果 index 是 datetime 类型
            if isinstance(self.long_term_pred.index, pd.DatetimeIndex):
                date_normalized = pd.Timestamp(date).normalize()
                long_pred_date = self.long_term_pred[self.long_term_pred.index.normalize() == date_normalized].copy()
                short_pred_date = self.short_term_pred[self.short_term_pred.index.normalize() == date_normalized].copy()
                long_pred_date.index.name = None
                short_pred_date.index.name = None
            else:
                # 尝试用字符串匹配
                long_pred_date = self.long_term_pred.filter(regex=f"^{date_str}", axis=0)
                short_pred_date = self.short_term_pred.filter(regex=f"^{date_str}", axis=0)

        # 如果预测是 DataFrame，取第一列
        if isinstance(long_pred_date, pd.DataFrame):
            if len(long_pred_date.columns) > 0:
                long_pred_date = long_pred_date.iloc[:, 0]
            else:
                long_pred_date = pd.Series()
        if isinstance(short_pred_date, pd.DataFrame):
            if len(short_pred_date.columns) > 0:
                short_pred_date = short_pred_date.iloc[:, 0]
            else:
                short_pred_date = pd.Series()

        return long_pred_date, short_pred_date


def load_predictions(long_pred_path: str, short_pred_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载预测数据

    Parameters
    ----------
    long_pred_path : str
        长期预测文件路径
    short_pred_path : str
        短期预测文件路径

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (长期预测, 短期预测)
    """
    print(f"加载长期预测: {long_pred_path}")
    with open(long_pred_path, "rb") as f:
        long_pred = pickle.load(f)

    print(f"加载短期预测: {short_pred_path}")
    with open(short_pred_path, "rb") as f:
        short_pred = pickle.load(f)

    print(f"长期预测形状: {long_pred.shape}")
    print(f"短期预测形状: {short_pred.shape}")

    return long_pred, short_pred


def initialize(context):
    """
    策略初始化函数

    Parameters
    ----------
    context
        策略上下文对象
    """
    # 从全局获取预测数据
    global long_term_pred, short_term_pred, strategy

    # 创建策略实例
    context.strategy = strategy

    # 设置基准
    context.set_benchmark("SH000300")

    # 设置股票池（使用预测数据中的所有股票）
    if isinstance(long_term_pred.index, pd.MultiIndex):
        all_stocks = list(long_term_pred.index.get_level_values(0).unique())
    else:
        all_stocks = list(long_term_pred.index.unique())

    # 过滤掉非标准股票代码（确保格式如 "SH000001" 或 "SZ000001"）
    valid_stocks = [s for s in all_stocks if isinstance(s, str) and (s.startswith(("SH", "SZ")))]

    print(f"设置股票池，共 {len(valid_stocks)} 只股票")
    context.set_universe(valid_stocks)

    # 设置佣金（0.03% + 5元最低）
    context.set_commission(commission_ratio=0.0003, min_commission=5.0)

    # 设置滑点（0.1%）
    context.set_slippage(slippage=0.1)

    # 初始化状态变量
    context.peak_account_value = context.account.total_value

    print("策略初始化完成")


def handle_data(context, data):
    """
    主策略逻辑函数，每日执行

    Parameters
    ----------
    context
        策略上下文对象
    data
        数据对象
    """
    strategy = context.strategy
    current_date = context.current_dt.replace(hour=0, minute=0, second=0, microsecond=0)

    print(f"=== {current_date.strftime('%Y-%m-%d')} 策略执行 ===")

    # 获取当前日期的预测值
    long_pred, short_pred = strategy.get_predictions_for_date(current_date)

    if long_pred.empty or short_pred.empty:
        print(f"无预测数据，跳过")
        return

    # 获取股票池（取交集）
    stock_pool = long_pred.index.intersection(short_pred.index)
    if len(stock_pool) == 0:
        print(f"无有效股票池，跳过")
        return

    long_pred = long_pred.reindex(stock_pool)
    short_pred = short_pred.reindex(stock_pool)

    # 获取当前持仓
    positions = context.get_positions()
    current_stock_list = [pos.stock_code for pos in positions if pos.amount > 0]

    # 更新持仓天数
    for code in current_stock_list:
        if code not in strategy.stock_hold_days:
            strategy.stock_hold_days[code] = 0
        strategy.stock_hold_days[code] += 1

    # 计算总资产和回撤
    total_value = context.account.total_value
    if strategy.peak_account_value is None or total_value > strategy.peak_account_value:
        strategy.peak_account_value = total_value

    risk_degree_multiplier = 1.0
    should_reduce_position = False

    if strategy.peak_account_value is not None and strategy.peak_account_value > 0:
        current_drawdown = (strategy.peak_account_value - total_value) / strategy.peak_account_value

        # 回撤预警
        drawdown_warning = strategy.max_drawdown * 0.7 if strategy.max_drawdown else None
        if drawdown_warning is not None and current_drawdown > drawdown_warning:
            excess_drawdown = current_drawdown - drawdown_warning
            risk_degree_multiplier = max(0.5, 1.0 - excess_drawdown / (drawdown_warning * 2))
            print(f"回撤预警: {current_drawdown*100:.2f}%, 降低仓位至 {risk_degree_multiplier*100:.1f}%")

            if current_drawdown > drawdown_warning * 2.0:
                cooldown_days = strategy.drawdown_reduce_cooldown_days
                if strategy.last_drawdown_reduce_date is None:
                    should_reduce_position = True
                    strategy.last_drawdown_reduce_date = current_date
                    print(f"回撤预警：主动减仓 {strategy.drawdown_reduce_ratio*100:.0f}%")
                elif (current_date - strategy.last_drawdown_reduce_date).days >= cooldown_days:
                    should_reduce_position = True
                    strategy.last_drawdown_reduce_date = current_date
                    print(f"回撤预警：主动减仓 {strategy.drawdown_reduce_ratio*100:.0f}%")

        # 回撤过大
        if strategy.max_drawdown is not None and current_drawdown > strategy.max_drawdown:
            excess_drawdown = current_drawdown - strategy.max_drawdown
            risk_degree_multiplier = max(0.3, 1.0 - excess_drawdown / strategy.max_drawdown)

            cooldown_days = strategy.drawdown_reduce_cooldown_days
            if strategy.last_drawdown_reduce_date is None:
                should_reduce_position = True
                strategy.last_drawdown_reduce_date = current_date
                print(f"回撤过大: {current_drawdown*100:.2f}%, 主动减仓 {strategy.drawdown_reduce_ratio*100:.0f}%")
            elif (current_date - strategy.last_drawdown_reduce_date).days >= cooldown_days:
                should_reduce_position = True
                strategy.last_drawdown_reduce_date = current_date
                print(f"回撤过大: {current_drawdown*100:.2f}%, 主动减仓 {strategy.drawdown_reduce_ratio*100:.0f}%")

    # ========== 1. TopkDrop 逻辑：基于长期预测值选择要买入和卖出的股票 ==========
    # 当前持仓按长期预测值排序
    current_stock_list_valid = [s for s in current_stock_list if s in long_pred.index]
    if current_stock_list_valid:
        long_pred_current = long_pred.reindex(current_stock_list_valid)
        last = long_pred_current.sort_values(ascending=False).index.tolist()
    else:
        last = []

    # 选择要买入的候选股票（基于长期预测值）
    available_for_buy = long_pred[~long_pred.index.isin(last)]
    if len(available_for_buy) > 0:
        n_candidates = min(strategy.n_drop + strategy.topk - len(last), len(available_for_buy))
        today_candidates = available_for_buy.sort_values(ascending=False).index.tolist()[:n_candidates]
    else:
        today_candidates = []

    # 合并候选股票和当前持仓
    comb = list(set(last).union(set(today_candidates)))
    if comb:
        long_pred_comb = long_pred.reindex(comb).sort_values(ascending=False)
        comb = long_pred_comb.index.tolist()

    # 选择要卖出的股票（基于长期预测值）
    sell_candidates = []
    if len(comb) > 0 and len(last) > 0:
        n_sell = min(strategy.n_drop, len(comb))
        bottom_stocks = long_pred.reindex(comb).sort_values(ascending=True).index.tolist()[:n_sell]
        sell_candidates = [s for s in last if s in bottom_stocks]

    # 实际要买入的股票数量
    buy_count = len(sell_candidates) + strategy.topk - len(last)
    buy_count = max(0, buy_count)

    # ========== 2. 买入筛选：只选择长期和短期预测都为正的股票 ==========
    buy_list = []
    backup_buy_list = []

    for stock_id in today_candidates[:buy_count]:
        if stock_id not in long_pred.index or stock_id not in short_pred.index:
            continue

        long_term_value = long_pred.loc[stock_id]
        short_term_value = short_pred.loc[stock_id]

        is_long_term_positive = strategy.is_prediction_positive(long_term_value, long_pred)
        is_short_term_positive = strategy.is_prediction_positive(short_term_value, short_pred)

        if is_long_term_positive and is_short_term_positive:
            buy_list.append(stock_id)
        elif is_long_term_positive:
            backup_buy_list.append(stock_id)

    # 如果符合条件的股票不够，使用备选方案
    if len(buy_list) < buy_count:
        needed = buy_count - len(buy_list)
        buy_list.extend(backup_buy_list[:needed])

    # ========== 3. 执行卖出订单 ==========
    print(f"当前持仓: {len(current_stock_list)} 只")
    print(f"候选卖出: {sell_candidates}")

    for code in current_stock_list:
        if code not in sell_candidates:
            continue

        # 检查最小持有天数
        hold_days = strategy.stock_hold_days.get(code, 0)
        if hold_days < strategy.hold_thresh:
            print(f"  {code} 持有天数 {hold_days} < {strategy.hold_thresh}，暂不卖出")
            continue

        # 卖出全部持仓
        pos = context.get_position(code)
        if pos and pos.amount > 0:
            context.order(code, 0)  # 清仓
            strategy.entry_prices.pop(code, None)
            strategy.stock_hold_days.pop(code, None)
            print(f"  卖出 {code}: {pos.amount} 股")

    # ========== 4. 执行买入订单 ==========
    # 获取可用现金
    available_cash = context.account.cash
    cash_for_buy = available_cash * strategy.risk_degree * risk_degree_multiplier

    print(f"买入候选: {buy_list[:buy_count]}, 可用资金: {cash_for_buy:,.2f}")

    if len(buy_list) > 0 and cash_for_buy > 0:
        # 按信号强度分配资金
        signal_scores = {}
        for code in buy_list:
            if code in long_pred.index and code in short_pred.index:
                long_score = long_pred.loc[code]
                short_score = short_pred.loc[code]
                # 归一化到0-1范围（使用排名）
                long_rank = (long_pred > long_score).sum() / len(long_pred)
                short_rank = (short_pred > short_score).sum() / len(short_pred)
                # 综合得分
                signal_scores[code] = 0.6 * (1 - long_rank) + 0.4 * (1 - short_rank)
            else:
                signal_scores[code] = 0.5

        # 计算单只股票最大资金
        max_single_value = total_value * strategy.max_single_stock_ratio

        # 按信号强度分配资金
        total_score = sum(signal_scores.values())
        if total_score > 0:
            stock_values = {
                code: min(cash_for_buy * signal_scores[code] / total_score, max_single_value)
                for code in buy_list
            }
        else:
            value_per_stock = min(cash_for_buy / len(buy_list), max_single_value)
            stock_values = {code: value_per_stock for code in buy_list}

        # 获取当前价格
        for code in buy_list:
            if code in long_pred.index and code in short_pred.index:
                try:
                    current_price = context.get_price(code, end_date=current_date, frequency="1d")
                    if current_price is None or len(current_price) == 0:
                        print(f"  {code} 无法获取价格，跳过")
                        continue
                    price = float(current_price.iloc[-1])

                    if price <= 0:
                        continue

                    stock_value = stock_values.get(code, 0)
                    if stock_value <= 0:
                        continue

                    # 计算买入数量
                    buy_amount = int(stock_value / price / 100) * 100  # 按手数买入

                    if buy_amount > 0:
                        context.order(code, buy_amount)
                        strategy.entry_prices[code] = price
                        strategy.stock_hold_days[code] = 0
                        print(f"  买入 {code}: {buy_amount} 股, 价格: {price:.2f}, 金额: {buy_amount * price:,.2f}")
                except Exception as e:
                    print(f"  {code} 买入失败: {e}")

    # ========== 5. 加仓/止盈逻辑 ==========
    # 更新持仓列表（卖出后）
    positions = context.get_positions()
    current_stock_list = [pos.stock_code for pos in positions if pos.amount > 0]

    for code in current_stock_list:
        if code not in long_pred.index or code not in short_pred.index:
            continue

        # 获取预测值
        long_term_value = long_pred.loc[code]
        short_term_value = short_pred.loc[code]
        is_long_term_positive = strategy.is_prediction_positive(long_term_value, long_pred)
        is_short_term_positive = strategy.is_prediction_positive(short_term_value, short_pred)

        # 获取当前持仓和价格
        pos = context.get_position(code)
        if not pos or pos.amount <= 0:
            continue

        try:
            price_series = context.get_price(code, end_date=current_date, frequency="1d")
            if price_series is None or len(price_series) == 0:
                continue
            current_price = float(price_series.iloc[-1])
        except:
            continue

        if current_price <= 0:
            continue

        # 获取入场价格
        entry_price = strategy.entry_prices.get(code, current_price)
        if entry_price is None or entry_price <= 0:
            entry_price = current_price
            strategy.entry_prices[code] = entry_price

        # 计算盈利比例
        profit_ratio = (current_price - entry_price) / entry_price if entry_price > 0 else 0

        # 止损判断
        if profit_ratio < strategy.stop_loss_threshold:
            context.order(code, 0)
            strategy.entry_prices.pop(code, None)
            strategy.stock_hold_days.pop(code, None)
            print(f"  {code} 亏损 {profit_ratio*100:.2f}%，止损卖出")
            continue

        # 长期预测为负：止损
        if not is_long_term_positive:
            context.order(code, 0)
            strategy.entry_prices.pop(code, None)
            strategy.stock_hold_days.pop(code, None)
            print(f"  {code} 长期预测为负，止损卖出")
            continue

        # 短期预测为正：加仓
        if is_short_term_positive:
            current_value = pos.amount * current_price
            max_single_stock_value = total_value * strategy.max_single_stock_ratio
            add_value = min(current_value * strategy.add_position_ratio, max_single_stock_value - current_value)

            # 检查可用资金
            cash_available = context.account.cash
            max_add_value = cash_available * strategy.risk_degree * risk_degree_multiplier
            add_value = min(add_value, max_add_value)

            if add_value > 0:
                add_amount = int(add_value / current_price / 100) * 100
                if add_amount > 0:
                    context.order(code, pos.amount + add_amount)
                    # 更新入场价格（加权平均）
                    old_price = strategy.entry_prices.get(code, current_price)
                    old_amount = pos.amount
                    new_price = (old_amount * old_price + add_amount * current_price) / (old_amount + add_amount)
                    strategy.entry_prices[code] = new_price
                    print(f"  {code} 加仓 {add_amount} 股，价格: {current_price:.2f}")

        # 短期预测为负且盈利：止盈
        elif not is_short_term_positive and profit_ratio > strategy.profit_threshold:
            reduce_amount = int(pos.amount * strategy.reduce_position_ratio / 100) * 100
            if reduce_amount > 0 and reduce_amount < pos.amount:
                context.order(code, pos.amount - reduce_amount)
                # 更新入场价格
                remaining_amount = pos.amount - reduce_amount
                if remaining_amount > 0:
                    strategy.entry_prices[code] = current_price
                print(f"  {code} 盈利 {profit_ratio*100:.2f}%，止盈减仓 {reduce_amount} 股")

    # 回撤控制：主动减仓
    if should_reduce_position and len(current_stock_list) > 0:
        print(f"执行回撤控制减仓")
        # 按长期预测值排序，卖出预测值最低的股票
        stocks_to_reduce = long_pred.reindex(current_stock_list).sort_values(ascending=True).index.tolist()
        reduce_count = max(1, int(len(current_stock_list) * strategy.drawdown_reduce_ratio))
        stocks_to_reduce = stocks_to_reduce[:reduce_count]

        for code in stocks_to_reduce:
            if code in sell_candidates:
                continue
            pos = context.get_position(code)
            if pos and pos.amount > 0:
                reduce_amount = int(pos.amount * strategy.drawdown_reduce_ratio / 100) * 100
                if reduce_amount > 0 and reduce_amount < pos.amount:
                    context.order(code, pos.amount - reduce_amount)
                    print(f"  回撤控制：{code} 减仓 {reduce_amount} 股")

    print(f"=== 当前资产: {context.account.total_value:,.2f}, 现金: {context.account.cash:,.2f} ===\n")


def after_trading_end(context, data):
    """
    盘后处理函数

    Parameters
    ----------
    context
        策略上下文对象
    data
        数据对象
    """
    current_date = context.current_dt.strftime("%Y-%m-%d")
    total_value = context.account.total_value
    print(f"\n{current_date} 盘后 - 总资产: {total_value:,.2f}, 持仓数: {len(context.get_positions())}")


def main():
    """
    主函数：用于加载预测数据并准备 ptrade 回测
    """
    parser = argparse.ArgumentParser(description="Ptrade 多空 GATs 策略回测")
    parser.add_argument(
        "--long_pred",
        type=str,
        required=True,
        help="长期预测文件路径（pkl 格式）",
    )
    parser.add_argument(
        "--short_pred",
        type=str,
        required=True,
        help="短期预测文件路径（pkl 格式）",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="持仓股票数量（默认 20）",
    )
    parser.add_argument(
        "--n_drop",
        type=int,
        default=5,
        help="每天替换的股票数量（默认 5）",
    )
    parser.add_argument(
        "--hold_thresh",
        type=int,
        default=10,
        help="最小持有天数（默认 10）",
    )
    parser.add_argument(
        "--add_position_ratio",
        type=float,
        default=0.03,
        help="加仓比例（默认 0.03）",
    )
    parser.add_argument(
        "--profit_threshold",
        type=float,
        default=0.05,
        help="止盈阈值（默认 0.05）",
    )
    parser.add_argument(
        "--reduce_position_ratio",
        type=float,
        default=0.2,
        help="止盈减仓比例（默认 0.2）",
    )
    parser.add_argument(
        "--max_single_stock_ratio",
        type=float,
        default=0.05,
        help="单只股票最大仓位比例（默认 0.05）",
    )
    parser.add_argument(
        "--stop_loss_threshold",
        type=float,
        default=-0.05,
        help="止损阈值（默认 -0.05）",
    )
    parser.add_argument(
        "--max_drawdown",
        type=float,
        default=None,
        help="最大回撤阈值（默认 None）",
    )
    parser.add_argument(
        "--risk_degree",
        type=float,
        default=0.95,
        help="风险仓位比例（默认 0.95）",
    )
    args = parser.parse_args()

    # 加载预测数据
    long_term_pred, short_term_pred = load_predictions(args.long_pred, args.short_pred)

    # 创建策略实例
    global strategy, long_term_pred_global, short_term_pred_global
    long_term_pred_global = long_term_pred
    short_term_pred_global = short_term_pred
    strategy = PtradeStrategy(
        long_term_pred=long_term_pred,
        short_term_pred=short_term_pred,
        topk=args.topk,
        n_drop=args.n_drop,
        hold_thresh=args.hold_thresh,
        add_position_ratio=args.add_position_ratio,
        profit_threshold=args.profit_threshold,
        reduce_position_ratio=args.reduce_position_ratio,
        max_single_stock_ratio=args.max_single_stock_ratio,
        stop_loss_threshold=args.stop_loss_threshold,
        max_drawdown=args.max_drawdown,
        risk_degree=args.risk_degree,
    )

    print("=" * 80)
    print("Ptrade 策略准备完成")
    print("=" * 80)
    print("请在 Ptrade 平台上运行此策略文件")
    print("策略参数：")
    print(f"  - 持仓股票数量: {args.topk}")
    print(f"  - 每天替换数量: {args.n_drop}")
    print(f"  - 最小持有天数: {args.hold_thresh}")
    print(f"  - 加仓比例: {args.add_position_ratio}")
    print(f"  - 止盈阈值: {args.profit_threshold}")
    print(f"  - 止盈减仓比例: {args.reduce_position_ratio}")
    print(f"  - 单只股票最大仓位: {args.max_single_stock_ratio}")
    print(f"  - 止损阈值: {args.stop_loss_threshold}")
    print(f"  - 最大回撤阈值: {args.max_drawdown}")
    print(f"  - 风险仓位比例: {args.risk_degree}")
    print("=" * 80)


if __name__ == "__main__":
    main()
