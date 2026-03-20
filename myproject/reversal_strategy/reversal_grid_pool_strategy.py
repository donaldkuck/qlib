"""
Reversal grid pool strategy — cleaned implementation.

This file is a simplified, corrected multi-stock pool strategy that follows
the single-stock reversal grid logic per-symbol, while enforcing a portfolio
level total position cap (<=100% of account value).

The implementation focuses on correctness and readability to avoid import-time
syntax/indentation errors encountered previously.
"""

import warnings
from datetime import timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from qlib.strategy.base import BaseStrategy
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.data import D
from qlib.log import get_module_logger

warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = get_module_logger("ReversalGridPoolStrategy")


class StockState:
    def __init__(self, stock_id: str, config: Dict):
        self.stock_id = stock_id
        # strategy params
        self.long_trend_period = int(config.get('long_trend_period', 15))
        self.short_trend_period = int(config.get('short_trend_period', 3))
        self.long_trend_atr_threshold = float(config.get('long_trend_atr_threshold', 0.1))
        self.short_trend_atr_threshold = float(config.get('short_trend_atr_threshold', 0.1))
        self.atr_period = int(config.get('atr_period', 14))
        self.atr_multiplier = float(config.get('atr_multiplier', 2.0))
        self.grid_atr_interval = float(config.get('grid_atr_interval', 0.2))
        self.entry_atr_multiplier = float(config.get('entry_atr_multiplier', 1.0))
        self.lookback_period = int(config.get('lookback_period', 5))
        self.grid_size = float(config.get('grid_size', 0.2))
        self.profit_atr_multiplier = float(config.get('profit_atr_multiplier', 0.5))
        self.enable_short = bool(config.get('enable_short', False))

        # position sizing
        self.initial_position_ratio = float(config.get('initial_position_ratio', 0.05))
        self.max_position_ratio = 1

        # runtime state
        self.position_direction = 0
        self.entry_price = None
        self.base_position_ratio = 0.0
        self.current_grid_level = 0
        self.highest_price = None
        self.lowest_price = None
        self.reversal_price = None
        
        # 参数验证
        max_possible_grid_levels = int(self.atr_multiplier / self.grid_atr_interval)
        if max_possible_grid_levels < 3:
            logger.warning(
                f"[{stock_id}] atr_multiplier ({self.atr_multiplier}) 相对于 grid_atr_interval ({self.grid_atr_interval}) 太小，"
                f"最多只能加仓 {max_possible_grid_levels} 次。建议将 atr_multiplier 设置为至少 {3 * self.grid_atr_interval}"
            )


class ReversalGridPoolStrategy(BaseStrategy):
    def __init__(self, stock_pool: Dict[str, Dict], max_holding_stocks: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.stock_states = {sid: StockState(sid, cfg) for sid, cfg in stock_pool.items()}
        self.max_holding_stocks = int(max_holding_stocks)
        logger.info(f"ReversalGridPoolStrategy init: pool_size={len(self.stock_states)}, max_hold={self.max_holding_stocks}")

    def _get_price_data(self, stock_id: str, current_time: pd.Timestamp) -> Optional[pd.DataFrame]:
        try:
            lookback_days =  max((s.long_trend_period for s in self.stock_states.values()), default=30)
            start_time = current_time - timedelta(days=lookback_days + 10)
            df = D.features([stock_id], ["$open", "$high", "$low", "$close"], start_time=start_time, end_time=current_time, freq="day", disk_cache=True)
            if df is None or len(df) == 0:
                return None
            if isinstance(df.index, pd.MultiIndex):
                if stock_id in df.index.get_level_values(0):
                    return df.loc[stock_id].dropna()
                return None
            return df.dropna()
        except Exception as e:
            logger.debug(f"_get_price_data {stock_id} err: {e}")
            return None

    def _calculate_atr(self, price_df: pd.DataFrame, atr_period: int) -> Optional[float]:
        try:
            high = price_df['$high']
            low = price_df['$low']
            close = price_df['$close']
            tr = np.maximum(high - low, np.maximum((high - close.shift(1)).abs(), (low - close.shift(1)).abs()))
            tr = tr.dropna()
            if len(tr) < atr_period:
                return None
            return float(tr.iloc[-atr_period:].mean())
        except Exception:
            return None

    def _get_trend(self, price_df: pd.DataFrame, period: int, threshold: float, atr: float) -> Optional[int]:
        """
        获取趋势方向（基于动量，动态ATR系数）
        
        使用整个时间段的动量来判断趋势，阈值根据ATR动态调整
        
        Parameters
        ----------
        price_df : pd.DataFrame
            价格数据
        period : int
            趋势周期
        threshold : float
            趋势判断的ATR系数（动量需要达到 threshold * ATR 才判定趋势）
        atr : float
            当前ATR值
            
        Returns
        -------
        int
            趋势方向：1=上涨，-1=下跌，0=持平，None=无法判断
        """
        try:
            close = price_df['$close']
            if len(close) < period + 1:
                return None
            
            if atr is None or atr <= 0:
                return None
            
            # 计算整个时间段内的动量（使用收益率计算）
            # 不仅仅看首尾价格，而是计算整个时间段的平均收益率方向
            period_close_prices = close.iloc[-period:]
            
            # 使用日收益率的累计和来衡量动量
            # 这样可以更好地反映整个时间段内的价格变化趋势
            daily_returns = period_close_prices.pct_change().dropna()
            
            if len(daily_returns) == 0:
                return None
            
            # 计算动量：整个时间段内的累计收益率转换为价格变化
            cumulative_return = (1 + daily_returns).prod() - 1
            momentum_in_price = period_close_prices.iloc[-1] * cumulative_return
            
            # 计算动态阈值：threshold * ATR
            dynamic_threshold = threshold * atr
            
            # 判断趋势
            if momentum_in_price > dynamic_threshold:
                return 1  # 上涨
            if momentum_in_price < -dynamic_threshold:
                return -1  # 下跌
            return 0  # 持平
        except Exception:
            return None

    def _get_recent_extreme_prices(self, price_df: pd.DataFrame, lookback: int) -> Tuple[Optional[float], Optional[float]]:
        try:
            close = price_df['$close']
            recent = close.iloc[-lookback:] if len(close) >= lookback else close
            if len(recent) == 0:
                return None, None
            return float(recent.min()), float(recent.max())
        except Exception:
            return None, None

    def _should_go_long(self, state: StockState, long_trend: int, short_trend: int, current_price: float, atr: float, price_df: pd.DataFrame) -> Tuple[bool, Optional[float]]:
        if short_trend != 1:
            return False, None
        if long_trend == 1:
            return False, None
        recent_low, _ = self._get_recent_extreme_prices(price_df, state.lookback_period)
        if recent_low is None:
            return False, None
        if current_price - recent_low >= state.entry_atr_multiplier * atr:
            return True, current_price - state.atr_multiplier * atr
        return False, None

    def _should_go_short(self, state: StockState, long_trend: int, short_trend: int, current_price: float, atr: float, price_df: pd.DataFrame) -> Tuple[bool, Optional[float]]:
        if not state.enable_short:
            return False, None
        if long_trend == -1:
            return False, None
        if short_trend != -1:
            return False, None
        _, recent_high = self._get_recent_extreme_prices(price_df, state.lookback_period)
        if recent_high is None:
            return False, None
        if recent_high - current_price >= state.entry_atr_multiplier * atr:
            return True, current_price + state.atr_multiplier * atr
        return False, None

    def _should_reverse(self, state: StockState, current_price: float, atr: float) -> Optional[int]:
        if state.position_direction == 0 or state.reversal_price is None:
            return None
        if state.position_direction == 1 and current_price <= state.reversal_price:
            return -1 if state.enable_short else None
        if state.position_direction == -1 and current_price >= state.reversal_price:
            return 1
        return None

    def _should_add_position(self, state: StockState, current_position_ratio: float, current_price: float, atr: float) -> bool:
        if current_position_ratio >= state.max_position_ratio:
            return False
        if state.position_direction == 1:
            highest = state.highest_price if state.highest_price is not None else state.entry_price or current_price
            drawdown = highest - current_price
            required = (state.current_grid_level + 1) * state.grid_atr_interval * atr
            if drawdown >= required and drawdown < state.atr_multiplier * atr:
                return True
        if state.position_direction == -1:
            lowest = state.lowest_price if state.lowest_price is not None else state.entry_price or current_price
            rally = current_price - lowest
            required = (state.current_grid_level + 1) * state.grid_atr_interval * atr
            if rally >= required and rally < state.atr_multiplier * atr:
                return True
        return False

    def _should_reduce_position_on_profit(self, state: StockState, current_position_ratio: float, current_price: float, atr: float) -> bool:
        if current_position_ratio <= state.base_position_ratio:
            return False
        if state.current_grid_level == 0:
            return False
        if state.position_direction == 1:
            if state.highest_price is None:
                return False
            drawdown = state.highest_price - current_price
            threshold = (state.current_grid_level - 1) * state.grid_atr_interval * atr
            return drawdown < threshold
        if state.position_direction == -1:
            if state.lowest_price is None:
                return False
            rally = current_price - state.lowest_price
            threshold = (state.current_grid_level - 1) * state.grid_atr_interval * atr
            return rally < threshold
        return False

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        current_time = trade_start_time

        account_value = self.trade_position.calculate_value()
        if account_value <= 0:
            return TradeDecisionWO([], self)

        # compute total current position ratio
        total_position_value = 0.0
        current_amounts = self.trade_position.get_stock_amount_dict()
        for sid, amt in current_amounts.items():
            try:
                p = self.trade_exchange.get_close(stock_id=sid, start_time=current_time, end_time=trade_end_time, method="ts_data_last")
                if hasattr(p, 'value'):
                    p = p.value
                elif isinstance(p, (pd.Series, pd.DataFrame)):
                    p = p.iloc[-1] if len(p) > 0 else None
                if p is None or pd.isna(p):
                    continue
                total_position_value += abs(amt) * float(p)
            except Exception:
                continue
        total_position_ratio = total_position_value / account_value if account_value > 0 else 0.0

        orders = []

        # limit candidates to existing holdings + a few candidates
        holdings = [s for s, a in current_amounts.items() if a > 0]
        candidates = list(self.stock_states.keys())
        stocks_to_process = list(set(holdings + candidates))
        
        logger.info(f"[{current_time}] 投资组合状态：总价值={account_value:.2f}, 持仓数={len(holdings)}, "
                   f"持仓比例={total_position_ratio:.2%}, 账户配置数量={len(stocks_to_process)}")

        for stock_id in stocks_to_process:
            state = self.stock_states.get(stock_id)
            if state is None:
                continue
            price_df = self._get_price_data(stock_id, current_time)
            if price_df is None:
                logger.debug(f"[{current_time}] {stock_id} 无法获取价格数据")
                continue
            atr = self._calculate_atr(price_df, state.atr_period)
            if atr is None or atr <= 0:
                logger.debug(f"[{current_time}] {stock_id} 无法计算ATR")
                continue
            long_trend = self._get_trend(price_df, state.long_trend_period, state.long_trend_atr_threshold, atr)
            short_trend = self._get_trend(price_df, state.short_trend_period, state.short_trend_atr_threshold, atr)
            if long_trend is None or short_trend is None:
                logger.debug(f"[{current_time}] {stock_id} 无法计算趋势")
                continue

            # get current price and amount
            try:
                current_price = self.trade_exchange.get_close(stock_id=stock_id, start_time=current_time, end_time=trade_end_time, method="ts_data_last")
                if hasattr(current_price, 'value'):
                    current_price = current_price.value
                elif isinstance(current_price, (pd.Series, pd.DataFrame)):
                    current_price = current_price.iloc[-1] if len(current_price) > 0 else None
            except Exception:
                logger.debug(f"[{current_time}] {stock_id} 无法获取当前价格")
                continue
            if current_price is None or pd.isna(current_price):
                logger.debug(f"[{current_time}] {stock_id} 价格为空")
                continue

            current_amount = self.trade_position.get_stock_amount(stock_id)
            current_position_value = abs(current_amount) * float(current_price)
            current_position_ratio = current_position_value / account_value if account_value > 0 else 0.0

            # update tracked prices
            if current_amount > 0:
                if state.position_direction == 0:
                    state.position_direction = 1
                    state.entry_price = float(current_price)
                    state.base_position_ratio = current_position_ratio
                    state.highest_price = float(current_price)
                if state.position_direction == 1 and (state.highest_price is None or current_price > state.highest_price):
                    state.highest_price = float(current_price)
                if state.position_direction == -1 and (state.lowest_price is None or current_price < state.lowest_price):
                    state.lowest_price = float(current_price)

            # if no position -> consider opening
            if current_amount == 0:
                should_long, rev_long = self._should_go_long(state, long_trend, short_trend, float(current_price), atr, price_df)
                if should_long and total_position_ratio < 1.0 and len(holdings) < self.max_holding_stocks:
                    allowed_ratio = min(state.initial_position_ratio, 1.0 - total_position_ratio)
                    target_value = account_value * allowed_ratio
                    amount = target_value / float(current_price)
                    if amount > 0:
                        orders.append(Order(stock_id=stock_id, amount=amount, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY, factor=1.0))
                        total_position_ratio += allowed_ratio
                        holdings.append(stock_id)
                        state.position_direction = 1
                        state.entry_price = float(current_price)
                        state.base_position_ratio = allowed_ratio
                        state.highest_price = float(current_price)
                        state.reversal_price = state.highest_price - state.atr_multiplier * atr
                        logger.info(
                            f"[{current_time}] {stock_id} 初始开仓（做多）："
                            f"价格={current_price:.4f}, ATR={atr:.4f}, 反转价格={state.reversal_price:.4f}, "
                            f"数量={amount:.0f}, 仓位比例={allowed_ratio:.2%}"
                        )
                        continue

                should_short, rev_short = self._should_go_short(state, long_trend, short_trend, float(current_price), atr, price_df)
                if should_short and total_position_ratio < 1.0 and len(holdings) < self.max_holding_stocks:
                    allowed_ratio = min(state.initial_position_ratio, 1.0 - total_position_ratio)
                    target_value = account_value * allowed_ratio
                    amount = target_value / float(current_price)
                    if amount > 0:
                        orders.append(Order(stock_id=stock_id, amount=amount, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY, factor=1.0))
                        total_position_ratio += allowed_ratio
                        holdings.append(stock_id)
                        state.position_direction = -1
                        state.entry_price = float(current_price)
                        state.base_position_ratio = allowed_ratio
                        state.lowest_price = float(current_price)
                        state.reversal_price = state.lowest_price + state.atr_multiplier * atr
                        logger.info(
                            f"[{current_time}] {stock_id} 初始开仓（做空）："
                            f"价格={current_price:.4f}, ATR={atr:.4f}, 反转价格={state.reversal_price:.4f}, "
                            f"数量={amount:.0f}, 仓位比例={allowed_ratio:.2%}"
                        )
                        continue

            else:
                # existing position: check reverse
                rev = self._should_reverse(state, float(current_price), atr)
                if rev is not None:
                    # sell current
                    if current_amount > 0:
                        orders.append(Order(stock_id=stock_id, amount=current_amount, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.SELL, factor=1.0))
                        direction_str = "做多" if state.position_direction == 1 else "做空"
                        logger.info(
                            f"[{current_time}] {stock_id} 止损平仓（{direction_str}）："
                            f"平仓价={current_price:.4f}, 反转价格={state.reversal_price:.4f}, "
                            f"数量={current_amount:.0f}, 盈亏={(current_price - state.entry_price) / state.entry_price * 100:.2f}%"
                        )
                    # open reversed if allowed and capacity exists
                    allowed_ratio = min(state.initial_position_ratio, max(0.0, 1.0 - (total_position_ratio - current_position_ratio)))
                    if rev == 1 or (rev == -1 and state.enable_short):
                        if allowed_ratio > 0:
                            target_value = account_value * allowed_ratio
                            amount = target_value / float(current_price)
                            if amount > 0:
                                orders.append(Order(stock_id=stock_id, amount=amount, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY, factor=1.0))
                                total_position_ratio += allowed_ratio
                                state.position_direction = rev
                                state.entry_price = float(current_price)
                                state.base_position_ratio = allowed_ratio
                                state.current_grid_level = 0
                                if rev == 1:
                                    state.highest_price = float(current_price)
                                    state.reversal_price = state.highest_price - state.atr_multiplier * atr
                                    logger.info(
                                        f"[{current_time}] {stock_id} 反手开仓（做多）："
                                        f"开仓价={current_price:.4f}, 反转价格={state.reversal_price:.4f}, "
                                        f"数量={amount:.0f}, 仓位比例={allowed_ratio:.2%}"
                                    )
                                else:
                                    state.lowest_price = float(current_price)
                                    state.reversal_price = state.lowest_price + state.atr_multiplier * atr
                                    logger.info(
                                        f"[{current_time}] {stock_id} 反手开仓（做空）："
                                        f"开仓价={current_price:.4f}, 反转价格={state.reversal_price:.4f}, "
                                        f"数量={amount:.0f}, 仓位比例={allowed_ratio:.2%}"
                                    )
                    else:
                        # if cannot open reverse, just keep position closed
                        state.position_direction = 0
                        state.entry_price = None
                        state.base_position_ratio = 0.0
                        state.current_grid_level = 0
                        state.highest_price = None
                        state.lowest_price = None
                        state.reversal_price = None
                        logger.info(f"[{current_time}] {stock_id} 做空被禁用，平仓后不反手")
                    continue

                # existing position: check add
                if self._should_add_position(state, current_position_ratio, float(current_price), atr):
                    add_ratio = state.initial_position_ratio * state.grid_size
                    allowed_add = min(add_ratio, state.max_position_ratio - current_position_ratio, max(0.0, 1.0 - total_position_ratio))
                    if allowed_add > 0:
                        add_value = account_value * allowed_add
                        add_amount = add_value / float(current_price)
                        if add_amount > 0:
                            orders.append(Order(stock_id=stock_id, amount=add_amount, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY, factor=1.0))
                            state.current_grid_level += 1
                            total_position_ratio += allowed_add
                            direction_str = "做多" if state.position_direction == 1 else "做空"
                            if state.position_direction == 1:
                                drawdown = state.highest_price - current_price if state.highest_price else 0
                                logger.info(
                                    f"[{current_time}] {stock_id} {direction_str}加仓（等级{state.current_grid_level}）："
                                    f"价格={current_price:.4f}, 最高价={state.highest_price:.4f}, 回调={drawdown:.4f}, "
                                    f"加仓数量={add_amount:.0f}, 加仓比例={allowed_add:.2%}"
                                )
                            else:
                                rally = current_price - state.lowest_price if state.lowest_price else 0
                                logger.info(
                                    f"[{current_time}] {stock_id} {direction_str}加仓（等级{state.current_grid_level}）："
                                    f"价格={current_price:.4f}, 最低价={state.lowest_price:.4f}, 上涨={rally:.4f}, "
                                    f"加仓数量={add_amount:.0f}, 加仓比例={allowed_add:.2%}"
                                )
                            continue

                # existing position: check reduce on profit
                if self._should_reduce_position_on_profit(state, current_position_ratio, float(current_price), atr):
                    reduce_ratio = max(0.0, current_position_ratio - state.base_position_ratio)
                    reduce_amount = (account_value * reduce_ratio) / float(current_price)
                    if reduce_amount > 0 and reduce_amount < current_amount:
                        orders.append(Order(stock_id=stock_id, amount=reduce_amount, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.SELL, factor=1.0))
                        state.current_grid_level = 0
                        state.entry_price = float(current_price)
                        total_position_ratio -= reduce_ratio
                        direction_str = "做多" if state.position_direction == 1 else "做空"
                        if state.position_direction == 1:
                            drawdown = state.highest_price - current_price if state.highest_price else 0
                            logger.info(
                                f"[{current_time}] {stock_id} {direction_str}盈利平仓："
                                f"价格={current_price:.4f}, 最高价={state.highest_price:.4f}, 回调={drawdown:.4f}, "
                                f"平仓数量={reduce_amount:.0f}, 平仓比例={reduce_ratio:.2%}"
                            )
                        else:
                            rally = current_price - state.lowest_price if state.lowest_price else 0
                            logger.info(
                                f"[{current_time}] {stock_id} {direction_str}盈利平仓："
                                f"价格={current_price:.4f}, 最低价={state.lowest_price:.4f}, 上涨={rally:.4f}, "
                                f"平仓数量={reduce_amount:.0f}, 平仓比例={reduce_ratio:.2%}"
                            )

        return TradeDecisionWO(orders, self)
        