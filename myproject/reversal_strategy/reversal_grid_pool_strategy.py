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
        self.trend_threshold = float(config.get('trend_threshold', 0.01))
        self.atr_period = int(config.get('atr_period', 14))
        self.atr_multiplier = float(config.get('atr_multiplier', 2.0))
        self.grid_atr_interval = float(config.get('grid_atr_interval', 0.2))
        self.entry_atr_multiplier = float(config.get('entry_atr_multiplier', 1.0))
        self.lookback_period = int(config.get('lookback_period', 5))
        self.grid_size = float(config.get('grid_size', 0.2))
        self.profit_threshold = float(config.get('profit_threshold', 0.03))
        self.enable_short = bool(config.get('enable_short', False))

        # position sizing
        self.initial_position_ratio = float(config.get('initial_position_ratio', 0.05))
        self.max_position_ratio = float(config.get('max_position_ratio', min(self.initial_position_ratio * 3, 0.8)))

        # runtime state
        self.position_direction = 0
        self.entry_price = None
        self.base_position_ratio = 0.0
        self.current_grid_level = 0
        self.highest_price = None
        self.lowest_price = None
        self.reversal_price = None


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

    def _get_trend(self, price_df: pd.DataFrame, period: int, trend_threshold: float) -> Optional[int]:
        try:
            close = price_df['$close']
            if len(close) < period + 1:
                return None
            momentum = (close.iloc[-1] - close.iloc[-(period + 1)]) / close.iloc[-(period + 1)]
            if momentum > trend_threshold:
                return 1
            if momentum < -trend_threshold:
                return -1
            return 0
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
        stocks_to_process = list(set(holdings + candidates))[: max(len(holdings), self.max_holding_stocks * 2)]

        for stock_id in stocks_to_process:
            state = self.stock_states.get(stock_id)
            if state is None:
                continue
            price_df = self._get_price_data(stock_id, current_time)
            if price_df is None:
                continue
            atr = self._calculate_atr(price_df, state.atr_period)
            if atr is None or atr <= 0:
                continue
            long_trend = self._get_trend(price_df, state.long_trend_period, state.trend_threshold)
            short_trend = self._get_trend(price_df, state.short_trend_period, state.trend_threshold)
            if long_trend is None or short_trend is None:
                continue

            # get current price and amount
            try:
                current_price = self.trade_exchange.get_close(stock_id=stock_id, start_time=current_time, end_time=trade_end_time, method="ts_data_last")
                if hasattr(current_price, 'value'):
                    current_price = current_price.value
                elif isinstance(current_price, (pd.Series, pd.DataFrame)):
                    current_price = current_price.iloc[-1] if len(current_price) > 0 else None
            except Exception:
                continue
            if current_price is None or pd.isna(current_price):
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
                        continue

            else:
                # existing position: check reverse
                rev = self._should_reverse(state, float(current_price), atr)
                if rev is not None:
                    # sell current
                    if current_amount > 0:
                        orders.append(Order(stock_id=stock_id, amount=current_amount, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.SELL, factor=1.0))
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
                                else:
                                    state.lowest_price = float(current_price)
                                    state.reversal_price = state.lowest_price + state.atr_multiplier * atr
                    else:
                        # if cannot open reverse, just keep position closed
                        state.position_direction = 0
                        state.entry_price = None
                        state.base_position_ratio = 0.0
                        state.current_grid_level = 0
                        state.highest_price = None
                        state.lowest_price = None
                        state.reversal_price = None
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

        return TradeDecisionWO(orders, self)
"""
反转网格股票池策略

基于单股反转网格策略扩展，支持多只股票同时运行，每只股票有独立的策略参数。

策略特点：
1. 从股票池中选择符合开仓条件的股票
2. 最多同时持仓N只股票
3. 每只股票有独立的策略参数（最高持仓金额、趋势参数、ATR参数等）
4. 统一的资金管理和风险控制

使用方法：
    在配置文件中定义股票池和每只股票的参数，然后运行回测。
    
Author: AI Assistant
Date: 2026-01-19
"""

import pandas as pd
import numpy as np
import warnings
from typing import List, Dict, Optional, Tuple
from datetime import timedelta

from qlib.strategy.base import BaseStrategy
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.backtest.signal import create_signal_from
from qlib.data import D
from qlib.log import get_module_logger

# 抑制 qlib 内部的 RuntimeWarning（空切片警告，不影响功能）
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in')

logger = get_module_logger("ReversalGridPoolStrategy")


class StockState:
    """单只股票的状态管理"""
    def __init__(self, stock_id: str, config: Dict):
        self.stock_id = stock_id
        self.config = config

        # 策略参数（从配置中读取）
        self.long_trend_period = config.get('long_trend_period', 20)
        self.short_trend_period = config.get('short_trend_period', 5)
        self.trend_threshold = config.get('trend_threshold', 0.001)
        self.atr_period = config.get('atr_period', 14)
        self.atr_multiplier = config.get('atr_multiplier', 3.0)
        self.grid_atr_interval = config.get('grid_atr_interval', 0.5)
        self.entry_atr_multiplier = config.get('entry_atr_multiplier', 1.0)
        self.lookback_period = config.get('lookback_period', 30)
        self.grid_size = config.get('grid_size', 0.5)
        self.profit_threshold = config.get('profit_threshold', 0.03)
        self.enable_short = config.get('enable_short', True)

        # 仓位配置（百分比）
        self.initial_position_ratio = config.get('initial_position_ratio', 0.1)
        self.max_position_ratio = config.get('max_position_ratio', 0.8)

        # 持仓状态
        self.position_direction = 0  # 0: 无持仓, 1: 做多, -1: 做空
        self.entry_price = None  # 开仓价格
        self.base_position_ratio = 0.0  # 基础仓位比例
        self.current_grid_level = 0  # 当前网格层级
        self.last_grid_price = None  # 上次加仓价格
        self.lowest_price = None  # 做多时的最低价
        self.highest_price = None  # 做多时的最高价
        self.reversal_price = None  # 反转价格（止损价格）
        self.entry_time = None  # 开仓时间


class ReversalGridPoolStrategy(BaseStrategy):
    """
    反转网格股票池策略
    
    直接在池策略中实现所有逻辑，复用单股策略的完整逻辑。
    """
    
    def __init__(
        self,
        stock_pool: Dict[str, Dict],  # 股票池配置：{stock_id: config}
        max_holding_stocks: int = 5,  # 最多同时持仓股票数
        **kwargs
    ):
        super().__init__(**kwargs)

        self.stock_pool = stock_pool
        self.max_holding_stocks = max_holding_stocks

        # 为每只股票创建状态对象
        self.stock_states: Dict[str, StockState] = {}
        for stock_id, config in stock_pool.items():
            self.stock_states[stock_id] = StockState(stock_id, config)

        # 组合级别的风险控制
        self.portfolio_high_water_mark = None  # 组合最高净值
        self.max_portfolio_drawdown = 0.0  # 组合最大回撤
        self.last_portfolio_drawdown = 0.0  # 上一个交易日的组合回撤
        self.force_reduce_triggered = False  # 是否已经触发过强制减仓

        logger.info(f"策略初始化完成：股票池大小={len(stock_pool)}, 最大持仓数={max_holding_stocks}")
        for stock_id in list(stock_pool.keys())[:3]:  # 只打印前3只股票的配置
            config = stock_pool[stock_id]
            logger.info(f"  {stock_id}: initial_ratio={config.get('initial_position_ratio', 0.05)}, "
                       f"atr_multiplier={config.get('atr_multiplier', 2.0)}")

    def _get_price_data(self, stock_id: str, current_time: pd.Timestamp) -> Optional[pd.DataFrame]:
        """获取价格数据"""
        try:
            # 首先尝试从 trade_exchange 获取数据
            price_df = None
            if hasattr(self, 'trade_exchange') and self.trade_exchange is not None:
                try:
                    if hasattr(self.trade_exchange, 'quote_df') and self.trade_exchange.quote_df is not None:
                        quote_df = self.trade_exchange.quote_df
                        if isinstance(quote_df.index, pd.MultiIndex):
                            instrument_level = quote_df.index.get_level_values(0)
                            if stock_id in instrument_level:
                                stock_data = quote_df.loc[stock_id]
                                stock_data = stock_data[stock_data.index <= current_time]
                                stock_data = stock_data.sort_index()
                                required_cols = ['$open', '$high', '$low', '$close']
                                if all(col in stock_data.columns for col in required_cols):
                                    price_df = stock_data[required_cols].copy()
                except Exception as e:
                    logger.debug(f"从 exchange 获取数据失败: {e}")
            
            # 如果从 exchange 获取失败，使用 D.features
            if price_df is None or len(price_df) == 0:
                state = self.stock_states.get(stock_id)
                if state is None:
                    return None
                
                lookback_days = max(state.long_trend_period, state.short_trend_period, state.atr_period) + 10
                start_time = current_time - timedelta(days=lookback_days)
                
                price_data = D.features(
                    [stock_id],
                    ["$open", "$high", "$low", "$close"],
                    start_time=start_time,
                    end_time=current_time,
                    freq="day",
                    disk_cache=True,
                )
                
                if price_data is None or len(price_data) == 0:
                    return None
                
                if isinstance(price_data.index, pd.MultiIndex):
                    if stock_id in price_data.index.get_level_values(0):
                        price_df = price_data.loc[stock_id].copy()
                    else:
                        price_df = price_data.iloc[:, :].copy()
                else:
                    price_df = price_data.copy()
            
            price_df = price_df.dropna()
            state = self.stock_states.get(stock_id)
            if state is None:
                return None
            
            min_required = max(state.long_trend_period, state.short_trend_period, state.atr_period) + 1
            if len(price_df) < min_required:
                return None
            
            return price_df
            
        except Exception as e:
            logger.debug(f"获取 {stock_id} 价格数据时出错: {e}")
            return None
    
    def _calculate_atr(self, price_df: pd.DataFrame, atr_period: int) -> Optional[float]:
        """计算ATR"""
        try:
            if price_df is None or len(price_df) < atr_period + 1:
                return None
            
            high = price_df['$high'] if '$high' in price_df.columns else price_df.iloc[:, 1]
            low = price_df['$low'] if '$low' in price_df.columns else price_df.iloc[:, 2]
            close = price_df['$close'] if '$close' in price_df.columns else price_df.iloc[:, 3]
            
            tr_list = []
            for i in range(1, len(price_df)):
                tr1 = high.iloc[i] - low.iloc[i]
                tr2 = abs(high.iloc[i] - close.iloc[i-1])
                tr3 = abs(low.iloc[i] - close.iloc[i-1])
                tr = max(tr1, tr2, tr3)
                tr_list.append(tr)
            
            if len(tr_list) < atr_period:
                return None
            
            recent_tr = tr_list[-atr_period:]
            atr = np.mean(recent_tr)
            return atr
            
        except Exception as e:
            logger.debug(f"计算ATR时出错: {e}")
            return None
    
    def _get_trend(self, price_df: pd.DataFrame, period: int, trend_threshold: float) -> Optional[int]:
        """
        获取趋势方向（基于动量，优化版本）
        
        返回整数：1=上涨，-1=下跌，0=持平，None=无法判断
        """
        try:
            if price_df is None or len(price_df) < period + 1:
                return None

            # 提取收盘价
            close = price_df['$close'] if '$close' in price_df.columns else price_df.iloc[:, -1]

            if len(close) < period + 1:
                return None

            # 使用收益率序列计算动量（与单股策略保持一致）
            returns = close.pct_change().dropna()
            if len(returns) < period:
                return None

            recent_returns = returns.iloc[-period:]
            momentum = recent_returns.mean()

            if momentum > trend_threshold:
                return 1
            elif momentum < -trend_threshold:
                return -1
            else:
                return 0

        except Exception as e:
            logger.debug(f"计算趋势时出错: {e}")
            return None
    
    def _get_recent_extreme_prices(self, price_df: pd.DataFrame, lookback_period: int) -> Tuple[Optional[float], Optional[float]]:
        """获取近期历史数据的最低点和最高点"""
        try:
            if price_df is None or len(price_df) == 0:
                return None, None
            
            close = price_df['$close'] if '$close' in price_df.columns else price_df.iloc[:, -1]
            
            if len(close) > lookback_period:
                recent_close = close.iloc[-lookback_period:]
            else:
                recent_close = close
            
            if len(recent_close) == 0:
                return None, None
            
            lowest_price = recent_close.min()
            highest_price = recent_close.max()
            
            return lowest_price, highest_price
            
        except Exception as e:
            logger.debug(f"计算近期极值价格时出错: {e}")
            return None, None
    
    def _should_go_long(self, state: StockState, long_trend: int, short_trend: int, current_price: float, atr: float, price_df: pd.DataFrame) -> Tuple[bool, Optional[float]]:
        """
        判断是否应该做多
        
        优化后的做多条件（恢复更严格的条件，提高开仓质量）：
        1. 短期趋势必须上涨
        2. 长期趋势下跌或持平（恢复原逻辑，避免追高）
        3. 较近期最低点有N个ATR涨幅（使用配置的entry_atr_multiplier，不降低）
        """
        # 短期趋势必须是上涨
        if short_trend != 1:
            return False, None
        
        # 恢复长期趋势要求：长期趋势必须是下跌或持平（避免追高）
        if long_trend == 1:  # 长期上涨，不符合做多条件（避免追高）
            return False, None
        
        # 基于历史数据计算近期最低点
        recent_lowest, _ = self._get_recent_extreme_prices(price_df, state.lookback_period)
        if recent_lowest is None:
            return False, None
        
        # 判断是否较近期最低点有N个ATR涨幅
        # 恢复使用配置的entry_atr_multiplier，不降低（提高开仓质量）
        price_rise = current_price - recent_lowest
        min_rise = state.entry_atr_multiplier * atr
        
        if price_rise >= min_rise:
            reversal_price = current_price - state.atr_multiplier * atr
            return True, reversal_price
        
        return False, None
    
    def _should_go_short(self, state: StockState, long_trend: int, short_trend: int, current_price: float, atr: float, price_df: pd.DataFrame) -> Tuple[bool, Optional[float]]:
        """
        判断是否应该做空
        
        做空条件：长期趋势上涨或持平 + 短期趋势下跌 + 较近期最高点有N个ATR下跌
        """
        if not state.enable_short:
            return False, None
        
        # 长期趋势必须是上涨或持平
        if long_trend == -1:  # 长期下跌，不符合做空条件
            return False, None
        
        # 短期趋势必须是下跌
        if short_trend != -1:
            return False, None
        
        # 基于历史数据计算近期最高点
        _, recent_highest = self._get_recent_extreme_prices(price_df, state.lookback_period)
        if recent_highest is None:
            return False, None
        
        # 判断是否较近期最高点有N个ATR下跌
        price_drop = recent_highest - current_price
        min_drop = state.entry_atr_multiplier * atr
        
        if price_drop >= min_drop:
            reversal_price = current_price + state.atr_multiplier * atr
            return True, reversal_price
        
        return False, None
    
    def _should_reverse(self, state: StockState, current_price: float, atr: float) -> Optional[int]:
        """
        判断是否应该反手
        
        反手逻辑：
        - 做多时：跟踪最高点，如果价格从最高点跌回N个ATR则反手做空
        - 做空时：跟踪最低点，如果价格从最低点涨回N个ATR则反手做多
        """
        if state.position_direction == 0 or state.reversal_price is None:
            return None
        
        if state.position_direction == 1:  # 当前做多
            if current_price <= state.reversal_price:
                if not state.enable_short:
                    return None
                return -1
        elif state.position_direction == -1:  # 当前做空
            if current_price >= state.reversal_price:
                return 1
        
        return None
    
    def _calculate_pnl_ratio(self, entry_price: float, current_price: float, direction: int) -> float:
        """计算盈亏比例"""
        if entry_price is None or entry_price <= 0:
            return 0.0
        
        if direction == 1:  # 做多
            return (current_price - entry_price) / entry_price
        elif direction == -1:  # 做空
            return (entry_price - current_price) / entry_price
        else:
            return 0.0
    
    def _should_add_position(self, state: StockState, current_position_ratio: float, current_price: float, atr: float) -> bool:
        """
        判断是否应该加仓（基于最高/最低点的回调幅度）
        """
        if current_position_ratio >= state.max_position_ratio:
            return False
        
        if state.position_direction == 1:  # 做多
            if state.highest_price is None:
                highest_price = state.entry_price if state.entry_price is not None else current_price
            else:
                highest_price = state.highest_price
            
            drawdown = highest_price - current_price
            required_drawdown = (state.current_grid_level + 1) * state.grid_atr_interval * atr
            reversal_drawdown = state.atr_multiplier * atr
            
            if drawdown >= required_drawdown and drawdown < reversal_drawdown:
                return True
                
        elif state.position_direction == -1:  # 做空
            if state.lowest_price is None:
                lowest_price = state.entry_price if state.entry_price is not None else current_price
            else:
                lowest_price = state.lowest_price
            
            rally = current_price - lowest_price
            required_rally = (state.current_grid_level + 1) * state.grid_atr_interval * atr
            reversal_rally = state.atr_multiplier * atr
            
            if rally >= required_rally and rally < reversal_rally:
                return True
        
        return False
    
    def _should_reduce_position_on_profit(self, state: StockState, current_position_ratio: float, current_price: float, atr: float) -> bool:
        """
        判断是否应该平掉加仓部分（基于峰值计算，与加仓逻辑一致）
        """
        if current_position_ratio <= state.base_position_ratio:
            return False
                    if should_long:
                        # 使用单股策略的固定初始仓位，但受组合剩余空间限制
                        desired_ratio = state.initial_position_ratio
                        allowed_ratio = min(desired_ratio, max(0.0, 1.0 - total_position_ratio))
                        if allowed_ratio <= 0:
                            logger.debug(f"[{current_time}] {stock_id} 无剩余资金以开新仓，跳过开仓")
                        else:
                            target_position_value = account_value * allowed_ratio
                            new_amount = target_position_value / current_price * 0.99

                            order = Order(
                                stock_id=stock_id,
                                amount=new_amount,
                                start_time=trade_start_time,
                                end_time=trade_end_time,
                                direction=OrderDir.BUY,
                                factor=factor,
                            )
                                # 只有当前仓位大于基础仓位时才平仓
                                if current_position_ratio <= state.base_position_ratio:
                                    return False

                                # 如果没有加仓（等级为0），不需要平仓
                                if state.current_grid_level == 0:
                                    return False

                                if state.position_direction == 1:  # 做多
                                    # 做多时，根据最高价回调来判断是否应该平仓
                                    if state.highest_price is None:
                                        return False

                                    # 计算当前回调幅度
                                    drawdown = state.highest_price - current_price

                                    # 如果当前回调幅度 < (当前加仓等级 - 1) * grid_atr_interval * atr
                                    # 说明价格已经恢复到接近最高点，应该平掉加仓部分
                                    threshold_drawdown = (state.current_grid_level - 1) * state.grid_atr_interval * atr

                                    if drawdown < threshold_drawdown:
                                        return True

                                elif state.position_direction == -1:  # 做空
                                    # 做空时，根据最低价上涨来判断是否应该平仓
                                    if state.lowest_price is None:
                                        return False

                                    # 计算当前上涨幅度
                                    rally = current_price - state.lowest_price

                                    # 如果当前上涨幅度 < (当前加仓等级 - 1) * grid_atr_interval * atr
                                    threshold_rally = (state.current_grid_level - 1) * state.grid_atr_interval * atr

                                    if rally < threshold_rally:
                                        return True

                                return False
                    )
                    if hasattr(p, 'value'):
                        p = p.value
                    elif isinstance(p, (pd.Series, pd.DataFrame)):
                        p = p.iloc[-1] if len(p) > 0 else None
                except Exception:
                    p = None
                if p is not None and not pd.isna(p):
                    total_position_value += amt * p
        except Exception:
            total_position_value = 0.0

        total_position_ratio = total_position_value / account_value if account_value > 0 else 0.0
        
        # 组合回撤控制
        force_reduce_position = False
        force_close_all = False
        reduce_ratio = 0.0
        
        # 优化：放宽组合回撤控制，只在回撤严重且持续恶化时才强制平仓
        # 如果回撤超过25%但正在恢复（回撤比上次好），允许继续持仓
        if portfolio_drawdown < -0.25:
            # 性能优化：如果已经没有持仓，直接返回，避免重复处理
            if len(current_positions) == 0:
                self.last_portfolio_drawdown = portfolio_drawdown
                return TradeDecisionWO([], self)
            
            # 优化：如果回撤正在恢复（比上次好0.05%以上），不强制平仓
            if self.last_portfolio_drawdown < 0 and portfolio_drawdown > self.last_portfolio_drawdown + 0.0005:
                # 回撤正在恢复，允许继续持仓
                logger.info(f"[{current_time}] 组合回撤超过25%（{portfolio_drawdown:.2%}），但正在恢复，暂不强制平仓")
                self.last_portfolio_drawdown = portfolio_drawdown
                # 继续执行后续逻辑，不强制平仓
            else:
                # 回撤严重且持续恶化，强制平仓
                force_close_all = True
                logger.warning(f"[{current_time}] ⚠️ 组合回撤超过25%（{portfolio_drawdown:.2%}）且持续恶化，强制平掉所有持仓！")
                
                for stock_id, amount in current_positions.items():
                    order = Order(
                        stock_id=stock_id,
                        amount=amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=OrderDir.SELL,
                        factor=1.0,
                    )
                    orders.append(order)
                    if stock_id in self.stock_states:
                        state = self.stock_states[stock_id]
                        state.position_direction = 0
                        state.entry_price = None
                        state.base_position_ratio = 0.0
                        state.current_grid_level = 0
                
                if portfolio_drawdown > self.last_portfolio_drawdown + 0.01:
                    self.force_reduce_triggered = False
                
                self.last_portfolio_drawdown = portfolio_drawdown
                return TradeDecisionWO(orders, self)
        
        # 优化：放宽组合回撤控制，只在回撤严重且持续恶化时才减仓
        elif portfolio_drawdown < -0.20:
            # 只有在回撤持续恶化时才减仓
            if portfolio_drawdown < self.last_portfolio_drawdown - 0.05:
                force_reduce_position = True
                reduce_ratio = 0.4  # 降低减仓幅度
                self.force_reduce_triggered = True
                logger.warning(f"[{current_time}] ⚠️ 组合回撤超过20%（{portfolio_drawdown:.2%}）且持续恶化，强制减仓40%！")
        elif portfolio_drawdown < -0.15:
            if portfolio_drawdown < self.last_portfolio_drawdown - 0.05:
                force_reduce_position = True
                reduce_ratio = 0.3  # 降低减仓幅度
                self.force_reduce_triggered = True
                logger.warning(f"[{current_time}] ⚠️ 组合回撤超过15%（{portfolio_drawdown:.2%}）且持续恶化，强制减仓30%！")
        # 取消10%和5%的减仓，避免过度干预
        
        if portfolio_drawdown > self.last_portfolio_drawdown + 0.01:
            self.force_reduce_triggered = False
        
        self.last_portfolio_drawdown = portfolio_drawdown
        
        # 处理每只股票
        candidate_new_positions = []
        
        # 只处理持仓股票和候选股票（限制数量以提升性能）
        stocks_to_process = list(current_holdings)
        if len(current_holdings) < self.max_holding_stocks:
            remaining_slots = self.max_holding_stocks - len(current_holdings)
            candidate_stocks = [s for s in self.stock_states.keys() if s not in current_holdings]
            # 进一步减少候选股票数量，只处理必要的数量
            stocks_to_process.extend(candidate_stocks[:min(remaining_slots * 2, 10)])
        
        for stock_id in stocks_to_process:
            state = self.stock_states.get(stock_id)
            if state is None:
                continue
            
            try:
                # 获取当前价格
                try:
                    current_price = self.trade_exchange.get_close(
                        stock_id=stock_id,
                        start_time=current_time,
                        end_time=trade_end_time,
                        method="ts_data_last"
                    )
                    if hasattr(current_price, 'value'):
                        current_price = current_price.value
                    elif isinstance(current_price, (pd.Series, pd.DataFrame)):
                        current_price = current_price.iloc[-1] if len(current_price) > 0 else None
                except Exception as e:
                    logger.debug(f"无法获取 {stock_id} 的价格: {e}")
                    continue
                
                if current_price is None or pd.isna(current_price) or account_value <= 0:
                    continue
                
                # 对于无持仓的候选股票，先快速筛选（只检查趋势，不计算完整逻辑）
                is_holding = stock_id in current_holdings
                
                # 获取价格数据
                price_df = self._get_price_data(stock_id, current_time)
                if price_df is None or len(price_df) == 0:
                    continue
                
                # 计算长期和短期趋势（先计算趋势，用于快速筛选）
                long_trend = self._get_trend(price_df, state.long_trend_period, state.trend_threshold)
                short_trend = self._get_trend(price_df, state.short_trend_period, state.trend_threshold)
                
                if long_trend is None or short_trend is None:
                    continue
                
                # 对于无持仓的候选股票，如果趋势不符合开仓条件，直接跳过（不计算ATR等）
                if not is_holding:
                    # 快速筛选：恢复更严格的条件，提高开仓质量
                    # 做多条件 = 长期趋势下跌或持平 + 短期趋势上涨
                    # 做空条件 = 长期趋势上涨或持平 + 短期趋势下跌（如果允许做空）
                    can_long = (long_trend <= 0 and short_trend == 1)  # 恢复长期趋势要求
                    can_short = (state.enable_short and long_trend >= 0 and short_trend == -1)
                    
                    if not can_long and not can_short:
                        continue  # 趋势不符合，跳过后续计算
                
                # 计算ATR（只有持仓股票或通过快速筛选的候选股票才计算）
                atr = self._calculate_atr(price_df, state.atr_period)
                if atr is None or atr <= 0:
                    continue
                
                # 获取当前持仓
                current_amount = self.trade_position.get_stock_amount(stock_id)
                current_position_value = abs(current_amount * current_price)
                current_position_ratio = current_position_value / account_value if account_value > 0 else 0
                
                # 更新状态
                if current_amount == 0 and state.position_direction != 0:
                    state.position_direction = 0
                    state.entry_price = None
                    # 重置为初始仓位比例（下次开仓时会动态调整）
                    state.base_position_ratio = state.initial_position_ratio
                    state.current_grid_level = 0
                    state.last_grid_price = None
                    state.lowest_price = None
                    state.highest_price = None
                    state.reversal_price = None
                
                if current_amount > 0 and state.position_direction == 0:
                    state.position_direction = 1
                    state.entry_price = current_price
                    # 只有当前仓位大于基础仓位时才平仓
                    if current_position_ratio <= state.base_position_ratio:
                        return False

                    # 如果没有加仓（等级为0），不需要平仓
                    if state.current_grid_level == 0:
                        return False

                    if state.position_direction == 1:  # 做多
                        # 做多时，根据最高价回调来判断是否应该平仓
                        if state.highest_price is None:
                            return False

                        # 计算当前回调幅度
                        drawdown = state.highest_price - current_price

                        # 如果当前回调幅度 < (当前加仓等级 - 1) * grid_atr_interval * atr
                        threshold_drawdown = (state.current_grid_level - 1) * state.grid_atr_interval * atr

                        if drawdown < threshold_drawdown:
                            return True

                    elif state.position_direction == -1:  # 做空
                        # 做空时，根据最低价上涨来判断是否应该平仓
                        if state.lowest_price is None:
                            return False

                        # 计算当前上涨幅度
                        rally = current_price - state.lowest_price

                        # 如果当前上涨幅度 < (当前加仓等级 - 1) * grid_atr_interval * atr
                        threshold_rally = (state.current_grid_level - 1) * state.grid_atr_interval * atr

                        if rally < threshold_rally:
                            return True

                    return False
                        state.current_grid_level = 0
                        state.last_grid_price = None
                        state.lowest_price = None
                        state.highest_price = None
                        state.reversal_price = None
                        reverse_direction = None
                    
                    if reverse_direction is not None:
                        # 反手逻辑（使用固定的 initial_position_ratio，与单股策略一致）
                        dynamic_position_ratio = state.initial_position_ratio

                        # 当反手会先平掉当前仓位，再开新仓，计算可用的组合仓位空间
                        effective_total_ratio = max(0.0, total_position_ratio - current_position_ratio)
                        allowed_ratio = min(dynamic_position_ratio, max(0.0, 1.0 - effective_total_ratio))

                        if allowed_ratio <= 0:
                            # 无可用资金，先平仓当前仓位不再反手
                            if current_amount > 0:
                                order = Order(
                                    stock_id=stock_id,
                                    amount=current_amount,
                                    start_time=trade_start_time,
                                    end_time=trade_end_time,
                                    direction=OrderDir.SELL,
                                    factor=factor,
                                )
                                stock_orders.append(order)

                                state.position_direction = 0
                                state.entry_price = None
                                state.base_position_ratio = 0.0
                                state.current_grid_level = 0
                                state.last_grid_price = None
                                state.lowest_price = None
                                state.highest_price = None
                                state.reversal_price = None
                                logger.info(f"[{current_time}] {stock_id} 无可用资金执行反手，已平仓但不再建反向仓位")
                        else:
                            # 以可用比例开新仓
                            estimated_cash = account_value - current_position_value * 0.9985
                            target_position_value = estimated_cash * allowed_ratio * 0.95
                            new_amount = target_position_value / current_price * 0.99
                            if new_amount <= 0:
                                target_position_value = account_value * allowed_ratio * 0.9
                                new_amount = target_position_value / current_price * 0.99

                            if current_amount > 0:
                                order = Order(
                                    stock_id=stock_id,
                                    amount=current_amount,
                                    start_time=trade_start_time,
                                    end_time=trade_end_time,
                                    direction=OrderDir.SELL,
                                    factor=factor,
                                )
                                stock_orders.append(order)

                            order = Order(
                                stock_id=stock_id,
                                amount=new_amount,
                                start_time=trade_start_time,
                                end_time=trade_end_time,
                                direction=OrderDir.BUY,
                                factor=factor,
                            )
                            stock_orders.append(order)

                            state.position_direction = reverse_direction
                            state.entry_price = current_price
                            state.base_position_ratio = allowed_ratio
                            state.current_grid_level = 0
                            state.last_grid_price = None
                            if reverse_direction == 1:
                                state.highest_price = current_price
                                state.lowest_price = None
                                state.reversal_price = state.highest_price - state.atr_multiplier * atr
                            else:
                                state.lowest_price = current_price
                                state.highest_price = None
                                state.reversal_price = state.lowest_price + state.atr_multiplier * atr
                            # 更新组合占比预估，避免当日内再开超额仓
                            total_position_ratio += allowed_ratio
                    
                    # 情况2：当前有持仓，检查是否需要加仓或平仓
                    else:
                        # 更新最低点和最高点
                        if state.position_direction == 1:  # 做多
                            if state.highest_price is None:
                                state.highest_price = current_price
                            elif current_price > state.highest_price:
                                state.highest_price = current_price
                            state.reversal_price = state.highest_price - state.atr_multiplier * atr
                        elif state.position_direction == -1:  # 做空
                            if state.lowest_price is None:
                                state.lowest_price = current_price
                            elif current_price < state.lowest_price:
                                state.lowest_price = current_price
                            state.reversal_price = state.lowest_price + state.atr_multiplier * atr
                        
                        if state.entry_price is None:
                            state.entry_price = current_price
                        
                        # 检查是否需要加仓
                        should_add = self._should_add_position(state, current_position_ratio, current_price, atr)
                        
                        if should_add:
                            add_position_ratio = state.initial_position_ratio * state.grid_size
                            target_position_ratio = min(
                                current_position_ratio + add_position_ratio,
                                state.max_position_ratio
                            )
                            
                            target_position_value = account_value * target_position_ratio
                            current_position_value = current_amount * current_price
                            add_value = target_position_value - current_position_value
                            
                            # 限制加仓金额不超过组合剩余仓位空间
                            remaining_capacity_value = max(0.0, account_value * (1.0 - total_position_ratio))
                            add_value = min(add_value, remaining_capacity_value)

                            if add_value > 0:
                                add_amount = add_value / current_price * 0.99

                                order = Order(
                                    stock_id=stock_id,
                                    amount=add_amount,
                                    start_time=trade_start_time,
                                    end_time=trade_end_time,
                                    direction=OrderDir.BUY,
                                    factor=factor,
                                )
                                stock_orders.append(order)

                                state.current_grid_level += 1
                                # 更新组合占比估算
                                total_position_ratio += add_value / account_value if account_value > 0 else 0.0
                        
                        # 检查是否需要平掉加仓部分
                        elif self._should_reduce_position_on_profit(state, current_position_ratio, current_price, atr):
                            reduce_ratio = current_position_ratio - state.base_position_ratio
                            reduce_value = account_value * reduce_ratio
                            reduce_amount = reduce_value / current_price
                            
                            if reduce_amount > 0 and reduce_amount < current_amount:
                                order = Order(
                                    stock_id=stock_id,
                                    amount=reduce_amount,
                                    start_time=trade_start_time,
                                    end_time=trade_end_time,
                                    direction=OrderDir.SELL,
                                    factor=factor,
                                )
                                stock_orders.append(order)
                                
                                state.current_grid_level = 0
                                state.last_grid_price = None
                                state.entry_price = current_price
                
                # 情况3：当前无仓位，判断是否开仓
                elif state.position_direction == 0:
                    should_long, reversal_price_long = self._should_go_long(state, long_trend, short_trend, current_price, atr, price_df)
                    if should_long:
                        # 优化：根据实际持仓数量动态调整每只股票的仓位，提高资金利用率
                        # 如果持仓数量少，每只股票可以分配更多资金
                        current_holding_count = len(current_holdings)
                        if current_holding_count < self.max_holding_stocks:
                            # 根据剩余仓位空间动态分配
                            remaining_slots = self.max_holding_stocks - current_holding_count
                            # 每只股票最多占总资金的 min(配置比例, 剩余仓位/剩余股票数)
                            dynamic_position_ratio = min(
                                state.initial_position_ratio,
                                0.95 / max(remaining_slots, 1)  # 预留5%现金，其余分配给新开仓股票
                            )
                        else:
                            # 持仓已满，使用最小仓位
                            dynamic_position_ratio = min(
                                state.initial_position_ratio,
                                1.0 / max(self.max_holding_stocks, 1)
                            )
                        target_position_value = account_value * dynamic_position_ratio
                        new_amount = target_position_value / current_price * 0.99
                        
                        order = Order(
                            stock_id=stock_id,
                            amount=new_amount,
                            start_time=trade_start_time,
                            end_time=trade_end_time,
                            direction=OrderDir.BUY,
                            factor=factor,
                        )
                        stock_orders.append(order)
                        
                        state.position_direction = 1
                        state.entry_price = current_price
                        # 使用动态调整后的仓位比例（已在上面计算）
                        state.base_position_ratio = dynamic_position_ratio
                        state.current_grid_level = 0
                        state.last_grid_price = None
                        state.highest_price = current_price
                        state.lowest_price = None
                        state.reversal_price = reversal_price_long
                    else:
                        should_short, reversal_price_short = self._should_go_short(state, long_trend, short_trend, current_price, atr, price_df)
                        if should_short:
                            # 优化：根据实际持仓数量动态调整每只股票的仓位
                            current_holding_count = len(current_holdings)
                            if current_holding_count < self.max_holding_stocks:
                                remaining_slots = self.max_holding_stocks - current_holding_count
                                dynamic_position_ratio = min(
                                    state.initial_position_ratio,
                                    0.95 / max(remaining_slots, 1)
                                )
                            else:
                                dynamic_position_ratio = min(
                                    state.initial_position_ratio,
                                    1.0 / max(self.max_holding_stocks, 1)
                                )
                            target_position_value = account_value * dynamic_position_ratio
                            new_amount = target_position_value / current_price * 0.99
                            
                            order = Order(
                                stock_id=stock_id,
                                amount=new_amount,
                                start_time=trade_start_time,
                                end_time=trade_end_time,
                                direction=OrderDir.BUY,
                                factor=factor,
                            )
                            stock_orders.append(order)
                            
                            state.position_direction = -1
                            state.entry_price = current_price
                            # 使用动态调整后的仓位比例（已在上面计算）
                            state.base_position_ratio = dynamic_position_ratio
                            state.current_grid_level = 0
                            state.last_grid_price = None
                            state.lowest_price = current_price
                            state.highest_price = None
                            state.reversal_price = reversal_price_short
                
                # 如果是强制减仓，修改订单
                if force_reduce_position and stock_id in current_holdings and current_amount > 0:
                    reduce_amount = current_amount * reduce_ratio
                    if reduce_amount > 0:
                        stock_orders = [o for o in stock_orders if o.direction != OrderDir.BUY]
                        existing_sell = sum(o.amount for o in stock_orders if o.direction == OrderDir.SELL)
                        if reduce_amount > existing_sell:
                            additional_reduce = reduce_amount - existing_sell
                            order = Order(
                                stock_id=stock_id,
                                amount=additional_reduce,
                                start_time=trade_start_time,
                                end_time=trade_end_time,
                                direction=OrderDir.SELL,
                                factor=factor,
                            )
                            stock_orders.append(order)
                
                # 检查是否是开仓订单
                is_new_position = stock_id not in current_holdings
                has_buy_order = any(o.direction == OrderDir.BUY for o in stock_orders)
                
                if is_new_position and has_buy_order and len(current_holdings) >= self.max_holding_stocks:
                    candidate_new_positions.append((stock_id, stock_orders))
                    continue
                
                if stock_orders:
                    orders.extend(stock_orders)
                
            except Exception as e:
                logger.error(f"处理股票 {stock_id} 时出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        # 处理候选新开仓
        if len(current_holdings) < self.max_holding_stocks:
            if portfolio_drawdown < -0.15:
                logger.info(f"[{current_time}] 组合回撤超过15%，暂停开新仓")
            else:
                for stock_id, stock_orders in candidate_new_positions:
                    if len(current_holdings) >= self.max_holding_stocks:
                        break
                    
                    adjusted_orders = []
                    for o in stock_orders:
                        if o.direction == OrderDir.BUY:
                            if portfolio_drawdown < -0.10:
                                o.amount = o.amount * 0.5
                            elif portfolio_drawdown < -0.05:
                                o.amount = o.amount * 0.7
                        adjusted_orders.append(o)
                    
                    orders.extend(adjusted_orders)
                    current_holdings.add(stock_id)
        
        return TradeDecisionWO(orders, self)
