# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
趋势反转网格策略（做多做空）

策略逻辑：
1. 永远持仓：一旦开仓后永久持仓，不是多就是空（如果enable_short=False则只做多）
2. 多空判断：使用长期和短期趋势来复合判断
   - 做多条件：长期趋势下跌或持平 + 短期趋势上涨 + 较近期最低点有N个ATR涨幅（基于历史数据）
   - 做多止损/反手：做多时跟踪最高点，如果价格从最高点跌回N个ATR则止损反手做空（如果enable_short=False则平仓）
   - 做空条件：长期趋势上涨或持平 + 短期趋势下跌 + 较近期最高点有N个ATR下跌（基于历史数据，需要enable_short=True）
   - 做空止损/反手：做空时跟踪最低点，如果价格从最低点涨回N个ATR则止损反手做多
3. 网格加仓：基于最高/最低点的回调幅度，每grid_atr_interval个ATR间隔加仓一次
4. 盈利平仓：基于峰值计算，当价格恢复到接近峰值时平掉加仓部分

这是一个基于趋势反转的双向网格策略，通过亏损加仓、盈利平仓来降低成本。
可以通过enable_short参数控制是否允许做空。
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import timedelta

from qlib.backtest.position import Position
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.strategy.base import BaseStrategy
from qlib.data import D
from qlib.log import get_module_logger

logger = get_module_logger("ReversalGridStrategy")


class ReversalGridStrategy(BaseStrategy):
    """
    趋势反转网格策略（做多做空）
    
    策略特点：
    - 针对单只股票
    - 永远持仓：一旦开仓后永久持仓，不是多就是空
    - 使用长期和短期趋势复合判断多空
    - 基于ATR判断趋势反转
    - 亏损时按网格加仓
    - 盈利超过1个点时平掉加仓部分
    """
    
    def __init__(
        self,
        *,
        # 股票代码（单只股票）
        stock_id: str,
        # 趋势计算参数
        long_trend_period: int = 20,  # 长期趋势周期（天）
        short_trend_period: int = 5,   # 短期趋势周期（天）
        trend_threshold: float = 0.001,  # 趋势判断阈值（绝对值小于此值视为持平）
        # ATR参数
        atr_period: int = 14,  # ATR计算周期
        atr_multiplier: float = 3.0,  # ATR倍数（用于判断反转/止损，默认3.0）
        grid_atr_interval: float = 0.5,  # 网格加仓间隔（ATR倍数，默认0.5）
        entry_atr_multiplier: float = 1.0,  # 开仓条件ATR倍数（用于判断开仓，默认1.0）
        # 历史数据回看参数
        lookback_period: int = 30,  # 计算最低点/最高点的历史回看周期（天）
        # 仓位管理参数
        max_position_ratio: float = 0.8,  # 最大仓位比例（不满仓，80%）
        grid_size: float = 0.5,  # 网格大小（每次加仓增加初始仓位的50%）
        initial_position_ratio: float = 0.1,  # 初始仓位比例（10%）
        profit_threshold: float = 0.03,  # 盈利平仓阈值（3个点，避免频繁平仓）
        enable_short: bool = True,  # 是否允许做空（默认True）
        # 交易参数
        risk_degree: float = 0.95,
        only_tradable: bool = False,
        forbid_all_trade_at_limit: bool = True,
        level_infra=None,
        common_infra=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        stock_id : str
            股票代码（单只股票）
        long_trend_period : int
            长期趋势周期（默认20天）
        short_trend_period : int
            短期趋势周期（默认5天）
        trend_threshold : float
            趋势判断阈值，绝对值小于此值视为持平（默认0.001，即0.1%）
        atr_period : int
            ATR计算周期（默认14天）
        atr_multiplier : float
            ATR倍数，用于判断反转/止损（默认3.0，即3个ATR）
        grid_atr_interval : float
            网格加仓ATR间隔（默认0.5，即每0.5个ATR加仓一次）
        entry_atr_multiplier : float
            开仓条件ATR倍数，用于判断开仓（默认1.0，即1个ATR）
            做多：从最低点上涨 >= entry_atr_multiplier * atr
            做空：从最高点下跌 >= entry_atr_multiplier * atr
        lookback_period : int
            计算最低点/最高点的历史回看周期（默认30天）
        max_position_ratio : float
            最大仓位比例（默认0.8，即80%）
        grid_size : float
            网格大小，每次加仓增加初始仓位的比例（默认0.5，即每次加仓增加初始仓位的50%）
            例如：如果initial_position_ratio=0.2，grid_size=0.5，则每次加仓增加10%的仓位
        initial_position_ratio : float
            初始仓位比例（默认0.1，即10%）
        profit_threshold : float
            盈利平仓阈值（默认0.01，即1个点）
        enable_short : bool
            是否允许做空（默认True）
            如果设置为False，策略将只做多，不会开空仓，也不会反手做空
        risk_degree : float
            风险度（默认0.95）
        """
        super().__init__(
            level_infra=level_infra,
            common_infra=common_infra
        )
        self.stock_id = stock_id
        self.long_trend_period = long_trend_period
        self.short_trend_period = short_trend_period
        self.trend_threshold = trend_threshold
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier  # 反转/止损ATR倍数
        self.grid_atr_interval = grid_atr_interval  # 网格加仓ATR间隔
        self.entry_atr_multiplier = entry_atr_multiplier  # 开仓条件ATR倍数
        self.lookback_period = lookback_period  # 历史回看周期
        self.max_position_ratio = max_position_ratio
        self.grid_size = grid_size
        self.initial_position_ratio = initial_position_ratio
        self.profit_threshold = profit_threshold
        self.enable_short = enable_short  # 是否允许做空
        self.risk_degree = risk_degree
        self.only_tradable = only_tradable
        self.forbid_all_trade_at_limit = forbid_all_trade_at_limit
        
        # 参数验证：确保atr_multiplier足够大以支持多次加仓
        # 最大加仓次数 = floor(atr_multiplier / grid_atr_interval)
        max_possible_grid_levels = int(self.atr_multiplier / self.grid_atr_interval)
        if max_possible_grid_levels < 3:
            logger.warning(
                f"atr_multiplier ({self.atr_multiplier}) 相对于 grid_atr_interval ({self.grid_atr_interval}) 太小，"
                f"最多只能加仓 {max_possible_grid_levels} 次。"
                f"建议将 atr_multiplier 设置为至少 {3 * self.grid_atr_interval} 以支持多次加仓。"
            )
        
        # 状态管理
        self.position_direction = 0  # 仓位方向：1=做多，-1=做空，0=无仓位
        self.current_grid_level = 0  # 当前网格等级（0表示初始仓位）
        self.entry_price = None  # 入场价格（用于计算盈亏）
        self.base_position_ratio = 0.0  # 基础仓位比例（初始开仓的仓位）
        self.lowest_price = None  # 做多时的最低点（持仓期间跟踪）
        self.highest_price = None  # 做空时的最高点（持仓期间跟踪）
        self.reversal_price = None  # 反转价格（止损价格，用于判断是否反手）
        self.last_grid_price = None  # 上次网格加仓的价格（用于ATR间隔判断）
        
        # 记录做多/做空状态历史（用于分析）
        self.position_history = []  # 记录每个时间点的状态: [(timestamp, direction, price, pnl_ratio), ...]
        
    def _get_price_data(self, stock_id: str, current_time: pd.Timestamp) -> Optional[pd.DataFrame]:
        """
        获取价格数据（包含open, high, low, close）
         
        Parameters
        ----------
        stock_id : str
            股票代码
        current_time : pd.Timestamp
            当前时间
            
        Returns
        -------
        pd.DataFrame
            包含open, high, low, close的价格数据，索引为datetime
        """
        try:
            # 首先尝试从 trade_exchange 获取数据（如果可用）
            price_df = None
            if hasattr(self, 'trade_exchange') and self.trade_exchange is not None:
                try:
                    # 从 exchange 的 quote_df 获取数据
                    if hasattr(self.trade_exchange, 'quote_df') and self.trade_exchange.quote_df is not None:
                        quote_df = self.trade_exchange.quote_df
                        if isinstance(quote_df.index, pd.MultiIndex):
                            # MultiIndex: (instrument, datetime)
                            instrument_level = quote_df.index.get_level_values(0)
                            if stock_id in instrument_level:
                                stock_data = quote_df.loc[stock_id]
                                # 只取当前时间之前的数据（包括当前时间）
                                stock_data = stock_data[stock_data.index <= current_time]
                                # 按时间排序
                                stock_data = stock_data.sort_index()
                                # 提取需要的列
                                required_cols = ['$open', '$high', '$low', '$close']
                                if all(col in stock_data.columns for col in required_cols):
                                    price_df = stock_data[required_cols].copy()
                except Exception as e:
                    logger.debug(f"从 exchange 获取数据失败: {e}")
            
            # 如果从 exchange 获取失败，使用 D.features
            if price_df is None or len(price_df) == 0:
                # 获取历史数据（需要足够的历史数据来计算ATR和趋势）
                lookback_days = max(self.long_trend_period, self.short_trend_period, self.atr_period) + 10
                start_time = current_time - timedelta(days=lookback_days)
                
                # 获取价格数据
                price_data = D.features(
                    [stock_id],
                    ["$open", "$high", "$low", "$close"],
                    start_time=start_time,
                    end_time=current_time,
                    freq="day",
                    disk_cache=True,
                )
                
                if price_data is None or len(price_data) == 0:
                    logger.debug(f"无法获取 {stock_id} 的价格数据")
                    return None
                
                # 提取数据
                if isinstance(price_data.index, pd.MultiIndex):
                    if stock_id in price_data.index.get_level_values(0):
                        price_df = price_data.loc[stock_id].copy()
                    else:
                        price_df = price_data.iloc[:, :].copy()
                else:
                    price_df = price_data.copy()
            
            # 去除NaN值
            price_df = price_df.dropna()
            
            # 检查数据是否足够
            min_required = max(self.long_trend_period, self.short_trend_period, self.atr_period) + 1
            if len(price_df) < min_required:
                logger.debug(f"{stock_id} 历史数据不足: 需要至少 {min_required} 个数据点，实际只有 {len(price_df)} 个")
                return None
            
            return price_df
            
        except Exception as e:
            logger.debug(f"获取 {stock_id} 价格数据时出错: {e}")
            return None
    
    def _calculate_atr(self, price_df: pd.DataFrame) -> Optional[float]:
        """
        计算ATR（Average True Range）
        
        Parameters
        ----------
        price_df : pd.DataFrame
            包含open, high, low, close的价格数据
            
        Returns
        -------
        float
            当前ATR值
        """
        try:
            if price_df is None or len(price_df) < self.atr_period + 1:
                return None
            
            # 提取数据
            if '$high' in price_df.columns:
                high = price_df['$high']
                low = price_df['$low']
                close = price_df['$close']
            else:
                # 尝试其他可能的列名
                high = price_df.iloc[:, 1] if len(price_df.columns) > 1 else None
                low = price_df.iloc[:, 2] if len(price_df.columns) > 2 else None
                close = price_df.iloc[:, 3] if len(price_df.columns) > 3 else None
            
            if high is None or low is None or close is None:
                return None
            
            # 计算True Range
            tr_list = []
            for i in range(1, len(price_df)):
                tr1 = high.iloc[i] - low.iloc[i]
                tr2 = abs(high.iloc[i] - close.iloc[i-1])
                tr3 = abs(low.iloc[i] - close.iloc[i-1])
                tr = max(tr1, tr2, tr3)
                tr_list.append(tr)
            
            if len(tr_list) < self.atr_period:
                return None
            
            # 计算ATR（简单移动平均）
            recent_tr = tr_list[-self.atr_period:]
            atr = np.mean(recent_tr)
            
            return atr
            
        except Exception as e:
            logger.debug(f"计算ATR时出错: {e}")
            return None
    
    def _get_recent_extreme_prices(self, price_df: pd.DataFrame, lookback_days: int) -> Tuple[Optional[float], Optional[float]]:
        """
        获取近期历史数据的最低点和最高点
        
        Parameters
        ----------
        price_df : pd.DataFrame
            价格数据
        lookback_days : int
            回看天数
            
        Returns
        -------
        Tuple[Optional[float], Optional[float]]
            (最低点, 最高点)
        """
        try:
            if price_df is None or len(price_df) == 0:
                return None, None
            
            # 提取收盘价
            if '$close' in price_df.columns:
                close = price_df['$close']
            else:
                close = price_df.iloc[:, -1]  # 假设最后一列是close
            
            # 只取最近lookback_days天的数据
            if len(close) > lookback_days:
                recent_close = close.iloc[-lookback_days:]
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
    
    def _get_trend(self, price_df: pd.DataFrame, period: int) -> Optional[int]:
        """
        获取趋势方向（基于动量）
        
        使用收益率序列计算动量，而不是直接用价格计算
        
        Parameters
        ----------
        price_df : pd.DataFrame
            价格数据
        period : int
            趋势周期
            
        Returns
        -------
        int
            趋势方向：1=上涨，-1=下跌，0=持平，None=无法判断
        """
        try:
            if price_df is None or len(price_df) < period + 1:
                return None
            
            # 提取收盘价
            if '$close' in price_df.columns:
                close = price_df['$close']
            else:
                close = price_df.iloc[:, -1]  # 假设最后一列是close
            
            if len(close) < period + 1:
                return None
            
            # 计算收益率序列（动量）
            returns = close.pct_change().dropna()
            
            if len(returns) < period:
                return None
            
            # 使用最近period期的收益率计算动量
            # 动量 = 最近N期收益率的平均值（或总和）
            recent_returns = returns.iloc[-period:]
            momentum = recent_returns.mean()  # 使用平均收益率作为动量
            
            # 判断趋势
            if momentum > self.trend_threshold:
                return 1  # 上涨
            elif momentum < -self.trend_threshold:
                return -1  # 下跌
            else:
                return 0  # 持平
                
        except Exception as e:
            logger.debug(f"计算趋势时出错: {e}")
            return None
    
    def _should_go_long(self, long_trend: int, short_trend: int, current_price: float, atr: float, price_df: pd.DataFrame) -> Tuple[bool, Optional[float]]:
        """
        判断是否应该做多
        
        做多条件：长期趋势下跌或持平 + 短期趋势上涨 + 较近期最低点有N个ATR涨幅
        
        Parameters
        ----------
        long_trend : int
            长期趋势：1=上涨，-1=下跌，0=持平
        short_trend : int
            短期趋势：1=上涨，-1=下跌，0=持平
        current_price : float
            当前价格
        atr : float
            当前ATR值
        price_df : pd.DataFrame
            价格数据，用于计算近期最低点
            
        Returns
        -------
        Tuple[bool, Optional[float]]
            (是否应该做多, 反转价格，如果做多则返回做空的反转价格)
        """
        # 长期趋势必须是下跌或持平
        if long_trend == 1:  # 长期上涨，不符合做多条件
            return False, None
        
        # 短期趋势必须是上涨
        if short_trend != 1:
            return False, None
        
        # 基于历史数据计算近期最低点
        recent_lowest, _ = self._get_recent_extreme_prices(price_df, self.lookback_period)
        if recent_lowest is None:
            return False, None
        
        # 判断是否较近期最低点有N个ATR涨幅
        price_rise = current_price - recent_lowest
        
        # 开仓条件：至少要有entry_atr_multiplier个ATR的涨幅，表明有明显的反弹信号
        # 不要求达到完整的atr_multiplier个ATR（那是止损距离，不是开仓距离）
        min_rise = self.entry_atr_multiplier * atr
        
        if price_rise >= min_rise:
            # 满足做多条件，返回反转价格（如果价格从最高点跌回N个ATR则反手做空/止损）
            reversal_price = current_price - self.atr_multiplier * atr
            return True, reversal_price
        
        return False, None
    
    def _should_go_short(self, long_trend: int, short_trend: int, current_price: float, atr: float, price_df: pd.DataFrame) -> Tuple[bool, Optional[float]]:
        """
        判断是否应该做空
        
        做空条件：长期趋势上涨或持平 + 短期趋势下跌 + 较近期最高点有N个ATR下跌
        
        Parameters
        ----------
        long_trend : int
            长期趋势：1=上涨，-1=下跌，0=持平
        short_trend : int
            短期趋势：1=上涨，-1=下跌，0=持平
        current_price : float
            当前价格
        atr : float
            当前ATR值
        price_df : pd.DataFrame
            价格数据，用于计算近期最高点
            
        Returns
        -------
        Tuple[bool, Optional[float]]
            (是否应该做空, 反转价格，如果做空则返回做多的反转价格)
        """
        # 如果做空功能被禁用，直接返回False
        if not self.enable_short:
            return False, None
        
        # 长期趋势必须是上涨或持平
        if long_trend == -1:  # 长期下跌，不符合做空条件
            return False, None
        
        # 短期趋势必须是下跌
        if short_trend != -1:
            return False, None
        
        # 基于历史数据计算近期最高点
        _, recent_highest = self._get_recent_extreme_prices(price_df, self.lookback_period)
        if recent_highest is None:
            return False, None
        
        # 判断是否较近期最高点有N个ATR下跌
        price_drop = recent_highest - current_price
        
        # 开仓条件：至少要有entry_atr_multiplier个ATR的下跌，表明有明显的回调信号
        # 不要求达到完整的atr_multiplier个ATR（那是止损距离，不是开仓距离）
        min_drop = self.entry_atr_multiplier * atr
        
        if price_drop >= min_drop:
            # 满足做空条件，返回反转价格（如果价格从最低点涨回N个ATR则反手做多/止损）
            reversal_price = current_price + self.atr_multiplier * atr
            return True, reversal_price
        
        return False, None
    
    def _should_reverse(self, current_price: float, atr: float) -> Optional[int]:
        """
        判断是否应该反手
        
        反手逻辑：
        - 做多时：跟踪最高点，如果价格从最高点跌回N个ATR则反手做空（如果enable_short=False则返回None，由外部逻辑处理平仓）
        - 做空时：跟踪最低点，如果价格从最低点涨回N个ATR则反手做多
        
        Parameters
        ----------
        current_price : float
            当前价格
        atr : float
            当前ATR值
            
        Returns
        -------
        Optional[int]
            如果应该反手，返回新的方向（1=做多，-1=做空），否则返回None
            如果enable_short=False且应该反手做空，返回None（由外部逻辑处理平仓）
        """
        if self.position_direction == 0 or self.reversal_price is None:
            return None
        
        if self.position_direction == 1:  # 当前做多
            # 如果价格从最高点跌回2ATR则反手做空
            if current_price <= self.reversal_price:
                # 如果做空功能被禁用，不允许反手做空，返回None（平仓）
                if not self.enable_short:
                    return None
                return -1
        elif self.position_direction == -1:  # 当前做空
            # 如果价格从最低点涨回2ATR则反手做多
            if current_price >= self.reversal_price:
                return 1
        
        return None
    
    def _calculate_pnl_ratio(self, entry_price: float, current_price: float, direction: int) -> float:
        """
        计算盈亏比例
        
        Parameters
        ----------
        entry_price : float
            入场价格
        current_price : float
            当前价格
        direction : int
            仓位方向：1=做多，-1=做空
            
        Returns
        -------
        float
            盈亏比例（正数表示盈利，负数表示亏损）
        """
        if entry_price is None or entry_price <= 0:
            return 0.0
        
        if direction == 1:  # 做多
            return (current_price - entry_price) / entry_price
        elif direction == -1:  # 做空
            return (entry_price - current_price) / entry_price
        else:
            return 0.0
    
    def _should_add_position(self, current_position_ratio: float, direction: int, 
                            current_price: float, atr: float) -> bool:
        """
        判断是否应该加仓（统一逻辑：基于最高/最低点的回调幅度）
        
        加仓逻辑（网格思想）：
        - 做多时：根据开仓后的最高价回调来判断加仓等级
          * 最高价 = max(开仓价, 开仓后的最高价)
          * 回调幅度 = 最高价 - 当前价格
          * 如果回调 >= (当前加仓等级 + 1) * grid_atr_interval * atr，则加仓
          * 注意：如果回调 >= atr_multiplier * atr，会先触发反转，而不是加仓
        - 做空时：根据开仓后的最低价上涨来判断加仓等级
          * 最低价 = min(开仓价, 开仓后的最低价)
          * 上涨幅度 = 当前价格 - 最低价
          * 如果上涨 >= (当前加仓等级 + 1) * grid_atr_interval * atr，则加仓
          * 注意：如果上涨 >= atr_multiplier * atr，会先触发反转，而不是加仓
        
        加仓等级说明：
        - 等级0：初始仓位
        - 等级1：回调 >= 0.5 * atr（如果grid_atr_interval=0.5）
        - 等级2：回调 >= 1.0 * atr
        - 等级3：回调 >= 1.5 * atr
        - 等级4：回调 >= 2.0 * atr
        - 等级5：回调 >= 2.5 * atr
        - 反转：回调 >= atr_multiplier * atr（默认3.0 * atr）
        
        因此，要支持加仓5次，需要 atr_multiplier > 5 * grid_atr_interval
        
        Parameters
        ----------
        current_position_ratio : float
            当前仓位比例（绝对值）
        direction : int
            仓位方向：1=做多，-1=做空
        current_price : float
            当前价格
        atr : float
            当前ATR值
            
        Returns
        -------
        bool
            是否应该加仓
        """
        # 检查是否达到最大仓位
        if current_position_ratio >= self.max_position_ratio:
            return False
        
        if direction == 1:  # 做多
            # 做多时，根据最高价回调来判断加仓
            # 最高价应该是开仓后的最高价（会动态更新）
            if self.highest_price is None:
                # 如果还没有最高价，使用开仓价
                highest_price = self.entry_price if self.entry_price is not None else current_price
            else:
                highest_price = self.highest_price
            
            # 计算回调幅度
            drawdown = highest_price - current_price
            
            # 计算应该达到的加仓等级（基于回调幅度）
            # 如果回调 >= (当前等级 + 1) * grid_atr_interval * atr，则应该加仓
            required_drawdown = (self.current_grid_level + 1) * self.grid_atr_interval * atr
            
            # 计算反转条件（如果回调达到这个值就会反转）
            reversal_drawdown = self.atr_multiplier * atr
            
            # 只有当回调达到加仓条件，且未达到反转条件时，才允许加仓
            # 注意：如果回调刚好等于反转条件，应该先反转，而不是加仓
            if drawdown >= required_drawdown and drawdown < reversal_drawdown:
                return True
                
        elif direction == -1:  # 做空
            # 做空时，根据最低价上涨来判断加仓
            # 最低价应该是开仓后的最低价（会动态更新）
            if self.lowest_price is None:
                # 如果还没有最低价，使用开仓价
                lowest_price = self.entry_price if self.entry_price is not None else current_price
            else:
                lowest_price = self.lowest_price
            
            # 计算上涨幅度
            rally = current_price - lowest_price
            
            # 计算应该达到的加仓等级（基于上涨幅度）
            # 如果上涨 >= (当前等级 + 1) * grid_atr_interval * atr，则应该加仓
            required_rally = (self.current_grid_level + 1) * self.grid_atr_interval * atr
            
            # 计算反转条件（如果上涨达到这个值就会反转）
            reversal_rally = self.atr_multiplier * atr
            
            # 只有当上涨达到加仓条件，且未达到反转条件时，才允许加仓
            # 注意：如果上涨刚好等于反转条件，应该先反转，而不是加仓
            if rally >= required_rally and rally < reversal_rally:
                return True
        
        return False
    
    def _should_reduce_position_on_profit(self, current_position_ratio: float, direction: int,
                                          current_price: float, atr: float) -> bool:
        """
        判断是否应该平掉加仓部分（基于峰值计算，与加仓逻辑一致）
        
        平仓逻辑（网格思想）：
        - 做多时：如果价格从最高点回调后，又涨回到接近最高点（回调幅度减小），应该平掉加仓部分
          * 如果当前回调幅度 < (当前加仓等级 - 1) * grid_atr_interval * atr，说明价格已经恢复，应该平掉加仓部分
        - 做空时：如果价格从最低点上涨后，又跌回到接近最低点（上涨幅度减小），应该平掉加仓部分
          * 如果当前上涨幅度 < (当前加仓等级 - 1) * grid_atr_interval * atr，说明价格已经恢复，应该平掉加仓部分
        
        Parameters
        ----------
        current_position_ratio : float
            当前仓位比例（绝对值）
        direction : int
            仓位方向：1=做多，-1=做空
        current_price : float
            当前价格
        atr : float
            当前ATR值
            
        Returns
        -------
        bool
            是否应该平掉加仓部分
        """
        # 只有当前仓位大于基础仓位时才平仓
        if current_position_ratio <= self.base_position_ratio:
            return False
        
        # 如果没有加仓（等级为0），不需要平仓
        if self.current_grid_level == 0:
            return False
        
        if direction == 1:  # 做多
            # 做多时，根据最高价回调来判断是否应该平仓
            if self.highest_price is None:
                return False
            
            # 计算当前回调幅度
            drawdown = self.highest_price - current_price
            
            # 如果当前回调幅度 < (当前加仓等级 - 1) * grid_atr_interval * atr
            # 说明价格已经恢复到接近最高点，应该平掉加仓部分
            # 例如：如果加仓等级是3（回调了1.5个ATR），现在回调只有0.5个ATR，说明价格已经恢复
            threshold_drawdown = (self.current_grid_level - 1) * self.grid_atr_interval * atr
            
            if drawdown < threshold_drawdown:
                return True
                
        elif direction == -1:  # 做空
            # 做空时，根据最低价上涨来判断是否应该平仓
            if self.lowest_price is None:
                return False
            
            # 计算当前上涨幅度
            rally = current_price - self.lowest_price
            
            # 如果当前上涨幅度 < (当前加仓等级 - 1) * grid_atr_interval * atr
            # 说明价格已经恢复到接近最低点，应该平掉加仓部分
            threshold_rally = (self.current_grid_level - 1) * self.grid_atr_interval * atr
            
            if rally < threshold_rally:
                return True
        
        return False
    
    def generate_trade_decision(self, execute_result=None):
        """
        生成交易决策
        
        策略逻辑：
        1. 永远持仓：一旦开仓后永久持仓，不是多就是空
        2. 多空判断：使用长期和短期趋势来复合判断
           - 做多条件：长期趋势下跌或持平 + 短期趋势上涨 + 较近期最低点有N个ATR涨幅（基于历史数据）
           - 做多止损/反手：做多时跟踪最高点，如果价格从最高点跌回N个ATR则止损反手做空
           - 做空条件：长期趋势上涨或持平 + 短期趋势下跌 + 较近期最高点有N个ATR下跌（基于历史数据）
           - 做空止损/反手：做空时跟踪最低点，如果价格从最低点涨回N个ATR则止损反手做多
        3. 网格加仓：在亏损时使用网格加仓，每0.5个ATR间隔加仓一次
        4. 盈利平仓：盈利超过1个点则把加仓平掉
        
        Parameters
        ----------
        execute_result : list, optional
            执行结果（本策略不使用）
            
        Returns
        -------
        TradeDecisionWO
            交易决策
        """
        # 获取交易时间
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        current_time = trade_start_time
        
        # 获取当前持仓
        current_temp: Position = self.trade_position
        current_amount = current_temp.get_stock_amount(self.stock_id)
        account_value = current_temp.calculate_value()
        
        # 获取当前价格
        try:
            current_price = self.trade_exchange.get_close(
                stock_id=self.stock_id,
                start_time=current_time,
                end_time=trade_end_time,
                method="ts_data_last"
            )
            # 如果返回的是 IndexData，提取值
            if hasattr(current_price, 'value'):
                current_price = current_price.value
            elif isinstance(current_price, (pd.Series, pd.DataFrame)):
                current_price = current_price.iloc[-1] if len(current_price) > 0 else None
            
            # 注意：qlib默认使用前复权价格，这是正常的
            # 如果价格看起来很低，可能是多次复权后的结果，这在回测中是正常的
        except Exception as e:
            logger.warning(f"无法获取 {self.stock_id} 的价格: {e}")
            return TradeDecisionWO([], self)
        
        if current_price is None or pd.isna(current_price) or account_value <= 0:
            return TradeDecisionWO([], self)
        
        # 获取价格数据
        price_df = self._get_price_data(self.stock_id, current_time)
        if price_df is None or len(price_df) == 0:
            return TradeDecisionWO([], self)
        
        # 计算ATR
        atr = self._calculate_atr(price_df)
        if atr is None or atr <= 0:
            return TradeDecisionWO([], self)
        
        # 计算长期和短期趋势
        long_trend = self._get_trend(price_df, self.long_trend_period)
        short_trend = self._get_trend(price_df, self.short_trend_period)
        
        if long_trend is None or short_trend is None:
            return TradeDecisionWO([], self)
        
        # 计算当前仓位（做多时为正，做空时为负，但 qlib 不支持负值，所以用方向标记）
        current_position_value = abs(current_amount * current_price)
        current_position_ratio = current_position_value / account_value if account_value > 0 else 0
        
        # 检查状态一致性：如果仓位为0但position_direction不为0，说明订单可能没有执行成功
        if current_amount == 0 and self.position_direction != 0:
            logger.warning(
                f"[{current_time}] {self.stock_id} 状态不一致：仓位为0但position_direction={self.position_direction}, "
                f"可能是订单未执行成功，重置状态"
            )
            # 重置状态
            self.position_direction = 0
            self.entry_price = None
            self.base_position_ratio = 0.0
            self.current_grid_level = 0
            self.last_grid_price = None
            self.lowest_price = None
            self.highest_price = None
            self.reversal_price = None
        
        # 如果当前有持仓但没有设置方向，根据持仓判断方向（做多）
        if current_amount > 0 and self.position_direction == 0:
            self.position_direction = 1  # 默认为做多
            self.entry_price = current_price
            self.base_position_ratio = current_position_ratio
            self.current_grid_level = 0
            self.last_grid_price = None
            # 做多时跟踪最高点
            self.highest_price = current_price
            self.lowest_price = None
            # 设置反转价格（止损价格：如果价格从最高点跌回N个ATR则反手做空）
            self.reversal_price = self.highest_price - self.atr_multiplier * atr
        
        # 如果当前没有持仓，初始化状态
        if current_amount == 0 and self.position_direction != 0:
            # 重置状态但保持方向（因为策略要求永久持仓）
            self.entry_price = None
            self.base_position_ratio = self.initial_position_ratio
            self.current_grid_level = 0
            self.lowest_price = None
            self.highest_price = None
            self.reversal_price = None
        
        orders = []
        
        try:
            factor = self.trade_exchange.get_factor(
                stock_id=self.stock_id,
                start_time=current_time,
                end_time=trade_end_time
            )
            if factor is None:
                factor = 1.0
        except Exception:
            factor = 1.0
        
        # 情况0：如果做空被禁用但当前持仓是做空，需要平仓
        if not self.enable_short and self.position_direction == -1 and current_amount > 0:
            # 平掉做空仓位
            order = Order(
                stock_id=self.stock_id,
                amount=current_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.SELL,
                factor=factor,
            )
            orders.append(order)
            
            # 重置状态
            self.position_direction = 0
            self.entry_price = None
            self.base_position_ratio = 0.0
            self.current_grid_level = 0
            self.last_grid_price = None
            self.lowest_price = None
            self.highest_price = None
            self.reversal_price = None
            
            logger.info(
                f"[{current_time}] {self.stock_id} 做空被禁用，平掉做空仓位："
                f"数量={current_amount:.0f}, 价格={current_price:.2f}"
            )
        
        # 情况1：检查是否需要反手或平仓
        reverse_direction = self._should_reverse(current_price, atr)
        
        # 如果做空被禁用且做多时达到反转条件，应该平仓而不是反手
        if (not self.enable_short and 
            self.position_direction == 1 and 
            current_amount > 0 and 
            self.reversal_price is not None and 
            current_price <= self.reversal_price):
            # 保存反转价格用于日志
            reversal_price_for_log = self.reversal_price
            
            # 平掉做多仓位（止损）
            order = Order(
                stock_id=self.stock_id,
                amount=current_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.SELL,
                factor=factor,
            )
            orders.append(order)
            
            # 重置状态
            self.position_direction = 0
            self.entry_price = None
            self.base_position_ratio = 0.0
            self.current_grid_level = 0
            self.last_grid_price = None
            self.lowest_price = None
            self.highest_price = None
            self.reversal_price = None
            
            logger.info(
                f"[{current_time}] {self.stock_id} 做多止损平仓（做空被禁用）："
                f"价格={current_price:.2f}, 反转价格={reversal_price_for_log:.2f}, 数量={current_amount:.0f}"
            )
            reverse_direction = None  # 阻止后续反手逻辑
        
        if reverse_direction is not None:
            # 计算目标仓位
            # 先估算平仓后的账户价值（考虑交易成本）
            current_position_value = current_amount * current_price if current_amount > 0 else 0
            # 估算平仓后的现金（考虑卖出成本0.15%）
            estimated_cash = account_value - current_position_value * 0.9985  # 卖出后剩余资金
            # 使用估算的现金计算目标仓位（更保守，确保有足够资金）
            target_position_value = estimated_cash * self.initial_position_ratio * 0.95  # 再预留5%安全边际
            new_amount = target_position_value / current_price * 0.99  # 预留1%给交易成本
            
            # 确保new_amount是正数且合理
            if new_amount <= 0:
                logger.warning(
                    f"[{current_time}] {self.stock_id} 反手开仓计算错误："
                    f"new_amount={new_amount:.0f}, estimated_cash={estimated_cash:.2f}"
                )
                # 如果计算错误，使用当前账户价值重新计算
                target_position_value = account_value * self.initial_position_ratio * 0.9
                new_amount = target_position_value / current_price * 0.99
            
            # 平掉当前方向的仓位
            if current_amount > 0:
                order = Order(
                    stock_id=self.stock_id,
                    amount=current_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.SELL,
                    factor=factor,
                )
                orders.append(order)
                logger.debug(
                    f"[{current_time}] {self.stock_id} 平掉当前仓位："
                    f"方向={self.position_direction}, 数量={current_amount:.0f}"
                )
            
            # 反向开仓（使用BUY订单，因为qlib不支持真正的做空）
            # 注意：在qlib中，无论是做多还是做空，我们都使用BUY订单来建立仓位
            # 做空方向通过position_direction=-1来标记，盈亏计算会考虑这个方向
            order = Order(
                stock_id=self.stock_id,
                amount=new_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.BUY,  # 在qlib中，即使是做空，我们也使用BUY订单
                factor=factor,
            )
            orders.append(order)
            
            # 重置状态
            self.position_direction = reverse_direction
            self.entry_price = current_price
            self.base_position_ratio = self.initial_position_ratio
            self.current_grid_level = 0
            # 反手后重置加仓价格，趋势跟踪加仓的参考价格会是entry_price
            self.last_grid_price = None
            if reverse_direction == 1:  # 反手做多
                # 做多时跟踪最高点
                self.highest_price = current_price
                self.lowest_price = None
                # 设置反转价格（止损价格：如果价格从最高点跌回N个ATR则反手做空）
                self.reversal_price = self.highest_price - self.atr_multiplier * atr
            else:  # 反手做空
                # 做空时跟踪最低点
                self.lowest_price = current_price
                self.highest_price = None
                # 设置反转价格（止损价格：如果价格从最低点涨回N个ATR则反手做多）
                self.reversal_price = self.lowest_price + self.atr_multiplier * atr
            
            direction_str = "做多" if reverse_direction == 1 else "做空"
            logger.info(
                f"[{current_time}] {self.stock_id} 反手开仓（{direction_str}）："
                f"价格={current_price:.2f}, ATR={atr:.2f}, 目标数量={new_amount:.0f}, "
                f"目标仓位={self.initial_position_ratio:.2%}, 当前账户价值={account_value:.2f}"
            )
        
        # 情况2：当前有持仓，检查是否需要加仓或平仓
        elif self.position_direction != 0:
            # 更新最低点和最高点
            if self.position_direction == 1:  # 做多
                # 做多时跟踪最高点（用于判断反手）
                if self.highest_price is None:
                    self.highest_price = current_price
                elif current_price > self.highest_price:
                    self.highest_price = current_price
                # 更新反转价格（如果价格从最高点跌回2ATR则反手做空）
                self.reversal_price = self.highest_price - self.atr_multiplier * atr
            elif self.position_direction == -1:  # 做空
                # 做空时跟踪最低点（用于判断反手）
                if self.lowest_price is None:
                    self.lowest_price = current_price
                elif current_price < self.lowest_price:
                    self.lowest_price = current_price
                # 更新反转价格（如果价格从最低点涨回2ATR则反手做多）
                self.reversal_price = self.lowest_price + self.atr_multiplier * atr
            
            # 计算盈亏比例
            if self.entry_price is None:
                self.entry_price = current_price
            
            pnl_ratio = self._calculate_pnl_ratio(self.entry_price, current_price, self.position_direction)
            
            # 记录状态历史（用于后续分析）
            # 注意：根据实际仓位（current_amount）来确定 direction，而不是策略状态（position_direction）
            # 这样 position_history 中的 direction 才能与 report_normal 的收益匹配
            # 如果 enable_short=False，不记录做空（即使 position_direction=-1）
            if current_amount > 0 and self.position_direction == 1:
                actual_direction = 1
            elif current_amount > 0 and self.position_direction == -1 and self.enable_short:
                actual_direction = -1
            else:
                actual_direction = 0
            self.position_history.append({
                'timestamp': current_time,
                'direction': actual_direction,  # 使用实际仓位方向（与 report_normal 匹配）
                'price': current_price,
                'pnl_ratio': pnl_ratio,
                'position_ratio': current_position_ratio,
                'grid_level': self.current_grid_level,
                'actual_amount': current_amount,  # 记录实际仓位数量
            })
            
            # 2.0 检查是否触发反向开仓信号（趋势转向）
            # 如果 enable_short=False，当触发做空信号时应该主动平掉多仓
            if self.position_direction == 1 and not self.enable_short:
                # 检查是否触发做空条件（趋势转为看空）
                # 做空条件：长期上涨或持平 + 短期下跌 + 较近期最高点有N个ATR下跌
                if long_trend >= 0 and short_trend == -1:  # 长期上涨或持平 + 短期下跌
                    _, recent_highest = self._get_recent_extreme_prices(price_df, self.lookback_period)
                    if recent_highest is not None:
                        price_drop = recent_highest - current_price
                        min_drop = self.entry_atr_multiplier * atr
                        if price_drop >= min_drop:
                            # 触发做空条件，平掉所有多仓
                            order = Order(
                                stock_id=self.stock_id,
                                amount=current_amount,
                                start_time=trade_start_time,
                                end_time=trade_end_time,
                                direction=OrderDir.SELL,
                                factor=factor,
                            )
                            orders.append(order)
                            
                            # 重置状态
                            self.position_direction = 0
                            self.entry_price = None
                            self.base_position_ratio = 0.0
                            self.current_grid_level = 0
                            self.last_grid_price = None
                            self.lowest_price = None
                            self.highest_price = None
                            self.reversal_price = None
                            
                            logger.info(
                                f"[{current_time}] {self.stock_id} 做多主动平仓（触发做空信号但做空被禁用）："
                                f"价格={current_price:.2f}, 近期最高={recent_highest:.2f}, 下跌={price_drop:.2f}, "
                                f"盈亏={pnl_ratio:.2%}, 数量={current_amount:.0f}"
                            )
                            return TradeDecisionWO(orders, self)
            
            # 对称情况：如果持有空仓且触发做多条件（理论上enable_short=False时不应出现）
            elif self.position_direction == -1 and not self.enable_short:
                # 检查是否触发做多条件（趋势转为看多）
                # 做多条件：长期下跌或持平 + 短期上涨 + 较近期最低点有N个ATR涨幅
                if long_trend <= 0 and short_trend == 1:  # 长期下跌或持平 + 短期上涨
                    recent_lowest, _ = self._get_recent_extreme_prices(price_df, self.lookback_period)
                    if recent_lowest is not None:
                        price_rise = current_price - recent_lowest
                        min_rise = self.entry_atr_multiplier * atr
                        if price_rise >= min_rise:
                            # 触发做多条件，平掉所有空仓
                            order = Order(
                                stock_id=self.stock_id,
                                amount=current_amount,
                                start_time=trade_start_time,
                                end_time=trade_end_time,
                                direction=OrderDir.SELL,
                                factor=factor,
                            )
                            orders.append(order)
                            
                            # 重置状态
                            self.position_direction = 0
                            self.entry_price = None
                            self.base_position_ratio = 0.0
                            self.current_grid_level = 0
                            self.last_grid_price = None
                            self.lowest_price = None
                            self.highest_price = None
                            self.reversal_price = None
                            
                            logger.info(
                                f"[{current_time}] {self.stock_id} 做空异常平仓（触发做多信号但做空被禁用）："
                                f"价格={current_price:.2f}, 近期最低={recent_lowest:.2f}, 上涨={price_rise:.2f}, "
                                f"盈亏={pnl_ratio:.2%}, 数量={current_amount:.0f}"
                            )
                            return TradeDecisionWO(orders, self)
            
            # 2.1 统一加仓逻辑：基于最高/最低点的回调幅度判断加仓等级
            should_add = self._should_add_position(current_position_ratio, self.position_direction, current_price, atr)
            
            # 调试信息：定期输出状态（每20个交易日输出一次）
            if trade_step % 20 == 0:
                if self.position_direction == 1:  # 做多
                    highest_price = self.highest_price if self.highest_price is not None else self.entry_price
                    drawdown = highest_price - current_price if highest_price is not None else 0
                    required_drawdown = (self.current_grid_level + 1) * self.grid_atr_interval * atr
                    logger.info(
                        f"[{current_time}] {self.stock_id} 做多状态检查："
                        f"价格={current_price:.4f}, 最高价={highest_price:.4f}, "
                        f"回调={drawdown:.4f}, 需要回调={required_drawdown:.4f} (ATR={atr:.4f}), "
                        f"盈亏={pnl_ratio:.2%}, 仓位={current_position_ratio:.2%}, "
                        f"加仓等级={self.current_grid_level}, should_add={should_add}"
                    )
                elif self.position_direction == -1:  # 做空
                    lowest_price = self.lowest_price if self.lowest_price is not None else self.entry_price
                    rally = current_price - lowest_price if lowest_price is not None else 0
                    required_rally = (self.current_grid_level + 1) * self.grid_atr_interval * atr
                    logger.info(
                        f"[{current_time}] {self.stock_id} 做空状态检查："
                        f"价格={current_price:.4f}, 最低价={lowest_price:.4f}, "
                        f"上涨={rally:.4f}, 需要上涨={required_rally:.4f} (ATR={atr:.4f}), "
                        f"盈亏={pnl_ratio:.2%}, 仓位={current_position_ratio:.2%}, "
                        f"加仓等级={self.current_grid_level}, should_add={should_add}"
                    )
            
            if should_add:
                # 基于ATR间隔加仓：每次加仓增加初始仓位的grid_size比例
                # 计算加仓数量（每次加仓增加初始仓位的grid_size比例）
                add_position_ratio = self.initial_position_ratio * self.grid_size  # 每次加仓增加初始仓位的grid_size比例
                target_position_ratio = min(
                    current_position_ratio + add_position_ratio,
                    self.max_position_ratio
                )
                
                # 计算需要加仓的金额
                target_position_value = account_value * target_position_ratio
                current_position_value = current_amount * current_price
                add_value = target_position_value - current_position_value
                
                if add_value > 0:
                    add_amount = add_value / current_price * 0.99  # 预留1%给交易成本
                    
                    # 注意：在qlib中，无论是做多还是做空，我们都使用BUY订单来加仓
                    # 做空方向通过position_direction=-1来标记，盈亏计算会考虑这个方向
                    order = Order(
                        stock_id=self.stock_id,
                        amount=add_amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=OrderDir.BUY,  # 在qlib中，即使是做空，我们也使用BUY订单
                        factor=factor,
                    )
                    orders.append(order)
                    
                    # 更新网格状态
                    self.current_grid_level += 1
                    # 注意：不再使用last_grid_price，因为加仓逻辑基于最高/最低点
                    
                    direction_str = "做多" if self.position_direction == 1 else "做空"
                    if self.position_direction == 1:
                        highest_price = self.highest_price if self.highest_price is not None else self.entry_price
                        drawdown = highest_price - current_price if highest_price is not None else 0
                        logger.info(
                            f"[{current_time}] {self.stock_id} {direction_str}加仓（ATR间隔{self.grid_atr_interval}，等级{self.current_grid_level}）："
                            f"最高价={highest_price:.4f}, 当前价={current_price:.4f}, 回调={drawdown:.4f}, "
                            f"盈亏={pnl_ratio:.2%}, 当前仓位={current_position_ratio:.2%}, 目标仓位={target_position_ratio:.2%}"
                        )
                    else:
                        lowest_price = self.lowest_price if self.lowest_price is not None else self.entry_price
                        rally = current_price - lowest_price if lowest_price is not None else 0
                        logger.info(
                            f"[{current_time}] {self.stock_id} {direction_str}加仓（ATR间隔{self.grid_atr_interval}，等级{self.current_grid_level}）："
                            f"最低价={lowest_price:.4f}, 当前价={current_price:.4f}, 上涨={rally:.4f}, "
                            f"盈亏={pnl_ratio:.2%}, 当前仓位={current_position_ratio:.2%}, 目标仓位={target_position_ratio:.2%}"
                        )
            
            # 2.2 盈利时平掉加仓部分（基于峰值计算，与加仓逻辑一致）
            elif self._should_reduce_position_on_profit(current_position_ratio, self.position_direction, current_price, atr):
                # 平掉加仓部分，保留基础仓位
                reduce_ratio = current_position_ratio - self.base_position_ratio
                reduce_value = account_value * reduce_ratio
                reduce_amount = reduce_value / current_price
                
                if reduce_amount > 0 and reduce_amount < current_amount:
                    # 注意：在qlib中，无论是做多还是做空，仓位都是正向的（通过BUY建立）
                    # 所以盈利平仓时，都用SELL来减少持仓
                    order = Order(
                        stock_id=self.stock_id,
                        amount=reduce_amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=OrderDir.SELL,  # 在qlib中，无论是做多还是做空，都用SELL来平仓
                        factor=factor,
                    )
                    orders.append(order)
                    
                    # 重置网格等级和加仓价格
                    self.current_grid_level = 0
                    self.last_grid_price = None
                    # 更新入场价格（保留基础仓位的入场价格）
                    self.entry_price = current_price
                    
                    direction_str = "做多" if self.position_direction == 1 else "做空"
                    if self.position_direction == 1:  # 做多
                        highest_price = self.highest_price if self.highest_price is not None else current_price
                        drawdown = highest_price - current_price if highest_price is not None else 0
                        logger.info(
                            f"[{current_time}] {self.stock_id} {direction_str}盈利平仓："
                            f"最高价={highest_price:.4f}, 当前价={current_price:.4f}, 回调={drawdown:.4f}, "
                            f"盈亏={pnl_ratio:.2%}, 平掉加仓部分={reduce_ratio:.2%}"
                        )
                    else:  # 做空
                        lowest_price = self.lowest_price if self.lowest_price is not None else current_price
                        rally = current_price - lowest_price if lowest_price is not None else 0
                        logger.info(
                            f"[{current_time}] {self.stock_id} {direction_str}盈利平仓："
                            f"最低价={lowest_price:.4f}, 当前价={current_price:.4f}, 上涨={rally:.4f}, "
                            f"盈亏={pnl_ratio:.2%}, 平掉加仓部分={reduce_ratio:.2%}"
                        )
        
        # 情况3：当前无仓位，判断是否开仓
        elif self.position_direction == 0:
            # 记录无仓位状态（根据实际仓位 current_amount 判断）
            # 注意：如果 current_amount == 0，direction = 0（无仓位）
            # 如果 enable_short=False，不记录做空（即使 position_direction=-1）
            if current_amount == 0:
                actual_direction_no_pos = 0
            elif self.position_direction == 1:
                actual_direction_no_pos = 1
            elif self.position_direction == -1 and self.enable_short:
                actual_direction_no_pos = -1
            else:
                actual_direction_no_pos = 0
            self.position_history.append({
                'timestamp': current_time,
                'direction': actual_direction_no_pos,  # 使用实际仓位方向
                'price': current_price,
                'pnl_ratio': 0.0,
                'position_ratio': current_position_ratio,  # 使用实际仓位比例
                'grid_level': 0,
                'actual_amount': current_amount,  # 记录实际仓位数量
            })
            # 判断是否应该做多
            should_long, reversal_price_long = self._should_go_long(long_trend, short_trend, current_price, atr, price_df)
            if should_long:
                target_position_value = account_value * self.initial_position_ratio
                new_amount = target_position_value / current_price * 0.99  # 预留1%给交易成本
                
                order = Order(
                    stock_id=self.stock_id,
                    amount=new_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.BUY,
                    factor=factor,
                )
                orders.append(order)
                
                # 设置状态
                self.position_direction = 1
                self.entry_price = current_price
                self.base_position_ratio = self.initial_position_ratio
                self.current_grid_level = 0
                self.last_grid_price = None  # 初始时没有加仓
                # 做多时跟踪最高点
                self.highest_price = current_price
                self.lowest_price = None
                # 设置反转价格（止损价格：如果价格从最高点跌回N个ATR则反手做空）
                self.reversal_price = self.highest_price - self.atr_multiplier * atr
                
                logger.info(
                    f"[{current_time}] {self.stock_id} 初始开仓（做多）："
                    f"价格={current_price:.2f}, ATR={atr:.2f}, 反转价格={reversal_price_long:.2f}"
                )
            else:
                # 判断是否应该做空
                should_short, reversal_price_short = self._should_go_short(long_trend, short_trend, current_price, atr, price_df)
                if should_short:
                    target_position_value = account_value * self.initial_position_ratio
                    new_amount = target_position_value / current_price * 0.99  # 预留1%给交易成本
                    
                    # 注意：qlib不支持真正的做空（没有持仓时无法使用SELL订单）
                    # 所以，做空时也使用BUY订单来建立仓位，通过position_direction=-1来标记做空方向
                    # 盈亏计算会考虑这个方向（见_calculate_pnl_ratio方法）
                    order = Order(
                        stock_id=self.stock_id,
                        amount=new_amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=OrderDir.BUY,  # 在qlib中，即使是做空，我们也使用BUY订单
                        factor=factor,
                    )
                    orders.append(order)
                    
                    # 设置状态
                    self.position_direction = -1
                    self.entry_price = current_price
                    self.base_position_ratio = self.initial_position_ratio
                    self.current_grid_level = 0
                    self.last_grid_price = None  # 初始时没有加仓
                    # 做空时跟踪最低点
                    self.lowest_price = current_price
                    self.highest_price = None
                    # 设置反转价格（止损价格：如果价格从最低点涨回N个ATR则反手做多）
                    self.reversal_price = self.lowest_price + self.atr_multiplier * atr
                    
                    logger.info(
                        f"[{current_time}] {self.stock_id} 初始开仓（做空）："
                        f"价格={current_price:.2f}, ATR={atr:.2f}, 反转价格={reversal_price_short:.2f}"
                    )
        
        return TradeDecisionWO(orders, self)
