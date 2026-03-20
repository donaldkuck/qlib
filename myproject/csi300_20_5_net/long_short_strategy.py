# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
基于 TopkDrop 思想的长期和短期因子结合策略

策略逻辑：
1. 买入和卖出基于长期预测值（使用 topkdrop 思想）
   - 持有 topk 只股票
   - 每天替换 n_drop 只股票
2. 买入条件：长期和短期预测都为正
3. 加仓/止盈：根据短期预测值对已持仓股票进行加仓或止盈
   - 短期预测为正：加仓
   - 短期预测为负且盈利：止盈（减仓）
4. 仓位分配：使用总资产计算等权仓位，确保每只股票的仓位固定
   - 不再基于可用现金动态分配，避免后续买入金额递减的问题
   - 每只股票的目标仓位 = 总资产 * risk_degree / topk
"""

import copy
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

from qlib.backtest.position import Position
from qlib.backtest.signal import Signal, create_signal_from
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
from qlib.data import D
from qlib.log import get_module_logger
from qlib.model.base import BaseModel
from qlib.data.dataset import Dataset

logger = get_module_logger("TopkDropLongShortStrategy")


class TopkDropLongShortStrategy(BaseSignalStrategy):
    """
    基于 TopkDrop 思想的长期和短期因子结合策略
    
    策略特点：
    - 使用长期预测值进行买入和卖出决策（topkdrop 逻辑）
    - 买入时要求长期和短期预测都为正
    - 根据短期预测值对已持仓股票进行加仓或止盈
    """
    
    def __init__(
        self,
        *,
        # 长期因子信号（用于买入和卖出决策）
        long_term_signal: Union[Signal, Tuple[BaseModel, Dataset], List, Dict, str, pd.Series, pd.DataFrame] = None,
        long_term_model=None,
        long_term_dataset=None,
        # 短期因子信号（用于买入筛选和加仓/止盈）
        short_term_signal: Union[Signal, Tuple[BaseModel, Dataset], List, Dict, str, pd.Series, pd.DataFrame] = None,
        short_term_model=None,
        short_term_dataset=None,
        # TopkDrop 参数
        topk: int = 20,  # 持仓股票数量
        n_drop: int = 5,  # 每天替换的股票数量
        method_sell: str = "bottom",  # 卖出方法：bottom（卖出预测值最低的）
        method_buy: str = "top",  # 买入方法：top（买入预测值最高的）
        hold_thresh: int = 1,  # 最小持有天数
        long_signal_quantile: float = 0.5,  # 长期信号为正的分位阈值
        short_buy_quantile: float = 0.6,  # 短期买入过滤分位阈值
        short_add_quantile: float = 0.7,  # 短期加仓分位阈值
        short_reduce_quantile: float = 0.4,  # 短期减仓/止盈分位阈值
        buy_score_long_weight: float = 0.7,  # 买入排序中长期信号权重
        buy_score_short_weight: float = 0.3,  # 买入排序中短期信号权重
        sell_score_long_weight: float = 0.8,  # 卖出排序中长期信号权重
        sell_score_short_weight: float = 0.2,  # 卖出排序中短期信号权重
        add_size_min_multiplier: float = 0.5,  # 加仓最小倍数（相对基础加仓单位）
        add_size_max_multiplier: float = 1.5,  # 加仓最大倍数（相对基础加仓单位）
        # 海龟突破参数
        enable_turtle_breakout: bool = False,  # 是否启用海龟突破过滤（默认关闭）
        turtle_breakout_period: int = 20,  # 海龟突破周期（N日）
        # 加仓/止盈参数
        add_position_ratio: float = 0.1,  # 加仓比例（相对于单只股票仓位），当 position_unit_ratio 为 None 时使用
        position_unit_ratio: float = 0.01,  # 加仓单位（相对总资产），如 0.01=1%
        open_position_ratio: float = 0.01,  # 开仓时单只仓位（相对总资产），如 0.01=1%
        profit_threshold: float = 0.02,  # 止盈阈值（盈利超过此比例时可以考虑止盈）
        reduce_position_ratio: float = 0.5,  # 止盈时减仓比例（相对于当前仓位）
        max_single_stock_ratio: float = 0.05,  # 单只股票最大仓位比例（相对于总资产）
        # 止损参数
        stop_loss_threshold: float = -0.05,  # 止损阈值（亏损超过此比例时止损，例如-0.05表示亏损5%时止损）
        # 回撤控制参数
        max_drawdown: float = None,  # 最大回撤阈值（例如0.15表示15%），超过此值降低仓位
        drawdown_warning: float = None,  # 回撤预警阈值（例如0.10表示10%），超过此值开始降低仓位
        drawdown_reduce_ratio: float = 0.3,  # 回撤时减仓比例（例如0.3表示减仓30%）
        # 其他参数
        risk_degree: float = 0.90,  # 总仓位上限（0.90=90%），留一定现金应对赎回/加仓
        only_tradable: bool = False,
        forbid_all_trade_at_limit: bool = True,
        # 固定股票池（可选）
        fixed_stock_pool: List[str] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        long_term_signal : Signal or tuple
            长期因子信号（用于买入和卖出决策）
        short_term_signal : Signal or tuple
            短期因子信号（用于买入筛选和加仓/止盈）
        topk : int
            持仓股票数量
        n_drop : int
            每天替换的股票数量
        method_sell : str
            卖出方法：bottom（卖出预测值最低的）
        method_buy : str
            买入方法：top（买入预测值最高的）
        hold_thresh : int
            最小持有天数
        add_position_ratio : float
            加仓比例（相对于单只股票仓位，例如0.1表示加仓10%）
        open_position_ratio : float
            开仓时单只仓位（相对总资产，例如0.01表示1%）
        profit_threshold : float
            止盈阈值（盈利超过此比例时可以考虑止盈）
        reduce_position_ratio : float
            止盈时减仓比例（相对于当前仓位，例如0.5表示减仓50%）
        fixed_stock_pool : List[str]
            固定股票池（可选，如果提供则只从该股票池中选择）
        """
        # 初始化长期和短期信号
        # 优先使用kwargs中的signal（PortAnaRecord可能会传入，可能是<PRED>占位符或实际预测数据）
        if 'signal' in kwargs and kwargs['signal'] is not None:
            long_term_signal = kwargs.pop('signal')
        
        # 保存recorder以便后续加载长期/短期信号
        self.recorder = kwargs.pop('recorder', None)
        self.long_term_signal_path = None
        self.short_term_signal_path = None
        
        # 提取不需要传递给父类的参数
        self.stats_output_dir = kwargs.pop('stats_output_dir', None)
        kwargs.pop('long_term_dataset', None)  # 不需要保存，只是用于传递
        kwargs.pop('short_term_dataset', None)  # 不需要保存，只是用于传递
        
        if long_term_model is not None and long_term_dataset is not None:
            long_term_signal = (long_term_model, long_term_dataset)
        if short_term_model is not None and short_term_dataset is not None:
            short_term_signal = (short_term_model, short_term_dataset)
        
        # 如果没有提供长期信号，使用默认值
        if long_term_signal is None:
            raise ValueError("必须提供长期信号（long_term_signal或signal参数）")

        # 如果长期信号是字符串（如 "<PRED>" 或 "*.pkl"），先从 recorder 显式加载，
        # 避免 BaseSignalStrategy 将其误判为模块配置并尝试 import。
        if isinstance(long_term_signal, str):
            self.long_term_signal_path = "pred.pkl" if long_term_signal == "<PRED>" else long_term_signal
            if self.recorder is None:
                raise ValueError(f"长期信号为路径 {self.long_term_signal_path}，但未提供 recorder")
            try:
                from qlib.backtest.signal import SignalWCache
                long_term_pred_data = self.recorder.load_object(self.long_term_signal_path)
                if long_term_pred_data is None:
                    raise ValueError(f"recorder 中未找到长期信号文件: {self.long_term_signal_path}")
                long_term_signal = SignalWCache(long_term_pred_data)
            except Exception as e:
                raise ValueError(f"无法加载长期信号 {self.long_term_signal_path}: {e}") from e
        
        # 使用长期信号初始化基类（用于兼容）
        super().__init__(
            signal=long_term_signal,
            risk_degree=risk_degree,
            **kwargs
        )
        
        # 初始化交易日志文件 handler（只写入文件，不输出到控制台）
        self._init_trade_logger()
        
        # 创建短期信号
        # 如果short_term_signal是字符串（文件路径），则延迟加载
        if short_term_signal is None:
            # 尝试从kwargs中获取短期信号路径
            if 'short_term_signal' in kwargs:
                short_term_signal = kwargs.pop('short_term_signal')
            else:
                raise ValueError("必须提供短期信号（short_term_signal参数）")
        
        # 如果short_term_signal是字符串（文件路径），延迟加载
        if isinstance(short_term_signal, str):
            self.short_term_signal_path = short_term_signal
            self.short_term_signal = None  # 延迟加载
        else:
            self.short_term_signal: Signal = create_signal_from(short_term_signal)
        
        # 保存策略参数
        self.topk = topk
        self.n_drop = n_drop
        self.method_sell = method_sell
        self.method_buy = method_buy
        self.hold_thresh = hold_thresh
        self.long_signal_quantile = long_signal_quantile
        self.short_buy_quantile = short_buy_quantile
        self.short_add_quantile = short_add_quantile
        self.short_reduce_quantile = short_reduce_quantile
        self.buy_score_long_weight = buy_score_long_weight
        self.buy_score_short_weight = buy_score_short_weight
        self.sell_score_long_weight = sell_score_long_weight
        self.sell_score_short_weight = sell_score_short_weight
        self.add_size_min_multiplier = add_size_min_multiplier
        self.add_size_max_multiplier = add_size_max_multiplier
        self.enable_turtle_breakout = enable_turtle_breakout
        self.turtle_breakout_period = turtle_breakout_period
        self.add_position_ratio = add_position_ratio
        self.position_unit_ratio = position_unit_ratio
        self.open_position_ratio = open_position_ratio  # 开仓单只仓位（总资产比例）
        self.profit_threshold = profit_threshold
        self.reduce_position_ratio = reduce_position_ratio
        self.max_single_stock_ratio = max_single_stock_ratio
        self.stop_loss_threshold = stop_loss_threshold
        self.max_drawdown = max_drawdown
        self.drawdown_warning = drawdown_warning if drawdown_warning is not None else (max_drawdown * 0.7 if max_drawdown is not None else None)
        self.drawdown_reduce_ratio = drawdown_reduce_ratio
        self.only_tradable = only_tradable
        self.forbid_all_trade_at_limit = forbid_all_trade_at_limit
        self.fixed_stock_pool = fixed_stock_pool
        
        # 记录每只股票的入场价格（用于计算盈利）
        self.entry_prices: Dict[str, float] = {}
        
        # 记录历史最高资产（用于计算回撤）
        self.peak_account_value: float = None
        
        # 回撤控制冷却期（避免频繁减仓）
        self.last_drawdown_reduce_date: pd.Timestamp = None
        self.drawdown_reduce_cooldown_days: int = 5  # 减仓后5天内不再减仓
    
    def _init_trade_logger(self):
        """
        初始化交易日志记录器（已禁用，不再写入 trade_records.log）
        """
        pass
    
    def _is_prediction_positive(self, pred_value: float, all_pred_values: pd.Series = None, quantile: float = 0.5) -> bool:
        """
        判断预测值是否为正（预测上涨）
        
        Parameters
        ----------
        pred_value : float
            预测值（原始收益率预测值，正值表示预测上涨，负值表示预测下跌）
        all_pred_values : pd.Series, optional
            所有预测值（用于计算相对阈值）
        
        Returns
        -------
        bool
            是否为正（预测上涨）
        """
        # 检查NaN
        if pd.isna(pred_value):
            return False
        
        # 如果有所有预测值，使用分位数阈值（相对排名）
        if all_pred_values is not None and len(all_pred_values) > 0:
            valid_values = all_pred_values.dropna()
            if len(valid_values) > 0:
                threshold = valid_values.quantile(quantile)
                return pred_value > threshold
        
        # 否则，直接判断是否大于0（预测正收益）
        return pred_value > 0

    def _build_combined_buy_score(self, long_term_pred: pd.Series, short_term_pred: pd.Series) -> pd.Series:
        """构建买入排序用的长期/短期组合分数。"""
        long_rank = long_term_pred.rank(pct=True)
        short_rank = short_term_pred.rank(pct=True)
        return long_rank * self.buy_score_long_weight + short_rank * self.buy_score_short_weight

    def _build_combined_sell_score(self, long_term_pred: pd.Series, short_term_pred: pd.Series) -> pd.Series:
        """构建卖出排序用的弱势分数，分数越低越优先卖出。"""
        long_rank = long_term_pred.rank(pct=True)
        short_rank = short_term_pred.rank(pct=True)
        return long_rank * self.sell_score_long_weight + short_rank * self.sell_score_short_weight

    def _get_add_size_multiplier(self, score: float) -> float:
        """根据信号强弱调整加仓倍数。"""
        if pd.isna(score):
            return self.add_size_min_multiplier
        clipped_score = min(max(float(score), 0.0), 1.0)
        return self.add_size_min_multiplier + (
            self.add_size_max_multiplier - self.add_size_min_multiplier
        ) * clipped_score
    
    def _is_turtle_breakout(self, stock_id: str, current_time: pd.Timestamp) -> bool:
        """
        判断是否发生海龟突破（价格突破N日最高价）
        
        Parameters
        ----------
        stock_id : str
            股票代码
        current_time : pd.Timestamp
            当前时间
        
        Returns
        -------
        bool
            是否发生突破（True表示突破，可以使用长期预测）
        """
        if not self.enable_turtle_breakout:
            return True  # 如果未启用突破过滤，默认返回True
        
        try:
            # 获取价格数据
            lookback_days = self.turtle_breakout_period + 5
            start_time = current_time - pd.Timedelta(days=lookback_days)
            
            price_data = D.features(
                [stock_id],
                ["$close", "$high"],
                start_time=start_time,
                end_time=current_time,
                freq="day",
                disk_cache=True,
            )
            
            if price_data is None or len(price_data) == 0:
                return False  # 数据不足，不允许交易
            
            # 提取价格序列
            if isinstance(price_data.index, pd.MultiIndex):
                if stock_id in price_data.index.get_level_values(0):
                    close_series = price_data.loc[stock_id, "$close"].dropna()
                    high_series = price_data.loc[stock_id, "$high"].dropna()
                else:
                    return False
            else:
                close_series = price_data.iloc[:, 0].dropna()
                high_series = price_data.iloc[:, 1].dropna() if len(price_data.columns) > 1 else close_series
            
            # 需要至少N+1个数据点（N个历史 + 当前）
            if len(high_series) < self.turtle_breakout_period + 1:
                return False
            
            current_price = close_series.iloc[-1]
            # 过去N日的最高价（不包括今天）
            past_high = high_series.iloc[-self.turtle_breakout_period-1:-1].max() if len(high_series) > 1 else high_series.iloc[-1]
            
            # 当前价格必须严格大于过去N日最高价（突破）
            is_breakout = current_price > past_high
            
            return is_breakout
            
        except Exception as e:
            logger.warning(f"检查股票 {stock_id} 海龟突破失败: {e}")
            return False  # 出错时保守处理，不允许交易
    
    def _get_stock_pool(self, long_term_pred: pd.Series, short_term_pred: pd.Series) -> pd.Index:
        """
        获取股票池（如果提供了固定股票池，则使用固定股票池）
        
        Parameters
        ----------
        long_term_pred : pd.Series
            长期因子预测值
        short_term_pred : pd.Series
            短期因子预测值
        
        Returns
        -------
        pd.Index
            股票池索引
        """
        if self.fixed_stock_pool is not None:
            # 使用固定股票池
            common_stocks = long_term_pred.index.intersection(short_term_pred.index)
            fixed_pool_stocks = [s for s in self.fixed_stock_pool if s in common_stocks]
            return pd.Index(fixed_pool_stocks)
        else:
            # 使用所有有预测值的股票
            return long_term_pred.index.intersection(short_term_pred.index)
    
    def generate_trade_decision(self, execute_result=None):
        """
        生成交易决策
        
        逻辑：
        1. 基于长期预测值，使用 topkdrop 逻辑选择要买入和卖出的股票
        2. 买入时，只选择长期和短期预测都为正的股票
        3. 对于已持仓的股票，根据短期预测值决定是否加仓或止盈
        """
        # 如果短期信号需要延迟加载，现在加载
        if self.short_term_signal is None and self.short_term_signal_path is not None and self.recorder is not None:
            try:
                short_term_pred_data = self.recorder.load_object(self.short_term_signal_path)
                # 将预测数据转换为Signal对象
                from qlib.backtest.signal import SignalWCache
                self.short_term_signal = SignalWCache(short_term_pred_data)
            except Exception as e:
                logger.error(f"从recorder加载短期信号失败: {e}")
                raise ValueError(f"无法加载短期信号: {e}")
        
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        
        # 获取长期和短期预测信号
        long_term_pred = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        short_term_pred = self.short_term_signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        
        if long_term_pred is None or short_term_pred is None:
            return TradeDecisionWO([], self)
        
        # 处理DataFrame格式的信号
        if isinstance(long_term_pred, pd.DataFrame):
            long_term_pred = long_term_pred.iloc[:, 0]
        if isinstance(short_term_pred, pd.DataFrame):
            short_term_pred = short_term_pred.iloc[:, 0]
        
        # 获取股票池
        stock_pool = self._get_stock_pool(long_term_pred, short_term_pred)
        if len(stock_pool) == 0:
            return TradeDecisionWO([], self)
        
        # 在股票池内筛选长期预测值
        long_term_pred_filtered = long_term_pred.reindex(stock_pool)
        short_term_pred_filtered = short_term_pred.reindex(stock_pool)
        
        # 辅助函数：获取前 N 个
        def get_first_n(li, n, reverse=False):
            """获取前 n 个元素"""
            if self.only_tradable:
                cur_n = 0
                res = []
                for si in reversed(li) if reverse else li:
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    ):
                        res.append(si)
                        cur_n += 1
                        if cur_n >= n:
                            break
                return res[::-1] if reverse else res
            else:
                li_list = list(li)
                return li_list[:n] if not reverse else li_list[-n:]
        
        def get_last_n(li, n):
            return get_first_n(li, n, reverse=True)
        
        def filter_stock(li):
            if self.only_tradable:
                return [
                    si
                    for si in li
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    )
                ]
            else:
                return li
        
        # 获取当前持仓
        current_temp: Position = copy.deepcopy(self.trade_position)
        current_stock_list = current_temp.get_stock_list()
        cash = current_temp.get_cash()
        
        # 计算当前总资产和回撤（用于仓位控制）
        total_value_current = current_temp.calculate_value()
        if self.peak_account_value is None or total_value_current > self.peak_account_value:
            self.peak_account_value = total_value_current
        
        current_drawdown = None
        risk_degree_multiplier = 1.0
        should_reduce_position = False  # 是否需要主动减仓
        
        if self.peak_account_value is not None and self.peak_account_value > 0:
            current_drawdown = (self.peak_account_value - total_value_current) / self.peak_account_value
            
            # 回撤预警：提前降低仓位（但不要频繁减仓）
            if self.drawdown_warning is not None and current_drawdown > self.drawdown_warning:
                # 回撤超过预警阈值，开始降低仓位
                excess_drawdown = current_drawdown - self.drawdown_warning
                # 适度的仓位降低：回撤越大，仓位降低越多
                risk_degree_multiplier = max(0.5, 1.0 - excess_drawdown / (self.drawdown_warning * 2))
                logger.warning(f"回撤预警: {current_drawdown*100:.2f}%, 超过预警阈值 {self.drawdown_warning*100:.2f}%, 降低仓位至 {risk_degree_multiplier*100:.1f}%")
                
                # 如果回撤超过预警阈值较多，且不在冷却期内，才考虑主动减仓
                if (current_drawdown > self.drawdown_warning * 2.0 and  # 超过预警阈值的2倍（10%）
                    (self.last_drawdown_reduce_date is None or 
                     (trade_start_time - self.last_drawdown_reduce_date).days >= self.drawdown_reduce_cooldown_days)):
                    should_reduce_position = True
                    logger.warning(f"回撤预警：回撤 {current_drawdown*100:.2f}% 超过预警阈值较多，主动减仓 {self.drawdown_reduce_ratio*100:.0f}%")
            
            # 回撤过大：更激进的减仓（但要有冷却期，避免频繁减仓）
            if self.max_drawdown is not None and current_drawdown > self.max_drawdown:
                # 回撤超过最大阈值，不仅降低买入，还要主动减仓
                excess_drawdown = current_drawdown - self.max_drawdown
                risk_degree_multiplier = max(0.3, 1.0 - excess_drawdown / self.max_drawdown)
                
                # 只有在冷却期外才减仓，避免频繁交易
                if (self.last_drawdown_reduce_date is None or 
                    (trade_start_time - self.last_drawdown_reduce_date).days >= self.drawdown_reduce_cooldown_days):
                    should_reduce_position = True  # 标记需要主动减仓
                    logger.warning(f"回撤过大: {current_drawdown*100:.2f}%, 超过阈值 {self.max_drawdown*100:.2f}%, 降低仓位至 {risk_degree_multiplier*100:.1f}%, 主动减仓 {self.drawdown_reduce_ratio*100:.0f}%")
                else:
                    logger.warning(f"回撤过大: {current_drawdown*100:.2f}%, 超过阈值 {self.max_drawdown*100:.2f}%, 但在冷却期内，仅降低仓位至 {risk_degree_multiplier*100:.1f}%")
        
        # ========== 1. TopkDrop 逻辑：基于长期预测值选择要买入和卖出的股票 ==========
        # 但只有在海龟突破时才使用长期预测
        
        # 过滤：只考虑发生海龟突破的股票（如果启用突破过滤）
        if self.enable_turtle_breakout:
            breakout_stocks = []
            for stock_id in stock_pool:
                if self._is_turtle_breakout(stock_id, trade_start_time):
                    breakout_stocks.append(stock_id)
            
            if len(breakout_stocks) == 0:
                logger.warning(f"没有股票发生海龟突破，不进行长期预测交易")
                # 如果没有突破，只使用短期预测进行加仓/止盈，不进行TopkDrop交易
                breakout_stocks = []
            else:
                logger.debug(f"发生海龟突破的股票数量: {len(breakout_stocks)}/{len(stock_pool)}")
            
            # 只在突破股票中筛选长期预测值
            long_term_pred_breakout = long_term_pred_filtered.reindex(breakout_stocks)
            short_term_pred_breakout = short_term_pred_filtered.reindex(breakout_stocks)
        else:
            # 未启用突破过滤，使用所有股票
            long_term_pred_breakout = long_term_pred_filtered
            short_term_pred_breakout = short_term_pred_filtered

        combined_buy_score = self._build_combined_buy_score(long_term_pred_breakout, short_term_pred_breakout)
        combined_sell_score = self._build_combined_sell_score(long_term_pred_breakout, short_term_pred_breakout)
        
        # 当前持仓按长期预测值排序（只考虑突破股票）
        current_stock_list_breakout = [s for s in current_stock_list if s in combined_sell_score.index]
        last = combined_sell_score.reindex(current_stock_list_breakout).sort_values(ascending=False).index
        
        # 选择要买入的候选股票（基于长期预测值，排除已持仓，只考虑突破股票）
        if self.method_buy == "top":
            available_for_buy = combined_buy_score[~combined_buy_score.index.isin(last)]
            if len(available_for_buy) > 0:
                today_candidates = get_first_n(
                    available_for_buy.sort_values(ascending=False).index,
                    self.n_drop + self.topk - len(last),
                )
            else:
                today_candidates = []
        else:
            raise NotImplementedError(f"买入方法 {self.method_buy} 不支持")
        
        # 合并候选股票和当前持仓，用于选择要卖出的股票
        if len(last) > 0 or len(today_candidates) > 0:
            comb = combined_sell_score.reindex(last.union(pd.Index(today_candidates))).sort_values(ascending=False).index
        else:
            comb = pd.Index([])
        
        # 选择要卖出的股票（基于长期预测值，只考虑突破股票）
        if self.method_sell == "bottom":
            if len(comb) > 0 and len(last) > 0:
                sell_candidates = last[last.isin(get_last_n(comb, min(self.n_drop, len(comb))))]
            else:
                sell_candidates = pd.Index([])
        else:
            raise NotImplementedError(f"卖出方法 {self.method_sell} 不支持")

        # 先基于最小持有期过滤计划卖出列表，避免后续买入数量按“计划卖出”高估
        time_per_step = self.trade_calendar.get_freq()
        executable_sell_candidates = []
        for code in sell_candidates:
            if current_temp.get_stock_count(code, bar=time_per_step) < self.hold_thresh:
                continue
            executable_sell_candidates.append(code)
        sell_candidates = pd.Index(executable_sell_candidates)
        
        # 实际要买入的股票数量
        buy_count = max(0, len(sell_candidates) + self.topk - len(last))
        buy_candidates = today_candidates[:buy_count] if len(today_candidates) > 0 else []
        
        # ========== 2. 买入筛选：只选择长期和短期预测都为正的股票 ==========
        buy_list = []
        backup_buy_list = []  # 备选股票（只要求长期预测为正）
        
        for stock_id in buy_candidates:
            if stock_id not in long_term_pred_filtered.index or stock_id not in short_term_pred_filtered.index:
                continue
            
            long_term_value = long_term_pred_filtered.loc[stock_id]
            short_term_value = short_term_pred_filtered.loc[stock_id]
            
            is_long_term_positive = self._is_prediction_positive(
                long_term_value, long_term_pred_filtered, quantile=self.long_signal_quantile
            )
            is_short_term_positive = self._is_prediction_positive(
                short_term_value, short_term_pred_filtered, quantile=self.short_buy_quantile
            )
            
            if is_long_term_positive and is_short_term_positive:
                buy_list.append(stock_id)
            elif is_long_term_positive:
                # 只要求长期预测为正（备选方案）
                backup_buy_list.append(stock_id)
        
        # 如果符合条件的股票不够，使用长期为正的备选股票补足，避免组合长期吃不满仓
        if len(buy_list) < buy_count:
            needed = buy_count - len(buy_list)
            fallback = backup_buy_list[:needed]
            buy_list.extend(fallback)
            if len(buy_list) < buy_count:
                logger.warning(
                    f"买入候选股票不足：需要 {buy_count} 只，实际只有 {len(buy_list)} 只 "
                    f"（双正信号 {len(buy_list) - len(fallback)} 只，仅长期为正 {len(fallback)} 只）"
                )
        
        # 统一交易记录（首次买入、加仓、换仓卖出、止损卖出、止盈减仓等）
        trade_records: List[Dict] = []
        
        # ========== 3. 生成卖出订单（换仓：TopkDrop 调出标的全部卖出） ==========
        sell_order_list = []
        for code in current_stock_list:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
            ):
                continue
            
            if code in sell_candidates:
                # 生成卖出订单
                sell_amount = current_temp.get_stock_amount(code=code)
                entry_price = self.entry_prices.get(code)
                sell_order = Order(
                    stock_id=code,
                    amount=sell_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.SELL,
                )
                
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                        sell_order, position=current_temp
                    )
                    cash += trade_val - trade_cost
                    # 换仓卖出：区分止盈/止损
                    if entry_price is not None and entry_price > 0 and trade_price is not None:
                        profit_ratio = (trade_price - entry_price) / entry_price
                        reason = f"换仓止盈{profit_ratio*100:.2f}%" if profit_ratio >= 0 else f"换仓止损{profit_ratio*100:.2f}%"
                    else:
                        reason = "TopkDrop调出"
                    trade_records.append({
                        "date": trade_start_time,
                        "type": "换仓卖出",
                        "stock_id": code,
                        "amount": sell_amount,
                        "price": trade_price,
                        "reason": reason,
                    })
                    self.entry_prices.pop(code, None)
        
        # ========== 4. 生成买入订单 ==========
        buy_order_list = []
        # 开仓：单只仓位 = 总资产 * open_position_ratio（默认 5%）
        total_value = current_temp.calculate_value()
        if len(buy_list) > 0:
            max_single_value = total_value * self.max_single_stock_ratio
            desired_value_per_stock = min(
                total_value * self.open_position_ratio * risk_degree_multiplier,
                max_single_value,
            )

            current_equity_value = total_value - cash
            target_total_exposure = total_value * self.risk_degree * risk_degree_multiplier
            remaining_capacity = max(0.0, target_total_exposure - current_equity_value)
            available_buy_budget = min(cash, remaining_capacity)

            if available_buy_budget <= 0:
                stock_values = {}
            else:
                buy_scores = combined_buy_score.reindex(buy_list).fillna(0.0)
                min_score = float(buy_scores.min()) if len(buy_scores) > 0 else 0.0
                shifted_scores = buy_scores - min_score
                positive_scores = shifted_scores + max(1e-6, shifted_scores.max() * 0.05)
                score_sum = float(positive_scores.sum())

                if score_sum <= 0:
                    total_desired_buy_value = desired_value_per_stock * len(buy_list)
                    scale = min(1.0, available_buy_budget / total_desired_buy_value) if total_desired_buy_value > 0 else 0.0
                    scaled_value_per_stock = desired_value_per_stock * scale
                    stock_values = {code: min(scaled_value_per_stock, max_single_value) for code in buy_list}
                else:
                    stock_values = {}
                    for code in buy_list:
                        raw_value = available_buy_budget * (positive_scores.loc[code] / score_sum)
                        stock_values[code] = min(raw_value, desired_value_per_stock, max_single_value)
        else:
            stock_values = {}
        
        for code in buy_list:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
            ):
                continue
            
            # 生成买入订单
            buy_price = self.trade_exchange.get_deal_price(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
            )
            if buy_price is None or buy_price <= 0:
                continue
            buy_amount = stock_values.get(code, 0) / buy_price
            factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            
            buy_order = Order(
                stock_id=code,
                amount=buy_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.BUY,
            )
            buy_order_list.append(buy_order)
            
            if code not in current_stock_list:
                self.entry_prices[code] = buy_price
                trade_records.append({
                    "date": trade_start_time,
                    "type": "首次买入",
                    "stock_id": code,
                    "amount": buy_amount,
                    "price": buy_price,
                    "reason": "新入选",
                })
        
        # ========== 5. 回撤控制：如果回撤过大，主动减仓 ==========
        adjust_order_list = []
        
        # 如果回撤过大，主动减仓（卖出部分持仓）
        if should_reduce_position and len(current_stock_list) > 0:
            logger.warning(f"执行回撤控制减仓：当前持仓 {len(current_stock_list)} 只股票，回撤 {current_drawdown*100:.2f}%")
            # 优先减掉组合分数较弱、且已有盈利的仓位
            reduce_candidates = []
            for code in current_stock_list:
                if code in sell_candidates or code not in combined_sell_score.index:
                    continue
                current_amount = current_temp.get_stock_amount(code)
                if current_amount <= 0:
                    continue
                current_price = self.trade_exchange.get_deal_price(
                    stock_id=code,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.SELL,
                )
                entry_price = self.entry_prices.get(code)
                profit_ratio = 0.0
                if (
                    entry_price is not None
                    and entry_price > 0
                    and current_price is not None
                    and current_price > 0
                ):
                    profit_ratio = (current_price - entry_price) / entry_price
                reduce_candidates.append((code, combined_sell_score.loc[code], profit_ratio))

            stocks_to_reduce = [
                code
                for code, _, _ in sorted(
                    reduce_candidates,
                    key=lambda item: (item[2] <= 0, item[1], -item[2]),
                )
            ]
            # 减仓数量：至少减仓 drawdown_reduce_ratio 比例的股票
            reduce_count = max(1, int(len(current_stock_list) * self.drawdown_reduce_ratio))
            stocks_to_reduce = stocks_to_reduce[:reduce_count]
            
            for code in stocks_to_reduce:
                if code in sell_candidates:  # 已经在卖出列表中，跳过
                    continue
                
                if not self.trade_exchange.is_stock_tradable(
                    stock_id=code,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
                ):
                    continue
                
                current_amount = current_temp.get_stock_amount(code)
                if current_amount <= 0:
                    continue
                
                # 减仓：卖出部分持仓（例如30%）
                reduce_amount = int(current_amount * self.drawdown_reduce_ratio)
                factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
                reduce_amount = self.trade_exchange.round_amount_by_trade_unit(reduce_amount, factor)
                
                if reduce_amount > 0 and reduce_amount < current_amount:
                    reduce_order = Order(
                        stock_id=code,
                        amount=reduce_amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=OrderDir.SELL,
                    )
                    if self.trade_exchange.check_order(reduce_order):
                        adjust_order_list.append(reduce_order)
                        logger.warning(f"回撤控制：股票 {code} 减仓 {reduce_amount} 股（{reduce_amount/current_amount*100:.1f}%）")
                        trade_records.append({
                            "date": trade_start_time,
                            "type": "回撤减仓",
                            "stock_id": code,
                            "amount": reduce_amount,
                            "reason": f"回撤{current_drawdown*100:.2f}%超阈值",
                        })
            
            # 如果执行了减仓，记录日期（用于冷却期）
            if len(adjust_order_list) > 0:
                self.last_drawdown_reduce_date = trade_start_time
                logger.warning(f"回撤控制：执行了 {len(adjust_order_list)} 笔减仓订单，进入冷却期 {self.drawdown_reduce_cooldown_days} 天")
            elif should_reduce_position:
                logger.warning(f"回撤控制：尝试减仓但未生成任何订单（可能是股票不可交易或已卖出，或在冷却期内）")
        
        # ========== 6. 加仓/止盈：根据长期和短期预测值对已持仓股票进行调整 ==========
        current_stock_list_after_sell = [s for s in current_stock_list if s not in sell_candidates]
        
        # 获取总资产（用于计算单只股票最大仓位）
        total_value = current_temp.calculate_value()
        remaining_adjust_cash = max(0.0, cash)
        
        for code in current_stock_list_after_sell:
            if code not in long_term_pred_filtered.index or code not in short_term_pred_filtered.index:
                continue
            
            # 获取长期和短期预测值
            long_term_value = long_term_pred_filtered.loc[code]
            short_term_value = short_term_pred_filtered.loc[code]
            is_long_term_positive = self._is_prediction_positive(
                long_term_value, long_term_pred_filtered, quantile=self.long_signal_quantile
            )
            is_short_term_positive = self._is_prediction_positive(
                short_term_value, short_term_pred_filtered, quantile=self.short_buy_quantile
            )
            is_short_term_strong = self._is_prediction_positive(
                short_term_value, short_term_pred_filtered, quantile=self.short_add_quantile
            )
            is_short_term_weak = not self._is_prediction_positive(
                short_term_value, short_term_pred_filtered, quantile=self.short_reduce_quantile
            )
            
            # 获取当前持仓和价格
            current_amount = current_temp.get_stock_amount(code)
            if current_amount <= 0:
                continue
            
            current_price = self.trade_exchange.get_deal_price(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.SELL
            )
            
            # 获取入场价格（如果之前没有记录，使用当前价格）
            entry_price = self.entry_prices.get(code, current_price)
            if current_price is None or current_price <= 0:
                continue  # 跳过无效价格
            if entry_price is None or entry_price <= 0:
                entry_price = current_price
                self.entry_prices[code] = entry_price
            
            # 计算盈利比例
            profit_ratio = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            
            # 决策逻辑：首先检查止损（基于亏损比例）
            if profit_ratio < self.stop_loss_threshold:
                # 亏损超过止损阈值：止损（全部卖出）
                if not self.trade_exchange.is_stock_tradable(
                    stock_id=code,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
                ):
                    continue
                
                stop_loss_order = Order(
                    stock_id=code,
                    amount=current_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.SELL,
                )
                if self.trade_exchange.check_order(stop_loss_order):
                    adjust_order_list.append(stop_loss_order)
                    self.entry_prices.pop(code, None)
                    trade_records.append({
                        "date": trade_start_time,
                        "type": "止损卖出",
                        "stock_id": code,
                        "amount": current_amount,
                        "price": current_price,
                        "reason": f"亏损{profit_ratio*100:.2f}%",
                    })
            
            # 然后检查长期预测
            elif not is_long_term_positive:
                # 长期预测为负：止损（全部卖出）
                if not self.trade_exchange.is_stock_tradable(
                    stock_id=code,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
                ):
                    continue
                
                stop_loss_order = Order(
                    stock_id=code,
                    amount=current_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.SELL,
                )
                if self.trade_exchange.check_order(stop_loss_order):
                    adjust_order_list.append(stop_loss_order)
                    self.entry_prices.pop(code, None)
                    trade_records.append({
                        "date": trade_start_time,
                        "type": "长期预测为负卖出",
                        "stock_id": code,
                        "amount": current_amount,
                        "price": current_price,
                        "reason": "长期预测为负",
                    })
            
            elif is_short_term_strong:
                # 长期预测为正，短期预测为正：加仓
                if not self.trade_exchange.is_stock_tradable(
                    stock_id=code,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
                ):
                    continue
                
                # 加仓金额：与开仓单位一致 = 总资产 * position_unit_ratio（如 1%）
                current_value = current_amount * current_price
                max_single_stock_value = total_value * self.max_single_stock_ratio
                add_score = combined_buy_score.get(code, np.nan)
                add_multiplier = self._get_add_size_multiplier(add_score)
                add_value = total_value * self.position_unit_ratio * risk_degree_multiplier * add_multiplier
                if add_value > max(0, max_single_stock_value - current_value):
                    add_value = max(0, max_single_stock_value - current_value)
                max_add_value = remaining_adjust_cash * self.risk_degree * risk_degree_multiplier
                if add_value > max_add_value:
                    add_value = max_add_value
                if add_value > 0:
                    add_amount = add_value / current_price
                    factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
                    add_amount = self.trade_exchange.round_amount_by_trade_unit(add_amount, factor)
                    if add_amount > 0:
                        add_order = Order(
                            stock_id=code,
                            amount=add_amount,
                            start_time=trade_start_time,
                            end_time=trade_end_time,
                            direction=OrderDir.BUY,
                        )
                        if self.trade_exchange.check_order(add_order):
                            adjust_order_list.append(add_order)
                            remaining_adjust_cash = max(0.0, remaining_adjust_cash - add_value)
                            old_price = self.entry_prices.get(code, current_price)
                            old_amount = current_amount
                            new_price = (old_amount * old_price + add_amount * current_price) / (old_amount + add_amount)
                            self.entry_prices[code] = new_price
                            trade_records.append({
                                "date": trade_start_time,
                                "type": "加仓",
                                "stock_id": code,
                                "amount": add_amount,
                                "price": current_price,
                                "reason": f"短期强势({add_multiplier:.2f}x)",
                            })
            
            elif is_short_term_weak and profit_ratio > self.profit_threshold:
                # 长期预测为正，短期预测为负且盈利：止盈（减仓）
                if not self.trade_exchange.is_stock_tradable(
                    stock_id=code,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
                ):
                    continue
                
                reduce_amount = int(current_amount * self.reduce_position_ratio)
                factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
                reduce_amount = self.trade_exchange.round_amount_by_trade_unit(reduce_amount, factor)
                
                if reduce_amount > 0 and reduce_amount < current_amount:
                    reduce_order = Order(
                        stock_id=code,
                        amount=reduce_amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=OrderDir.SELL,
                    )
                    if self.trade_exchange.check_order(reduce_order):
                        adjust_order_list.append(reduce_order)
                        # 更新入场价格（保留的仓位使用当前价格作为新的入场价格）
                        remaining_amount = current_amount - reduce_amount
                        if remaining_amount > 0:
                            self.entry_prices[code] = current_price
                        trade_records.append({
                            "date": trade_start_time,
                            "type": "止盈减仓",
                            "stock_id": code,
                            "amount": reduce_amount,
                            "price": current_price,
                            "reason": f"盈利{profit_ratio*100:.2f}%",
                        })

        return TradeDecisionWO(sell_order_list + buy_order_list + adjust_order_list, self)
