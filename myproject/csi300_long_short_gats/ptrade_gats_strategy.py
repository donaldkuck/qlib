# -*- coding: utf-8 -*-
"""
Ptrade 版本的多空 GATs 策略

完全独立于 Qlib，直接在 ptrade 平台上运行：
1. 从 ptrade 获取历史数据
2. 按照原始因子配置计算因子
3. 加载预训练的 GATs 模型进行预测
4. 基于 TopkDrop 思想执行多空交易策略

用法：
    1. 将训练好的模型文件（.pkl）放在同一目录下：
       - short_term_params.pkl / short_term_all_models.pkl
       - long_term_params.pkl / long_term_all_models.pkl

    2. 确保因子配置文件存在：
       - factor_config.yaml（短期因子）
       - factor_config_longterm.yaml（长期因子）

    3. 在 ptrade 平台上运行此策略
"""

import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# 导入自定义模块
from ptrade_factors import (
    FactorLibrary, load_factors_from_config,
    calculate_factor_series, prepare_time_series_data, normalize_features
)
from ptrade_gats_model import PtradeGATsPredictor, load_model_from_pickle


class PtradeGATsStrategy:
    """
    Ptrade 多空 GATs 策略

    完全独立于 Qlib，使用 ptrade API 进行数据获取和交易
    """

    def __init__(
        self,
        # 模型路径
        short_term_model_path: Optional[str] = None,
        long_term_model_path: Optional[str] = None,
        # 因子配置
        short_factor_config: Optional[str] = None,
        long_factor_config: Optional[str] = None,
        # 模型参数
        short_step_len: int = 20,
        long_step_len: int = 40,
        short_d_feat: int = 143,
        long_d_feat: int = 150,
        # TopkDrop 参数
        topk: int = 20,
        n_drop: int = 5,
        hold_thresh: int = 10,
        # 加仓/止盈参数
        add_position_ratio: float = 0.03,
        profit_threshold: float = 0.05,
        reduce_position_ratio: float = 0.2,
        max_single_stock_ratio: float = 0.05,
        # 止损参数
        stop_loss_threshold: float = -0.05,
        # 回撤控制参数
        max_drawdown: Optional[float] = None,
        drawdown_reduce_ratio: float = 0.3,
        # 其他参数
        risk_degree: float = 0.95,
        # 设备
        device: str = "cpu",
    ):
        """
        初始化策略

        Parameters
        ----------
        short_term_model_path : str
            短期模型文件路径
        long_term_model_path : str
            长期模型文件路径
        short_factor_config : str
            短期因子配置文件路径
        long_factor_config : str
            长期因子配置文件路径
        short_step_len : int
            短期时间序列长度
        long_step_len : int
            长期时间序列长度
        short_d_feat : int
            短期特征维度
        long_d_feat : int
            长期特征维度
        其他参数同原始策略
        """
        self.short_term_model_path = short_term_model_path or "short_term_params.pkl"
        self.long_term_model_path = long_term_model_path or "long_term_params.pkl"
        self.short_factor_config = short_factor_config or "factor_config.yaml"
        self.long_factor_config = long_factor_config or "factor_config_longterm.yaml"

        self.short_step_len = short_step_len
        self.long_step_len = long_step_len
        self.short_d_feat = short_d_feat
        self.long_d_feat = long_d_feat

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
        self.device = device

        # 状态变量
        self.entry_prices: Dict[str, float] = {}
        self.peak_account_value: Optional[float] = None
        self.last_drawdown_reduce_date: Optional[datetime] = None
        self.drawdown_reduce_cooldown_days: int = 5
        self.stock_hold_days: Dict[str, int] = {}

        # 因子配置
        self.short_factors: Dict[str, str] = {}
        self.long_factors: Dict[str, str] = {}

        # 因子缓存（避免重复计算）
        self.factor_cache: Dict[str, pd.DataFrame] = {}
        self.last_factor_calc_date: Optional[datetime] = None

        # 数据缓存（股票的历史数据）
        self.data_cache: Dict[str, pd.DataFrame] = {}

        # 因子名称（按顺序）
        self.short_factor_names: List[str] = []
        self.long_factor_names: List[str] = []

        # 已初始化标志
        self._initialized = False

    def initialize(self):
        """初始化策略（加载模型和因子配置）"""
        if self._initialized:
            return

        print("=" * 80)
        print("Ptrade GATs 策略初始化")
        print("=" * 80)

        # 加载因子配置
        self.short_factors = load_factors_from_config(self.short_factor_config)
        self.long_factors = load_factors_from_config(self.long_factor_config)

        self.short_factor_names = list(self.short_factors.keys())[:self.short_d_feat]
        self.long_factor_names = list(self.long_factors.keys())[:self.long_d_feat]

        print(f"短期因子: {len(self.short_factor_names)} 个")
        print(f"长期因子: {len(self.long_factor_names)} 个")

        # 加载模型
        print(f"\n加载短期模型: {self.short_term_model_path}")
        try:
            self.short_term_model = load_model_from_pickle(
                self.short_term_model_path,
                device=self.device
            )
        except Exception as e:
            print(f"⚠️ 无法加载短期模型: {e}")
            self.short_term_model = None

        print(f"加载长期模型: {self.long_term_model_path}")
        try:
            self.long_term_model = load_model_from_pickle(
                self.long_term_model_path,
                device=self.device
            )
        except Exception as e:
            print(f"⚠️ 无法加载长期模型: {e}")
            self.long_term_model = None

        if self.short_term_model is None or self.long_term_model is None:
            raise RuntimeError("模型加载失败，请检查模型文件路径")

        print("=" * 80)
        print("策略初始化完成")
        print("=" * 80)

        self._initialized = True

    def is_prediction_positive(self, pred_value: float, all_pred_values: pd.Series = None) -> bool:
        """判断预测值是否为正"""
        if pd.isna(pred_value):
            return False

        # 使用中位数作为阈值（相对排名）
        if all_pred_values is not None and len(all_pred_values) > 0:
            valid_values = all_pred_values.dropna()
            if len(valid_values) > 0:
                threshold = valid_values.quantile(0.5)
                return pred_value > threshold

        return pred_value > 0

    def fetch_stock_data(self, context, stock_code: str, lookback_days: int = 100) -> pd.DataFrame:
        """
        从 ptrade 获取股票历史数据

        Parameters
        ----------
        context
            Ptrade 上下文对象
        stock_code : str
            股票代码
        lookback_days : int
            获取历史天数

        Returns
        -------
        pd.DataFrame
            OHLCV 数据
        """
        cache_key = f"{stock_code}_{lookback_days}"
        current_date = context.current_dt.date()

        # 检查缓存
        if cache_key in self.data_cache:
            cached_data = self.data_cache[cache_key]
            last_date = cached_data.index[-1].date() if len(cached_data) > 0 else None
            if last_date == current_date:
                return cached_data

        # 从 ptrade 获取数据
        try:
            start_date = current_date - timedelta(days=lookback_days * 2)
            price_data = context.get_price(
                stock_code,
                start_date=start_date,
                end_date=current_date,
                frequency="1d"
            )

            if price_data is None or len(price_data) == 0:
                return pd.DataFrame()

            # 转换为 DataFrame
            df = pd.DataFrame({
                'close': price_data['close'].values,
                'high': price_data.get('high', price_data['close']).values,
                'low': price_data.get('low', price_data['close']).values,
                'open': price_data.get('open', price_data['close']).values,
                'volume': price_data.get('volume', pd.Series(index=price_data.index)).values,
            }, index=price_data.index)

            # 只保留最近 lookback_days 行
            df = df.iloc[-lookback_days:].copy()

            # 缓存数据
            self.data_cache[cache_key] = df

            return df

        except Exception as e:
            print(f"获取 {stock_code} 数据失败: {e}")
            return pd.DataFrame()

    def predict_stock(self, stock_code: str, context, model_type: str = "short_term") -> Optional[float]:
        """
        预测单只股票的收益

        Parameters
        ----------
        stock_code : str
            股票代码
        context
            Ptrade 上下文对象
        model_type : str
            模型类型（short_term/long_term）

        Returns
        -------
        float or None
            预测值
        """
        # 获取模型参数
        if model_type == "short_term":
            model = self.short_term_model
            step_len = self.short_step_len
            d_feat = self.short_d_feat
            factor_names = self.short_factor_names
        else:
            model = self.long_term_model
            step_len = self.long_step_len
            d_feat = self.long_d_feat
            factor_names = self.long_factor_names

        if model is None:
            return None

        # 获取股票数据
        lookback = max(step_len + 50, 100)  # 确保有足够数据计算因子
        stock_data = self.fetch_stock_data(context, stock_code, lookback_days=lookback)

        if stock_data is None or len(stock_data) < step_len:
            return None

        # 计算因子
        try:
            factors = calculate_factor_series(stock_data, {
                k: v for k, v in (self.short_factors if model_type == "short_term" else self.long_factors).items()
                if k in factor_names
            })

            # 确保所有因子都存在
            missing_factors = [f for f in factor_names if f not in factors.columns]
            if missing_factors:
                for f in missing_factors:
                    factors[f] = np.nan

            factors = factors[factor_names]

            # 准备时间序列数据
            ts_data = prepare_time_series_data(factors, step_len, d_feat, factor_names)

            # 归一化
            ts_data = normalize_features(ts_data)

            # 预测
            pred = model.predict_single(ts_data)

            return pred

        except Exception as e:
            print(f"预测 {stock_code} ({model_type}) 失败: {e}")
            return None

    def predict_all_stocks(self, context, stock_pool: List[str], model_type: str = "short_term") -> pd.Series:
        """
        预测所有股票池的收益

        Parameters
        ----------
        context
            Ptrade 上下文对象
        stock_pool : List[str]
            股票池
        model_type : str
            模型类型（short_term/long_term）

        Returns
        -------
        pd.Series
            预测结果，index 为股票代码
        """
        predictions = {}

        for stock_code in stock_pool:
            pred = self.predict_stock(stock_code, context, model_type)
            if pred is not None and not np.isnan(pred):
                predictions[stock_code] = pred

        return pd.Series(predictions)

    def generate_trade_decision(self, context):
        """
        生成交易决策（在 handle_data 中调用）

        Parameters
        ----------
        context
            Ptrade 上下文对象
        """
        if not self._initialized:
            self.initialize()

        current_date = context.current_dt

        print(f"\n=== {current_date.strftime('%Y-%m-%d')} 策略执行 ===")

        # 获取股票池
        stock_pool = context.get_universe()
        if not stock_pool:
            print("无股票池，跳过")
            return

        # 预测长期和短期收益
        print(f"预测 {len(stock_pool)} 只股票的收益...")

        long_term_pred = self.predict_all_stocks(context, stock_pool, model_type="long_term")
        short_term_pred = self.predict_all_stocks(context, stock_pool, model_type="short_term")

        print(f"长期预测: {len(long_term_pred)} 只有效")
        print(f"短期预测: {len(short_term_pred)} 只有效")

        if long_term_pred.empty or short_term_pred.empty:
            print("预测结果为空，跳过")
            return

        # 获取股票池（取交集）
        stock_pool_valid = long_term_pred.index.intersection(short_term_pred.index)
        long_term_pred = long_term_pred.reindex(stock_pool_valid)
        short_term_pred = short_term_pred.reindex(stock_pool_valid)

        if len(stock_pool_valid) == 0:
            print("无有效股票池，跳过")
            return

        # 获取当前持仓
        positions = context.get_positions()
        current_stock_list = [pos.stock_code for pos in positions if pos.amount > 0]

        # 更新持仓天数
        for code in current_stock_list:
            if code not in self.stock_hold_days:
                self.stock_hold_days[code] = 0
            self.stock_hold_days[code] += 1

        # 计算总资产和回撤
        total_value = context.account.total_value
        if self.peak_account_value is None or total_value > self.peak_account_value:
            self.peak_account_value = total_value

        risk_degree_multiplier = 1.0
        should_reduce_position = False

        if self.peak_account_value is not None and self.peak_account_value > 0:
            current_drawdown = (self.peak_account_value - total_value) / self.peak_account_value

            # 回撤预警
            drawdown_warning = self.max_drawdown * 0.7 if self.max_drawdown else None
            if drawdown_warning is not None and current_drawdown > drawdown_warning:
                excess_drawdown = current_drawdown - drawdown_warning
                risk_degree_multiplier = max(0.5, 1.0 - excess_drawdown / (drawdown_warning * 2))
                print(f"回撤预警: {current_drawdown*100:.2f}%, 降低仓位至 {risk_degree_multiplier*100:.1f}%")

            # 回撤过大
            if self.max_drawdown is not None and current_drawdown > self.max_drawdown:
                excess_drawdown = current_drawdown - self.max_drawdown
                risk_degree_multiplier = max(0.3, 1.0 - excess_drawdown / self.max_drawdown)

                cooldown_days = self.drawdown_reduce_cooldown_days
                if self.last_drawdown_reduce_date is None:
                    should_reduce_position = True
                    self.last_drawdown_reduce_date = current_date
                    print(f"回撤过大: {current_drawdown*100:.2f}%, 主动减仓 {self.drawdown_reduce_ratio*100:.0f}%")
                elif (current_date - self.last_drawdown_reduce_date).days >= cooldown_days:
                    should_reduce_position = True
                    self.last_drawdown_reduce_date = current_date
                    print(f"回撤过大: {current_drawdown*100:.2f}%, 主动减仓 {self.drawdown_reduce_ratio*100:.0f}%")

        # ========== 1. TopkDrop 逻辑 ==========
        current_stock_list_valid = [s for s in current_stock_list if s in long_term_pred.index]
        if current_stock_list_valid:
            long_pred_current = long_term_pred.reindex(current_stock_list_valid)
            last = long_pred_current.sort_values(ascending=False).index.tolist()
        else:
            last = []

        available_for_buy = long_term_pred[~long_term_pred.index.isin(last)]
        if len(available_for_buy) > 0:
            n_candidates = min(self.n_drop + self.topk - len(last), len(available_for_buy))
            today_candidates = available_for_buy.sort_values(ascending=False).index.tolist()[:n_candidates]
        else:
            today_candidates = []

        comb = list(set(last).union(set(today_candidates)))
        if comb:
            long_pred_comb = long_term_pred.reindex(comb).sort_values(ascending=False)
            comb = long_pred_comb.index.tolist()

        sell_candidates = []
        if len(comb) > 0 and len(last) > 0:
            n_sell = min(self.n_drop, len(comb))
            bottom_stocks = long_term_pred.reindex(comb).sort_values(ascending=True).index.tolist()[:n_sell]
            sell_candidates = [s for s in last if s in bottom_stocks]

        buy_count = len(sell_candidates) + self.topk - len(last)
        buy_count = max(0, buy_count)

        # ========== 2. 买入筛选 ==========
        buy_list = []
        backup_buy_list = []

        for stock_id in today_candidates[:buy_count]:
            if stock_id not in long_term_pred.index or stock_id not in short_term_pred.index:
                continue

            long_term_value = long_term_pred.loc[stock_id]
            short_term_value = short_term_pred.loc[stock_id]

            is_long_term_positive = self.is_prediction_positive(long_term_value, long_term_pred)
            is_short_term_positive = self.is_prediction_positive(short_term_value, short_term_pred)

            if is_long_term_positive and is_short_term_positive:
                buy_list.append(stock_id)
            elif is_long_term_positive:
                backup_buy_list.append(stock_id)

        if len(buy_list) < buy_count:
            needed = buy_count - len(buy_list)
            buy_list.extend(backup_buy_list[:needed])

        # ========== 3. 执行卖出订单 ==========
        print(f"当前持仓: {len(current_stock_list)} 只")
        print(f"候选卖出: {sell_candidates}")

        for code in current_stock_list:
            if code not in sell_candidates:
                continue

            hold_days = self.stock_hold_days.get(code, 0)
            if hold_days < self.hold_thresh:
                print(f"  {code} 持有天数 {hold_days} < {self.hold_thresh}，暂不卖出")
                continue

            pos = context.get_position(code)
            if pos and pos.amount > 0:
                context.order(code, 0)
                self.entry_prices.pop(code, None)
                self.stock_hold_days.pop(code, None)
                print(f"  卖出 {code}: {pos.amount} 股")

        # ========== 4. 执行买入订单 ==========
        available_cash = context.account.cash
        cash_for_buy = available_cash * self.risk_degree * risk_degree_multiplier

        print(f"买入候选: {buy_list[:buy_count]}, 可用资金: {cash_for_buy:,.2f}")

        if len(buy_list) > 0 and cash_for_buy > 0:
            # 按信号强度分配资金
            signal_scores = {}
            for code in buy_list:
                if code in long_term_pred.index and code in short_term_pred.index:
                    long_score = long_term_pred.loc[code]
                    short_score = short_term_pred.loc[code]
                    long_rank = (long_term_pred > long_score).sum() / len(long_term_pred)
                    short_rank = (short_term_pred > short_score).sum() / len(short_term_pred)
                    signal_scores[code] = 0.6 * (1 - long_rank) + 0.4 * (1 - short_rank)
                else:
                    signal_scores[code] = 0.5

            max_single_value = total_value * self.max_single_stock_ratio
            total_score = sum(signal_scores.values())

            if total_score > 0:
                stock_values = {
                    code: min(cash_for_buy * signal_scores[code] / total_score, max_single_value)
                    for code in buy_list
                }
            else:
                value_per_stock = min(cash_for_buy / len(buy_list), max_single_value)
                stock_values = {code: value_per_stock for code in buy_list}

            for code in buy_list:
                if code in long_term_pred.index and code in short_term_pred.index:
                    try:
                        price_data = context.get_price(code, end_date=current_date, frequency="1d")
                        if price_data is None or len(price_data) == 0:
                            continue
                        price = float(price_data['close'].iloc[-1])

                        if price <= 0:
                            continue

                        stock_value = stock_values.get(code, 0)
                        if stock_value <= 0:
                            continue

                        buy_amount = int(stock_value / price / 100) * 100

                        if buy_amount > 0:
                            context.order(code, buy_amount)
                            self.entry_prices[code] = price
                            self.stock_hold_days[code] = 0
                            print(f"  买入 {code}: {buy_amount} 股, 价格: {price:.2f}, 金额: {buy_amount * price:,.2f}")
                    except Exception as e:
                        print(f"  {code} 买入失败: {e}")

        # ========== 5. 加仓/止盈逻辑 ==========
        positions = context.get_positions()
        current_stock_list = [pos.stock_code for pos in positions if pos.amount > 0]

        for code in current_stock_list:
            if code not in long_term_pred.index or code not in short_term_pred.index:
                continue

            long_term_value = long_term_pred.loc[code]
            short_term_value = short_term_pred.loc[code]
            is_long_term_positive = self.is_prediction_positive(long_term_value, long_term_pred)
            is_short_term_positive = self.is_prediction_positive(short_term_value, short_term_pred)

            pos = context.get_position(code)
            if not pos or pos.amount <= 0:
                continue

            try:
                price_data = context.get_price(code, end_date=current_date, frequency="1d")
                if price_data is None or len(price_data) == 0:
                    continue
                current_price = float(price_data['close'].iloc[-1])
            except:
                continue

            if current_price <= 0:
                continue

            entry_price = self.entry_prices.get(code, current_price)
            if entry_price is None or entry_price <= 0:
                entry_price = current_price
                self.entry_prices[code] = entry_price

            profit_ratio = (current_price - entry_price) / entry_price if entry_price > 0 else 0

            # 止损判断
            if profit_ratio < self.stop_loss_threshold:
                context.order(code, 0)
                self.entry_prices.pop(code, None)
                self.stock_hold_days.pop(code, None)
                print(f"  {code} 亏损 {profit_ratio*100:.2f}%，止损卖出")
                continue

            if not is_long_term_positive:
                context.order(code, 0)
                self.entry_prices.pop(code, None)
                self.stock_hold_days.pop(code, None)
                print(f"  {code} 长期预测为负，止损卖出")
                continue

            if is_short_term_positive:
                current_value = pos.amount * current_price
                max_single_stock_value = total_value * self.max_single_stock_ratio
                add_value = min(current_value * self.add_position_ratio, max_single_stock_value - current_value)

                cash_available = context.account.cash
                max_add_value = cash_available * self.risk_degree * risk_degree_multiplier
                add_value = min(add_value, max_add_value)

                if add_value > 0:
                    add_amount = int(add_value / current_price / 100) * 100
                    if add_amount > 0:
                        context.order(code, pos.amount + add_amount)
                        old_price = self.entry_prices.get(code, current_price)
                        old_amount = pos.amount
                        new_price = (old_amount * old_price + add_amount * current_price) / (old_amount + add_amount)
                        self.entry_prices[code] = new_price
                        print(f"  {code} 加仓 {add_amount} 股，价格: {current_price:.2f}")

            elif not is_short_term_positive and profit_ratio > self.profit_threshold:
                reduce_amount = int(pos.amount * self.reduce_position_ratio / 100) * 100
                if reduce_amount > 0 and reduce_amount < pos.amount:
                    context.order(code, pos.amount - reduce_amount)
                    remaining_amount = pos.amount - reduce_amount
                    if remaining_amount > 0:
                        self.entry_prices[code] = current_price
                    print(f"  {code} 盈利 {profit_ratio*100:.2f}%，止盈减仓 {reduce_amount} 股")

        # 回撤控制
        if should_reduce_position and len(current_stock_list) > 0:
            print(f"执行回撤控制减仓")
            stocks_to_reduce = long_term_pred.reindex(current_stock_list).sort_values(ascending=True).index.tolist()
            reduce_count = max(1, int(len(current_stock_list) * self.drawdown_reduce_ratio))
            stocks_to_reduce = stocks_to_reduce[:reduce_count]

            for code in stocks_to_reduce:
                if code in sell_candidates:
                    continue
                pos = context.get_position(code)
                if pos and pos.amount > 0:
                    reduce_amount = int(pos.amount * self.drawdown_reduce_ratio / 100) * 100
                    if reduce_amount > 0 and reduce_amount < pos.amount:
                        context.order(code, pos.amount - reduce_amount)
                        print(f"  回撤控制：{code} 减仓 {reduce_amount} 股")

        print(f"=== 当前资产: {context.account.total_value:,.2f}, 现金: {context.account.cash:,.2f} ===\n")


# 全局策略实例
strategy = None


def initialize(context):
    """
    Ptrade 策略初始化函数

    Parameters
    ----------
    context
        Ptrade 上下文对象
    """
    global strategy

    # 设置基准
    context.set_benchmark("SH000300")

    # 设置佣金
    context.set_commission(commission_ratio=0.0003, min_commission=5.0)

    # 设置滑点
    context.set_slippage(slippage=0.1)

    # 创建策略实例
    strategy = PtradeGATsStrategy(
        topk=20,
        n_drop=5,
        hold_thresh=10,
        add_position_ratio=0.03,
        profit_threshold=0.05,
        reduce_position_ratio=0.2,
        max_single_stock_ratio=0.05,
        stop_loss_threshold=-0.05,
        max_drawdown=None,
        drawdown_reduce_ratio=0.3,
        risk_degree=0.95,
        device="cpu",
    )

    # 初始化策略
    strategy.initialize()

    print("Ptrade 策略初始化完成")


def handle_data(context, data):
    """
    Ptrade 主策略逻辑函数，每日执行

    Parameters
    ----------
    context
        Ptrade 上下文对象
    data
        数据对象
    """
    global strategy

    if strategy is None:
        return

    strategy.generate_trade_decision(context)


def after_trading_end(context, data):
    """
    盘后处理函数

    Parameters
    ----------
    context
        Ptrade 上下文对象
    data
        数据对象
    """
    current_date = context.current_dt.strftime("%Y-%m-%d")
    total_value = context.account.total_value
    print(f"\n{current_date} 盘后 - 总资产: {total_value:,.2f}, 持仓数: {len(context.get_positions())}")


if __name__ == "__main__":
    print("Ptrade GATs 策略")
    print("请在 Ptrade 平台上运行此策略文件")
