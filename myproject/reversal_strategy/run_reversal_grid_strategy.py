#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行单股反转网格策略

策略逻辑：
1. 开仓条件：长期动量 <= 0 或接近 0，且短期动量 > 0（反转信号）
2. 仓位管理：不满仓，使用网格加仓（每次加固定比例）
3. 退出条件：短期动量 < 0（趋势反转）
"""

import sys
import time
import warnings
from pathlib import Path
import pandas as pd
import numpy as np

# 过滤警告
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Mean of empty slice.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')

import logging
logging.getLogger('qlib.online operator').setLevel(logging.ERROR)
logging.getLogger('qlib.BaseExecutor').setLevel(logging.ERROR)

# 确保使用安装的qlib包
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
qlib_local_path = project_root / 'qlib'

paths_to_remove = []
for p in sys.path:
    p_str = str(p)
    if str(qlib_local_path) in p_str or (str(project_root) in p_str and 'qlib' in p_str.lower()):
        paths_to_remove.append(p_str)
    if 'myqlib' in p_str.lower() and 'qlib' in p_str.lower():
        paths_to_remove.append(p_str)

for p in paths_to_remove:
    if p in sys.path:
        sys.path.remove(p)

import qlib

# 添加当前目录和项目根目录到sys.path，以便导入reversal_strategy模块
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from qlib.workflow import R
from qlib.utils import init_instance_by_config
from qlib.backtest import backtest as normal_backtest
from qlib.contrib.evaluate import risk_analysis
from qlib.data import D
import yaml


def format_time(seconds: float) -> str:
    """格式化时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}小时{minutes}分钟{secs}秒"
    elif minutes > 0:
        return f"{minutes}分钟{secs}秒"
    else:
        return f"{secs}秒"


def main():
    """运行单股反转网格策略"""
    start_time = time.time()
    
    try:
        # 初始化 qlib
        print("=" * 80)
        print("初始化 Qlib...")
        qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn')
        print("✓ Qlib 初始化成功")
        
        # 加载配置文件
        config_path = Path(__file__).parent / 'workflow_config_reversal_grid.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        print(f"\n加载配置文件: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 获取股票代码（可以从命令行参数获取，或使用配置文件中的）
        stock_id = None
        if len(sys.argv) > 1:
            stock_id = sys.argv[1]
            print(f"\n从命令行参数获取股票代码: {stock_id}")
        else:
            stock_id = config.get('port_analysis_config', {}).get('strategy', {}).get('kwargs', {}).get('stock_id', 'SH600000')
            print(f"\n使用配置文件中的股票代码: {stock_id}")
        
        # 更新配置中的股票代码
        if stock_id:
            config['port_analysis_config']['strategy']['kwargs']['stock_id'] = stock_id
        
        print("=" * 80)
        print(f"单股反转网格策略回测")
        print(f"股票代码: {stock_id}")
        print("=" * 80)
        
        # 运行工作流
        experiment_name = 'reversal_grid_strategy'
        print(f"\n开始运行实验: {experiment_name}")
        
        with R.start(experiment_name=experiment_name):
            recorder = R.get_recorder()
            
            # 初始化策略（在外部作用域，以便后续分析使用）
            port_analysis_config = config.get('port_analysis_config', {})
            strategy = None
            
            # 运行回测
            print("\n[1/1] 运行回测...")
            backtest_start = time.time()
            try:
                # 初始化策略和执行器
                strategy = init_instance_by_config(port_analysis_config.get('strategy', {}))
                executor_config = port_analysis_config.get('executor', {
                    "class": "SimulatorExecutor",
                    "module_path": "qlib.backtest.executor",
                    "kwargs": {
                        "time_per_step": "day",
                        "generate_portfolio_metrics": True,
                        "verbose": False,  # 禁用进度条，方便日志重定向
                    },
                })
                executor = init_instance_by_config(executor_config)
                
                # 获取回测配置
                backtest_config = port_analysis_config.get('backtest', {})
                
                # 直接运行回测（不依赖SignalRecord）
                portfolio_metric_dict, indicator_dict = normal_backtest(
                    executor=executor,
                    strategy=strategy,
                    **backtest_config
                )
                
                # 保存回测结果
                artifact_objects = {}
                for _freq, (report_normal, positions_normal) in portfolio_metric_dict.items():
                    artifact_objects.update({f"report_normal_{_freq}.pkl": report_normal})
                    artifact_objects.update({f"positions_normal_{_freq}.pkl": positions_normal})
                
                for _freq, indicators_normal in indicator_dict.items():
                    artifact_objects.update({f"indicators_normal_{_freq}.pkl": indicators_normal[0]})
                    artifact_objects.update({f"indicators_normal_{_freq}_obj.pkl": indicators_normal[1]})
                
                # 进行风险分析
                analysis_objects = {}
                for _freq, (report_normal, _) in portfolio_metric_dict.items():
                    analysis = dict()
                    analysis["excess_return_without_cost"] = risk_analysis(
                        report_normal["return"] - report_normal["bench"], freq=_freq
                    )
                    analysis["excess_return_with_cost"] = risk_analysis(
                        report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=_freq
                    )
                    analysis_df = pd.concat(analysis)
                    analysis_objects[f"port_analysis_{_freq}.pkl"] = analysis_df
                
                # 保存所有结果
                recorder.save_objects(**artifact_objects, **analysis_objects)
                
                backtest_time = time.time() - backtest_start
                print(f"✓ 回测完成 (耗时: {format_time(backtest_time)})")
            except Exception as e:
                print(f"❌ 回测失败: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            # 分析回测结果
            print("\n[2/2] 分析回测结果...")
            try:
                # 尝试加载回测结果（可能有不同的频率）
                report_normal = None
                positions_normal = None
                port_analysis = None
                
                possible_paths = [
                    ("report_normal_1day.pkl", "positions_normal_1day.pkl", "port_analysis_1day.pkl"),
                    ("report_normal_day.pkl", "positions_normal_day.pkl", "port_analysis_day.pkl"),
                ]
                
                for report_path, pos_path, analysis_path in possible_paths:
                    try:
                        report_normal = recorder.load_object(report_path)
                        positions_normal = recorder.load_object(pos_path)
                        port_analysis = recorder.load_object(analysis_path)
                        if report_normal is not None:
                            break
                    except Exception:
                        continue
                
                if report_normal is None:
                    print("⚠️  无法加载回测结果")
                    return
                
                # 打印基本统计信息
                if 'return' in report_normal.columns:
                    returns = report_normal['return']
                    cost = report_normal.get('cost', pd.Series(0, index=returns.index))
                    net_returns = returns - cost
                    cum_returns = (net_returns + 1).cumprod()
                    
                    total_return = (net_returns + 1).prod() - 1
                    annual_return = (net_returns + 1).prod() ** (252 / len(net_returns)) - 1 if len(net_returns) > 0 else 0
                    sharpe_ratio = net_returns.mean() / net_returns.std() * np.sqrt(252) if net_returns.std() > 0 else 0
                    max_drawdown = (net_returns.cumsum() - net_returns.cumsum().expanding().max()).min()
                    
                    # 计算波动率
                    volatility = net_returns.std() * np.sqrt(252) if net_returns.std() > 0 else 0
                    
                    # 计算卡玛比率（年化收益率 / 最大回撤）
                    calmar_ratio = abs(annual_return / max_drawdown) if max_drawdown != 0 else 0
                    
                    # 计算买入并持有收益（Buy and Hold）
                    start_date = report_normal.index[0]
                    end_date = report_normal.index[-1]
                    start_price = None
                    end_price = None
                    hold_return = None
                    
                    # 格式化日期为字符串（如果需要）
                    if isinstance(start_date, pd.Timestamp):
                        start_date_str = start_date.strftime('%Y-%m-%d')
                    else:
                        start_date_str = str(start_date)
                        start_date = pd.Timestamp(start_date_str)
                    
                    if isinstance(end_date, pd.Timestamp):
                        end_date_str = end_date.strftime('%Y-%m-%d')
                    else:
                        end_date_str = str(end_date)
                        end_date = pd.Timestamp(end_date_str)
                    
                    try:
                        # 获取回测区间的开始和结束价格
                        # 扩展日期范围以确保能获取到数据
                        price_start = start_date - pd.Timedelta(days=5)  # 提前5天以确保能获取到开始日期的价格
                        price_end = end_date + pd.Timedelta(days=5)  # 延后5天以确保能获取到结束日期的价格
                        
                        price_data = D.features(
                            [stock_id],
                            ["$close"],
                            start_time=price_start,
                            end_time=price_end,
                            freq="day",
                            disk_cache=True,
                        )
                        
                        if price_data is not None and len(price_data) > 0:
                            # 处理 MultiIndex 的情况
                            if isinstance(price_data.index, pd.MultiIndex):
                                if stock_id in price_data.index.get_level_values(0):
                                    close_series = price_data.loc[stock_id, "$close"]
                                else:
                                    close_series = price_data.iloc[:, 0]
                            else:
                                close_series = price_data.iloc[:, 0] if len(price_data.columns) > 0 else price_data.squeeze()
                            
                            # 去除 NaN 值并按日期排序
                            close_series = close_series.dropna().sort_index()
                            
                            if len(close_series) > 0:
                                # 找到最接近开始日期和结束日期的价格
                                # 使用第一个和最后一个有效价格
                                start_price = close_series.iloc[0]
                                end_price = close_series.iloc[-1]
                                
                                # 或者尝试找到精确日期的价格
                                try:
                                    if start_date in close_series.index:
                                        start_price = close_series.loc[start_date]
                                    elif len(close_series[close_series.index >= start_date]) > 0:
                                        start_price = close_series[close_series.index >= start_date].iloc[0]
                                except Exception:
                                    pass
                                
                                try:
                                    if end_date in close_series.index:
                                        end_price = close_series.loc[end_date]
                                    elif len(close_series[close_series.index <= end_date]) > 0:
                                        end_price = close_series[close_series.index <= end_date].iloc[-1]
                                except Exception:
                                    pass
                                
                                # 计算持有收益
                                if start_price is not None and end_price is not None and start_price > 0:
                                    hold_return = (end_price - start_price) / start_price
                    except Exception as e:
                        # 如果获取价格失败，不影响其他分析
                        pass
                    
                    print(f"\n回测结果统计:")
                    print(f"  - 回测区间: {start_date_str} 至 {end_date_str}")
                    if start_price is not None:
                        print(f"  - 开始价格: {start_price:.4f}")
                    if end_price is not None:
                        print(f"  - 结束价格: {end_price:.4f}")
                    if hold_return is not None:
                        print(f"  - 买入并持有收益: {hold_return:.2%}")
                        # 计算策略收益与持有收益的对比
                        excess_return = total_return - hold_return
                        print(f"  - 策略超额收益: {excess_return:.2%} ({'+' if excess_return >= 0 else ''}{excess_return:.2%})")
                    print(f"  - 总收益率: {total_return:.2%}")
                    print(f"  - 年化收益率: {annual_return:.2%}")
                    print(f"  - 年化波动率: {volatility:.2%}")
                    print(f"  - 夏普比率: {sharpe_ratio:.4f}")
                    print(f"  - 卡玛比率: {calmar_ratio:.4f}")
                    print(f"  - 最大回撤: {max_drawdown:.2%}")
                    print(f"  - 交易天数: {len(net_returns)}")
                    
                    # 计算回撤分析
                    drawdown_series = cum_returns / cum_returns.expanding().max() - 1
                    max_dd_duration = 0
                    current_dd_duration = 0
                    for dd in drawdown_series:
                        if dd < 0:
                            current_dd_duration += 1
                            max_dd_duration = max(max_dd_duration, current_dd_duration)
                        else:
                            current_dd_duration = 0
                    
                    # 计算最大回撤的时间区间
                    # 找到最大回撤的最低点
                    max_dd_idx = drawdown_series.idxmin()
                    max_dd_value = drawdown_series.min()
                    
                    # 找到最大回撤开始的时间（在最低点之前的最高点）
                    # 在最低点之前，找到累积收益的最高点
                    cum_returns_before_dd = cum_returns.loc[:max_dd_idx]
                    if len(cum_returns_before_dd) > 0:
                        peak_idx = cum_returns_before_dd.idxmax()
                        peak_value = cum_returns_before_dd.max()
                        dd_start_date = peak_idx
                        dd_end_date = max_dd_idx
                        
                        # 计算持续时间
                        if isinstance(dd_start_date, pd.Timestamp) and isinstance(dd_end_date, pd.Timestamp):
                            dd_duration_days = (dd_end_date - dd_start_date).days
                        else:
                            # 如果不是时间戳，计算索引差值
                            start_pos = cum_returns.index.get_loc(dd_start_date)
                            end_pos = cum_returns.index.get_loc(dd_end_date)
                            dd_duration_days = end_pos - start_pos
                        
                        # 格式化日期输出
                        if isinstance(dd_start_date, pd.Timestamp):
                            start_date_str = dd_start_date.strftime('%Y-%m-%d')
                        else:
                            start_date_str = str(dd_start_date)
                        
                        if isinstance(dd_end_date, pd.Timestamp):
                            end_date_str = dd_end_date.strftime('%Y-%m-%d')
                        else:
                            end_date_str = str(dd_end_date)
                        
                        print(f"  - 最大回撤持续时间: {max_dd_duration} 天")
                        print(f"  - 最大回撤时间区间:")
                        print(f"     开始时间: {start_date_str} (峰值)")
                        print(f"     结束时间: {end_date_str} (最低点)")
                        print(f"     持续时间: {dd_duration_days} 天")
                        print(f"     峰值累积收益: {peak_value:.4f}")
                        print(f"     最低点累积收益: {cum_returns.loc[dd_end_date]:.4f}")
                        print(f"     回撤幅度: {max_dd_value:.2%}")
                    else:
                        print(f"  - 最大回撤持续时间: {max_dd_duration} 天")
                    
                    # 计算盈利/亏损天数
                    profit_days = (net_returns > 0).sum()
                    loss_days = (net_returns < 0).sum()
                    flat_days = (net_returns == 0).sum()
                    win_rate = profit_days / len(net_returns) if len(net_returns) > 0 else 0
                    
                    print(f"\n交易日统计:")
                    print(f"  - 盈利天数: {profit_days} ({win_rate:.2%})")
                    print(f"  - 亏损天数: {loss_days} ({(loss_days/len(net_returns)):.2%})" if len(net_returns) > 0 else "")
                    print(f"  - 持平天数: {flat_days}")
                    
                    # 计算平均盈亏
                    avg_profit = net_returns[net_returns > 0].mean() if (net_returns > 0).any() else 0
                    avg_loss = net_returns[net_returns < 0].mean() if (net_returns < 0).any() else 0
                    profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
                    
                    print(f"\n盈亏分析:")
                    print(f"  - 平均盈利: {avg_profit:.4%}")
                    print(f"  - 平均亏损: {avg_loss:.4%}")
                    print(f"  - 盈亏比: {profit_loss_ratio:.2f}")
                    
                    # 计算最大单日盈亏
                    max_daily_profit = net_returns.max()
                    max_daily_loss = net_returns.min()
                    print(f"  - 最大单日盈利: {max_daily_profit:.4%}")
                    print(f"  - 最大单日亏损: {max_daily_loss:.4%}")
                    
                    # 计算连续盈利/亏损
                    def calc_consecutive(series, condition):
                        max_consecutive = 0
                        current_consecutive = 0
                        for val in series:
                            if condition(val):
                                current_consecutive += 1
                                max_consecutive = max(max_consecutive, current_consecutive)
                            else:
                                current_consecutive = 0
                        return max_consecutive
                    
                    max_consecutive_profit = calc_consecutive(net_returns, lambda x: x > 0)
                    max_consecutive_loss = calc_consecutive(net_returns, lambda x: x < 0)
                    
                    print(f"  - 最大连续盈利天数: {max_consecutive_profit}")
                    print(f"  - 最大连续亏损天数: {max_consecutive_loss}")
                
                # 打印持仓信息
                if positions_normal is not None:
                    print(f"\n持仓信息:")
                    if hasattr(positions_normal, '__len__'):
                        print(f"  - 持仓记录数: {len(positions_normal)}")
                        
                        # 分析持仓变化
                        if isinstance(positions_normal, pd.DataFrame):
                            # 计算持仓价值
                            if 'amount' in positions_normal.columns and 'price' in positions_normal.columns:
                                position_values = positions_normal['amount'] * positions_normal['price']
                                avg_position_value = position_values.mean()
                                max_position_value = position_values.max()
                                min_position_value = position_values.min()
                                
                                print(f"  - 平均持仓价值: {avg_position_value:,.2f}")
                                print(f"  - 最大持仓价值: {max_position_value:,.2f}")
                                print(f"  - 最小持仓价值: {min_position_value:,.2f}")
                            
                            # 计算持仓比例
                            if 'amount' in positions_normal.columns:
                                non_zero_positions = (positions_normal['amount'] != 0).sum()
                                print(f"  - 有持仓天数: {non_zero_positions} ({non_zero_positions/len(positions_normal):.2%})")
                    else:
                        print(f"  - 持仓记录数: N/A")
                
                # 计算月度/季度收益
                if 'return' in report_normal.columns:
                    report_normal['date'] = report_normal.index
                    report_normal['month'] = pd.to_datetime(report_normal['date']).dt.to_period('M')
                    report_normal['quarter'] = pd.to_datetime(report_normal['date']).dt.to_period('Q')
                    
                    monthly_returns = report_normal.groupby('month')['return'].apply(lambda x: (x + 1).prod() - 1)
                    quarterly_returns = report_normal.groupby('quarter')['return'].apply(lambda x: (x + 1).prod() - 1)
                    
                    print(f"\n月度收益分析:")
                    print(f"  - 平均月度收益: {monthly_returns.mean():.2%}")
                    print(f"  - 月度收益标准差: {monthly_returns.std():.2%}")
                    print(f"  - 最佳月度收益: {monthly_returns.max():.2%} ({monthly_returns.idxmax()})")
                    print(f"  - 最差月度收益: {monthly_returns.min():.2%} ({monthly_returns.idxmin()})")
                    print(f"  - 盈利月数: {(monthly_returns > 0).sum()} / {len(monthly_returns)}")
                    
                    print(f"\n季度收益分析:")
                    print(f"  - 平均季度收益: {quarterly_returns.mean():.2%}")
                    print(f"  - 季度收益标准差: {quarterly_returns.std():.2%}")
                    print(f"  - 最佳季度收益: {quarterly_returns.max():.2%} ({quarterly_returns.idxmax()})")
                    print(f"  - 最差季度收益: {quarterly_returns.min():.2%} ({quarterly_returns.idxmin()})")
                    print(f"  - 盈利季度数: {(quarterly_returns > 0).sum()} / {len(quarterly_returns)}")
                
                # 计算交易成本分析
                if 'cost' in report_normal.columns:
                    total_cost = report_normal['cost'].sum()
                    total_return_value = (net_returns + 1).prod() - 1
                    cost_ratio = abs(total_cost / total_return_value) if total_return_value != 0 else 0
                    
                    print(f"\n交易成本分析:")
                    print(f"  - 总交易成本: {total_cost:.4%}")
                    print(f"  - 平均每日成本: {report_normal['cost'].mean():.6%}")
                    print(f"  - 成本/收益比: {cost_ratio:.2%}")
                    print(f"  - 最大单日成本: {report_normal['cost'].max():.4%}")
                
                # 收益分布分析
                if 'return' in report_normal.columns:
                    print(f"\n收益分布分析:")
                    print(f"  - 收益中位数: {net_returns.median():.4%}")
                    print(f"  - 收益25%分位数: {net_returns.quantile(0.25):.4%}")
                    print(f"  - 收益75%分位数: {net_returns.quantile(0.75):.4%}")
                    print(f"  - 收益偏度: {net_returns.skew():.4f}")
                    print(f"  - 收益峰度: {net_returns.kurtosis():.4f}")
                    
                    # 收益区间分布
                    bins = [-np.inf, -0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05, np.inf]
                    labels = ['<-5%', '-5%~-2%', '-2%~-1%', '-1%~0%', '0%~1%', '1%~2%', '2%~5%', '>5%']
                    return_dist = pd.cut(net_returns, bins=bins, labels=labels)
                    dist_counts = return_dist.value_counts().sort_index()
                    
                    print(f"\n收益区间分布:")
                    for label, count in dist_counts.items():
                        pct = count / len(net_returns) * 100
                        print(f"  - {label}: {count} 天 ({pct:.1f}%)")
                
                # 计算滚动统计
                if 'return' in report_normal.columns and len(net_returns) >= 60:
                    rolling_30d = net_returns.rolling(30).apply(lambda x: (x + 1).prod() - 1)
                    rolling_60d = net_returns.rolling(60).apply(lambda x: (x + 1).prod() - 1)
                    
                    print(f"\n滚动收益分析 (30天窗口):")
                    print(f"  - 平均30天收益: {rolling_30d.mean():.2%}")
                    print(f"  - 最佳30天收益: {rolling_30d.max():.2%}")
                    print(f"  - 最差30天收益: {rolling_30d.min():.2%}")
                    
                    if len(net_returns) >= 60:
                        print(f"\n滚动收益分析 (60天窗口):")
                        print(f"  - 平均60天收益: {rolling_60d.mean():.2%}")
                        print(f"  - 最佳60天收益: {rolling_60d.max():.2%}")
                        print(f"  - 最差60天收益: {rolling_60d.min():.2%}")
                
                # 打印风险分析结果
                if port_analysis is not None:
                    print(f"\n风险分析结果:")
                    print(port_analysis)
                
                # 做多/做空单独分析
                try:
                    if strategy and hasattr(strategy, 'position_history') and strategy.position_history:
                        position_df = pd.DataFrame(strategy.position_history)
                        position_df['timestamp'] = pd.to_datetime(position_df['timestamp'])
                        position_df = position_df.set_index('timestamp')
                        
                        # 合并收益数据
                        if 'return' in report_normal.columns:
                            position_df = position_df.join(report_normal[['return', 'cost']], how='left')
                            position_df['net_return'] = position_df['return'] - position_df.get('cost', 0)
                            
                            # 做多分析：使用与回测结果统计相同的数据集（report_normal）
                            # 从 position_df 获取做多天数的时间戳，然后从 report_normal 中筛选这些天数的收益
                            long_timestamps = position_df[position_df['direction'] == 1].index
                            if len(long_timestamps) > 0:
                                # 从 report_normal 中筛选出做多天数的收益
                                long_net_returns = net_returns[net_returns.index.isin(long_timestamps)]
                                if len(long_net_returns) > 0:
                                    long_total_return = (long_net_returns + 1).prod() - 1
                                else:
                                    # 如果无法匹配，使用 position_df 的数据（向后兼容）
                                    long_data = position_df[position_df['direction'] == 1]
                                    long_returns = long_data['net_return']
                                    long_total_return = (long_returns + 1).prod() - 1
                                    long_net_returns = long_returns
                            else:
                                long_net_returns = pd.Series(dtype=float)
                                long_total_return = 0
                            
                            # 计算无仓位天数的收益，用于验证
                            no_position_timestamps = position_df[position_df['direction'] == 0].index
                            # 从 report_normal 中获取无仓位天数的收益
                            no_position_net_returns = net_returns[net_returns.index.isin(no_position_timestamps)]
                            
                            # 调试：检查无仓位天数是否真的无仓位（可能是 position_history 记录的问题）
                            # 如果 position_history 中 direction=0，但实际有仓位，说明记录有问题
                            
                            # 验证：如果关闭了做空，总收益应该等于做多总收益（因为无仓位天数的收益应该为0）
                            if len(no_position_net_returns) > 0:
                                no_position_total_return = (no_position_net_returns + 1).prod() - 1
                                # 验证：总收益 ≈ 做多总收益 * (1 + 无仓位总收益)
                                # 如果无仓位收益为0，则 总收益 ≈ 做多总收益
                                expected_total_return_from_long = long_total_return if len(no_position_net_returns) == 0 else ((long_net_returns + 1).prod() * (no_position_net_returns + 1).prod() - 1) if len(long_net_returns) > 0 else 0
                            else:
                                no_position_total_return = 0
                                expected_total_return_from_long = long_total_return
                            
                            if len(long_net_returns) > 0:
                                long_days = len(long_net_returns)
                                long_win_rate = (long_net_returns > 0).sum() / len(long_net_returns) if len(long_net_returns) > 0 else 0
                                long_avg_return = long_net_returns.mean()
                                long_sharpe = long_net_returns.mean() / long_net_returns.std() * np.sqrt(252) if long_net_returns.std() > 0 else 0
                                
                                print(f"\n做多分析:")
                                print(f"  - 做多天数: {long_days} ({long_days/len(net_returns):.2%})")
                                print(f"  - 做多总收益: {long_total_return:.2%} (仅做多天数的复利收益)")
                                if len(no_position_net_returns) > 0:
                                    no_position_days = len(no_position_net_returns)
                                    print(f"  - 无仓位天数: {no_position_days} ({no_position_days/len(net_returns):.2%})")
                                    print(f"  - 无仓位总收益: {no_position_total_return:.2%} (无仓位天数的复利收益)")
                                    print(f"  - 无仓位平均日收益: {no_position_net_returns.mean():.6%}")
                                    print(f"  - 无仓位最大单日收益: {no_position_net_returns.max():.4%}")
                                    print(f"  - 无仓位最大单日亏损: {no_position_net_returns.min():.4%}")
                                    print(f"  - 无仓位收益非零天数: {(no_position_net_returns != 0).sum()} / {no_position_days}")
                                    # 验证：按时间顺序重新组合收益，验证是否等于总收益率
                                    # 关键：总收益率是所有528天按时间顺序的复利
                                    # 做多总收益只是206个做多天的复利（不按时间顺序）
                                    # 两者不能直接比较，因为天数不同且顺序不同
                                    
                                    # 验证：根据 position_df 的时间戳，按时间顺序重建收益序列
                                    if len(position_df) > 0:
                                        # 从 net_returns 中获取 position_df 对应时间戳的收益
                                        # 注意：position_df 的时间戳可能与 net_returns 不完全匹配
                                        position_returns_aligned = net_returns[net_returns.index.isin(position_df.index)]
                                        
                                        if len(position_returns_aligned) > 0:
                                            # 按时间顺序计算复利
                                            reconstructed_return = (position_returns_aligned + 1).prod() - 1
                                            
                                            print(f"  - 验证: 基于position_history的时间戳按顺序计算的收益 = {reconstructed_return:.2%}")
                                            print(f"          实际总收益率 = {total_return:.2%}")
                                            print(f"          差异 = {abs(total_return - reconstructed_return):.4%}")
                                            
                                            # 检查覆盖天数
                                            coverage = len(position_returns_aligned) / len(net_returns)
                                            print(f"  - position_history覆盖天数: {len(position_returns_aligned)}/{len(net_returns)} ({coverage:.2%})")
                                            
                                            if coverage < 0.99:
                                                print(f"  - ⚠️  position_history可能缺少{len(net_returns) - len(position_returns_aligned)}天的记录")
                                        
                                    # 说明
                                    print(f"  - 说明: 总收益率({total_return:.2%}) = 所有{len(net_returns)}天按时间顺序的复利")
                                    print(f"          做多总收益({long_total_return:.2%}) = 仅{long_days}个做多天的复利（天数少、不按时间顺序）")
                                    print(f"          做多总收益更高是因为：1) 只计算了收益最好的做多天数 2) 忽略了无仓位和做空损失")
                                print(f"  - 做多年化收益: {(long_total_return + 1) ** (252/long_days) - 1:.2%}" if long_days > 0 else "  - 做多年化收益: N/A")
                                print(f"  - 做多胜率: {long_win_rate:.2%}")
                                print(f"  - 做多平均日收益: {long_avg_return:.4%}")
                                print(f"  - 做多夏普比率: {long_sharpe:.4f}")
                                print(f"  - 做多最大单日收益: {long_net_returns.max():.4%}")
                                print(f"  - 做多最大单日亏损: {long_net_returns.min():.4%}")
                                
                                # 做多加仓统计（从 position_df 获取）
                                long_data = position_df[position_df['direction'] == 1]
                                if len(long_data) > 0:
                                    long_grid_levels = long_data['grid_level']
                                    if len(long_grid_levels) > 0:
                                        print(f"  - 做多平均加仓等级: {long_grid_levels.mean():.2f}")
                                        print(f"  - 做多最大加仓等级: {long_grid_levels.max()}")
                            
                            # 做空分析
                            short_data = position_df[position_df['direction'] == -1]
                            if len(short_data) > 0:
                                short_returns = short_data['net_return']
                                short_total_return = (short_returns + 1).prod() - 1
                                short_days = len(short_returns)
                                short_win_rate = (short_returns > 0).sum() / len(short_returns) if len(short_returns) > 0 else 0
                                short_avg_return = short_returns.mean()
                                short_sharpe = short_returns.mean() / short_returns.std() * np.sqrt(252) if short_returns.std() > 0 else 0
                                
                                print(f"\n做空分析:")
                                print(f"  - 做空天数: {short_days} ({short_days/len(position_df):.2%})")
                                print(f"  - 做空总收益: {short_total_return:.2%}")
                                print(f"  - 做空年化收益: {(short_total_return + 1) ** (252/short_days) - 1:.2%}" if short_days > 0 else "  - 做空年化收益: N/A")
                                print(f"  - 做空胜率: {short_win_rate:.2%}")
                                print(f"  - 做空平均日收益: {short_avg_return:.4%}")
                                print(f"  - 做空夏普比率: {short_sharpe:.4f}")
                                print(f"  - 做空最大单日收益: {short_returns.max():.4%}")
                                print(f"  - 做空最大单日亏损: {short_returns.min():.4%}")
                                
                                # 做空加仓统计
                                short_grid_levels = short_data['grid_level']
                                if len(short_grid_levels) > 0:
                                    print(f"  - 做空平均加仓等级: {short_grid_levels.mean():.2f}")
                                    print(f"  - 做空最大加仓等级: {short_grid_levels.max()}")
                            
                            # 无仓位分析
                            no_position_data = position_df[position_df['direction'] == 0]
                            if len(no_position_data) > 0:
                                no_position_days = len(no_position_data)
                                print(f"\n无仓位分析:")
                                print(f"  - 无仓位天数: {no_position_days} ({no_position_days/len(position_df):.2%})")
                            
                            # 对比分析
                            if len(long_data) > 0 and len(short_data) > 0:
                                print(f"\n做多 vs 做空对比:")
                                print(f"  - 天数比: 做多 {long_days} 天 vs 做空 {short_days} 天")
                                print(f"  - 总收益比: 做多 {long_total_return:.2%} vs 做空 {short_total_return:.2%}")
                                print(f"  - 胜率比: 做多 {long_win_rate:.2%} vs 做空 {short_win_rate:.2%}")
                                print(f"  - 平均日收益比: 做多 {long_avg_return:.4%} vs 做空 {short_avg_return:.4%}")
                                print(f"  - 夏普比率比: 做多 {long_sharpe:.4f} vs 做空 {short_sharpe:.4f}")
                except Exception as e:
                    # 如果无法获取策略对象或分析失败，不影响其他分析
                    pass
                
            except Exception as e:
                print(f"⚠️  分析回测结果时出错: {e}")
                import traceback
                traceback.print_exc()
        
        total_time = time.time() - start_time
        print(f"\n" + "=" * 80)
        print(f"策略运行完成 (总耗时: {format_time(total_time)})")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
