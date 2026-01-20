"""
运行反转网格股票池策略的脚本

支持多只股票同时运行反转网格策略，每只股票有独立的参数配置。

使用方法：
    python myproject/run_reversal_grid_pool_strategy.py myproject/workflow_config_reversal_grid_pool.yaml
    
Author: AI Assistant
Date: 2026-01-19
"""

import sys
import yaml
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

# 配置日志（在导入qlib之前）
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
# 设置qlib相关日志级别
logging.getLogger('qlib.BaseExecutor').setLevel(logging.ERROR)
logging.getLogger('qlib.Executor').setLevel(logging.ERROR)
logging.getLogger('qlib.timer').setLevel(logging.ERROR)

import qlib
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.backtest import backtest as normal_backtest


def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"


def analyze_results(report_normal_df: pd.DataFrame, positions_df, 
                    account: float, stock_pool: Dict, benchmark_id: str = None):
    """
    分析回测结果
    
    Parameters
    ----------
    report_normal_df : pd.DataFrame
        回测报告数据
    positions_df : pd.DataFrame or dict
        持仓数据（可能是字典或DataFrame）
    account : float
        初始资金
    stock_pool : Dict
        股票池配置
    benchmark_id : str
        基准指数代码
    """
    # 如果 positions_df 是字典，转换为 DataFrame
    if isinstance(positions_df, dict):
        try:
            from qlib.contrib.report.analysis_position.parse_position import parse_position
            positions_df = parse_position(positions_df)
        except Exception as e:
            print(f"警告: 无法解析持仓数据: {e}")
            positions_df = pd.DataFrame()  # 使用空DataFrame
    
    print("\n" + "="*80)
    print("回测结果统计")
    print("="*80)
    
    # 基础统计
    total_days = len(report_normal_df)
    
    # 计算收益
    net_returns = report_normal_df['return'].dropna()
    total_return = (1 + net_returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(net_returns)) - 1 if len(net_returns) > 0 else 0
    
    # 计算夏普比率
    daily_mean = net_returns.mean()
    daily_std = net_returns.std()
    sharpe_ratio = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0
    
    # 计算最大回撤
    cumulative_returns = (1 + net_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # 找到最大回撤的时间区间
    max_dd_end_idx = drawdown.idxmin()
    max_dd_end_date = max_dd_end_idx
    # 找到回撤开始点（之前的最高点）
    max_dd_start_idx = cumulative_returns[:max_dd_end_idx].idxmax()
    max_dd_start_date = max_dd_start_idx
    
    print(f"- 总收益率: {total_return:.2%}")
    print(f"- 年化收益率: {annualized_return:.2%}")
    print(f"- 夏普比率: {sharpe_ratio:.4f}")
    print(f"- 最大回撤: {max_drawdown:.2%}")
    print(f"- 最大回撤区间: {max_dd_start_date} 至 {max_dd_end_date}")
    print(f"- 交易天数: {total_days}")
    
    # Buy and Hold 分析
    if benchmark_id:
        try:
            from qlib.data import D
            backtest_start = report_normal_df.index[0]
            backtest_end = report_normal_df.index[-1]
            
            benchmark_data = D.features(
                [benchmark_id],
                ['$close'],
                start_time=backtest_start,
                end_time=backtest_end
            )
            
            if not benchmark_data.empty:
                start_price = benchmark_data.iloc[0]['$close']
                end_price = benchmark_data.iloc[-1]['$close']
                hold_return = (end_price - start_price) / start_price
                excess_return = total_return - hold_return
                
                print(f"\n基准指数 ({benchmark_id}) 分析:")
                print(f"- 起始价格: {start_price:.2f}")
                print(f"- 结束价格: {end_price:.2f}")
                print(f"- Buy & Hold 收益: {hold_return:.2%}")
                print(f"- 策略超额收益: {excess_return:.2%}")
        except Exception as e:
            print(f"\n基准指数分析失败: {str(e)}")
    
    # 持仓分析
    print(f"\n持仓信息:")
    print(f"- 股票池大小: {len(stock_pool)}")
    
    # 统计每只股票的交易情况
    if isinstance(positions_df, pd.DataFrame) and not positions_df.empty:
        stock_trading_stats = {}
        
        # 检查是否是 MultiIndex (instrument, datetime)
        if isinstance(positions_df.index, pd.MultiIndex):
            # 使用索引的第一层（instrument）来筛选
            for stock_id in stock_pool.keys():
                if stock_id in positions_df.index.get_level_values(0):
                    stock_positions = positions_df.loc[stock_id]
                    if len(stock_positions) > 0:
                        # 如果 stock_positions 是 Series，转换为 DataFrame
                        if isinstance(stock_positions, pd.Series):
                            stock_positions = stock_positions.to_frame().T
                        trading_days = len(stock_positions[stock_positions['amount'] > 0]) if 'amount' in stock_positions.columns else 0
                        max_position = stock_positions['amount'].max() if 'amount' in stock_positions.columns and len(stock_positions) > 0 else 0
                        stock_trading_stats[stock_id] = {
                            'trading_days': trading_days,
                            'max_position': max_position,
                            'traded': trading_days > 0
                        }
        elif 'stock_id' in positions_df.columns:
            # 如果使用 stock_id 列
            for stock_id in stock_pool.keys():
                stock_positions = positions_df[positions_df['stock_id'] == stock_id]
                if len(stock_positions) > 0:
                    trading_days = len(stock_positions[stock_positions['amount'] > 0])
                    max_position = stock_positions['amount'].max() if len(stock_positions) > 0 else 0
                    stock_trading_stats[stock_id] = {
                        'trading_days': trading_days,
                        'max_position': max_position,
                        'traded': trading_days > 0
                    }
        
        traded_stocks = [s for s, stats in stock_trading_stats.items() if stats['traded']]
        print(f"- 实际交易股票数: {len(traded_stocks)}")
        
        if traded_stocks:
            print(f"\n交易明细（按交易天数排序）:")
            sorted_stocks = sorted(
                [(s, stats) for s, stats in stock_trading_stats.items() if stats['traded']],
                key=lambda x: x[1]['trading_days'],
                reverse=True
            )
            for stock_id, stats in sorted_stocks[:10]:  # 只显示前10只
                print(f"  {stock_id}: 交易天数={stats['trading_days']}, "
                      f"最大持仓={stats['max_position']:.0f}股, "
                      f"最高持仓金额={stock_pool[stock_id].get('max_position_value', 0)/10000:.1f}万")
    
    # 统计持仓天数分布和总仓位
    print(f"\n整体持仓统计:")
    if isinstance(positions_df, pd.DataFrame) and not positions_df.empty and 'amount' in positions_df.columns:
        total_positions = positions_df[positions_df['amount'] > 0]
        if len(total_positions) > 0:
            # 按日期统计持仓股票数
            if isinstance(total_positions.index, pd.MultiIndex):
                # MultiIndex: (instrument, datetime)，按 datetime (level=1) 分组
                daily_stock_count = total_positions.groupby(level=1).size()
            else:
                # 单层索引，按日期分组
                daily_stock_count = total_positions.groupby(total_positions.index).size()
            
            print(f"- 平均每日持仓股票数: {daily_stock_count.mean():.1f}")
            print(f"- 最多同时持仓: {daily_stock_count.max()}只")
            print(f"- 最少同时持仓: {daily_stock_count.min()}只")
            print(f"- 空仓天数: {(daily_stock_count == 0).sum()}")
    
    # 统计总仓位（资金利用率）
    print(f"\n资金利用率统计:")
    try:
        # 计算每日总持仓市值
        daily_position_value = {}
        daily_account_value = {}
        
        # 获取账户总资产（从report_normal_df）
        # 注意：account列是账户总资产（现金+股票），value列只是股票价值
        # 我们应该使用account列来计算资金利用率
        if 'account' in report_normal_df.columns:
            account_value_col = 'account'  # 账户总资产（现金+股票）
            account_value_series = None
        elif 'account_value' in report_normal_df.columns:
            account_value_col = 'account_value'
            account_value_series = None
        elif 'value' in report_normal_df.columns and 'cash' in report_normal_df.columns:
            # 如果只有value和cash列，需要相加得到总资产
            account_value_col = None
            account_value_series = report_normal_df['value'] + report_normal_df['cash']
        elif 'value' in report_normal_df.columns:
            # 如果只有value列，说明可能value就是总资产（需要验证）
            account_value_col = 'value'
            account_value_series = None
        else:
            # 如果没有相关列，使用初始资金和累计收益计算
            account_value_col = None
            cumulative_returns = (1 + report_normal_df['return']).cumprod()
            account_value_series = account * cumulative_returns
        
        # 检查positions_df的结构
        if isinstance(positions_df, pd.DataFrame) and not positions_df.empty:
            # 检查是否有必要的列
            has_amount = 'amount' in positions_df.columns
            has_price = 'price' in positions_df.columns
            has_close = 'close' in positions_df.columns
            has_value = 'value' in positions_df.columns
        
        # 初始化所有日期的持仓市值为0
        for date in report_normal_df.index:
            daily_position_value[date] = 0
        
        # 计算每日持仓市值
        if isinstance(positions_df, pd.DataFrame) and not positions_df.empty:
            if isinstance(positions_df.index, pd.MultiIndex):
                # MultiIndex: (instrument, datetime)
                for date in positions_df.index.get_level_values(1).unique():
                    if date not in report_normal_df.index:
                        continue
                    try:
                        date_positions = positions_df.xs(date, level=1)
                        if isinstance(date_positions, pd.Series):
                            date_positions = pd.DataFrame([date_positions]).T
                        date_positions = date_positions[date_positions['amount'] > 0]
                        
                        if len(date_positions) > 0:
                            # 计算持仓市值：amount * price
                            position_value = 0
                            if 'price' in date_positions.columns:
                                position_value = (date_positions['amount'] * date_positions['price']).sum()
                            elif 'close' in date_positions.columns:
                                position_value = (date_positions['amount'] * date_positions['close']).sum()
                            elif 'value' in date_positions.columns:
                                # 如果有value列，直接使用
                                position_value = date_positions['value'].sum()
                            else:
                                # 如果没有价格列，尝试从数据获取
                                try:
                                    from qlib.data import D
                                    # 获取股票列表
                                    if isinstance(date_positions.index, pd.MultiIndex):
                                        stocks = date_positions.index.get_level_values(0).unique().tolist()
                                    else:
                                        stocks = date_positions.index.tolist()
                                    
                                    if stocks:
                                        prices = D.features(stocks, ['$close'], start_time=date, end_time=date)
                                        if not prices.empty:
                                            # 处理价格数据
                                            if isinstance(prices.index, pd.MultiIndex):
                                                prices_dict = prices.groupby(level=0)['$close'].first().to_dict()
                                            else:
                                                prices_dict = prices['$close'].to_dict()
                                            
                                            # 计算持仓市值
                                            for stock in stocks:
                                                try:
                                                    if isinstance(date_positions.index, pd.MultiIndex):
                                                        if stock in date_positions.index.get_level_values(0):
                                                            stock_pos = date_positions.loc[stock]
                                                            if isinstance(stock_pos, pd.Series):
                                                                stock_amount = stock_pos['amount']
                                                            else:
                                                                stock_amount = stock_pos['amount'].sum()
                                                            stock_price = prices_dict.get(stock, 0)
                                                            position_value += stock_amount * stock_price
                                                    else:
                                                        if stock in date_positions.index:
                                                            stock_amount = date_positions.loc[stock, 'amount']
                                                            stock_price = prices_dict.get(stock, 0)
                                                            position_value += stock_amount * stock_price
                                                except:
                                                    pass
                                except Exception as e:
                                    position_value = 0
                            
                            daily_position_value[date] = position_value
                    except Exception as e:
                        daily_position_value[date] = 0
            else:
                # 单层索引
                for date in positions_df.index.unique():
                    if date not in report_normal_df.index:
                        continue
                    try:
                        date_positions = positions_df.loc[date]
                        if isinstance(date_positions, pd.Series):
                            date_positions = pd.DataFrame([date_positions]).T
                        date_positions = date_positions[date_positions['amount'] > 0]
                        
                        if len(date_positions) > 0:
                            position_value = 0
                            if 'value' in date_positions.columns:
                                # 如果有value列（持仓价值），直接使用（这是最准确的）
                                position_value = date_positions['value'].sum()
                            elif 'price' in date_positions.columns:
                                # 使用持仓记录中的价格
                                position_value = (date_positions['amount'] * date_positions['price']).sum()
                            elif 'close' in date_positions.columns:
                                # 使用持仓记录中的收盘价
                                position_value = (date_positions['amount'] * date_positions['close']).sum()
                            else:
                                # 尝试从数据获取价格
                                try:
                                    from qlib.data import D
                                    stocks = date_positions.index.tolist()
                                    if stocks:
                                        prices = D.features(stocks, ['$close'], start_time=date, end_time=date)
                                        if not prices.empty:
                                            if isinstance(prices.index, pd.MultiIndex):
                                                prices_dict = prices.groupby(level=0)['$close'].first().to_dict()
                                            else:
                                                prices_dict = prices['$close'].to_dict()
                                            for stock in stocks:
                                                if stock in date_positions.index:
                                                    stock_amount = date_positions.loc[stock, 'amount']
                                                    stock_price = prices_dict.get(stock, 0)
                                                    position_value += stock_amount * stock_price
                                except:
                                    position_value = 0
                            daily_position_value[date] = position_value
                    except Exception as e:
                        daily_position_value[date] = 0
        
        # 获取每日账户总资产
        if account_value_col:
            for date in report_normal_df.index:
                daily_account_value[date] = report_normal_df.loc[date, account_value_col]
        elif account_value_series is not None:
            for date in report_normal_df.index:
                daily_account_value[date] = account_value_series.loc[date]
        else:
            # 使用初始资金和累计收益计算
            cumulative_returns = (1 + report_normal_df['return']).cumprod()
            account_value_series = account * cumulative_returns
            for date in report_normal_df.index:
                daily_account_value[date] = account_value_series.loc[date]
        
        # 计算总仓位比例
        position_ratios = []
        for date in report_normal_df.index:
            pos_value = daily_position_value.get(date, 0)
            acc_value = daily_account_value.get(date, account)
            if acc_value > 0:
                ratio = pos_value / acc_value
                position_ratios.append(ratio)
            else:
                position_ratios.append(0)
        
        if position_ratios:
            position_ratios_series = pd.Series(position_ratios, index=report_normal_df.index)
            avg_position_ratio = position_ratios_series.mean()
            max_position_ratio = position_ratios_series.max()
            min_position_ratio = position_ratios_series.min()
            median_position_ratio = position_ratios_series.median()
            
            print(f"- 平均总仓位比例: {avg_position_ratio:.2%}")
            print(f"- 最高总仓位比例: {max_position_ratio:.2%}")
            print(f"- 最低总仓位比例: {min_position_ratio:.2%}")
            print(f"- 中位数总仓位比例: {median_position_ratio:.2%}")
            
            # 如果仓位比例超过100%，给出警告
            if max_position_ratio > 1.05:
                print(f"\n⚠️  警告: 检测到仓位比例超过100%（最高{max_position_ratio:.2%}）")
                print(f"   可能原因:")
                print(f"   1. 使用了融资融券（策略代码中未实现）")
                print(f"   2. 持仓市值计算使用了错误的价格或日期")
                print(f"   3. 账户总资产计算不准确")
                print(f"   4. 数据时间不匹配")
            
            # 计算平均持仓市值
            position_values = [daily_position_value.get(date, 0) for date in report_normal_df.index]
            avg_position_value = np.mean(position_values)
            avg_account_value = np.mean([daily_account_value.get(date, account) for date in report_normal_df.index])
            
            # 计算平均现金
            avg_cash = avg_account_value - avg_position_value
            
            print(f"\n- 平均持仓市值: {avg_position_value/10000:.1f}万元")
            print(f"- 平均账户总资产: {avg_account_value/10000:.1f}万元")
            print(f"- 平均现金: {avg_cash/10000:.1f}万元")
            print(f"- 平均现金比例: {(1 - avg_position_ratio):.2%}")
            
            # 如果report_normal_df有cash列，也显示一下
            if 'cash' in report_normal_df.columns:
                avg_cash_from_report = report_normal_df['cash'].mean()
                print(f"- 平均现金（从报告）: {avg_cash_from_report/10000:.1f}万元")
                if abs(avg_cash - avg_cash_from_report) > avg_account_value * 0.01:  # 差异超过1%
                    print(f"  ⚠️  现金计算不一致，差异: {abs(avg_cash - avg_cash_from_report)/10000:.1f}万元")
            
            # 额外统计：仓位分布
            low_position_days = (position_ratios_series < 0.5).sum()
            medium_position_days = ((position_ratios_series >= 0.5) & (position_ratios_series < 0.8)).sum()
            high_position_days = (position_ratios_series >= 0.8).sum()
            total_days = len(position_ratios_series)
            
            print(f"\n仓位分布统计:")
            print(f"- 低仓位天数（<50%）: {low_position_days}天 ({low_position_days/total_days:.1%})")
            print(f"- 中仓位天数（50%-80%）: {medium_position_days}天 ({medium_position_days/total_days:.1%})")
            print(f"- 高仓位天数（>=80%）: {high_position_days}天 ({high_position_days/total_days:.1%})")
        else:
            print("- 无法计算总仓位比例（无有效数据）")
                
    except Exception as e:
        print(f"- 资金利用率统计失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)


def analyze_risk(report_normal_df: pd.DataFrame, benchmark_id: str = None):
    """
    详细的风险分析
    
    Parameters
    ----------
    report_normal_df : pd.DataFrame
        回测报告数据
    benchmark_id : str
        基准指数代码
    """
    print("\n" + "="*80)
    print("风险分析")
    print("="*80)
    
    try:
        net_returns = report_normal_df['return'].dropna()
        if len(net_returns) == 0:
            print("警告: 无有效收益数据")
            return
        
        # 基础风险指标
        daily_mean = net_returns.mean()
        daily_std = net_returns.std()
        annualized_return = (1 + net_returns).prod() ** (252 / len(net_returns)) - 1
        annualized_vol = daily_std * np.sqrt(252)
        
        # 计算累计收益和回撤
        cumulative_returns = (1 + net_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 1. 波动率分析
        print("\n1. 波动率分析:")
        print(f"   - 日收益率均值: {daily_mean:.4%}")
        print(f"   - 日收益率标准差: {daily_std:.4%}")
        print(f"   - 年化波动率: {annualized_vol:.2%}")
        print(f"   - 波动率/收益率比: {annualized_vol/abs(annualized_return) if annualized_return != 0 else 0:.2f}")
        
        # 2. 下行风险分析
        negative_returns = net_returns[net_returns < 0]
        if len(negative_returns) > 0:
            downside_std = negative_returns.std()
            downside_vol = downside_std * np.sqrt(252)
            downside_ratio = len(negative_returns) / len(net_returns)
        else:
            downside_std = 0
            downside_vol = 0
            downside_ratio = 0
        
        print("\n2. 下行风险分析:")
        print(f"   - 下行波动率（年化）: {downside_vol:.2%}")
        print(f"   - 亏损交易日占比: {downside_ratio:.2%}")
        print(f"   - 平均亏损: {negative_returns.mean():.4%}" if len(negative_returns) > 0 else "   - 平均亏损: 0.00%")
        print(f"   - 最大单日亏损: {net_returns.min():.2%}")
        
        # 3. Sortino比率（只考虑下行风险）
        if downside_vol > 0:
            sortino_ratio = (annualized_return / downside_vol) if downside_vol > 0 else 0
        else:
            sortino_ratio = float('inf') if annualized_return > 0 else 0
        print(f"   - Sortino比率: {sortino_ratio:.4f}")
        
        # 4. VaR和CVaR分析
        print("\n3. VaR (Value at Risk) 分析:")
        var_95 = np.percentile(net_returns, 5)  # 95%置信度下的VaR
        var_99 = np.percentile(net_returns, 1)  # 99%置信度下的VaR
        print(f"   - VaR (95%置信度): {var_95:.2%}")
        print(f"   - VaR (99%置信度): {var_99:.2%}")
        
        # CVaR (Conditional VaR)
        cvar_95 = net_returns[net_returns <= var_95].mean() if len(net_returns[net_returns <= var_95]) > 0 else 0
        cvar_99 = net_returns[net_returns <= var_99].mean() if len(net_returns[net_returns <= var_99]) > 0 else 0
        print(f"   - CVaR (95%置信度): {cvar_95:.2%}")
        print(f"   - CVaR (99%置信度): {cvar_99:.2%}")
        
        # 5. 回撤分析
        print("\n4. 回撤分析:")
        print(f"   - 最大回撤: {max_drawdown:.2%}")
        
        # 计算回撤持续时间
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_dd_start = None
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and current_dd_start is None:
                current_dd_start = i
            elif not is_dd and current_dd_start is not None:
                drawdown_periods.append(i - current_dd_start)
                current_dd_start = None
        if current_dd_start is not None:
            drawdown_periods.append(len(in_drawdown) - current_dd_start)
        
        if drawdown_periods:
            avg_dd_duration = np.mean(drawdown_periods)
            max_dd_duration = max(drawdown_periods)
            print(f"   - 平均回撤持续时间: {avg_dd_duration:.1f}天")
            print(f"   - 最长回撤持续时间: {max_dd_duration:.0f}天")
            print(f"   - 回撤次数: {len(drawdown_periods)}次")
        
        # 计算回撤恢复时间
        recovery_periods = []
        for i in range(1, len(drawdown_periods)):
            if i < len(drawdown_periods):
                recovery_periods.append(drawdown_periods[i] - drawdown_periods[i-1])
        
        if recovery_periods:
            avg_recovery = np.mean(recovery_periods)
            print(f"   - 平均恢复时间: {avg_recovery:.1f}天")
        
        # 6. 收益风险比
        print("\n5. 收益风险比:")
        sharpe_ratio = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0
        calmar_ratio = (annualized_return / abs(max_drawdown)) if max_drawdown != 0 else 0
        print(f"   - 夏普比率: {sharpe_ratio:.4f}")
        print(f"   - Calmar比率: {calmar_ratio:.4f}")
        print(f"   - 收益/最大回撤比: {annualized_return/abs(max_drawdown) if max_drawdown != 0 else 0:.2f}")
        
        # 7. 胜率分析
        print("\n6. 胜率分析:")
        positive_returns = net_returns[net_returns > 0]
        win_rate = len(positive_returns) / len(net_returns) if len(net_returns) > 0 else 0
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        print(f"   - 胜率: {win_rate:.2%}")
        print(f"   - 平均盈利: {avg_win:.4%}")
        print(f"   - 平均亏损: {avg_loss:.4%}")
        print(f"   - 盈亏比: {profit_loss_ratio:.2f}")
        
        # 8. 月度收益分析
        print("\n7. 月度收益分析:")
        try:
            # 确保索引是DatetimeIndex
            if isinstance(report_normal_df.index, pd.DatetimeIndex):
                net_returns_indexed = pd.Series(net_returns.values, index=report_normal_df.index[report_normal_df['return'].notna()])
            else:
                # 尝试转换索引
                net_returns_indexed = pd.Series(net_returns.values, index=pd.to_datetime(report_normal_df.index[report_normal_df['return'].notna()]))
            
            monthly_returns = net_returns_indexed.resample('M').apply(lambda x: (1 + x).prod() - 1)
            if len(monthly_returns) > 0:
                print(f"   - 平均月度收益: {monthly_returns.mean():.2%}")
                print(f"   - 月度收益标准差: {monthly_returns.std():.2%}")
                print(f"   - 最佳月度收益: {monthly_returns.max():.2%}")
                print(f"   - 最差月度收益: {monthly_returns.min():.2%}")
                print(f"   - 正收益月数: {(monthly_returns > 0).sum()}/{len(monthly_returns)}")
                print(f"   - 月度胜率: {(monthly_returns > 0).sum()/len(monthly_returns):.2%}")
        except Exception as e:
            print(f"   - 月度收益分析失败: {str(e)}")
        
        # 9. 基准对比（如果有）
        if benchmark_id:
            try:
                from qlib.data import D
                backtest_start = report_normal_df.index[0]
                backtest_end = report_normal_df.index[-1]
                
                benchmark_data = D.features(
                    [benchmark_id],
                    ['$close'],
                    start_time=backtest_start,
                    end_time=backtest_end
                )
                
                if not benchmark_data.empty:
                    benchmark_returns = benchmark_data['$close'].pct_change().dropna()
                    benchmark_annual_return = (1 + benchmark_returns).prod() ** (252 / len(benchmark_returns)) - 1
                    benchmark_vol = benchmark_returns.std() * np.sqrt(252)
                    benchmark_sharpe = (benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252)) if benchmark_returns.std() > 0 else 0
                    
                    print("\n8. 与基准对比:")
                    print(f"   - 策略年化收益: {annualized_return:.2%} vs 基准: {benchmark_annual_return:.2%}")
                    print(f"   - 策略波动率: {annualized_vol:.2%} vs 基准: {benchmark_vol:.2%}")
                    print(f"   - 策略夏普比率: {sharpe_ratio:.4f} vs 基准: {benchmark_sharpe:.4f}")
                    
                    # 计算跟踪误差
                    if len(benchmark_returns) == len(net_returns):
                        tracking_error = (net_returns - benchmark_returns).std() * np.sqrt(252)
                        print(f"   - 跟踪误差: {tracking_error:.2%}")
            except Exception as e:
                print(f"\n8. 基准对比失败: {str(e)}")
        
        # 10. 风险等级评估
        print("\n9. 风险等级评估:")
        if annualized_vol < 0.15:
            risk_level = "低风险"
        elif annualized_vol < 0.25:
            risk_level = "中低风险"
        elif annualized_vol < 0.35:
            risk_level = "中风险"
        elif annualized_vol < 0.50:
            risk_level = "中高风险"
        else:
            risk_level = "高风险"
        
        print(f"   - 风险等级: {risk_level}")
        print(f"   - 风险评分: {min(10, int(annualized_vol * 20))}/10")
        
    except Exception as e:
        print(f"风险分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)


def main():
    if len(sys.argv) < 2:
        print("使用方法: python run_reversal_grid_pool_strategy.py <config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    print("="*80)
    print("反转网格股票池策略回测")
    print("="*80)
    print(f"配置文件: {config_path}")
    
    # 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 初始化qlib
    print("\n正在初始化Qlib...")
    qlib_config = config.get('qlib_init', {})
    provider_uri = qlib_config.get('provider_uri', '~/.qlib/qlib_data/cn_data')
    region = qlib_config.get('region', 'cn')
    
    qlib.init(
        provider_uri=provider_uri,
        region=region,
        auto_mount=True,
    )
    print("✓ Qlib初始化完成")
    
    # 开始回测
    port_analysis_config = config.get('port_analysis_config', {})
    
    if port_analysis_config:
        try:
            print("\n[1/1] 运行回测...")
            backtest_start = time.time()
            
            # 初始化策略和执行器
            strategy = init_instance_by_config(port_analysis_config.get('strategy', {}))
            executor_config = port_analysis_config.get('executor', {
                "class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {
                    "time_per_step": "day",
                    "generate_portfolio_metrics": True,
                    "verbose": False,
                },
            })
            executor = init_instance_by_config(executor_config)
            
            # 获取回测配置
            backtest_config = port_analysis_config.get('backtest', {})
            account = backtest_config.get('account', 100000000)
            benchmark_id = backtest_config.get('benchmark', None)
            
            # 直接运行回测
            portfolio_metric_dict, indicator_dict = normal_backtest(
                executor=executor,
                strategy=strategy,
                **backtest_config
            )
            
            backtest_time = time.time() - backtest_start
            print(f"✓ 回测完成 (耗时: {format_time(backtest_time)})")
            
            # 分析结果
            for _freq, (report_normal, positions_normal) in portfolio_metric_dict.items():
                report_df = report_normal
                positions_df = positions_normal
                
                # 获取股票池配置
                stock_pool = port_analysis_config.get('strategy', {}).get('kwargs', {}).get('stock_pool', {})
                
                # 分析结果
                analyze_results(report_df, positions_df, account, stock_pool, benchmark_id)
            
            # 详细风险分析
            analyze_risk(report_df, benchmark_id)
            
            # 打印qlib自带的风险分析结果
            if indicator_dict:
                print("\nQlib风险分析结果:")
                for key, value in indicator_dict.items():
                    if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                        print(f"{key}")
                        print(value.to_string())
                        print()
            
        except Exception as e:
            print(f"\n✗ 回测失败: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    total_time = time.time() - backtest_start
    print("="*80)
    print(f"策略运行完成 (总耗时: {format_time(total_time)})")
    print("="*80)


if __name__ == "__main__":
    main()
