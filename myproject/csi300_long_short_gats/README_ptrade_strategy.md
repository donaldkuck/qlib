# Ptrade 多空 GATs 策略使用说明

## 概述

这是一个完全独立于 Qlib 的 ptrade 回测策略，直接在 ptrade 平台上运行。策略实现了与原始 Qlib 版本相同的逻辑：

- **TopkDrop 交易逻辑**：持有 topk 只股票，每天替换 n_drop 只
- **多空结合**：使用长期和短期因子进行预测
- **加仓/止盈**：根据短期预测值对已持仓股票进行加仓或止盈
- **止损/回撤控制**：保护本金，控制风险

## 文件结构

```
ptrade_gats_strategy.py      # Ptrade 策略主文件（在 ptrade 上运行）
ptrade_factors.py            # 因子计算模块
ptrade_gats_model.py         # GATs 模型加载和预测模块
```

## 准备工作

### 1. 安装依赖

在 ptrade 环境中安装以下依赖：

```bash
pip install numpy pandas scipy torch pyyaml
```

### 2. 准备模型文件

从 Qlib 的 mlruns 目录中复制训练好的模型文件：

```bash
# 找到 run_id（32 位十六进制，如 0b8e2395033a4264803061c0f76b63c8）
ls mlruns/673062759933473454/

# 复制模型文件到策略目录
cp mlruns/673062759933473454/<run_id>/short_term_params.pkl ./
cp mlruns/673062759933473454/<run_id>/long_term_params.pkl ./
```

### 3. 准备因子配置文件

将因子配置文件复制到策略目录：

```bash
cp factor_config.yaml ./
cp factor_config_longterm.yaml ./
```

### 4. 确保策略文件在正确位置

将以下文件放在 ptrade 策略目录中：

- `ptrade_gats_strategy.py`
- `ptrade_factors.py`
- `ptrade_gats_model.py`
- `short_term_params.pkl`
- `long_term_params.pkl`
- `factor_config.yaml`
- `factor_config_longterm.yaml`

## 策略参数

在 `ptrade_gats_strategy.py` 的 `initialize` 函数中可以调整策略参数：

```python
strategy = PtradeGATsStrategy(
    # 模型路径
    short_term_model_path="short_term_params.pkl",
    long_term_model_path="long_term_params.pkl",
    # 因子配置
    short_factor_config="factor_config.yaml",
    long_factor_config="factor_config_longterm.yaml",
    # 模型参数
    short_step_len=20,      # 短期时间序列长度
    long_step_len=40,       # 长期时间序列长度
    short_d_feat=143,       # 短期特征维度
    long_d_feat=150,        # 长期特征维度
    # TopkDrop 参数
    topk=20,               # 持仓股票数量
    n_drop=5,               # 每天替换的股票数量
    hold_thresh=10,         # 最小持有天数
    # 加仓/止盈参数
    add_position_ratio=0.03, # 加仓比例
    profit_threshold=0.05,  # 止盈阈值（5%）
    reduce_position_ratio=0.2, # 止盈时减仓比例（20%）
    max_single_stock_ratio=0.05, # 单只股票最大仓位（5%）
    # 止损参数
    stop_loss_threshold=-0.05,  # 止损阈值（-5%）
    # 回撤控制参数
    max_drawdown=None,       # 最大回撤阈值（None 表示不启用）
    drawdown_reduce_ratio=0.3, # 回撤时减仓比例
    # 其他参数
    risk_degree=0.95,       # 风险仓位比例
    device="cpu",           # 设备（cpu/cuda）
)
```

## 在 Ptrade 上运行

### 方法 1：通过 Ptrade 平台上传

1. 登录 ptrade 平台
2. 进入策略管理
3. 上传 `ptrade_gats_strategy.py` 文件
4. 配置回测参数（回测周期、初始资金等）
5. 开始回测

### 方法 2：本地调试

如果 ptrade 支持本地运行，可以直接执行：

```bash
python ptrade_gats_strategy.py
```

## 策略逻辑说明

### 1. 因子计算

- 从 ptrade 获取 OHLCV 数据
- 按照原始因子配置计算所有因子
- 支持 Qlib 表达式语法（Ref, Mean, Std, Sum, Min, Max, Quantile, Slope 等）

### 2. 模型预测

- 加载预训练的 GATs TS 模型
- 使用时间序列数据（短期 20 天，长期 40 天）
- 归一化特征后输入模型
- 输出未来收益预测值

### 3. 交易决策

**买入逻辑**：
- 基于长期预测值选择 topk 只股票
- 每天替换 n_drop 只长期预测值最低的股票
- 只买入长期和短期预测都为正的股票

**卖出逻辑**：
- 卖出长期预测值最低的股票
- 必须满足最小持有天数（hold_thresh）

**加仓逻辑**：
- 已持仓股票短期预测为正：加仓
- 考虑单只股票最大仓位限制

**止盈逻辑**：
- 已持仓股票短期预测为负且盈利超过阈值：止盈减仓

**止损逻辑**：
- 亏损超过止损阈值：全部卖出
- 长期预测为负：止损卖出

**回撤控制**：
- 回撤超过阈值：降低仓位
- 主动卖出部分持仓

## 因子表达式语法

策略支持以下 Qlib 风格的表达式函数：

| 函数 | 说明 | 示例 |
|------|------|------|
| `Ref(series, n)` | 引用 n 天前的值 | `Ref($close, 1)` |
| `Mean(series, n)` | n 天移动平均 | `Mean($close, 20)` |
| `Std(series, n)` | n 天滚动标准差 | `Std($close, 20)` |
| `Sum(series, n)` | n 天滚动求和 | `Sum(($close-Ref($close,1))/Ref($close,1), 5)` |
| `Min(series, n)` | n 天滚动最小值 | `Min($close, 20)` |
| `Max(series, n)` | n 天滚动最大值 | `Max($close, 20)` |
| `Quantile(series, n, q)` | n 天分位数 | `Quantile($close, 20, 0.5)` |
| `Slope(series, n)` | n 天线性回归斜率 | `Slope($close, 20)` |
| `If(cond, t, f)` | 条件判断 | `If($close>Mean($close,5), 1, 0)` |
| `Sign(series)` | 符号函数 | `Sign($close-Ref($close,1))` |
| `Abs(series)` | 绝对值 | `Abs($close-Ref($close,1))` |
| `Resi(series, n)` | 残差 | `Resi($close, 20)` |

字段名：
- `$close`: 收盘价
- `$open`: 开盘价
- `$high`: 最高价
- `$low`: 最低价
- `$volume`: 成交量

## 常见问题

### Q: 模型加载失败怎么办？

A: 检查模型文件路径是否正确，确保模型文件是从正确的 mlruns 目录复制的。模型文件应该是 `.pkl` 格式。

### Q: 因子计算出错怎么办？

A: 检查因子配置文件路径是否正确，确保 ptrade 能够获取到足够的历史数据（至少 100 天）。

### Q: 预测结果为空怎么办？

A: 可能的原因：
1. 股票数据不足（需要至少 step_len + 50 天的数据）
2. 因子计算失败（检查因子表达式是否正确）
3. 模型加载失败

### Q: 如何调整交易频率？

A: 调整以下参数：
- `topk`: 增大减少调仓频率，减小增加调仓频率
- `n_drop`: 减小降低调仓频率
- `hold_thresh`: 增大增加最小持有天数

### Q: 如何降低风险？

A: 调整以下参数：
- `stop_loss_threshold`: 设置止损阈值（如 -0.05 表示亏损 5% 时止损）
- `max_drawdown`: 设置最大回撤阈值
- `max_single_stock_ratio`: 降低单只股票最大仓位比例
- `risk_degree`: 降低风险仓位比例

## 模块说明

### ptrade_factors.py

因子计算模块，实现：
- `FactorCalculator`: 因子计算器，包含所有 Qlib 函数的实现
- `FactorLibrary`: 因子库
- `calculate_factor_series()`: 计算单个股票的因子序列
- `prepare_time_series_data()`: 准备时间序列数据
- `normalize_features()`: 特征归一化

### ptrade_gats_model.py

模型加载和预测模块，实现：
- `GATModel`: GATs TS 模型定义
- `PtradeGATsPredictor`: 预测器
- `load_model_from_pickle()`: 从 pickle 文件加载模型
- `ModelEnsemble`: 模型集成（支持 n-fold）

### ptrade_gats_strategy.py

策略主文件，实现：
- `PtradeGATsStrategy`: 策略类
- `initialize()`: ptrade 初始化函数
- `handle_data()`: ptrade 主策略逻辑
- `after_trading_end()`: 盘后处理

## 注意事项

1. **数据完整性**：确保 ptrade 能够获取到完整的 OHLCV 数据
2. **计算时间**：因子计算和模型预测需要时间，可能影响回测速度
3. **内存使用**：模型加载和数据缓存会占用内存
4. **GPU 支持**：如果有 GPU，设置 `device="cuda"` 可以加速预测

## 性能优化建议

1. **缓存因子**：因子计算结果可以缓存，避免重复计算
2. **批量预测**：尽量使用批量预测而不是单个预测
3. **减少特征**：如果不需要全部特征，可以减少 `d_feat`
4. **使用 GPU**：模型预测在 GPU 上会更快
