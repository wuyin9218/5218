# 项目生成总结

## 生成的文件列表

### 配置文件 (config/)
- `global.yaml` - 全局配置（项目设置、数据参数、回测参数）
- `symbols.yaml` - 交易标的配置
- `risk.yaml` - 风险管理配置
- `news_filter.yaml` - 新闻过滤配置
- `loader.py` - 配置加载器（使用 dataclass + YAML）
- `__init__.py` - 包初始化文件

### SMC 引擎 (smc_engine/)
- `indicators.py` - SMC 指标封装（封装 smartmoneyconcepts.smc 的 8 个函数）
  - fvg, swing_highs_lows, bos_choch, ob, liquidity, previous_high_low, sessions, retracements
- `structure.py` - 统一的结构输出 dataclass（供后续策略消费）
- `__init__.py` - 包初始化文件

### 数据层 (data_layer/)
- `binance_rest.py` - Binance USD-M Futures REST API 客户端
  - 支持 5m/15m/1h/4h/1d 等时间周期
  - 支持批量获取 klines
- `cache.py` - 本地缓存（使用 Parquet 格式）
  - 防止重复拉取数据
  - 支持时间范围过滤
- `__init__.py` - 包初始化文件

### 回测模块 (backtest/)
- `runner.py` - 事件驱动回测骨架
  - 基于 candle close 价格（避免未来函数）
  - 支持风险管理规则
  - 支持止损/止盈
- `slippage_fee.py` - 手续费/滑点模型（可配置）
- `metrics.py` - 性能指标统计
  - winrate, PF, DD, expectancy 等
- `walkforward.py` - 滚动窗口分析接口（Week 3 占位符）
- `__init__.py` - 包初始化文件

### 脚本 (scripts/)
- `run_backtest.py` - 命令行回测运行脚本
  - 读取配置文件
  - 输出 `backtests/<phase>/summary.json` 和 `trades.csv`
- `__init__.py` - 包初始化文件

### 测试 (tests/)
- `test_backtest_regression.py` - 回归测试
  - 验证同一配置连续跑 3 次结果一致
  - 验证输出 schema 正确性
- `__init__.py` - 包初始化文件

### 其他文件
- `requirements.txt` - Python 依赖包列表
- `README.md` - 项目说明文档
- `pytest.ini` - pytest 配置文件
- `.gitignore` - Git 忽略文件配置

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行测试

```bash
pytest
```

### 3. 运行回测

```bash
python scripts/run_backtest.py --config config/global.yaml
```

## 关键特性

1. **配置管理**: 使用 dataclass + YAML，避免 dict 飘来飘去
2. **无未来函数**: 所有决策基于 candle close 价格
3. **可配置性**: 费用/滑点/时间周期/标的均可配置
4. **稳定输出**: summary.json 和 trades.csv 字段固定
5. **完整文档**: 所有函数和类都有清晰的 docstring

## 输出文件格式

### summary.json
```json
{
  "total_trades": 0,
  "winning_trades": 0,
  "losing_trades": 0,
  "win_rate": 0.0,
  "profit_factor": 0.0,
  "total_pnl": 0.0,
  "total_fees": 0.0,
  "net_pnl": 0.0,
  "max_drawdown": 0.0,
  "max_drawdown_pct": 0.0,
  "expectancy": 0.0,
  "avg_win": 0.0,
  "avg_loss": 0.0,
  "largest_win": 0.0,
  "largest_loss": 0.0,
  "sharpe_ratio": null,
  "calmar_ratio": null,
  "symbols": []
}
```

### trades.csv
包含以下列：
- entry_time, exit_time, symbol, side
- entry_price, exit_price, quantity
- pnl, pnl_pct, fees, rr

## 注意事项

1. 当前实现使用 `dummy_signal_generator`，不生成实际交易信号（骨架版本）
2. `walkforward.py` 是占位符，Week 3 才实现
3. `smc_engine/indicators.py` 中的函数需要实际安装 `smart-money-concepts==0.0.26` 才能正常工作
4. 数据缓存使用 Parquet 格式，需要安装 `pyarrow`

## 下一步开发

- Week 2: 实现具体的 SMC 信号生成策略
- Week 3: 实现滚动窗口分析（walkforward）







