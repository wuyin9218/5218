# SMC Signal System

半自动 Smart Money Concepts (SMC) 信号系统骨架。

## 项目结构

```
smc_signal_system/
├── config/              # 配置文件
│   ├── global.yaml      # 全局配置
│   ├── symbols.yaml     # 交易标的配置
│   ├── risk.yaml        # 风险管理配置
│   └── news_filter.yaml # 新闻过滤配置
├── smc_engine/          # SMC 引擎
│   ├── indicators.py    # SMC 指标封装
│   └── structure.py     # 统一结构输出
├── data_layer/          # 数据层
│   ├── binance_rest.py  # Binance API 客户端
│   └── cache.py         # 本地缓存
├── backtest/            # 回测模块
│   ├── runner.py        # 回测运行器
│   ├── slippage_fee.py  # 手续费/滑点模型
│   ├── metrics.py       # 性能指标
│   └── walkforward.py   # 滚动窗口分析（Week 3）
├── scripts/              # 脚本
│   └── run_backtest.py  # 回测运行脚本
└── tests/               # 测试
    └── test_backtest_regression.py  # 回归测试
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 运行回测

```bash
python scripts/run_backtest.py --config config/global.yaml
```

### 运行测试

```bash
pytest
```

## 配置说明

### global.yaml
- 项目基本配置
- 数据获取参数
- 回测参数（初始资金、手续费、滑点等）

### symbols.yaml
- 交易标的列表
- 市场类型

### risk.yaml
- 每笔交易风险比例
- 每日最大亏损限制
- 连续亏损限制
- 最小风险回报比

### news_filter.yaml
- 新闻过滤开关
- 黑名单时间段配置

## 输出文件

回测结果保存在 `backtests/<phase>/` 目录下：

- `summary.json`: 回测汇总指标
- `trades.csv`: 所有交易记录

## 注意事项

- 所有决策基于 candle close，避免未来函数
- 配置使用 dataclass + YAML 统一加载
- 费用/滑点/时间周期/标的均可配置
- 输出 schema 固定，便于后续分析

## 开发状态

- ✅ Week 1: 骨架搭建完成
- ⏳ Week 2: 策略实现（待开发）
- ⏳ Week 3: 滚动窗口分析（待开发）







