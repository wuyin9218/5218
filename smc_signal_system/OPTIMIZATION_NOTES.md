# Code Optimization Notes

## 修复日期: 2025-11-24

### 致命问题修复 (Critical Fixes)

#### C1. 数据层 - 剔除未完成的最后一根K线
**文件**: `data_layer/binance_rest.py`
**问题**: Binance API 可能返回正在形成中的K线，导致前视偏差
**修复**: 
- 在 `fetch_klines()` 返回前检查最后一根K线的 `close_time`
- 如果 `close_time > end_time`，剔除该K线
- 添加日志记录剔除操作

#### C2. 策略层 - 移除baseline模式的合成OB逻辑
**文件**: `strategy_engine/model_a.py`
**问题**: 
- 合成OB使用固定百分比阈值，对不同币种不适用
- 方向判断过于简单，产生大量虚假信号
**修复**: 
- 完全移除合成OB逻辑（第168-190行）
- baseline模式仍使用真实SMC结构，只是不应用趋势/FVG过滤

#### C4. 风控层 - 修复日亏损熔断计算
**文件**: `backtest/runner.py`
**问题**: 使用 `initial_balance` 计算日亏损限额，不随账户余额变化
**修复**: 改为使用 `self.balance` (当前余额)

---

### 高风险问题修复 (High Priority Fixes)

#### H1. 数据层 - 统一空DataFrame处理
**文件**: `data_layer/binance_rest.py`
**问题**: 空DataFrame未设置正确的index name
**修复**: 确保空DataFrame也设置 `index.name = "timestamp"`

#### H3. 策略层 - 改进OB排序异常处理
**文件**: `strategy_engine/model_a.py`
**问题**: 
- 使用裸 `except:` 捕获所有异常
- 排序失败时使用未排序列表，可能选择错误OB
**修复**: 
- 改为捕获 `(TypeError, AttributeError)`
- 排序失败时返回 `None` 而不是继续处理

#### H6. 风控层 - 优化冷却期逻辑
**文件**: `backtest/runner.py`
**问题**: 连续亏损后进入冷却期，但冷却期结束后 `consecutive_losses` 未重置
**修复**: 
- 在盈利交易时重置 `cooldown_until = None`
- 添加冷却期触发和重置的日志

---

### 中风险问题修复 (Medium Priority Fixes)

#### M3. 策略层 - 添加最小risk阈值
**文件**: `strategy_engine/model_a.py`
**问题**: 极小的risk值导致不合理的交易（手续费占比过大）
**修复**: 
- 添加最小risk阈值检查：`min_risk_threshold = entry_price * 0.002` (0.2%)
- 同时应用于baseline和strict模式

#### M6. 策略层 - 改进FVG验证逻辑
**文件**: `strategy_engine/model_a.py`
**问题**: 使用已填补的FVG作为确认
**修复**: 在 `_has_fvg_confirmation()` 中过滤 `f.filled == True` 的FVG

---

### 代码质量改进 (Code Quality Improvements)

#### 异常处理改进
**文件**: `smc_engine/indicators.py`, `strategy_engine/model_a.py`
**修复**: 
- 所有 `except Exception` 改为具体异常类型：`(ValueError, KeyError, AttributeError)`
- 警告信息统一使用 `[WARN]` 或 `[ERROR]` 前缀

#### 文档注释改进
**文件**: `backtest/runner.py`
**修复**: 
- 添加资金模型说明（保证金制，1x杠杆等效）
- 明确设计假设和局限性

#### 权益曲线完善
**文件**: `backtest/runner.py`
**修复**: 
- 在 `enter_position()` 后也记录权益点
- 确保开仓时的余额变化被正确跟踪

---

## 未修复但需注意的问题

### M1. 时间戳对齐问题
**影响**: 微小的前视偏差（使用open_time而非close_time作为索引）
**状态**: 未修复（需要较大重构）
**缓解措施**: 策略中使用 `data.loc[:current_time]` 避免使用未来数据

### M4. 仓位大小未考虑杠杆
**影响**: 对于杠杆交易，仓位计算可能不准确
**状态**: 已添加文档说明（假设1x杠杆）
**建议**: 如需杠杆交易，在 `calculate_position_size()` 中添加 `leverage` 参数

### L1. offline_fallback 默认开启
**影响**: 可能误用虚假数据
**状态**: 未修改（因当前实现默认使用真实数据）
**建议**: 在production环境明确设置 `offline_fallback=False`

---

## 测试建议

### 必须测试的场景
1. **数据边界测试**:
   - 请求数据的 `end_time` 是当前时间
   - 验证最后一根K线被正确剔除

2. **风控测试**:
   - 连续3次亏损 → 冷却期 → 盈利交易 → 冷却解除
   - 日亏损达到限额（使用当前余额计算）

3. **策略测试**:
   - baseline模式：无有效OB时不产生信号（不使用合成OB）
   - strict模式：已填补的FVG不被用于确认

4. **极端情况测试**:
   - risk < 0.2% 的信号被过滤
   - OB时间排序失败时不产生信号

---

## 性能优化建议（未实施）

1. **增量计算SMC指标**: 当前每根K线重新计算全部指标，效率较低
2. **缓存swing_highs_lows结果**: 避免重复计算
3. **使用numba加速**: 关键循环可用numba.jit编译

---

## 架构改进建议（未实施）

1. **添加单元测试**: 覆盖关键逻辑（OB解析、风控规则）
2. **统一日志系统**: 使用 `logging` 模块替代 `print()`
3. **配置验证**: 在加载配置时验证参数合法性
4. **提取重复代码**: `run_backtest.py` 和 `debug_model_a_signals.py` 的数据加载逻辑

---

## 版本信息
- 优化前版本: Initial
- 优化后版本: v1.0-optimized
- 优化者: AI Code Reviewer
- 优化日期: 2025-11-24
