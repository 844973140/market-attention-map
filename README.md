# Latch AGP Competition Agent

这是一个为 AGP（Agent Grand Prix）比赛构建的自主参赛 Agent。仓库用于说明它的决策架构、策略能力和模拟表现，不将其定位为面向通用场景的 Agent 产品、SDK 或公共服务。

它的目标是在有限 USDC 预算下找到隐藏 checkpoint：每次调用裁判前评估预期信息增益、成本和风险，在继续查询的价值低于成本时停止搜索并提交最可能的答案。

## 核心功能

- **信息价值搜索**：估算问题的信息增益、成功率提升和 `information_gain / cost`，优先执行单位预算价值更高的问题。
- **比赛节奏控制**：根据 Early、Mid、Late 阶段动态切换探索、收敛和终局策略，避免过早耗尽预算。
- **策略与权限分离**：Strategy 决定想做什么，Policy Engine 审核预算、重复问题、低价值动作和低置信度提交。
- **自适应决策**：结合剩余预算、当前置信度、问题数量、候选规模和比赛进度选择 `EXPLORATION`、`OPTIMIZATION` 或 `FINISH`。
- **比赛记忆**：记录历史问题表现与状态动作结果，用于调整信息增益阈值、提交阈值和预算分配。
- **运行时适配**：通过独立 AGP Client 处理状态查询、Judge 提问、预算扣减和最终答案提交，便于替换为正式比赛接口。
- **可追踪执行**：为每次允许或拒绝的动作记录时间、成本、决策和原因，保留完整审计链路。
- **统计评估**：通过批量模拟、策略 benchmark 和 Tournament 排行榜比较胜率、平均成本、问题数量与预算效率。

## 策略

Agent 维护所有候选 checkpoint 的贝叶斯概率分布，并把裁判视为一个可能有少量误差的二元信息源。

1. **类别定位**：先询问最有区分度的类别问题，快速缩小范围。
2. **高价值二分**：在剩余候选上构造接近 50/50 概率质量的集合问题，同时评估更短的单点确认与属性问题。
3. **动态停止**：置信度达到阈值后立即提交；如果问题的信息增益、成功率提升、单位成本价值或剩余预算不足，也会提前停止并猜测。

每个问题都会计算：

- Shannon 信息增益（bits）
- 预计输入/输出 token 和 USDC 成本
- 每 USDC 信息价值（bits/USDC）
- 发问后的预期成功率提升

## 文件结构

- `agent.py`：运行入口、预算账本、本地裁判、AGP HTTP 和 Latch 支付接口
- `agp_client.py`：AGP Runtime 抽象接口与本地 Mock 实现
- `simulation.py`：随机 AGP 训练赛、批量统计和策略 benchmark
- `adaptive_strategy.py`：根据实时预算、置信度和比赛进度调整模式
- `strategy_memory.py`：问题表现、状态动作经验、JSON 持久化和 Memory benchmark
- `strategy.py`：候选概率、问题生成、信息增益评分和停止策略
- `race_strategy.py`：Early/Mid/Late 比赛节奏与动态预算分配
- `policy_engine.py`：预算控制、动作授权、风险策略和审计日志
- `config.py`：环境变量与所有可调阈值
- `test_policy.py`：Policy Engine 的标准库单元测试
- `test_race_strategy.py`：Race Strategy 阶段、预算和 Policy 集成测试
- `test_agp_client.py`：Runtime 调用、预算扣减和提交测试
- `test_simulation.py`：批量赛局、预算上限和提交测试
- `test_adaptive_strategy.py`：Adaptive 模式、终局动作和 benchmark 测试
- `test_strategy_memory.py`：Memory 保存、读取和历史策略调整测试
- `requirements.txt`：最小运行依赖

## 本地验证

需要 Python 3.11 或更高版本。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
python agent.py
```

默认运行离线模拟，隐藏答案为 `checkpoint-delta`。也可以指定其他内置候选：

```bash
python agent.py --mode local --hidden checkpoint-hotel --budget 0.05
```

## 候选数据

真实比赛应提供 JSON 候选文件。最简单格式是字符串数组：

```json
[
  "checkpoint-alpha",
  "checkpoint-bravo",
  "checkpoint-charlie"
]
```

推荐提供类别、先验和可验证属性，使 Agent 能提出更短、更便宜的问题：

```json
[
  {
    "answer": "checkpoint-alpha",
    "category": "city",
    "prior": 2.0,
    "attributes": {"region": "north", "indoor": false}
  },
  {
    "answer": "checkpoint-delta",
    "category": "nature",
    "prior": 1.0,
    "attributes": {"region": "south", "indoor": false}
  }
]
```

`prior` 可以来自历史题目、公开线索或外部检索。缺省为 `1.0`，即均匀先验。候选必须唯一，且真实 checkpoint 必须在候选集合内；这是当前简单策略的边界。

## AGP Runtime 适配

设置环境变量后运行远程模式：

```bash
set AGP_MODE=remote
set AGP_API_BASE_URL=https://your-agp-host.example
set AGP_API_KEY=your-token
set AGP_SESSION_ID=game-or-session-id
python agent.py --candidates candidates.json
```

macOS / Linux 请将 `set NAME=value` 替换为 `export NAME=value`。

当前 HTTP 适配器的默认约定为：

- `POST /v1/judge/questions`，请求体包含 `question`、`max_tokens` 和可选 `session_id`
- `POST /v1/judge/submissions`，请求体包含 `answer` 和可选 `session_id`
- 问答响应从 `answer`、`reply`、`message` 或 `content` 字段读取文本
- 可选返回 `usage.prompt_tokens`、`usage.completion_tokens` 和 `cost_usdc`

正式 AGP 协议发布后，只需调整 `HttpAGPJudgeClient`，策略层不需要修改。路径也可通过 `AGP_QUESTION_PATH` 和 `AGP_SUBMIT_PATH` 覆盖。

## Latch 策略接口

`PaymentGate` 是预留的支付边界。每次向裁判提问前先授权预计成本，响应后再按实际成本结算。设置以下变量会启用通用 HTTP 适配器：

```bash
set LATCH_API_BASE_URL=https://your-latch-host.example
set LATCH_API_KEY=your-token
```

默认请求：

- `POST /v1/payments/authorize`
- `POST /v1/payments/settle`

由于最终 Latch API 字段可能不同，正式接入时可替换 `HttpLatchPaymentGate`，或用官方 SDK 实现同一个 `PaymentGate` 协议。

## Latch Inspired Architecture

This agent implements a lightweight policy layer inspired by Latch.

The goal is not maximum computation, but optimal decisions under constraints.

The policy layer provides:

- **Budget awareness**：每次请求前检查预计成本、当前支出和最终提交预留预算。
- **Controlled execution**：低价值、超预算、重复或无效问题不会发送给裁判；低置信度猜测不会提交。
- **Transparent decisions**：所有问题和猜测都生成带 UTC 时间戳、预计成本、ALLOW/DENY 结论及原因的审计记录。

执行流程：

```text
Observe
  -> Race Strategy and Budget Pacing (race_strategy.py)
  -> Strategy Selection and Information Value (strategy.py)
  -> Policy Check (policy_engine.py)
  -> Optional Latch Authorization
  -> AGP Client Execution (agp_client.py)
  -> Update Belief and Actual Spend
```

Policy Engine 会重新计算核心指标：

```text
information_efficiency_score = expected_information_gain / estimated_cost_usdc
```

这避免执行层盲目信任外部评分。默认审计记录输出到控制台。设置 `AGP_POLICY_AUDIT_LOG=policy-audit.jsonl` 后，还会生成便于机器分析的追加式 JSONL 日志。`PaymentGate` 继续作为真实 Latch API 的远程授权接口，Policy Engine 则提供调用远程服务之前的本地安全边界。

## AGP Racing Strategy

The agent does not maximize questions. It optimizes decisions under limited resources.

Race Strategy Layer 把一次比赛划分为三个单向推进的阶段：

- **Early game — exploration**：降低最低信息效率门槛，允许略高的探索成本，但只能使用 `exploration_budget`。
- **Mid game — convergence**：提高 `information_gain / cost` 要求，优先执行单位 USDC 信息价值最高的问题。
- **Late game — risk control**：提高最终提交置信度要求，限制单次问题成本，用额外验证减少错误提交风险。

总预算会动态拆分成：

- `exploration_budget`：候选越多，探索占比越高，但有明确上限。
- `optimization_budget`：用于中盘收敛和终局验证；未使用的探索预算会滚入此预算。
- `final_guess_budget`：始终保留给最终提交，不允许前期搜索占用。

每轮执行顺序为：

```text
Race Strategy
  -> Strategy Selection
  -> Policy Engine Approval
  -> AGP Client
  -> Belief Update
  -> Final Submit
```

Race Layer 给出本阶段的成本上限、信息效率阈值和提交置信度；Strategy 从候选问题中选出最优动作；Policy Engine 再独立复核预算、价值、重复性和风险。任何一层拒绝，付费裁判调用都不会发生。

## Runtime Architecture

Agent logic is separated from execution layer.

Strategy decides what to do.

Policy decides whether it is allowed.

Client handles external interaction.

`AGPClient` 定义四个稳定接口：

- `get_status()`：读取比赛是否仍在运行、已提问次数和提交状态。
- `get_budget()`：读取 Runtime 侧权威剩余预算。
- `ask_question(question)`：发送经过 Race 与 Policy 审核的问题。
- `submit_answer(answer)`：提交最终 checkpoint。

`MockAGPClient` 用于本地比赛模拟，负责 Judge 回答、问题成本扣减、超预算拒绝、预算更新和最终答案验证。主循环不会读取隐藏答案，也不会直接调用 Judge；它只消费 `AGPClient` 返回的标准结果。

未来接入正式 AGP 环境时，只需继承 `AGPClient` 并实现上述四个方法，然后将新 Client 传给 `run_agent()`。Race Strategy、信息搜索、Policy Engine 和信念更新都不需要修改。现有 HTTP Judge 通过兼容适配器继续可用。

## Adaptive Race Strategy

Adaptive Race Strategy 根据每轮比赛的实时状态重新决定节奏，而不是只按照固定问题数切换阶段。输入包括：

- `remaining_budget`
- `current_confidence`
- `questions_used`
- `candidate_count`
- `race_progress`

它输出以下模式和约束：

- **EXPLORATION**：比赛早期、有效候选较多且置信度低时，优先最大化 `information_gain / cost`，同时限制单次探索成本。
- **OPTIMIZATION**：候选已经缩小或出现领先答案时，提高信息效率门槛，用最低成本提升置信度。
- **FINISH**：置信度足够、预算接近下限或比赛进度较晚时停止常规搜索，根据风险选择立即提交或进行最后一次有成本上限的确认。

Adaptive Layer 只输出决策约束，不绕过 Strategy 或 Policy Engine。低置信度提交、超预算问题、重复问题仍由 Policy 独立拒绝。

运行 Current Race Strategy 与 Adaptive Race Strategy 的 10000 局对比：

```bash
python simulation.py --episodes 10000 --adaptive-benchmark
```

`budget efficiency` 定义为 `win_rate / average_cost_usdc`，用于衡量每 USDC 获得成功结果的效率。

默认模拟参数和固定种子下的 10000 局结果：

| Strategy | Win rate | Avg cost | Avg questions | Avg confidence | Budget efficiency |
|---|---:|---:|---:|---:|---:|
| Current Race | 91.23% | 0.003610 | 3.53 | 93.70% | 252.74 |
| Adaptive Race | 92.37% | 0.003361 | 3.28 | 93.58% | 274.82 |

在这组模拟条件下，Adaptive Race 平均成本降低约 6.9%，预算效率提高约 8.7%，同时胜率提高 1.14 个百分点。该结论是统计模拟结果，不代表真实 AGP 环境中的固定收益。

## Strategy Memory Layer

Agent improves through previous race experience.

`StrategyMemory` 记录两类可解释经验：

- **Question Performance**：按 `question_type` 保存调用次数、平均信息增益、平均成本、成功率和 information efficiency。
- **Decision Memory**：按置信度区间、剩余预算比例和有效候选数分桶，记录每个动作的样本数、成功率和平均奖励。

例如，`confidence85-90|budget>50|candidates<=3` 会保存该状态下 `submit`、`continue` 等动作的历史结果。`MemoryEnhancedAdaptiveStrategy` 在每次选择模式时读取这些记录，并在样本量足够后调整：

- information gain threshold
- submission threshold
- 单次问题成本上限
- exploration / optimization / final guess budget allocation

记忆使用版本化 JSON 格式持久化。以下命令会读取已有 `memory.json`、追加训练经验并重新保存：

```bash
python strategy_memory.py --episodes 10000 --training-episodes 2000 --memory memory.json
```

默认条件下的 10000 局结果：

| Strategy | Win rate | Avg cost | Questions | Budget efficiency |
|---|---:|---:|---:|---:|
| Adaptive Strategy | 92.37% | 0.003361 | 3.28 | 274.82 |
| Adaptive + Memory | 92.81% | 0.003200 | 3.13 | 290.02 |

Memory 从独立训练局中学习到 87% 提交阈值：只有相关状态至少积累 20 个提交样本且历史成功率不低于 85%，才会下调默认阈值，并保留额外安全边际。在这组模拟中，Memory 将胜率提高 0.44 个百分点，平均成本降低约 4.8%，预算效率提高约 5.5%。这些规则仍受现有 Policy Engine 约束，不使用机器学习模型。

## Simulation Training

Agent strategies can be evaluated through simulated AGP races before entering a competition.

`AGPSimulator` 会为每局比赛随机选择隐藏 checkpoint，通过带噪声的 Mock Judge 返回 `YES`、`NO` 或 `UNKNOWN`，同时模拟问题成本波动、信息收益、预算扣减和最终提交。它不使用机器学习模型，只通过可重复的统计模拟比较策略。

运行 1000 局当前 Race Strategy：

```bash
python simulation.py --episodes 1000 --strategy race
```

输出包括：

- win rate
- average cost
- average questions
- average confidence
- average information gain
- budget efficiency
- failure reasons

比较全部策略：

```bash
python simulation.py --episodes 1000 --benchmark
```

- **Strategy A — Fixed Budget Exploration**：把搜索预算按预期问题数平均分配。
- **Strategy B — Current Race Strategy**：使用当前 Early/Mid/Late 动态阶段和预算桶。
- **Strategy C — Aggressive Search**：降低问题价值门槛、保留更少终局预算，并追求更高提交置信度。
- **Strategy D — Adaptive Race Strategy**：根据剩余预算、有效候选、置信度和进度动态选择探索、优化或终局。

默认种子下的一次 1000 局基准示例：

| Strategy | Win rate | Avg cost | Avg questions | Avg confidence | Budget efficiency |
|---|---:|---:|---:|---:|---:|
| Strategy A | 92.90% | 0.003368 | 3.29 | 93.83% | 275.81 |
| Strategy B | 91.40% | 0.003607 | 3.53 | 94.03% | 253.38 |
| Strategy C | 98.80% | 0.004203 | 4.11 | 99.00% | 235.05 |
| Strategy D | 92.90% | 0.003368 | 3.29 | 93.83% | 275.81 |

这些统计可用于搜索更优的 `budget allocation`、`question selection` 和 `submission threshold`。结果取决于候选规模、Judge 准确率、未知回答概率和成本模型，不应直接视为真实比赛胜率。

## Tournament Evaluation

The tournament layer evaluates multiple agents under the same budget, candidate set, Judge model, episode count, and deterministic seed. It compares:

- **Agent A — Random Search**: randomly selects non-repeated candidate partitions without optimizing information efficiency.
- **Agent B — Fixed Budget**: spreads the searchable budget across a fixed expected question count.
- **Agent C — Adaptive Strategy**: changes exploration, optimization, and finish behavior from live race state.
- **Agent D — Adaptive + Memory**: applies historical question and decision performance to the adaptive strategy.

Run the public 10000-episode benchmark:

```bash
python tournament.py --episodes 10000 --output tournament_report.json
```

The leaderboard is ranked by win rate, then budget efficiency, then lower average cost. `tournament_report.json` records the complete parameters, strategy versions, generation time, wins, win rate, average cost, average questions, and budget efficiency for reproducibility.

Export an auditable Markdown performance report for the competition agent:

```bash
python export_report.py \
  --input tournament_report.json \
  --output AGP_AGENT_PERFORMANCE_REPORT.md \
  --strategy "Adaptive + Memory"
```

The generated report contains the selected Agent benchmark, the complete tournament leaderboard, and the parameters needed to interpret the result. Tournament results are statistical simulations rather than guaranteed real AGP outcomes.

## 测试 Policy Engine

测试仅使用 Python 标准库：

```bash
python -m unittest discover -v
```

测试覆盖超预算拒绝、低价值拒绝、高价值放行、审计日志持久化、重复问题、低置信度猜测、Race 阶段切换、预算隔离、Runtime 问答、批量模拟与最终提交。

## 常用配置

| 环境变量 | 默认值 | 作用 |
|---|---:|---|
| `AGP_BUDGET_USDC` | `0.25` | 单局总预算 |
| `AGP_SUBMISSION_RESERVE_USDC` | `0.005` | 为最终提交保留的预算 |
| `AGP_CONFIDENCE_THRESHOLD` | `0.92` | 达到后停止提问 |
| `AGP_CATEGORY_CONFIDENCE_THRESHOLD` | `0.85` | 从类别阶段切换到二分阶段 |
| `AGP_JUDGE_ACCURACY` | `0.98` | 贝叶斯更新采用的裁判可靠度 |
| `AGP_MAX_QUESTIONS` | `12` | 硬性提问上限 |
| `AGP_QUESTION_BASE_COST_USDC` | `0.001` | 每次提问固定成本 |
| `AGP_INPUT_USDC_PER_1K_TOKENS` | `0.001` | 输入 token 单价 |
| `AGP_OUTPUT_USDC_PER_1K_TOKENS` | `0.003` | 输出 token 单价 |
| `AGP_MIN_INFORMATION_GAIN_BITS` | `0.03` | 最低可接受信息增益 |
| `AGP_MIN_VALUE_PER_USDC` | `10.0` | 最低 bits/USDC |
| `AGP_MIN_EXPECTED_SUCCESS_GAIN` | `0.0` | 最低预期成功率提升；Policy 会继续约束低置信度提交 |
| `AGP_POLICY_AUDIT_LOG` | 空 | 可选的 JSONL 审计日志路径 |

比赛前应按真实 AGP/Latch 计费更新单价。Agent 会用 `max_tokens` 做保守的提问前估价，并优先保留最终提交预算；如果服务端返回 `cost_usdc`，账本会以实际值为准。

## 参赛运行环境

Agent 不依赖数据库或复杂框架，可在支持 Python 3.11+ 的比赛容器或虚拟机中运行。正式参赛时通过环境变量注入 AGP、Latch 和会话配置；敏感密钥不进入代码、日志或 Git 历史。
