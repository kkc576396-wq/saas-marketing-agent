# SmartPush Marketing Agent 交接文档

更新时间：2026-08-11（Asia/Shanghai）
项目目录：仓库根目录

## 1. 当前结论

这是一个面向 SHOPLINE SmartPush 的 Research → Content → Reflection →
Memory 营销 Agent。完整主链使用 `LangGraph StateGraph(MarketingState)`；
Research 是有界 ReAct 状态循环，Content 是运行在 LangGraph 中的
Plan-and-Solve 流程。

目前已完成：

- Rewriter 同时识别 Research 目标和 Content 交付意图；
- Research Agent 多来源检索、验证、评分、Top 5 筛选；
- Content Planner + Executor 自循环和 `execution_history`；
- 产品、受众、品牌、平台、合规 RAG；
- Reflection 风险门控、两模型 CoVe、修订回流；
- 英文内容硬合同、Reddit Top 5 独立回复、时间窗口硬过滤；
- 三级记忆：短期 State、中期 SQLite、长期 Brand RAG；
- 本地网页完整链路展示。

当前没有硬性运行阻塞。主要后续工作是长期品牌资产候选的审核/晋升
界面、自动发布、内容效果反馈，以及清理不可移植的 `requirements.txt`。

最新验证结果：

```text
131 passed in 15.76s
Python compileall passed
web/static/app.js syntax passed
git diff --check passed
```

本地页面已经重启并加载当前代码：

```text
http://127.0.0.1:8001
```

同一项目还有另一个 8000 服务，未在本轮中停止或修改。

## 2. 当前完整架构

```text
START
  ↓
planning
  ├── planner_node
  ├── query_rewriter_node
  └── source_router_node
  ↓
  ├──────────── research_agent ↔ tools ─────── research_done ──┐
  ├──────────── rag_prefetch ──────────────────────────────────┤
  └──────────── memory_prefetch ───────────────────────────────┤
                                                               ↓
                                                          evaluation
                                                               ↓
                                            ┌──────────────────┴──────────────┐
                                            │                                 │
                                      research_only                    content_planner
                                            │                                 ↓
                                            │                         content_executor
                                            │                    select/organize ↻ write
                                            │                                 ↓
                                            │                         draft_checkpoint
                                            │                                 ↓
                                            │                     reflection_risk_gate
                                            │                                 ↓
                                            │              reflection_question_planner
                                            │                                 ↓
                                            │               reflection_verification
                                            │                    ↓ pass      ↓ revision
                                            │                    │      content_executor
                                            └────────────── memory_commit ←───┘
                                                               ↓
                                                              save
                                                               ↓
                                                              END
```

关键说明：

- `planning` 和 `evaluation` 是组合节点，内部按顺序调用多个 Python 阶段；
- Research 节点、工具循环、Content 节点、Reflection 节点、Memory 节点均由
  LangGraph 路由；
- Research、RAG、Memory Prefetch 是三个并行分支，在 `evaluation` 前汇合；
- `MarketingState` 是 Research 和 Content 的显式共享数据总线；
- LangChain 负责消息、模型封装、`bind_tools()` 和 `@tool`；
- Python Harness 负责格式、时效、Top 5、证据、超时和状态保护。

## 3. 模型分工

| 环节 | 当前模型 |
|---|---|
| Query Rewriter 首次识别 | `qwen3.7-flash` |
| Query Rewriter 条件修复 | 同一个 `qwen3.7-flash` |
| Research narrow / Reddit | `qwen3.7-flash` |
| Research broad / 竞品与市场 | `qwen3.7-plus` |
| Content Planner | `qwen3.7-flash` |
| Executor 选择/组织阶段 | `qwen3.7-flash` |
| Executor 普通最终写作 | `qwen3.6-plus` |
| Homepage / Competitor Report | `qwen3.6-plus` |
| Reflection Question Planner | `deepseek-v4-flash` |
| Verification | `qwen3.7-plus` |
| JSON Repair | `qwen-flash` |
| RAG / Memory Embedding | `qwen3.7-text-embedding` |

所有非写作节点通过共享模型加载器关闭 Thinking。模型 API Key 和 Base URL
统一复用 `.env` 中的 OpenAI-compatible 配置。

## 4. Rewriter 与意图合同

Rewriter 一次调用同时输出 Research 和 Content 意图，原始用户请求始终保存在：

```text
MarketingState.raw_user_request
```

Content 意图核心字段：

```text
requested
deliverable_type
deliverable_description
request_evidence
platform
language
audience
tone
requires_brand_rag
```

当前交付类型：

```text
homepage_promotion
reddit_promotion
reddit_reply
competitor_report
```

兼容旧状态时：

```text
competitor_research → competitor_report
research_only → requested=false
```

不要继续添加中文关键词规则来扩充意图识别。当前策略是 LLM 语义合同 + Python
结构一致性检查。只有输出内部矛盾、缺少证据摘录或 schema 不合法时，才在同一个
Rewriter 节点里调用一次相同的 `qwen3.7-flash` 修复。

## 5. Research Agent

Research Agent 绑定：

```text
anysearch_search
anysearch_batch_search
agent_reach_search(channel=reddit|web|rss)
```

执行特征：

- 普通 Research 最多 5 轮；
- Reddit Research 最多 2 轮；
- 一轮是一个工具调用批次，不是一个查询；
- 同一轮最多并行执行 5 个独立 Tool Call；
- Reddit/OpenCLI 最大并发为 2；
- 下一轮必须等待上一轮观察，因此多轮之间串行；
- narrow 和 broad 模型绑定同一套搜索工具，只是模型能力不同。

Research 输出原始文档后，`evaluation` 顺序执行：

```text
analyzer_node
→ verifier_node
→ scoring_node
→ opportunity_classifier_node
```

正式洞察必须满足：

```text
verification.passed == true
AND scoring.total_score >= 60
```

产品更新还必须具备明确发布证据、发布日期和足够的主题匹配分。

最终交给 Content 的集合：

- `eligible_insights`：最多 5 条；
- `alternative_insights`：最多 5 条；
- `rejected_insights`：保留兼容输出；
- 原始 `documents` 和完整 `tool_results` 不写入最终 JSON。

明确时间要求，例如“30 天内”，由 Python 在 Verification 阶段作为硬过滤执行，
不是交给 LLM 自觉遵守。

## 6. Content Agent：Plan-and-Solve

`content_planner` 一次 LLM 调用输出 2～3 个计划步骤；Python 将其收敛成两个
Executor 语义阶段：

```text
1. 选择并组织 Research、RAG、Memory 和约束
2. 生成最终完整内容
```

Executor 每次只完成当前一步，并把结果写入：

```text
execution_history
execution_artifacts
current_step_index
executor_iterations
```

Executor 当前不允许在写作阶段再次调用工具。它只能使用：

- 原始用户请求；
- 完整计划；
- 已完成步骤结果；
- verified Research；
- Prefetched Brand RAG；
- 中期 Memory；
- Reflection 返回的受控 revision steps。

内容硬合同：

- 所有对外交付内容必须为英文；
- `reddit_reply` 为最多 5 个经过验证且在时间窗口内的帖子分别生成回复；
- 前端格式为“一条原帖链接 + 一条可复制回复”；
- 没有验证到 5 个帖子时，只输出实际验证到的帖子，不能伪造；
- Homepage 和竞品报告允许一份完整长文；
- 不能编造 URL、日期、指标、产品能力、客户案例或竞品事实。

## 7. RAG 与长期品牌知识

现有知识库位于：

```text
knowledge/product/
knowledge/audience/
knowledge/brand/
knowledge/platforms/
knowledge/compliance/
```

RAG Prefetch 与外部 Research 并行，内部继续并行检索：

```text
public brand/platform/compliance
public product
internal audience
```

正式索引：

```text
data/rag/brand_index.json
```

重建命令：

```bash
venv/bin/python scripts/build_brand_rag_index.py --force
```

Internal-only 内容只能指导定位，不能直接出现在公开内容或支持公开事实声明。

## 8. Reflection / CoVe

完成初稿后先进入 `draft_checkpoint`，确保 Reflection 超时不会丢正文。

风险门控：

- 无数字、价格、竞品事实、产品能力、市场趋势的 Reddit 建议回复：轻量审查；
- 出现价格、指标、竞品、产品能力、市场趋势：完整两模型 CoVe；
- Homepage 和 Competitor Report：始终完整 CoVe。

完整 CoVe：

```text
Reflection Question Planner
→ 生成原子事实问题与质量问题
→ Verification 仅基于封闭 Research/RAG 证据回答
→ 生成 revision_steps
→ Executor revision mode
```

默认：

```text
max_reflection_iterations = 1
```

Reflection 超时会保存 checkpoint 正文并记录 warning，不会让完整任务失败。

## 9. 三级记忆

### 9.1 短期记忆

短期记忆就是一次运行内的 `MarketingState`：

```text
messages
research_tool_history
documents
execution_history
rag_prefetch_results
reflection_history
revision_history
final_content
```

它服务于当前任务，不自动成为下一次任务的可检索知识。

### 9.2 中期记忆

中期任务记忆存放于本地 SQLite：

```text
data/memory/memory.sqlite3
```

数据库在第一次真实写入时自动创建，不需要外部数据库服务。运行时文件和 WAL/SHM
文件已经加入 `.gitignore`。

核心实现：

```text
workflow/memory_manager.py
    SQLiteMemoryStore
    MemoryManager

workflow/memory_tools.py
    MemoryTool.execute()
    memory_add
    memory_search
    memory_forget

workflow/memory_nodes.py
    memory_prefetch
    memory_commit
```

Memory Tool 使用方式：

```python
memory_tool.execute("add", ...)
memory_tool.execute("search", ...)
memory_tool.execute("forget", ...)
```

Search 先执行品牌、用户、namespace、状态、有效期过滤，再使用
Embedding + 词法回退排序。返回的中期记忆明确标记：

```text
not_fact_evidence = true
```

它只能指导用户偏好和历史任务策略，不能支持市场、竞品、价格、趋势、指标或产品
能力声明。

`memory_commit` 在最终 `save` 前写入一条紧凑、默认 180 天过期的任务经历。
Memory 读取或写入失败只记录状态，不影响正文保存。

### 9.3 遗忘策略

支持：

```python
memory_tool.execute(
    "forget",
    strategy="importance_based",
    threshold=0.2,
)

memory_tool.execute(
    "forget",
    strategy="time_based",
    max_age_days=30,
)

memory_tool.execute(
    "forget",
    strategy="capacity_based",
    threshold=0.3,
)
```

容量策略未传 `max_records` 时，默认读取：

```text
MEMORY_MAX_ACTIVE_PER_BRAND=5000
```

遗忘行为是软删除：

```text
status = forgotten
```

同时写入 `memory_events` 审计记录。支持 `dry_run=true` 预览。Pinned 记录不会被
自动遗忘。

### 9.4 长期记忆

批准后的长期品牌资产继续从 Brand RAG 读取。普通 `memory_add` 如果指定
`memory_layer="long_term"`，会被强制保存为 SQLite 候选：

```text
status = candidate
approved_for_external_use = false
```

长期候选必须包含 `source_refs`。自动 importance/time/capacity forgetting 只允许
清理候选，不会删除批准后的 Brand RAG 资产。

当前安全边界：Agent 不能自动把候选晋升为正式品牌资产。下一阶段可实现候选审核
界面，审核通过后生成知识 Markdown 并重建 Brand RAG。

## 10. Memory 在图中的数据流

```text
planning
  ↓
memory_prefetch ─────────────┐
research_agent/tools ────────┼── evaluation → planner → executor
rag_prefetch ────────────────┘
                                      ↓
                                memory_commit
                                      ↓
                                     save
```

Planner 和 Executor 都会收到：

```text
medium_term_memory.status
medium_term_memory.results
medium_term_memory.errors
medium_term_memory.usage_boundary
```

Reflection 仍然只用 Research/RAG 支持事实判断，不使用中期记忆作为事实来源。

## 11. 重要配置

`.env.example` 已包含：

```text
MEMORY_ENABLED=true
MEMORY_DB_PATH=data/memory/memory.sqlite3
MEMORY_DEFAULT_BRAND_ID=smartpush
MEMORY_DEFAULT_NAMESPACE=marketing
MEMORY_MAX_ACTIVE_PER_BRAND=5000
MEMORY_PREFETCH_TOP_K=5
MEMORY_TASK_RETENTION_DAYS=180
MEMORY_EMBEDDING_ENABLED=true
MEMORY_EMBEDDING_MODEL=qwen3.7-text-embedding
MEMORY_EMBEDDING_TIMEOUT_SECONDS=15
```

当前实际 `.env` 即使尚未写入这些字段，代码也有相同默认值。不要把 `.env` 或任何
API Key 写进交接文档或提交到 Git。

## 12. 前端状态

本地网页现在展示：

- `5 stages ready`；
- Memory 与 RAG/Research 的并行状态；
- Memory Prefetch / Commit 进度；
- 最终结果中的 Memory 状态卡；
- Reddit 每条链接和回复的独立复制卡片。

Competitor 示例提示已经统一要求英文报告，避免与全局英文合同冲突。

启动：

```bash
venv/bin/python -m web.app --host 127.0.0.1 --port 8001
```

## 13. 测试和验证

运行全量测试：

```bash
venv/bin/python -m pytest -q
```

当前结果：

```text
131 passed in 15.76s
```

新增 Memory 测试覆盖：

- Add 与精确去重；
- 品牌/用户作用域隔离；
- Embedding/词法检索；
- 长期新增强制 candidate；
- 已批准 Brand RAG 搜索；
- importance-based soft forget；
- time-based soft forget；
- capacity-based soft forget；
- Memory Prefetch 和 Commit；
- Memory Graph 并行汇合；
- Memory Tool schema。

受限沙箱中 HTTP 测试可能因为禁止绑定随机 localhost 端口而报
`PermissionError: [Errno 1] Operation not permitted`。这是测试环境权限，不是业务
代码失败；在允许本地端口的环境运行即可。

## 14. 关键文件

```text
workflow/marketing_graph.py
    完整 Marketing LangGraph

workflow/state.py
    ResearchState / MarketingState

workflow/query_rewriter.py
    Research + Content 联合意图识别

workflow/content_intent.py
    deliverable_type 规范化与旧状态兼容

workflow/research_agent.py
    Research ReAct、模型路由、并行工具批次

workflow/research_graph.py
    Planning、Evaluation、Save 和最终输出合同

workflow/content_planner.py
    Plan-and-Solve Planning Phase

workflow/content_executor.py
    Executor 自循环、execution_history、英文/Reddit 合同

workflow/content_tools.py
    Brand RAG Prefetch

workflow/rag_store.py
    Brand RAG 索引与检索

workflow/reflection.py
    风险门控、问题规划、Verification、Revision

workflow/memory_manager.py
    SQLiteMemoryStore / MemoryManager

workflow/memory_tools.py
    MemoryTool 和 LangChain Tools

workflow/memory_nodes.py
    Memory Prefetch / Commit Graph Nodes

prompts/
    Rewriter、Research、Planner、Executor、Reflection Prompts

web/app.py
    本地 HTTP 服务、任务队列和进度流

web/static/
    完整链路网页

tests/test_memory.py
    Memory 单元和节点测试
```

## 15. 已知待办

按优先级建议：

1. 为长期资产候选增加审核、批准、撤回和 RAG 重建界面；
2. 增加用户身份/品牌空间管理，避免未来多用户共享默认作用域；
3. 增加内容发布适配器和人工确认门；
4. 收集发布后的效果反馈，并形成受控的中期记忆；
5. 将当前庞大的 Conda `requirements.txt` 整理为最小可移植依赖；
6. 审核当前大量未跟踪文件后建立一次基线提交。

## 16. 不要重踩的边界

- 不要用固定关键词词表代替 Rewriter 的语义意图识别；
- 不要把 `competitor_report` 改回 `competitor_research`；
- 不要让 Executor 在 RAG Prefetch 后再次自由调用搜索工具；
- 不要把中期 Memory 当作当前事实证据；
- 不要让普通 LLM 自动批准长期品牌资产；
- 不要绕过明确的 30 天等时间硬过滤；
- 不要把一个 Reddit 用户体验扩大成市场普遍结论；
- 不要因为 Reflection 超时丢弃 checkpoint 正文；
- 不要把 Router 推荐来源写成实际调用来源；
- 不要把 5 次 Research 迭代理解为 5 个查询；
- 不要强制 `tool_choice=required`；
- 不要提交 `.env`、API Key、SQLite 运行库、日志或输出 JSON；
- 不要运行 `git reset --hard` 或 `git clean -fd`，当前工作区包含大量真实成果。

## 17. 新任务开始时

```bash
cd /path/to/saas-marketing-agent
source venv/bin/activate
git status --short
venv/bin/python -m pytest -q
lsof -nP -iTCP:8001 -sTCP:LISTEN
```

建议优先阅读：

1. `HANDOFF.md`
2. `workflow/README.md`
3. `workflow/marketing_graph.py`
4. `workflow/state.py`
5. `workflow/memory_manager.py`
6. `workflow/memory_nodes.py`
7. `workflow/content_executor.py`
8. `workflow/reflection.py`
