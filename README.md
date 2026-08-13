# AI SaaS Marketing Research Agent

Initial Python and LangGraph project structure for an AI agent that supports SaaS marketing research.

## Project structure

```text
.
├── agents/      # Agent definitions and role boundaries
├── tools/       # Research tools and external integrations
├── workflow/    # LangGraph state and graph composition
├── prompts/     # System prompts and reusable prompt templates
├── data/        # Research output and generated local RAG index
├── knowledge/   # Curated product, audience, brand, platform, compliance RAG
├── scripts/     # Index maintenance commands
├── docs/        # Architecture and project documentation
├── tests/       # Automated tests
├── .env.example # Environment variable template
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

The current LangGraph workflow connects bounded Research to a Plan-and-Solve
Content Agent with Reflection and three-level memory. Automated publishing is
still a separate later phase.

## Agent-Reach integration

Agent-Reach is installed from the official GitHub source into the project
virtual environment. Its official role is to install, diagnose, and route to
upstream channel tools; it is not a unified search API. The LangGraph wrapper
is in `tools/agent_reach_tool.py` and supports at most 5 queries per call.

Install and check channel availability:

```bash
venv/bin/pip install --upgrade \
  https://github.com/Panniantong/agent-reach/archive/main.zip
venv/bin/python3 venv/bin/agent-reach install --env=auto --safe
venv/bin/python3 venv/bin/agent-reach doctor --json
```

Use the tools in a LangGraph agent:

```python
from langgraph.prebuilt import create_react_agent
from tools.agent_reach_tool import AGENT_REACH_TOOLS

research_agent = create_react_agent(model, tools=AGENT_REACH_TOOLS)
```

Agent-Reach and AnySearch remain separate research providers. The existing
AnySearch integration is unchanged.

## SmartPush multi-source research architecture

The research layer is routed specifically for SHOPLINE SmartPush: email
automation, segmentation, lifecycle marketing, e-commerce CRM, retention, and
North American Shopify/DTC merchants.

```text
planning
   ├── initialize ResearchState
   ├── translate and rewrite the request in English
   ├── detect entities and intent facets
   └── provide a non-binding source recommendation
   ↓
research_agent (Reason)
   ├── inspect the goal and compact prior observations
   ├── identify the next evidence gap
   └── select one or more tools
   ↓
tools (Act / Observe)
   ├── AnySearch                 market intelligence and source discovery
   ├── Agent-Reach Reddit        merchant pain points and sentiment
   ├── Agent-Reach Web           extraction from discovered URLs
   └── Agent-Reach RSS           extraction from known feed URLs
   └── normalize full results while returning compact observations to the LLM
   ↓
research_agent again, or evaluation after completion / max_iterations
   ↓
evaluation
   ├── semantically deduplicate documents
   ├── verify relevance, evidence, bias, and freshness
   ├── score SmartPush business opportunity
   └── classify opportunity type and channels
   ↓
save
   └── ranked Top 5 plus five verified alternatives
```

The retrieval layer is a bounded, tool-using ReAct loop. The model chooses
tools after observing real results instead of following a fixed Python source
schedule. Multiple independent tool calls from one model response execute in
parallel; URL-dependent Web/RSS extraction happens in a later round. The
router remains an initial recommendation and no longer prevents the model from
using another approved source when evidence requires it.

The state records router advice in `recommended_sources`; `selected_sources`
contains only providers the Research Agent actually called. This keeps routing
diagnostics separate from retrieval evidence.

The harness still controls execution: only allowlisted tools are bound, one
tool batch counts as one search round, `max_iterations=5` is enforced for
general research, Reddit research is capped at two rounds, Reddit/OpenCLI
concurrency is capped at two, and full provider payloads
are normalized before deterministic evaluation. The LLM receives only compact
observations to control context growth. Verification, scoring, classification,
eligibility, and the final JSON contract remain outside the ReAct loop.

The configured model must support OpenAI-compatible `tool_calls`. Research
uses automatic tool selection with Thinking disabled; do not force
`tool_choice=required` on the compatible endpoint.

Query rewriting uses a hybrid strategy. When configured, an LLM translates
the request and generates platform-specific queries. Reddit queries are then
validated as short keyword phrases, while the deterministic competitor
templates are used only to fill missing or invalid results. The templates are
fallbacks, not a vocabulary restriction. Set `QUERY_REWRITER_USE_LLM=1` to
also rewrite English input with the LLM; Chinese input is translated
automatically when `OPENAI_API_KEY` is available.

`QUERY_REWRITER_MODEL` reuses the same `OPENAI_API_KEY` and `OPENAI_BASE_URL`
and defaults to `qwen3.7-flash`. Research dynamically routes narrow and Reddit
work to `RESEARCH_FAST_MODEL=qwen3.7-flash`, while broad market and competitor
work uses `RESEARCH_BROAD_MODEL=qwen3.7-plus`.

The rewriter also emits validated `detected_entities` and `intent_facets`.
Entity-to-category relationships are re-derived from `workflow/domain_context.py`
rather than trusted directly from model output, while intent facets are limited
to a fixed enum and supplemented by deterministic fallback detection. Router,
verifier, scoring, and opportunity classification consume the same competitor
knowledge without adding another model call.

Time-sensitive requests such as trends, latest updates, news, and pricing are
anchored to `research_date` with a default 365-day freshness window. Generated
queries targeting an older year are corrected unless the user explicitly asks
for that year or historical research.

An explicit request such as `搜索30天内` or `within the last two weeks` overrides
that default. The Rewriter stores the exact window in state, adds an absolute
cutoff date to search queries, and Verification rejects every result older than
the cutoff before scoring. Old posts therefore cannot reach the Content Agent.

Provider responses remain available in the in-memory workflow state while a
run is active, but raw `tool_results` and documents are no longer persisted in
the final JSON. Reddit YAML, AnySearch Markdown, and Agent-Reach Web/RSS JSON
are converted into documents with a stable ID, title, summary, URL, and
publication time. Duplicate documents returned across search iterations are
removed before verification and scoring.

Search concurrency stays inside the tools stage. Independent calls can run in
parallel, while Reddit is limited to two workers because OpenCLI shares a
browser bridge. Web/RSS extraction receives only exact URLs and therefore
normally follows discovery in a later ReAct round.

Verified insights are scored for SmartPush prioritization with:

```text
total_score = 0.25 topic alignment
            + 0.20 business relevance
            + 0.25 customer pain
            + 0.20 content opportunity
            + 0.10 freshness
```

`topic_alignment_score` measures whether an insight answers the current user
query and its structured intent facets, rather than merely belonging to the
SmartPush product category. Known-competitor bonuses are capped at 15 relevance
points and 5 content-opportunity points for explicit competitor research. In
broad market or trend research, a competitor mention contributes at most a
small auxiliary signal, preventing brand-heavy pages from dominating ranking.

Freshness is calculated from normalized source dates rather than words such as
"latest" or "2026" in the title. Documents retain `published_at`,
`retrieved_at`, `date_status`, `date_confidence`, page type, and source bias in
the in-memory state. Undated articles receive a low freshness score; official
pricing and product pages may use a clearly labeled current-page snapshot.

The verifier applies hard gates before scoring:

- product and competitor relevance use weighted SmartPush-specific signals;
- known competitors such as Klaviyo, Omnisend, and Mailchimp satisfy product
  relevance through a deterministic entity-to-category mapping even when a
  community post does not repeat the phrase `email marketing`;
- evidence requires a valid URL, meaningful summary, and claim support;
- time-sensitive requests require a freshness score of at least 60;
- promotional content is rejected;
- `product_update_research` requires an explicit launch, release, changelog,
  rollout, shipped-feature, or `what's new` signal plus a publication date;
- generic reviews, comparisons, migrations, and best-tool lists do not satisfy
  product-update evidence merely because their body mentions new features;
- broad trend and market claims require corroboration from another domain;
- individual community experiences do not require cross-domain corroboration,
  but remain explicitly scoped as a single-user observation;
- vendor-authored sources are labeled separately from independent and
  community sources.

Verification output includes relevance, evidence, source-quality and freshness
scores, source bias, independent-domain count, confidence, and explicit
rejection reasons.

For `product_update_research`, downstream eligibility also requires
`topic_alignment_score >= 60`. This intent-specific gate prevents pain,
business-relevance, and content-opportunity scores from promoting an off-topic
review or migration article above an actual dated product release.

The competitor mapping is local and shared across routing, verification, and
opportunity classification. Ambiguous names such as Drip require email or
e-commerce context. This improves recall without adding a LangGraph node,
model call, embedding, or external request.

Community findings also include `claim_type=community_observation`,
`claim_scope=single_user_experience`, and downstream usage constraints. This
keeps authentic merchant feedback available while preventing the Content
Agent from presenting one person's experience as a market-wide conclusion.

After scoring, verified insights are classified into SmartPush opportunity
types and mapped to recommended channels. This layer only produces structured
research metadata; it does not generate or publish content.

## Research output contract

`data/research_output.json` exposes two bounded downstream collections:

- `eligible_insights`: verified insights with `verification.passed == true` and
  `scoring.total_score >= 60`, sorted by `total_score` descending and capped at
  the top 5.
- `alternative_insights`: the next five highest-scoring, verified insights
  with valid evidence. These are backups and are not sent automatically to the
  Content Planner and Executor.
- `rejected_insights`: backwards-compatible alias of `alternative_insights`.
  Unverified and lower-priority records are omitted from the persisted JSON.

Each eligible insight contains:

```text
insight_id, title, summary, source_type, sources,
verification, scoring, opportunity_type, matched_signals,
recommended_channels
```

The legacy `insights` field is an alias of `eligible_insights`. The Content
Agent consumes the bounded verified collections and never raw provider
payloads. The output also includes aggregate retrieval counts.

## Content Agent Plan-and-Solve and brand RAG

The marketing graph continues after Research evaluation when the Rewriter has
detected a content-generation intent:

Content intent is a semantic LLM contract, not a growing keyword classifier.
`requested` records whether a downstream artifact is required;
`deliverable_type` records the normalized artifact and uses
`competitor_report` for competitor-facing reports. The model must preserve an
exact `request_evidence` excerpt from the original message. Python validates
the schema, excerpt grounding, and internal consistency only. Contradictory
output receives one conditional repair call using the same `qwen3.7-flash`
Rewriter model inside the same
Rewriter graph node; unresolved intent fails explicitly instead of silently
becoming research-only. Legacy `competitor_research` state is read through an
alias but is never written by new runs.

When extending intent behavior, prefer a semantic schema or consistency check
over adding vocabulary rules. Add production keywords only when a deterministic
protocol requirement cannot be represented semantically, and keep each change
to the smallest affected surface.

```text
planning
    ├── research_agent <-> tools ──┐
    └── rag_prefetch ──────────────┤
    └── memory_prefetch ───────────┤
                                   ↓
                              evaluation
                                   ↓
content_planner (qwen3.7-flash; 2～3 steps, consolidated to 2 phases)
    ↓
content_executor (qwen3.7-flash; select and organize prefetched context)
    ↓
final writing -> qwen3.6-plus
             └── homepage/competitor long-form -> qwen3.6-plus
                     ↓
                 draft_checkpoint
                     ↓
                 reflection_risk_gate
                    /                  \
     light Reddit review once      full two-model CoVe
                    \                  /
                 reflection_question_planner
                              ↓ deepseek-v4-flash
                     reflection_verification
                              ↓ qwen3.7-plus
                     pass ────────────────> memory_commit -> save
                     revision_steps -> content_executor (revision mode)
                                          ↓
                              revise once at most, then save
```

The Executor receives the immutable original request, complete plan, current
step, verified Research output, prefetched RAG, prior `execution_history`, and
accumulated artifacts. The graph owns step advancement. Normal execution has
only two semantic LLM steps: select/organize context, then final writing.

Every public deliverable is written in English, even when the instruction is
Chinese. Python validates the completed draft and makes one bounded correction
call if the writer violates this contract. For `reddit_reply`, the Executor
generates one directly copyable English reply for each of the first five
verified, fresh Reddit posts and preserves each exact post URL. It never invents
missing targets; if Research finds fewer than five eligible posts, it returns
only the verified targets that actually exist.

Reflection first applies a deterministic fact-risk gate. A Reddit advisory
reply with no numbers, pricing, competitor facts, product capability claims, or
market-trend claims receives one lightweight quality review. Pricing, metrics,
competitors, product capabilities, and market trends trigger full two-model
CoVe. Homepage copy and competitor reports always use full CoVe.

Full Reflection uses two isolated model calls. The Question Planner sees the full
draft but may only extract atomic factual questions and non-factual quality
issues. Verification does not receive the full draft; it receives those atomic
excerpts plus a closed pack of verified Research records and exact public RAG
evidence. A completed draft is persisted before Reflection starts. If either
Reflection call times out, the graph records a review warning and saves the
checkpointed draft instead of failing the whole content job.
Verification may cite RAG chunks previously retrieved during execution. A
supported factual verdict must
cite an allowlisted ID, otherwise Python downgrades it to insufficient
evidence. Internal-only RAG cannot support public claims.

Verification emits bounded `revision_steps`. The existing Executor switches to
revision mode, cannot call tools or introduce new evidence, and returns the
complete revised deliverable after each step. The default
`max_reflection_iterations=1` permits at most one review/revision cycle.

Curated knowledge lives under `knowledge/` and is split into product,
audience, brand, platform, and compliance corpora. Each document declares its
authority, visibility, external-use permission, source, retrieval date, and
usage constraints. Confidential ICP material is reduced to audience strategy
and is not eligible as public evidence.

Build the local index with the configured OpenAI-compatible endpoint:

```bash
venv/bin/python scripts/build_brand_rag_index.py --force
```

The required default model is `qwen3.7-text-embedding`. The generated JSON
index is local and ignored by Git; rebuild it after changing knowledge files.
The embedding endpoint used by this project accepts at most 20 inputs per
batch, which is enforced by the client.

## Three-level memory

The current run's `MarketingState` is short-term working memory. Medium-term
task episodes are stored in the local SQLite file
`data/memory/memory.sqlite3`; approved long-term brand assets remain in the
curated Brand RAG. The graph starts `memory_prefetch` in parallel with external
Research and RAG, supplies those records to Planner and Executor as preferences
and prior task strategy, and runs `memory_commit` immediately before the final
save. Medium-term memory is explicitly not factual evidence.

`MemoryManager` implements scoped add, hybrid search, and recoverable soft
forget. `MemoryTool.execute()` provides the Python dispatcher, while the
separate LangChain tools are `memory_add`, `memory_search`, and
`memory_forget`. Long-term additions are forced to `candidate` status and need
the existing brand-asset review and RAG rebuild process before public use.

Supported forgetting policies are:

```python
memory_tool.execute("forget", strategy="importance_based", threshold=0.2)
memory_tool.execute("forget", strategy="time_based", max_age_days=30)
memory_tool.execute(
    "forget",
    strategy="capacity_based",
    threshold=0.3,
    max_records=5000,
)
```

Time-based forgetting uses the last access time when present. Capacity-based
forgetting removes the least important eligible records only until the scoped
limit is restored. Pinned records and approved Brand RAG assets are protected.
All automated deletion is a recoverable status change to `forgotten`; event
records remain in SQLite for audit.

Eligible insights must also contain normalized evidence: a human-readable title
and summary, at least one valid source URL, and no raw JSON/YAML provider
payload in the title or summary.

## Local research web interface

The project includes a local browser interface for running the complete real
Research Agent without entering terminal commands for every query. It keeps
`max_iterations=5` for general research and caps Reddit research at two rounds,
uses the existing AnySearch and Agent-Reach integrations,
and displays live workflow status followed by the router decision, retrieval
counts, Top 5 eligible insights, score breakdowns, source URLs, and five
alternatives.

Reddit reply results are rendered as separate cards: one clickable source link
plus one reply with its own copy button. Other content types remain single
deliverables.

Start it from the project directory with the virtual environment activated:

```bash
venv/bin/python -m web.app
```

Then open `http://127.0.0.1:8000`. Research jobs are executed one at a time
because Agent-Reach Reddit shares the local OpenCLI browser bridge. The web
server uses only the Python standard library and does not expose `.env` values
to the browser.

### Publish to Feishu Docs

The MVP web UI can publish competitor reports, homepage marketing articles,
research reports, and other non-Reddit final deliverables directly to a Feishu
document. The final content is sent to the local Feishu CLI as Markdown. Reddit
posts and comments remain copy-only for manual posting.

Install the project-local CLI once:

```bash
npm install --prefix tools/feishu-cli
```

The CLI reuses the machine-level login under `~/.lark-cli`; no Feishu app secret
is stored by this web application. Verify the existing login with:

```bash
tools/feishu-cli/node_modules/@larksuite/cli/bin/lark-cli auth status --json --verify
```

The bundled Feishu authoring references are installed under `.agents/skills`.
`FEISHU_CLI_PATH`, `FEISHU_CLI_IDENTITY` (`bot` or `user`), and
`FEISHU_CLI_TIMEOUT_SECONDS` are optional overrides. Restart the web server
after changing them.
