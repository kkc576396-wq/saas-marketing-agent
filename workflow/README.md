# Workflow

Contains the LangGraph state, nodes, edges, and graph assembly for the
SmartPush Research → Plan-and-Solve Content workflow.

Current flow:

```text
planning -> research_agent <-> tools ---- research_done --┐
     |     (max 5 general / max 2 Reddit rounds)           |
     └--------------> rag_prefetch ------------------------┤
     └--------------> memory_prefetch ---------------------┤
                                                          v
                                                     evaluation
                         |
             content requested?
                  /             \
                no              yes
                |                |
       memory_commit      content_planner
                |                 |
              save                |
                                  |
                           content_executor
                    select/organize -> final writing
                                  |
                           plan complete
                                  |
                          draft_checkpoint
                                  |
                        reflection_risk_gate
                       /                    \
                light review             full CoVe
                       \                    /
                  reflection_question_planner
                                  |
                  reflection_verification
                        /                 \
                      pass          revision_steps
                       |                  |
              memory_commit    content_executor (revision)
                     |                    |
                   save           bounded recheck
                                          |
                                  bounded recheck / save
```

`research_agent` is an LLM reasoning node bound to AnySearch and the unified
Agent-Reach tool. `tools` executes independent calls concurrently, stores full
normalized evidence, and returns compact observations to the model. The
evaluation stage remains deterministic: analyzer, verifier, scoring, and
opportunity classification run before the Top 5 plus five alternatives are
passed downstream.

`rag_prefetch` runs concurrently with external Research and retrieves the
relevant product, audience, brand, platform, and compliance context before the
branches join at evaluation. `memory_prefetch` is the third parallel branch and
reads scoped medium-term episodes from local SQLite. These episodes may guide
preferences and prior task strategy but are never current factual evidence.
`content_planner` creates a 2～3 step plan in one
LLM call and Python consolidates it into two Executor phases. The Rewriter,
narrow Research, Planner, and non-writing Executor steps use `qwen3.7-flash`;
broad market/competitor Research and Verification use `qwen3.7-plus`.
The Rewriter emits `requested`, `deliverable_type`,
`deliverable_description`, and an exact grounded `request_evidence` excerpt.
Python performs structural consistency checks rather than expanding a content
keyword list. Only inconsistent output invokes one bounded repair call using
the same `qwen3.7-flash` Rewriter model, within the same graph node. The current competitor artifact enum is
`competitor_report`; the retired `competitor_research` value is accepted only
as a persisted-state alias.
`content_executor` first selects/organizes Research and prefetched RAG, then
writes the final deliverable. It records both results in `execution_history`.
Normal final writing, homepage copy, and competitor long-form writing all use
`qwen3.6-plus`. The Executor does not initiate another
RAG tool round. The graph, not the LLM, advances the step index.

Generated deliverables have an English-only output contract. `reddit_reply`
uses up to five verified, in-window Reddit targets and returns a structured
`replies` list containing the exact post URL and one independently copyable
reply per target. Explicit freshness windows are deterministic hard filters in
Verification, not suggestions left to either Research or Content models.

After the plan completes, `draft_checkpoint` persists the usable content before
review. `reflection_risk_gate` sends fact-free Reddit advisory replies through
one lightweight quality review. Numbers, pricing, competitor facts, product
capabilities, market trends, homepage copy, and competitor reports use full
two-model CoVe. `reflection_question_planner` uses `deepseek-v4-flash` to extract atomic
verification questions and quality issues
without answering them. `reflection_verification` uses `qwen3.7-plus` with a
closed Research/RAG evidence pack. Python validates every cited ID and creates
safe fallback revisions for unsupported claims. The Executor applies revisions
in a separate mode with no tool access or new evidence. Reflection defaults to
one review round. All configured graph models run with Thinking
disabled. Malformed structured responses receive one bounded `qwen-flash` JSON
repair attempt. A Reflection timeout routes directly to final save with a review
warning, preserving the checkpointed draft.

`memory_commit` runs before final save and writes one compact, expiring task
episode through `MemoryManager`. Additions to the long-term layer are stored as
unapproved candidates; approved brand assets continue to be read from the
curated RAG. Automated importance, time, and capacity forgetting soft-deletes
only eligible SQLite records, preserving audit events and protecting pinned or
approved long-term assets.
