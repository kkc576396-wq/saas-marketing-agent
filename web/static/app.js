const form = document.querySelector("#research-form");
const topicInput = document.querySelector("#topic");
const submitButton = document.querySelector("#submit-button");
const progressSection = document.querySelector("#progress-section");
const resultsSection = document.querySelector("#results-section");
const errorSection = document.querySelector("#error-section");
const stageLabel = document.querySelector("#stage-label");
const stageDetail = document.querySelector("#stage-detail");
const elapsedTime = document.querySelector("#elapsed-time");
const sourcePreview = document.querySelector("#source-preview");
const liveSourceChips = document.querySelector("#live-source-chips");

let pollTimer = null;
let finalContentText = "";
let currentJobId = "";
let feishuConfigured = false;

const stageDescriptions = {
  queued: "任务已进入本地执行队列。",
  planning: "Rewriter 正在同时识别 Research 目标与 Content 生成意图。",
  research_agent: "Research Agent 正在检查证据缺口并决定下一批工具调用。",
  tools: "正在通过 AnySearch 或 Agent-Reach 获取真实外部资料。",
  rag_prefetch: "正在与外部 Research 并行预取品牌、产品、平台和合规知识。",
  memory_prefetch: "正在并行读取 SQLite 中的中期任务经验与用户偏好。",
  research_done: "Research 已完成，正在等待并行 RAG 分支汇合。",
  evaluation: "正在去重、验证、评分并筛选可交给 Content Agent 的洞察。",
  content_planner: "Qwen 正在生成 2～3 步计划，并收敛为两个 Executor 阶段。",
  content_executor: "Executor 正在执行选择/组织或最终写作阶段。",
  draft_checkpoint: "已保存可恢复正文，接下来进入事实风险门控。",
  reflection_risk_gate: "正在根据数字、价格、竞品、产品能力和市场趋势选择审查强度。",
  reflection_question_planner: "DeepSeek 正在拆分事实声明与质量审查问题。",
  reflection_verification: "Qwen 正在仅依据封闭证据包核查声明并决定是否修订。",
  memory_commit: "正在把本次可复用任务经验以可恢复记录写入 SQLite。",
  save: "正在保存最终内容、执行轨迹与 Reflection 审计状态。",
  complete: "完整营销链路已经执行完成。",
};

const pipelineStages = [
  "planning",
  "research",
  "rag",
  "evaluation",
  "content",
  "execute",
  "reflection",
  "save",
];

const stageToPipeline = {
  queued: "planning",
  planning: "planning",
  research_agent: "research",
  tools: "research",
  rag_prefetch: "rag",
  memory_prefetch: "rag",
  research_done: "research",
  evaluation: "evaluation",
  content_planner: "content",
  content_executor: "execute",
  draft_checkpoint: "save",
  reflection_risk_gate: "reflection",
  reflection_question_planner: "reflection",
  reflection_verification: "reflection",
  memory_commit: "save",
  save: "save",
  complete: "save",
};

function text(value) {
  return String(value ?? "");
}

function setHidden(element, hidden) {
  element.classList.toggle("hidden", hidden);
}

function element(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = text(content);
  return node;
}

function formatElapsed(seconds) {
  const value = Math.max(0, Math.floor(Number(seconds) || 0));
  const minutes = String(Math.floor(value / 60)).padStart(2, "0");
  const remaining = String(value % 60).padStart(2, "0");
  return `${minutes}:${remaining}`;
}

function formatStatus(value) {
  return text(value || "unknown").replaceAll("_", " ");
}

function renderChips(container, items, tone = "default") {
  container.replaceChildren();
  (items || []).filter(Boolean).forEach((item) => {
    container.append(element("span", `chip chip-${tone}`, item));
  });
}

function updatePipeline(job) {
  const activeName = stageToPipeline[job.stage] || "planning";
  const activeIndex = pipelineStages.indexOf(activeName);
  document.querySelectorAll("[data-pipeline]").forEach((item) => {
    const itemIndex = pipelineStages.indexOf(item.dataset.pipeline);
    item.classList.toggle("active", itemIndex === activeIndex && job.stage !== "complete");
    item.classList.toggle("complete", itemIndex < activeIndex || job.stage === "complete");
    item.classList.toggle(
      "skipped",
      item.dataset.pipeline === "rag" && activeIndex > itemIndex && !job.rag_call_count,
    );
  });
}

function showProgress(job) {
  setHidden(progressSection, false);
  setHidden(resultsSection, true);
  setHidden(errorSection, true);
  stageLabel.textContent = job.stage_label || "Agent 正在执行";
  stageDetail.textContent = stageDescriptions[job.stage] || "正在处理当前状态。";
  elapsedTime.textContent = formatElapsed(job.elapsed_seconds);
  document.querySelector("#iteration-count").textContent = String(job.search_iterations || 0);
  document.querySelector("#step-count").textContent = String(job.content_plan_steps || 0);
  document.querySelector("#executor-count").textContent = String(job.executor_iterations || 0);
  document.querySelector("#rag-count").textContent = String(job.rag_call_count || 0);
  document.querySelector("#reflection-count").textContent = String(job.reflection_iterations || 0);
  const modeBadge = document.querySelector("#mode-badge");
  const revisionMode = job.executor_mode === "revision";
  modeBadge.textContent = revisionMode ? "REVISION MODE" : "PLAN MODE";
  modeBadge.classList.toggle("revision", revisionMode);
  updatePipeline(job);
  if (job.selected_sources?.length) {
    renderChips(liveSourceChips, job.selected_sources);
    setHidden(sourcePreview, false);
  } else {
    setHidden(sourcePreview, true);
  }
}

function metric(label, value, note) {
  const card = element("article");
  card.append(
    element("strong", "", value),
    element("span", "", label),
    element("small", "", note),
  );
  return card;
}

function statusCard(label, value, note, tone = "pass") {
  const card = element("article", `status-card ${tone}`);
  const header = element("div");
  header.append(element("span", "", label), element("strong", "", value));
  card.append(header, element("p", "", note));
  return card;
}

function scoreBar(label, value) {
  const row = element("div", "score-row");
  const caption = element("div");
  caption.append(
    element("span", "", label),
    element("strong", "", Math.round(Number(value) || 0)),
  );
  const track = element("div", "score-track");
  const fill = element("span");
  fill.style.width = `${Math.max(0, Math.min(100, Number(value) || 0))}%`;
  track.append(fill);
  row.append(caption, track);
  return row;
}

function sourceLink(source) {
  if (!source?.url) return null;
  const link = element("a");
  link.href = source.url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  let fallback = source.url;
  try {
    fallback = new URL(source.url).hostname;
  } catch (_) {
    fallback = source.url;
  }
  link.append(
    document.createTextNode(source.title || fallback),
    element(
      "small",
      "",
      source.published_at
        ? new Date(source.published_at).toLocaleDateString("zh-CN")
        : "日期未知",
    ),
  );
  return link;
}

function insightCard(insight, index, compact = false) {
  const card = element("article", compact ? "insight-card compact" : "insight-card");
  const rank = element("div", "rank", String(index).padStart(2, "0"));
  const body = element("div", "insight-body");
  const meta = element("div", "insight-meta");
  meta.append(
    element("span", "", formatStatus(insight.opportunity_type || "unclassified")),
    element("span", "", `${Math.round((insight.verification?.confidence || 0) * 100)}% 可信度`),
  );
  body.append(
    meta,
    element("h3", "", insight.title || "未命名洞察"),
    element("p", "", insight.summary || "暂无摘要"),
  );

  const sources = element("div", "source-links");
  (insight.sources || []).slice(0, 3).forEach((source) => {
    const link = sourceLink(source);
    if (link) sources.append(link);
  });
  if (sources.childElementCount) body.append(sources);
  const channels = element("div", "chips channel-chips");
  renderChips(channels, insight.recommended_channels || [], "soft");
  body.append(channels);

  const scorePanel = element("aside", "score-panel");
  const total = element("div", "total-score");
  total.append(
    element("strong", "", Math.round(insight.scoring?.total_score || 0)),
    element("span", "", "总分"),
  );
  scorePanel.append(total);
  if (!compact) {
    scorePanel.append(
      scoreBar("业务相关", insight.scoring?.business_relevance_score),
      scoreBar("客户痛点", insight.scoring?.customer_pain_score),
      scoreBar("内容机会", insight.scoring?.content_opportunity_score),
      scoreBar("时效", insight.scoring?.freshness_score),
    );
  }
  card.append(rank, body, scorePanel);
  return card;
}

function formatContent(value) {
  if (value === null || value === undefined || value === "") return "";
  if (typeof value === "string") return value;
  if (typeof value === "object") {
    if (Array.isArray(value.replies)) {
      return value.replies.map((item, index) => [
        `Reply ${index + 1}`,
        item.post_title || "Reddit post",
        item.post_url || "",
        "",
        item.reply || "",
      ].join("\n")).join("\n\n---\n\n");
    }
    const preferred = ["content", "report", "draft", "reply", "article"];
    for (const key of preferred) {
      if (typeof value[key] === "string") {
        const title = typeof value.title === "string" ? `${value.title}\n\n` : "";
        return `${title}${value[key]}`;
      }
    }
  }
  return JSON.stringify(value, null, 2);
}

function renderFinalContent(value) {
  const container = document.querySelector("#final-content");
  container.replaceChildren();
  const replies = Array.isArray(value?.replies) ? value.replies : [];
  container.classList.toggle("reddit-reply-list", Boolean(replies.length));
  if (!replies.length) {
    container.textContent = formatContent(value) || "本次没有生成 final_content。请检查 Content intent、Executor 和 Reflection 状态。";
    return;
  }

  replies.forEach((item, index) => {
    const card = element("article", "reddit-reply-card");
    const heading = element("div", "reddit-reply-heading");
    const title = element("div");
    title.append(
      element("span", "card-kicker", `REDDIT REPLY ${String(index + 1).padStart(2, "0")}`),
      element("strong", "", item.post_title || "Reddit post"),
    );
    const copy = element("button", "reply-copy-button", "复制这条回复");
    copy.type = "button";
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(item.reply || "");
        copy.textContent = "已复制";
      } catch (_) {
        copy.textContent = "复制失败";
      }
      setTimeout(() => { copy.textContent = "复制这条回复"; }, 1600);
    });
    heading.append(title, copy);
    const link = element("a", "reddit-post-link", item.post_url || "查看原帖");
    link.href = item.post_url || "#";
    link.target = "_blank";
    link.rel = "noreferrer noopener";
    const reply = element("div", "reddit-reply-text", item.reply || "");
    card.append(heading, link, reply);
    container.append(card);
  });
}

function renderAcceptance(data, job) {
  const container = document.querySelector("#acceptance-summary");
  const contentType = data.content_intent?.deliverable_type || data.content_intent?.type || "";
  const hasIntent = data.content_intent?.requested === true || Boolean(contentType && contentType !== "research_only");
  const hasPlan = data.content_planner_status === "planned";
  const executorOk = ["plan_completed", "revision_completed"].includes(data.executor_status);
  const reflectionOk = ["passed", "revision_applied_at_limit"].includes(data.reflection_status);
  const checkpointReady = data.draft_checkpoint_status === "saved";
  const memoryReady = ["saved", "disabled", "skipped"].includes(data.memory_commit_status);
  const finalReady = Boolean(formatContent(data.final_content));
  container.replaceChildren(
    statusCard("Intent", hasIntent ? "已穿透" : "仅调研", contentType || "not requested", hasIntent ? "pass" : "warn"),
    statusCard("Research", `${data.search_iterations || 0}/${data.max_iterations || 5}`, `${(data.eligible_insights || []).length} 条正式洞察`, "pass"),
    statusCard("Plan", hasPlan ? `${data.content_plan?.steps?.length || 0} steps` : "skipped", data.content_planner_status || "unknown", hasPlan ? "pass" : "warn"),
    statusCard("Executor", executorOk ? "完成" : formatStatus(data.executor_status), `${data.executor_iterations || 0} 次 LLM 调用`, executorOk ? "pass" : "warn"),
    statusCard("Draft", checkpointReady ? "已 checkpoint" : "未保存", "Reflection 前持久化", checkpointReady ? "pass" : "warn"),
    statusCard("Reflection", reflectionOk ? "通过" : formatStatus(data.reflection_status), `${data.reflection_mode || "unknown"} · ${data.reflection_iterations || 0} 轮`, reflectionOk ? "pass" : "warn"),
    statusCard("Memory", memoryReady ? formatStatus(data.memory_commit_status) : "未保存", `${data.memory_prefetch_status || "not started"} · 中期记忆`, memoryReady ? "pass" : "warn"),
    statusCard("Final", finalReady ? "已保存" : "无内容", `总耗时 ${formatElapsed(job.elapsed_seconds)}`, finalReady ? "pass" : "warn"),
  );
}

function renderIntent(data) {
  document.querySelector("#original-request").textContent = data.raw_user_request || data.original_query || "—";
  document.querySelector("#research-objective").textContent = data.research_objective || data.translated_query || "—";
  const intent = document.querySelector("#content-intent");
  intent.replaceChildren();
  const fields = [
    ["requested", data.content_intent?.requested],
    ["deliverable", data.content_intent?.deliverable_type || data.content_intent?.type],
    ["platform", data.content_intent?.platform],
    ["language", data.content_intent?.language],
    ["audience", data.content_intent?.audience],
    ["tone", data.content_intent?.tone],
  ];
  fields.filter(([, value]) => value).forEach(([key, value]) => {
    const row = element("div");
    row.append(element("span", "", key), element("strong", "", Array.isArray(value) ? value.join(", ") : value));
    intent.append(row);
  });
  if (!intent.childElementCount) intent.append(element("p", "muted-copy", "未检测到内容生成意图。"));
}

function timelineItem(index, title, note, metaItems, tone = "default") {
  const item = element("article", `timeline-item ${tone}`);
  item.append(element("span", "timeline-index", String(index).padStart(2, "0")));
  const body = element("div");
  body.append(element("strong", "", title), element("p", "", note || "—"));
  const meta = element("div", "timeline-meta");
  (metaItems || []).filter(Boolean).forEach((value) => meta.append(element("span", "", value)));
  if (meta.childElementCount) body.append(meta);
  item.append(body);
  return item;
}

function renderTrace(data) {
  const plan = document.querySelector("#content-plan");
  plan.replaceChildren();
  (data.content_plan?.steps || []).forEach((step, index) => {
    plan.append(
      timelineItem(index + 1, step.objective || step.step_id, `输出：${step.expected_output || "未指定"}`, step.suggested_tools || []),
    );
  });
  if (!plan.childElementCount) plan.append(element("div", "empty-state", "本次任务没有进入 Content Planner。"));

  const history = document.querySelector("#execution-history");
  history.replaceChildren();
  (data.execution_history || []).forEach((record, index) => {
    history.append(
      timelineItem(
        index + 1,
        `${record.step_id || "step"} · ${formatStatus(record.status)}`,
        record.execution_summary || `生成 ${record.result_type || "结果"}`,
        [
          ...(record.used_evidence_ids || []).map((id) => `Evidence ${id}`),
          ...(record.used_rag_chunk_ids || []).map((id) => `RAG ${id}`),
        ],
        record.status === "completed" ? "success" : "warning",
      ),
    );
  });
  if (!history.childElementCount) history.append(element("div", "empty-state", "暂无 Content Executor 记录。"));
  document.querySelector("#trace-summary").textContent = `${data.execution_history?.length || 0} 个计划步骤 · ${data.executor_iterations || 0} 次 Executor 调用`;

  const rag = document.querySelector("#rag-audit");
  rag.replaceChildren();
  const ragHistory = data.rag_tool_history || [];
  const title = element("div", "rag-audit-title");
  title.append(element("span", "card-kicker", "BRAND RAG"), element("strong", "", `${ragHistory.length} 次工具调用`));
  rag.append(title);
  if (ragHistory.length) {
    const calls = element("div", "rag-call-list");
    ragHistory.forEach((record) => {
      calls.append(
        timelineItem(
          Number(record.step_index || 0) + 1,
          record.tool_name || "brand_rag_search",
          record.error || record.arguments?.query || "已返回品牌知识",
          (record.chunk_ids || []).map((id) => id),
          record.error ? "warning" : "success",
        ),
      );
    });
    rag.append(calls);
  } else {
    rag.append(element("p", "muted-copy", "本次计划没有调用品牌 RAG。"));
  }
}

function verificationCard(result) {
  const id = result.claim_id || result.issue_id || "check";
  const verdict = result.verdict || "unknown";
  const card = element("article", `check-card ${verdict}`);
  const heading = element("div");
  heading.append(element("strong", "", id), element("span", "", formatStatus(verdict)));
  card.append(heading, element("p", "", result.answer || result.explanation || "无补充说明。"));
  const ids = [...(result.evidence_ids || []), ...(result.rag_chunk_ids || [])];
  if (ids.length) {
    const chips = element("div", "chips");
    renderChips(chips, ids, "soft");
    card.append(chips);
  }
  return card;
}

function renderReflection(data) {
  const badge = document.querySelector("#reflection-status-badge");
  badge.textContent = formatStatus(data.reflection_status || "not started");
  badge.className = `large-status ${data.reflection_status || "unknown"}`;
  document.querySelector("#verification-summary").textContent = data.verification_summary || "暂无 Verification 摘要。";

  const results = document.querySelector("#verification-results");
  results.replaceChildren();
  const rounds = data.reflection_history || [];
  rounds.forEach((round) => {
    const roundBlock = element("section", "reflection-round");
    const heading = element("div", "round-heading");
    heading.append(
      element("strong", "", `Round ${round.round || 1}`),
      element("span", "", formatStatus(round.status)),
    );
    roundBlock.append(heading);
    (round.verification_results || []).forEach((item) => roundBlock.append(verificationCard(item)));
    if (!(round.verification_results || []).length) {
      roundBlock.append(element("p", "muted-copy", "没有发现需要核查或修订的问题。"));
    }
    results.append(roundBlock);
  });
  if (!results.childElementCount) results.append(element("div", "empty-state", "本次没有 Reflection 审计记录。"));

  const revisions = document.querySelector("#revision-history");
  revisions.replaceChildren();
  (data.revision_history || []).forEach((record, index) => {
    revisions.append(
      timelineItem(
        index + 1,
        `${record.step_id || "revision"} · ${formatStatus(record.status)}`,
        record.execution_summary || "已应用修订。",
        [...(record.used_evidence_ids || []), ...(record.used_rag_chunk_ids || [])],
        record.status === "completed" ? "success" : "warning",
      ),
    );
  });
  if (!revisions.childElementCount) revisions.append(element("div", "empty-state", "初稿通过，无需进入 revision mode。"));
}

function renderResearch(data, job) {
  const summary = data.retrieval_summary || {};
  document.querySelector("#metrics").replaceChildren(
    metric("原始文档", summary.document_count || 0, "多来源去重后"),
    metric("候选洞察", summary.candidate_insight_count || 0, "分析器生成"),
    metric("通过验证", summary.verification_passed_count || 0, "证据与时效合格"),
    metric("最终入选", (data.eligible_insights || []).length, `耗时 ${formatElapsed(job.elapsed_seconds)}`),
  );
  renderChips(document.querySelector("#source-chips"), data.selected_sources || []);
  document.querySelector("#source-reasoning").textContent = data.source_reasoning || "来源由 Research Agent 根据证据缺口动态选择。";

  const eligible = document.querySelector("#eligible-insights");
  eligible.replaceChildren();
  (data.eligible_insights || []).forEach((item, index) => eligible.append(insightCard(item, index + 1)));
  if (!eligible.childElementCount) eligible.append(element("div", "empty-state", "没有洞察通过正式资格门槛。"));

  const alternatives = document.querySelector("#alternative-insights");
  alternatives.replaceChildren();
  (data.alternative_insights || []).forEach((item, index) => alternatives.append(insightCard(item, index + 6, true)));
  if (!alternatives.childElementCount) alternatives.append(element("div", "empty-state", "没有备选洞察。"));
}

function renderResults(job) {
  const data = job.result || {};
  setHidden(progressSection, true);
  setHidden(errorSection, true);
  setHidden(resultsSection, false);
  document.querySelector("#result-title").textContent = data.content_plan?.final_goal || data.topic || "完整链路结果";
  renderAcceptance(data, job);

  finalContentText = formatContent(data.final_content);
  currentJobId = job.job_id || "";
  renderFinalContent(data.final_content);
  renderFeishuPublishing(data, job.feishu_publication);
  renderIntent(data);
  renderTrace(data);
  renderReflection(data);
  renderResearch(data, job);
  document.querySelector("#raw-state").textContent = JSON.stringify(data, null, 2);
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderFeishuPublishing(data, publication) {
  const button = document.querySelector("#publish-feishu");
  const status = document.querySelector("#feishu-publication");
  const contentType = text(data.content_intent?.deliverable_type || data.content_intent?.type).toLowerCase();
  const platform = text(data.content_intent?.platform).toLowerCase();
  const publishable = Boolean(finalContentText) && platform !== "reddit" && !contentType.startsWith("reddit_");
  setHidden(button, !publishable);
  setHidden(status, true);
  status.classList.remove("error");
  status.replaceChildren();
  if (!publishable) return;
  button.disabled = !feishuConfigured;
  button.textContent = feishuConfigured ? "发布到飞书" : "飞书 CLI 未就绪";
  button.title = feishuConfigured ? "通过飞书 CLI 导入 Markdown 文档" : "请先安装并登录飞书 CLI";
  if (publication?.status === "published" && publication.document_url) {
    button.disabled = true;
    button.textContent = "已发布到飞书";
    const link = element("a", "", "打开飞书文档");
    link.href = publication.document_url;
    link.target = "_blank";
    link.rel = "noreferrer noopener";
    status.append("文档已保存。", link);
    setHidden(status, false);
  }
}

function showError(message) {
  setHidden(progressSection, true);
  setHidden(resultsSection, true);
  setHidden(errorSection, false);
  document.querySelector("#error-message").textContent = message || "未知错误";
  submitButton.disabled = false;
}

async function pollJob(jobId) {
  try {
    const response = await fetch(`/api/research/${jobId}`, { cache: "no-store" });
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || "无法读取任务状态");
    if (job.status === "succeeded") {
      clearTimeout(pollTimer);
      submitButton.disabled = false;
      renderResults(job);
      return;
    }
    if (job.status === "failed") {
      clearTimeout(pollTimer);
      showError(job.error);
      return;
    }
    showProgress(job);
    pollTimer = setTimeout(() => pollJob(jobId), 1500);
  } catch (error) {
    clearTimeout(pollTimer);
    showError(error.message);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const topic = topicInput.value.trim();
  if (!topic) return topicInput.focus();
  submitButton.disabled = true;
  setHidden(resultsSection, true);
  setHidden(errorSection, true);
  showProgress({ stage: "queued", stage_label: "正在提交完整任务", search_iterations: 0 });
  try {
    const response = await fetch("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic }),
    });
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || "任务提交失败");
    localStorage.setItem("smartpush:last-topic", topic);
    pollJob(job.job_id);
  } catch (error) {
    showError(error.message);
  }
});

document.querySelectorAll(".scenario-chip").forEach((button) => {
  button.addEventListener("click", () => {
    topicInput.value = button.dataset.prompt || "";
    topicInput.focus();
  });
});

document.querySelector("#copy-final").addEventListener("click", async (event) => {
  if (!finalContentText) return;
  const button = event.currentTarget;
  try {
    await navigator.clipboard.writeText(finalContentText);
    button.textContent = "已复制";
  } catch (_) {
    button.textContent = "复制失败，请手动选择";
  }
  setTimeout(() => { button.textContent = "复制最终内容"; }, 1800);
});

document.querySelector("#publish-feishu").addEventListener("click", async (event) => {
  if (!currentJobId || !finalContentText) return;
  const button = event.currentTarget;
  const status = document.querySelector("#feishu-publication");
  button.disabled = true;
  button.textContent = "正在发布…";
  status.textContent = "正在通过飞书 CLI 导入 Markdown 文档。";
  setHidden(status, false);
  try {
    const response = await fetch(`/api/research/${currentJobId}/publish/feishu`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const publication = await response.json();
    if (!response.ok) throw new Error(publication.error || "飞书发布失败");
    button.textContent = "已发布到飞书";
    status.classList.remove("error");
    status.replaceChildren();
    const link = element("a", "", "打开飞书文档");
    link.href = publication.document_url;
    link.target = "_blank";
    link.rel = "noreferrer noopener";
    status.append("文档已保存。", link);
  } catch (error) {
    button.disabled = false;
    button.textContent = "重新发布到飞书";
    status.textContent = error.message;
    status.classList.add("error");
  }
});

document.querySelector("#new-research").addEventListener("click", () => {
  setHidden(resultsSection, true);
  topicInput.focus();
  window.scrollTo({ top: 0, behavior: "smooth" });
});

document.querySelector("#retry-button").addEventListener("click", () => {
  setHidden(errorSection, true);
  topicInput.focus();
});

const previousTopic = localStorage.getItem("smartpush:last-topic");
if (previousTopic) topicInput.value = previousTopic;

fetch("/api/health", { cache: "no-store" })
  .then((response) => response.json())
  .then((health) => { feishuConfigured = Boolean(health.feishu_configured); })
  .catch(() => { feishuConfigured = false; });
