const statusEl = document.getElementById("status");
const uploadForm = document.getElementById("upload-form");
const chatForm = document.getElementById("chat-form");
const uploadResult = document.getElementById("upload-result");
const ragEvalResult = document.getElementById("rag-eval-result");
const ragasEvalResult = document.getElementById("ragas-eval-result");
const ragEvalProgress = document.getElementById("rag-eval-progress");
const ragEvalProgressLabel = document.getElementById("rag-eval-progress-label");
const ragEvalJobsEl = document.getElementById("rag-eval-jobs");
const ragEvaluationHistory = document.getElementById("rag-evaluation-history");
const refreshRagJobsButton = document.getElementById("refresh-rag-jobs-button");
const clearRagJobsButton = document.getElementById("clear-rag-jobs-button");
const messages = document.getElementById("messages");
const authStatus = document.getElementById("auth-status");
const sessionIdEl = document.getElementById("session-id");
const file = document.getElementById("file");
const uploadControls = document.getElementById("upload-controls");
const message = document.getElementById("message");
const signInButton = document.getElementById("sign-in-button");
const signOutButton = document.getElementById("sign-out-button");
const uploadButton = document.getElementById("upload-button");
const ragEvalButton = document.getElementById("rag-eval-button");
const ragasEvalButton = document.getElementById("ragas-eval-button");
const stopRagasButton = document.getElementById("stop-ragas-button");
const chatButton = document.getElementById("chat-button");
const newChatButton = document.getElementById("new-chat-button");
const idTokenStorageKey = "pra_cognito_token";
const accessTokenStorageKey = "pra_cognito_access_token";
const refreshTokenStorageKey = "pra_cognito_refresh_token";
const pkceVerifierStorageKey = "pra_pkce_verifier";
const pkceStateStorageKey = "pra_pkce_state";
const ragEvalJobsStorageKey = "pra_rag_eval_jobs";
const ragasEvalJobsStorageKey = "pra_ragas_eval_jobs";
const signInRequiredMessage = "You must sign in to use these features";
const modelUnavailableMessage = "The model is temporarily unavailable. Please try again.";
const retryableChatStatuses = new Set([429, 500, 502, 503, 504]);
const automaticRetryableAttempts = 1;
const automaticRetryDelayMilliseconds = 500;
let authConfig = null;
let documentUploadEnabled = true;
let retrievalDiagnosticsEnabled = false;
const browserTimeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
let currentSessionId = createSessionId();
let activeChatRequestKey = null;
let activeRagasJobId = null;
let ragasInteractionStarted = false;
const activeRagasPollers = new Map();
const cancelledRagasJobs = new Set();

function setStatus(text, kind = "") {
  statusEl.textContent = text;
  statusEl.className = `status ${kind}`;
}

function createSessionId() {
  return crypto.randomUUID();
}

function updateSessionIdDisplay() {
  sessionIdEl.textContent = `Active session: ${currentSessionId}`;
}

function showAuthRequired(target = uploadResult) {
  if (target) {
    target.textContent = signInRequiredMessage;
    target.className = "meta error";
  }
  setStatus(signInRequiredMessage, "error");
}

function formatSignInTimestamp(date = new Date()) {
  return formatTimestamp(date);
}

function formatTimestamp(date = new Date()) {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: browserTimeZone,
    month: "2-digit",
    day: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
  })
    .formatToParts(date)
    .reduce((values, part) => {
      values[part.type] = part.value;
      return values;
    }, {});
  return `${parts.month}/${parts.day}/${parts.year} ${parts.hour}:${parts.minute} ${parts.dayPeriod}`;
}

function buildActionTimingSummary(startedAt, finishedAt = null, label = "") {
  const lines = label ? [`${label}:`] : [];
  lines.push(`started: ${formatTimestamp(startedAt)}`);
  if (finishedAt) {
    lines.push(`finished: ${formatTimestamp(finishedAt)}`);
  }
  return lines.join("\n");
}

function appendActionTiming(text, startedAt, finishedAt = null, label = "") {
  return [text, buildActionTimingSummary(startedAt, finishedAt, label)]
    .filter(Boolean)
    .join("\n");
}

function formatTimestampFallback(date = new Date()) {
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  const year = date.getFullYear();
  const rawHours = date.getHours();
  const hours = String(rawHours % 12 || 12).padStart(2, "0");
  const minutes = String(date.getMinutes()).padStart(2, "0");
  const meridiem = rawHours >= 12 ? "PM" : "AM";
  return `${month}/${day}/${year} ${hours}:${minutes} ${meridiem}`;
}

function signInAgainMessage() {
  return `Please Sign in Again. ${formatSignInTimestamp()}`;
}

function showSignInAgain(target = uploadResult) {
  const text = signInAgainMessage();
  if (target) {
    target.textContent = text;
    target.className = "meta error";
  }
  setStatus(text, "error");
  return text;
}

function addMessage(text, role, meta = "", scoreMeta = "", debugMeta = "") {
  const item = document.createElement("div");
  item.className = `message ${role}`;
  appendMessageContent(item, text, meta, scoreMeta, debugMeta);
  messages.appendChild(item);
  messages.scrollTop = messages.scrollHeight;
  return item;
}

function appendMessageContent(item, text, meta = "", scoreMeta = "", debugMeta = "") {
  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = text;
  item.appendChild(body);

  if (meta) {
    const metaEl = document.createElement("div");
    metaEl.className = "meta";
    metaEl.textContent = meta;
    item.appendChild(metaEl);
  }

  if (scoreMeta) {
    const scoreEl = document.createElement("div");
    scoreEl.className = "message-scores";
    scoreEl.textContent = scoreMeta;
    item.appendChild(scoreEl);
  }

  if (debugMeta) {
    const debugEl = document.createElement("div");
    debugEl.className = "message-debug";
    debugEl.textContent = debugMeta;
    item.appendChild(debugEl);
  }

  const timestamp = document.createElement("div");
  timestamp.className = "message-time";
  timestamp.textContent = safeFormatTimestamp();
  timestamp.title = browserTimeZone || "Local time";
  item.appendChild(timestamp);
}

function safeFormatTimestamp(date = new Date()) {
  try {
    return formatTimestamp(date);
  } catch {
    return formatTimestampFallback(date);
  }
}

function addThinkingMessage() {
  const item = document.createElement("div");
  item.className = "message assistant pending";
  item.setAttribute("aria-live", "polite");

  const label = document.createElement("span");
  label.textContent = "Thinking";
  const dots = document.createElement("span");
  dots.className = "thinking-dots";
  dots.setAttribute("aria-hidden", "true");
  dots.textContent = "...";

  item.appendChild(label);
  item.appendChild(dots);
  const timestamp = document.createElement("div");
  timestamp.className = "message-time";
  timestamp.textContent = safeFormatTimestamp();
  timestamp.title = browserTimeZone || "Local time";
  item.appendChild(timestamp);
  messages.appendChild(item);
  messages.scrollTop = messages.scrollHeight;
  return item;
}

function replaceMessage(item, text, role, meta = "", scoreMeta = "", debugMeta = "") {
  if (!item) {
    addMessage(text, role, meta, scoreMeta, debugMeta);
    return;
  }

  item.className = `message ${role}`;
  item.removeAttribute("aria-live");
  item.replaceChildren();
  appendMessageContent(item, text, meta, scoreMeta, debugMeta);
  messages.scrollTop = messages.scrollHeight;
}

function addRetryButton(item, text) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "retry-button";
  button.textContent = "Try again";
  button.addEventListener("click", () => {
    item.remove();
    sendChatMessage(text, { addUserMessage: false });
  });
  item.appendChild(button);
}

function isRetryableChatFailure(status, messageText = "") {
  return retryableChatStatuses.has(status) || messageText === modelUnavailableMessage;
}

function chatErrorMessage(status, payload = {}) {
  const serverMessage = payload.error || payload.detail || "";
  if (isRetryableChatFailure(status, serverMessage)) {
    return modelUnavailableMessage;
  }
  return serverMessage || "Chat failed.";
}

function sleepMilliseconds(delay) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, delay);
  });
}

function formatScore(value) {
  return Number.isFinite(value) ? value.toFixed(2) : "none";
}

function buildChatScoreMeta(payload) {
  if (!retrievalDiagnosticsEnabled) {
    return "";
  }
  const topRetrieval = payload.retrieval_details?.[0] || {};
  return [
    `semantic: ${formatScore(topRetrieval.semantic_similarity_score)}`,
    `bm25: ${formatScore(topRetrieval.bm25_score)}`,
    `rrf: ${formatScore(topRetrieval.rrf_score)}`,
    `knowledge graph: ${payload.knowledge_graph_enabled ? "enabled" : "disabled"}`,
    `graph hits: ${payload.knowledge_graph_hits ?? 0}`,
    `top graph score: ${formatScore(payload.top_knowledge_graph_score)}`,
    `graph contribution: ${payload.knowledge_graph_contribution ? "yes" : "no"}`,
    `hallucination risk: ${formatScore(payload.hallucination_score)}`,
    `model: ${payload.selected_model || "unknown"}`,
    `topic reset: ${payload.topic_reset_applied ? "yes" : "no"}`,
  ].join(" | ");
}

function buildChatMeta(payload) {
  if (!retrievalDiagnosticsEnabled) {
    return "";
  }
  const uniqueCitations = [...new Set(payload.citations || [])];
  const citations = uniqueCitations.length
    ? `citations: ${uniqueCitations.join(", ")}`
    : "no citations";
  const chunkLabels = (payload.retrieved_chunk_labels || payload.retrieval_details || [])
    .map((detail) => (typeof detail === "string" ? detail : (detail.chunk_label || detail.filename)))
    .filter(Boolean)
    .slice(0, 5);
  if (!chunkLabels.length) {
    return citations;
  }
  return `${citations} | chunks: ${chunkLabels.join(", ")}`;
}

function buildChatDebugMeta(payload) {
  if (!retrievalDiagnosticsEnabled) {
    return "";
  }
  const parts = [
    `trace: ${payload.trace_id || "unknown"}`,
    `session: ${payload.session_id || currentSessionId}`,
    `hash: ${payload.message_hash || "unknown"}`,
  ];
  if (payload.top_chunk_label) {
    parts.push(`top chunk: ${payload.top_chunk_label}`);
  }
  return parts.join(" | ");
}

function formatMetricValue(value) {
  return Number.isFinite(value) ? value.toFixed(4) : "n/a";
}

function formatApiError(payload = {}, fallbackMessage = "Request failed.") {
  const message = payload.error || payload.detail || fallbackMessage;
  const traceId = payload.trace_id ? `trace_id: ${payload.trace_id}` : "";
  return [message, traceId].filter(Boolean).join("\n");
}

function buildRagEvaluationSummary(payload) {
  return [
    `dataset: ${payload.dataset}`,
    `total questions: ${payload.total_questions}`,
    `labeled questions: ${payload.labeled_questions}`,
    `unlabeled questions: ${payload.unlabeled_questions}`,
    `failed question count: ${payload.failed_question_count}`,
    `mean Recall@5: ${formatMetricValue(payload.metrics.recall_at_5)}`,
    `mean Precision@5: ${formatMetricValue(payload.metrics.precision_at_5)}`,
    `mean Recall@10: ${formatMetricValue(payload.metrics.recall_at_10)}`,
    `mean Precision@10: ${formatMetricValue(payload.metrics.precision_at_10)}`,
    `mean Recall@25: ${formatMetricValue(payload.metrics.recall_at_25)}`,
    `mean Precision@25: ${formatMetricValue(payload.metrics.precision_at_25)}`,
    `average retrieval latency: ${Number(payload.latency.avg_retrieval_latency_ms || 0).toFixed(2)} ms`,
  ].join("\n");
}

function buildRagEvaluationDiagnostics(payload) {
  if (!payload) {
    return "";
  }

  const lines = [
    "",
    "diagnostic evaluation",
    `run: ${payload.name}`,
    `chunk config: ${payload.chunk_config.chunk_size}/${payload.chunk_config.chunk_overlap}`,
    `expected relevant IDs present in index: ${payload.loss_summary.present_in_index}`,
    `expected relevant IDs missing from index: ${payload.loss_summary.missing_from_index}`,
    `BM25-only Recall@5: ${formatMetricValue(payload.stage_metric_means.bm25?.recall_at_5)}`,
    `BM25-only Precision@5: ${formatMetricValue(payload.stage_metric_means.bm25?.precision_at_5)}`,
    `BM25-only Recall@10: ${formatMetricValue(payload.stage_metric_means.bm25?.recall_at_10)}`,
    `BM25-only Precision@10: ${formatMetricValue(payload.stage_metric_means.bm25?.precision_at_10)}`,
    `BM25-only Recall@25: ${formatMetricValue(payload.stage_metric_means.bm25?.recall_at_25)}`,
    `BM25-only Precision@25: ${formatMetricValue(payload.stage_metric_means.bm25?.precision_at_25)}`,
    `vector-only Recall@5: ${formatMetricValue(payload.stage_metric_means.vector?.recall_at_5)}`,
    `vector-only Precision@5: ${formatMetricValue(payload.stage_metric_means.vector?.precision_at_5)}`,
    `vector-only Recall@10: ${formatMetricValue(payload.stage_metric_means.vector?.recall_at_10)}`,
    `vector-only Precision@10: ${formatMetricValue(payload.stage_metric_means.vector?.precision_at_10)}`,
    `vector-only Recall@25: ${formatMetricValue(payload.stage_metric_means.vector?.recall_at_25)}`,
    `vector-only Precision@25: ${formatMetricValue(payload.stage_metric_means.vector?.precision_at_25)}`,
    `hybrid pre-rerank Recall@5: ${formatMetricValue(payload.stage_metric_means.hybrid_pre_rerank?.recall_at_5)}`,
    `hybrid pre-rerank Precision@5: ${formatMetricValue(payload.stage_metric_means.hybrid_pre_rerank?.precision_at_5)}`,
    `hybrid pre-rerank Recall@10: ${formatMetricValue(payload.stage_metric_means.hybrid_pre_rerank?.recall_at_10)}`,
    `hybrid pre-rerank Precision@10: ${formatMetricValue(payload.stage_metric_means.hybrid_pre_rerank?.precision_at_10)}`,
    `hybrid pre-rerank Recall@25: ${formatMetricValue(payload.stage_metric_means.hybrid_pre_rerank?.recall_at_25)}`,
    `hybrid pre-rerank Precision@25: ${formatMetricValue(payload.stage_metric_means.hybrid_pre_rerank?.precision_at_25)}`,
    `hybrid post-rerank Recall@5: ${formatMetricValue(payload.stage_metric_means.hybrid_post_rerank?.recall_at_5)}`,
    `hybrid post-rerank Precision@5: ${formatMetricValue(payload.stage_metric_means.hybrid_post_rerank?.precision_at_5)}`,
    `hybrid post-rerank Recall@10: ${formatMetricValue(payload.stage_metric_means.hybrid_post_rerank?.recall_at_10)}`,
    `hybrid post-rerank Precision@10: ${formatMetricValue(payload.stage_metric_means.hybrid_post_rerank?.precision_at_10)}`,
    `hybrid post-rerank Recall@25: ${formatMetricValue(payload.stage_metric_means.hybrid_post_rerank?.recall_at_25)}`,
    `hybrid post-rerank Precision@25: ${formatMetricValue(payload.stage_metric_means.hybrid_post_rerank?.precision_at_25)}`,
    "",
    "loss summary",
    `missing in BM25: ${payload.loss_summary.missing_in_bm25}`,
    `missing in vector: ${payload.loss_summary.missing_in_vector}`,
    `lost before RRF output: ${payload.loss_summary.missing_after_rrf}`,
    `lost before rerank output: ${payload.loss_summary.missing_after_rerank}`,
    `lost before final context: ${payload.loss_summary.missing_from_final_context}`,
  ];

  payload.questions.forEach((question) => {
    lines.push("");
    lines.push(`question ${question.id}: ${question.question}`);
    lines.push(
      `present IDs: ${question.label_integrity.present_ids.length ? question.label_integrity.present_ids.join(", ") : "none"}`
    );
    lines.push(
      `missing IDs: ${question.label_integrity.missing_ids.length ? question.label_integrity.missing_ids.join(", ") : "none"}`
    );
    if (Object.keys(question.label_integrity.source_file_mismatches).length) {
      lines.push(
        `source mismatches: ${Object.entries(question.label_integrity.source_file_mismatches)
          .map(([label, files]) => `${label} -> ${files.join(", ")}`)
          .join(" | ")}`
      );
    }
    question.relevant_chunk_survival.forEach((chunk) => {
      lines.push(
        [
          `relevant ${chunk.relevant_chunk_id}`,
          `bm25 rank: ${chunk.bm25_rank ?? "missing"}`,
          `vector rank: ${chunk.vector_rank ?? "missing"}`,
          `rrf rank: ${chunk.rrf_rank ?? "missing"}`,
          `reranked rank: ${chunk.reranked_rank ?? "missing"}`,
          `final context: ${chunk.included_in_final_llm_context ? "yes" : "no"}`,
        ].join(" | ")
      );
    });
    lines.push(
      [
        `latency`,
        `bm25 ${Number(question.latency.bm25_latency_ms || 0).toFixed(2)} ms`,
        `vector ${Number(question.latency.vector_latency_ms || 0).toFixed(2)} ms`,
        `rrf ${Number(question.latency.rrf_merge_latency_ms || 0).toFixed(2)} ms`,
        `rerank ${Number(question.latency.reranker_latency_ms || 0).toFixed(2)} ms`,
        `total ${Number(question.latency.total_retrieval_latency_ms || 0).toFixed(2)} ms`,
      ].join(" | ")
    );
  });

  return lines.join("\n");
}

function buildRagEvaluationReport(record) {
  const parts = [];
  if (record?.summary) {
    parts.push(buildRagEvaluationSummary(record.summary));
  }
  if (record?.diagnostics) {
    parts.push(buildRagEvaluationDiagnostics(record.diagnostics));
  }
  if (parts.length) {
    return parts.join("\n");
  }
  return buildRagEvaluationJobSummary(record);
}

function buildRagEvaluationJobSummary(record) {
  if (record?.summary || record?.diagnostics) {
    return buildRagEvaluationReport(record);
  }

  if (record?.status === "failed") {
    return [
      record.error || "RAG evaluation failed.",
      record.trace_id ? `trace_id: ${record.trace_id}` : "",
      record.job_id ? `job_id: ${record.job_id}` : "",
    ]
      .filter(Boolean)
      .join("\n");
  }

  return [
    `job_id: ${record?.job_id || "unknown"}`,
    `status: ${record?.status || "queued"}`,
    `dataset: ${record?.request?.dataset || "default"}`,
  ].join("\n");
}

function getStoredRagEvaluationJobs() {
  try {
    const raw = localStorage.getItem(ragEvalJobsStorageKey);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveStoredRagEvaluationJobs(jobs) {
  localStorage.setItem(ragEvalJobsStorageKey, JSON.stringify(jobs.slice(0, 10)));
}

function clearStoredRagEvaluationJobs() {
  localStorage.removeItem(ragEvalJobsStorageKey);
  renderRagEvaluationJobs();
}

function clearStoredRagasEvaluationJobs() {
  localStorage.removeItem(ragasEvalJobsStorageKey);
}

function upsertStoredRagEvaluationJob(record) {
  const jobs = getStoredRagEvaluationJobs();
  const nextJobs = [
    {
      job_id: record.job_id,
      status: record.status,
      dataset: record.request?.dataset || "default",
      created_at: record.created_at || new Date().toISOString(),
      updated_at: record.updated_at || record.created_at || new Date().toISOString(),
      summary: record.summary || null,
      diagnostics: record.diagnostics || null,
      error: record.error || null,
    },
    ...jobs.filter((job) => job.job_id !== record.job_id),
  ];
  saveStoredRagEvaluationJobs(nextJobs);
  renderRagEvaluationJobs();
}

function updateStoredRagEvaluationJob(jobId, updates) {
  const jobs = getStoredRagEvaluationJobs();
  const nextJobs = jobs.map((job) =>
    job.job_id === jobId ? { ...job, ...updates } : job,
  );
  saveStoredRagEvaluationJobs(nextJobs);
  renderRagEvaluationJobs();
}

function renderRagEvaluationJobs() {
  const jobs = getStoredRagEvaluationJobs();
  ragEvalJobsEl.replaceChildren();

  if (!jobs.length) {
    const empty = document.createElement("div");
    empty.className = "recent-job empty";
    empty.textContent = "No saved evaluations yet.";
    ragEvalJobsEl.appendChild(empty);
    return;
  }

  jobs.forEach((job) => {
    const card = document.createElement("div");
    card.className = "recent-job";

    const topLine = document.createElement("div");
    topLine.className = "recent-job-topline";

    const title = document.createElement("div");
    title.className = "recent-job-title";
    title.textContent = job.dataset || "default";

    const status = document.createElement("div");
    status.className = "recent-job-status";
    status.textContent = job.status || "queued";

    topLine.appendChild(title);
    topLine.appendChild(status);
    card.appendChild(topLine);

    const meta = document.createElement("div");
    meta.className = "recent-job-meta";
    meta.textContent = [
      `job_id: ${job.job_id}`,
      job.updated_at ? `updated: ${formatStoredJobTime(job.updated_at)}` : "",
      job.error ? `error: ${job.error}` : "",
    ]
      .filter(Boolean)
      .join("\n");
    card.appendChild(meta);

    const actions = document.createElement("div");
    actions.className = "recent-job-actions";

    const checkButton = document.createElement("button");
    checkButton.type = "button";
    checkButton.className = "secondary-button compact-button";
    checkButton.textContent = "Check status";
    checkButton.addEventListener("click", async () => {
      await checkStoredRagEvaluationJob(job.job_id);
    });
    actions.appendChild(checkButton);

    const showButton = document.createElement("button");
    showButton.type = "button";
    showButton.className = "secondary-button compact-button";
    showButton.textContent = "Show summary";
    showButton.addEventListener("click", () => {
      ragEvalResult.textContent = buildRagEvaluationJobSummary(job);
      ragEvalResult.className = job.status === "failed" ? "meta error" : "meta";
    });
    actions.appendChild(showButton);

    card.appendChild(actions);
    ragEvalJobsEl.appendChild(card);
  });
}

function updateEvaluationVisibility() {
  const ragEvaluationEnabled = authConfig?.rag_evaluation_enabled === true;
  const ragasEnabled = authConfig?.ragas_enabled === true;
  ragEvalResult.hidden = !ragEvaluationEnabled;
  ragasEvalResult.hidden = !ragasEnabled;
  ragEvaluationHistory.hidden = !ragEvaluationEnabled && !ragasEnabled;
  if (!ragEvaluationEnabled) ragEvalResult.textContent = "";
  if (!ragasEnabled) ragasEvalResult.textContent = "";
}

function formatStoredJobTime(value) {
  try {
    return safeFormatTimestamp(new Date(value));
  } catch {
    return value;
  }
}

function formatElapsedDuration(seconds) {
  const wholeSeconds = Math.max(0, Math.floor(seconds));
  const minutes = String(Math.floor(wholeSeconds / 60)).padStart(2, "0");
  const remainder = String(wholeSeconds % 60).padStart(2, "0");
  return `${minutes}:${remainder}`;
}

function setRagEvaluationProgress(visible, text = "") {
  ragEvalProgress.classList.toggle("hidden", !visible);
  ragEvalProgress.hidden = !visible;
  ragEvalProgressLabel.textContent = text;
}

function updateRagEvaluationProgress(record, startedAt) {
  const elapsed = formatElapsedDuration((Date.now() - startedAt) / 1000);
  const statusText =
    record?.status === "running"
      ? `RAG evaluation running in worker... ${elapsed}`
      : `RAG evaluation queued... ${elapsed}`;
  setRagEvaluationProgress(true, statusText);
}

function clearTokens() {
  sessionStorage.removeItem(idTokenStorageKey);
  sessionStorage.removeItem(accessTokenStorageKey);
  sessionStorage.removeItem(refreshTokenStorageKey);
}

function tokenClaims(token) {
  const [, payload] = token.split(".");
  const normalized = payload.replace(/-/g, "+").replace(/_/g, "/");
  return JSON.parse(atob(normalized));
}

function tokenExpiresSoon(token, bufferSeconds = 60) {
  try {
    const claims = tokenClaims(token);
    if (!Number.isFinite(claims.exp)) {
      return true;
    }
    return claims.exp * 1000 <= Date.now() + bufferSeconds * 1000;
  } catch {
    return true;
  }
}

function storeTokens(payload) {
  if (payload.id_token) {
    sessionStorage.setItem(idTokenStorageKey, payload.id_token);
  }
  if (payload.access_token) {
    sessionStorage.setItem(accessTokenStorageKey, payload.access_token);
  }
  if (payload.refresh_token) {
    sessionStorage.setItem(refreshTokenStorageKey, payload.refresh_token);
  }
}

function cognitoUrl(path) {
  return `${authConfig.cognito_hosted_ui_base_url.replace(/\/$/, "")}${path}`;
}

async function refreshTokens() {
  const refreshToken = sessionStorage.getItem(refreshTokenStorageKey);
  if (!refreshToken || !authConfig?.cognito_hosted_ui_base_url) {
    throw new Error("No refresh token.");
  }

  const body = new URLSearchParams({
    grant_type: "refresh_token",
    client_id: authConfig.cognito_client_id,
    refresh_token: refreshToken,
  });
  const response = await fetch(cognitoUrl("/oauth2/token"), {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error_description || payload.error || "Token refresh failed.");
  }
  storeTokens(payload);
}

async function ensureValidToken() {
  const token = sessionStorage.getItem(idTokenStorageKey);
  if (!token) {
    return false;
  }
  if (!tokenExpiresSoon(token)) {
    return true;
  }

  try {
    await refreshTokens();
    return Boolean(sessionStorage.getItem(idTokenStorageKey));
  } catch {
    clearTokens();
    updateAuthState();
    return false;
  }
}

async function authHeadersOrPrompt(target = uploadResult) {
  if (!(await ensureValidToken())) {
    return {
      headers: null,
      message: showSignInAgain(target),
    };
  }
  return {
    headers: { Authorization: `Bearer ${sessionStorage.getItem(idTokenStorageKey)}` },
    message: "",
  };
}

function tokenSubject(token) {
  try {
    const claims = tokenClaims(token);
    return claims.email || claims["cognito:username"] || claims.sub || "signed-in user";
  } catch {
    return "signed-in user";
  }
}

function base64UrlEncode(bytes) {
  const binary = Array.from(bytes, (byte) => String.fromCharCode(byte)).join("");
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function generateCodeVerifier() {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  return base64UrlEncode(bytes);
}

async function codeChallenge(verifier) {
  const encoded = new TextEncoder().encode(verifier);
  const digest = await crypto.subtle.digest("SHA-256", encoded);
  return base64UrlEncode(new Uint8Array(digest));
}

async function exchangeAuthorizationCode(code) {
  const verifier = sessionStorage.getItem(pkceVerifierStorageKey);
  if (!verifier) {
    throw new Error("Missing PKCE verifier.");
  }

  const body = new URLSearchParams({
    grant_type: "authorization_code",
    client_id: authConfig.cognito_client_id,
    code,
    redirect_uri: authRedirectUri(),
    code_verifier: verifier,
  });
  const response = await fetch(cognitoUrl("/oauth2/token"), {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error_description || payload.error || "Token exchange failed.");
  }
  storeTokens(payload);
  sessionStorage.removeItem(pkceVerifierStorageKey);
  sessionStorage.removeItem(pkceStateStorageKey);
}

async function handleAuthorizationResponse() {
  const params = new URLSearchParams(window.location.search);
  const code = params.get("code");
  const state = params.get("state");
  if (!code) {
    return;
  }

  if (state !== sessionStorage.getItem(pkceStateStorageKey)) {
    throw new Error("Invalid sign-in state.");
  }
  await exchangeAuthorizationCode(code);
  history.replaceState(null, "", window.location.pathname);
}

function updateAuthState() {
  const token = sessionStorage.getItem(idTokenStorageKey);
  const signedIn = Boolean(token);
  authStatus.textContent = signedIn
    ? `Signed in as ${tokenSubject(token)}`
    : "Not signed in.";
  authStatus.className = `meta ${signedIn ? "ok" : "error"}`;
  signOutButton.disabled = !signedIn;
  uploadButton.disabled = !documentUploadEnabled;
  chatButton.disabled = false;
}

function requireAuth(target) {
  if (!sessionStorage.getItem(idTokenStorageKey)) {
    showAuthRequired(target);
    return false;
  }
  return true;
}

function authRedirectUri() {
  return authConfig?.redirect_uri || window.location.origin + window.location.pathname;
}

async function loadAuthConfig() {
  const response = await fetch("auth/config");
  authConfig = await response.json();
  documentUploadEnabled = authConfig.document_upload_enabled === true;
  retrievalDiagnosticsEnabled = authConfig.retrieval_diagnostics_enabled === true;
  uploadControls.hidden = !documentUploadEnabled;
  ragEvalButton.hidden = authConfig.rag_evaluation_enabled !== true;
  ragasEvalButton.hidden = authConfig.ragas_enabled !== true;
  stopRagasButton.hidden = authConfig.ragas_enabled !== true || !activeRagasJobId;
  updateEvaluationVisibility();
  await handleAuthorizationResponse();
  await ensureValidToken();
  updateAuthState();
}

async function fetchRagEvaluationJob(jobId, authorization) {
  const response = await fetch(`rag/evaluate/${jobId}`, {
    headers: {
      ...authorization.headers,
    },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, "RAG evaluation failed."));
  }
  return payload;
}

async function fetchRagasEvaluationJob(jobId, authorization) {
  const response = await fetch(`rag/evaluate/ragas/${jobId}`, {
    headers: { ...authorization.headers },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, "RAGAS evaluation failed."));
  }
  return payload;
}

async function cancelRagasEvaluationJob(jobId, authorization) {
  const response = await fetch(`rag/evaluate/ragas/${jobId}/cancel`, {
    method: "POST",
    headers: { ...authorization.headers },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(formatApiError(payload, "Unable to stop RAGAS evaluation."));
  return payload;
}

function buildRagasEvaluationSummary(record) {
  if (record?.status === "failed") {
    return [record.error || "RAGAS evaluation failed.", record.job_id ? `job_id: ${record.job_id}` : ""]
      .filter(Boolean).join("\n");
  }
  const summary = record?.summary;
  if (!summary) {
    return [
      `job_id: ${record?.job_id || "unknown"}`,
      `status: ${record?.status || "queued"}`,
      record?.orchestration_message || (record?.status === "queued"
        ? "Waiting for RAGAS worker capacity. If the worker is busy, the evaluation retries automatically."
        : ""),
      record?.total_questions != null
        ? `progress: ${record.completed_questions || 0}/${record.total_questions}`
        : "",
    ].filter(Boolean).join("\n");
  }
  return [
    `mode: ${record.request?.mode || "full"}`,
    `RAGAS questions: ${summary.total_questions}`,
    `completed: ${summary.completed_questions}`,
    `failed: ${summary.failed_questions}`,
    `success rate: ${formatMetricValue(summary.evaluation_success_rate)}`,
    summary.first_error ? `error: ${summary.first_error}` : "",
    `faithfulness: ${formatMetricValue(summary.metrics?.faithfulness)}`,
    `context precision: ${formatMetricValue(summary.metrics?.context_precision)}`,
    `context recall: ${formatMetricValue(summary.metrics?.context_recall)}`,
    `answer relevancy: ${formatMetricValue(summary.metrics?.answer_relevancy)}`,
    `answer correctness: ${formatMetricValue(summary.metrics?.answer_correctness)}`,
    `job_id: ${record.job_id}`,
  ].join("\n");
}

function getStoredRagasEvaluationJobs() {
  try {
    const raw = localStorage.getItem(ragasEvalJobsStorageKey);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveStoredRagasEvaluationJobs(jobs) {
  localStorage.setItem(ragasEvalJobsStorageKey, JSON.stringify(jobs.slice(0, 10)));
}

function persistRagasEvaluationJob(record) {
  const jobs = getStoredRagasEvaluationJobs();
  saveStoredRagasEvaluationJobs([
    {
      job_id: record.job_id,
      status: record.status,
      created_at: record.created_at || new Date().toISOString(),
      updated_at: record.updated_at || record.created_at || new Date().toISOString(),
      mode: record.request?.mode || record.mode || "full",
      completed_questions: record.completed_questions || 0,
      total_questions: record.total_questions || record.summary?.total_questions || null,
      summary: record.summary || null,
      error: record.error || null,
    },
    ...jobs.filter((job) => job.job_id !== record.job_id),
  ]);
}

function updateStoredRagasEvaluationJob(record) {
  const jobs = getStoredRagasEvaluationJobs();
  if (!jobs.some((job) => job.job_id === record.job_id)) {
    persistRagasEvaluationJob(record);
    return;
  }
  saveStoredRagasEvaluationJobs(jobs.map((job) => job.job_id === record.job_id
    ? { ...job, ...record, mode: record.request?.mode || record.mode || job.mode }
    : job));
}

function updateRagasControls(record) {
  const active = record && ["queued", "running"].includes(record.status);
  activeRagasJobId = active ? record.job_id : (activeRagasJobId === record?.job_id ? null : activeRagasJobId);
  ragasEvalButton.disabled = Boolean(active);
  stopRagasButton.disabled = !(active && ["queued", "running"].includes(record.status));
  stopRagasButton.hidden = authConfig?.ragas_enabled !== true || !active;
}

function renderRagasRecord(record, timing = {}) {
  if (authConfig?.ragas_enabled !== true) {
    ragasEvalResult.textContent = "";
    ragasEvalResult.hidden = true;
    updateRagasControls(null);
    return;
  }
  updateStoredRagasEvaluationJob(record);
  ragasEvalResult.textContent = timing.startedAt
    ? appendActionTiming(
      buildRagasEvaluationSummary(record),
      timing.startedAt,
      timing.finishedAt || null,
      "RAGAS test timing",
    )
    : buildRagasEvaluationSummary(record);
  ragasEvalResult.className = ["failed", "cancelled"].includes(record.status) ? "meta error" : "meta ok";
  updateRagasControls(record);
}

async function restoreStoredRagasEvaluationJobs() {
  if (authConfig?.ragas_enabled !== true) {
    ragasEvalResult.textContent = "";
    ragasEvalResult.hidden = true;
    return;
  }
  if (!sessionStorage.getItem(idTokenStorageKey)) return;
  const jobs = getStoredRagasEvaluationJobs();
  for (const job of jobs.slice(0, 10)) {
    try {
      const authorization = await authHeadersOrPrompt(ragasEvalResult);
      if (!authorization.headers) return;
      const payload = await fetchRagasEvaluationJob(job.job_id, authorization);
      updateStoredRagasEvaluationJob(payload);
      if (payload.status !== "cancelled") renderRagasRecord(payload);
      if (["queued", "running", "cancel_requested"].includes(payload.status)) {
        pollRagasEvaluationJob(payload.job_id, authorization, renderRagasRecord);
      }
    } catch (error) {
      ragasEvalResult.textContent = error.message;
    }
  }
}

async function pollRagasEvaluationJob(jobId, authorization, onUpdate = () => {}) {
  cancelledRagasJobs.delete(jobId);
  if (activeRagasPollers.has(jobId)) return activeRagasPollers.get(jobId);
  const poller = (async () => {
    let payload = null;
    for (let attempt = 0; attempt < 900; attempt += 1) {
      if (cancelledRagasJobs.has(jobId)) return payload;
      const currentAuthorization = await authHeadersOrPrompt(ragasEvalResult);
      if (!currentAuthorization.headers) throw new Error(currentAuthorization.message || "Your sign-in session expired.");
      payload = await fetchRagasEvaluationJob(jobId, currentAuthorization);
      persistRagasEvaluationJob(payload);
      onUpdate(payload, attempt);
      if (["completed", "failed", "cancelled"].includes(payload.status)) return payload;
      if (cancelledRagasJobs.has(jobId)) return payload;
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
    return payload;
  })().finally(() => activeRagasPollers.delete(jobId));
  activeRagasPollers.set(jobId, poller);
  return poller;
}

async function checkStoredRagEvaluationJob(jobId) {
  if (!requireAuth(ragEvalResult)) {
    return;
  }
  const authorization = await authHeadersOrPrompt(ragEvalResult);
  if (!authorization.headers) {
    return;
  }

  setStatus("Checking RAG evaluation status...");
  const payload = await fetchRagEvaluationJob(jobId, authorization);
  updateStoredRagEvaluationJob(jobId, {
    status: payload.status,
    updated_at: payload.updated_at || new Date().toISOString(),
    summary: payload.summary || null,
    diagnostics: payload.diagnostics || null,
    error: payload.error || null,
  });
  ragEvalResult.textContent = buildRagEvaluationJobSummary(payload);
  ragEvalResult.className = payload.status === "failed" ? "meta error" : "meta ok";
  setRagEvaluationProgress(false);
  setStatus(
    payload.status === "completed"
      ? "RAG evaluation is ready."
      : payload.status === "failed"
        ? "RAG evaluation failed."
        : "RAG evaluation is still running. Check back later.",
    payload.status === "failed" ? "error" : "ok",
  );
}

async function fetchRagEvaluationJobOnce(jobId, authorization) {
  const payload = await fetchRagEvaluationJob(jobId, authorization);
  updateStoredRagEvaluationJob(jobId, {
    status: payload.status,
    updated_at: payload.updated_at || new Date().toISOString(),
    summary: payload.summary || null,
    diagnostics: payload.diagnostics || null,
    error: payload.error || null,
  });
  return payload;
}

async function pollRagEvaluationJob(jobId, authorization, onUpdate = () => {}) {
  const maxAttempts = 60;
  const delayMilliseconds = 1000;
  let payload = null;

  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    payload = await fetchRagEvaluationJobOnce(jobId, authorization);
    onUpdate(payload, attempt);
    if (payload.status === "completed" || payload.status === "failed") {
      return payload;
    }
    await new Promise((resolve) => setTimeout(resolve, delayMilliseconds));
  }

  return payload;
}

async function resetServerSession(sessionIdToClear) {
  const authorization = await authHeadersOrPrompt(null);
  if (!authorization.headers) {
    return false;
  }
  const response = await fetch("chat/reset", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-trace-id": crypto.randomUUID(),
      ...authorization.headers,
    },
    body: JSON.stringify({ session_id: sessionIdToClear }),
  });
  return response.ok;
}

async function startNewChat() {
  const previousSessionId = currentSessionId;
  currentSessionId = createSessionId();
  updateSessionIdDisplay();
  messages.replaceChildren();
  message.value = "";

  if (sessionStorage.getItem(idTokenStorageKey)) {
    const cleared = await resetServerSession(previousSessionId);
    setStatus(cleared ? "New chat ready." : "New local chat ready.", "ok");
    return;
  }

  setStatus("New chat ready.", "ok");
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!documentUploadEnabled) {
    setStatus("Document upload is disabled.", "error");
    return;
  }
  if (!requireAuth(uploadResult)) {
    return;
  }
  const authorization = await authHeadersOrPrompt(uploadResult);
  if (!authorization.headers) {
    return;
  }
  if (!file.files.length) {
    uploadResult.textContent = "Choose a file.";
    uploadResult.className = "meta error";
    return;
  }

  uploadButton.disabled = true;
  setStatus("Uploading...");
  const data = new FormData();
  data.append("file", file.files[0]);

  try {
    const response = await fetch("documents/upload", {
      method: "POST",
      headers: authorization.headers,
      body: data,
    });
    const payload = await response.json();
    if (!response.ok) {
      if (response.status === 401) {
        clearTokens();
        updateAuthState();
        throw new Error(signInAgainMessage());
      }
      throw new Error(payload.error || payload.detail || "Upload failed.");
    }
    if (payload.already_uploaded) {
      uploadResult.textContent = payload.message;
      uploadResult.className = "meta ok";
      setStatus("Already uploaded.", "ok");
    } else if (payload.indexed === false) {
      uploadResult.textContent = `${payload.filename}: uploaded, but indexing is temporarily unavailable. Please try again later.`;
      uploadResult.className = "meta error";
      setStatus("Uploaded. Indexing unavailable.", "error");
    } else {
      uploadResult.textContent = `${payload.filename}: ${payload.chunk_count} chunk(s) indexed.`;
      uploadResult.className = "meta ok";
      setStatus("Document indexed.", "ok");
    }
  } catch (error) {
    uploadResult.textContent = error.message;
    uploadResult.className = "meta error";
    setStatus("Upload failed.", "error");
  } finally {
    uploadButton.disabled = false;
  }
});

ragEvalButton.addEventListener("click", async () => {
  if (!requireAuth(ragEvalResult)) {
    return;
  }
  const authorization = await authHeadersOrPrompt(ragEvalResult);
  if (!authorization.headers) {
    return;
  }

  const actionStartedAt = new Date();
  ragEvalButton.disabled = true;
  ragEvalResult.textContent = appendActionTiming(
    "Running retrieval-only RAG performance test...",
    actionStartedAt,
    null,
    "RAG performance test timing",
  );
  ragEvalResult.className = "meta";
  setRagEvaluationProgress(true, "Preparing RAG evaluation...");
  setStatus("Running RAG performance test...");

  try {
    const response = await fetch("rag/evaluate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...authorization.headers,
      },
      body: JSON.stringify({
        dataset: "default",
        top_k: 25,
        eval_k: [5, 10, 25],
      }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      if (response.status === 401) {
        clearTokens();
        updateAuthState();
        throw new Error(signInAgainMessage());
      }
      throw new Error(formatApiError(payload, "RAG evaluation failed."));
    }
    upsertStoredRagEvaluationJob(payload);
    const finalPayload = await pollRagEvaluationJob(payload.job_id, authorization, (job) => {
      ragEvalResult.textContent = appendActionTiming(
        buildRagEvaluationJobSummary(job),
        actionStartedAt,
        null,
        "RAG performance test timing",
      );
      ragEvalResult.className = job.status === "failed" ? "meta error" : "meta ok";
      setRagEvaluationProgress(
        job.status !== "completed" && job.status !== "failed",
        job.status === "running"
          ? "RAG evaluation running in worker..."
          : "Preparing RAG evaluation...",
      );
    });
    ragEvalResult.textContent = appendActionTiming(
      buildRagEvaluationJobSummary(finalPayload),
      actionStartedAt,
      new Date(),
      "RAG performance test timing",
    );
    ragEvalResult.className = finalPayload.status === "failed" ? "meta error" : "meta ok";
    setStatus(
      finalPayload.status === "completed"
        ? "RAG performance test complete."
        : "RAG performance test submitted. Check back later.",
      finalPayload.status === "completed" ? "ok" : "ok",
    );
  } catch (error) {
    setRagEvaluationProgress(false);
    ragEvalResult.textContent = appendActionTiming(
      error.message,
      actionStartedAt,
      new Date(),
      "RAG performance test timing",
    );
    ragEvalResult.className = "meta error";
    setStatus(error.message || "RAG performance test failed.", "error");
  } finally {
    setRagEvaluationProgress(false);
    ragEvalButton.disabled = false;
  }
});

ragasEvalButton.addEventListener("click", async () => {
  ragasInteractionStarted = true;
  const actionStartedAt = new Date();
  const existingJob = getStoredRagasEvaluationJobs().find((job) =>
    ["queued", "running"].includes(job.status)
  );
  if (existingJob) {
    renderRagasRecord(existingJob, { startedAt: actionStartedAt });
    pollRagasEvaluationJob(existingJob.job_id, null, (job) => {
      renderRagasRecord(job, { startedAt: actionStartedAt });
    });
    return;
  }
  if (!requireAuth(ragasEvalResult)) return;
  const authorization = await authHeadersOrPrompt(ragasEvalResult);
  if (!authorization.headers) return;

  ragasEvalButton.disabled = true;
  ragasEvalResult.textContent = appendActionTiming(
    "Running RAGAS answer-quality test...",
    actionStartedAt,
    null,
    "RAGAS test timing",
  );
  ragasEvalResult.className = "meta";
  setRagEvaluationProgress(true, "Preparing RAGAS evaluation...");
  setStatus("Running RAGAS test...");
  try {
    const response = await fetch("rag/evaluate/ragas", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authorization.headers },
      body: JSON.stringify({
        dataset: "default",
        mode: "full",
        top_k: 15,
        max_questions: 12,
      }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(formatApiError(payload, "RAGAS evaluation failed."));
    persistRagasEvaluationJob(payload);
    updateRagasControls(payload);
    const finalPayload = await pollRagasEvaluationJob(payload.job_id, authorization, (job) => {
      renderRagasRecord(job, { startedAt: actionStartedAt });
      setRagEvaluationProgress(!["completed", "failed", "cancelled"].includes(job.status), "RAGAS evaluation running in worker...");
    });
    renderRagasRecord(finalPayload, {
      startedAt: actionStartedAt,
      finishedAt: new Date(),
    });
    const finalStatusMessage = finalPayload.status === "completed"
      ? "RAGAS test complete."
      : finalPayload.status === "failed"
        ? "RAGAS test failed."
        : finalPayload.status === "cancelled"
          ? "RAGAS test stopped."
        : "RAGAS test is still running. Check back later.";
    setStatus(finalStatusMessage, finalPayload.status === "failed" ? "error" : "ok");
  } catch (error) {
    ragasEvalResult.textContent = appendActionTiming(
      error.message,
      actionStartedAt,
      new Date(),
      "RAGAS test timing",
    );
    ragasEvalResult.className = "meta error";
    setStatus(error.message || "RAGAS test failed.", "error");
  } finally {
    setRagEvaluationProgress(false);
    ragasEvalButton.disabled = false;
  }
});

stopRagasButton.addEventListener("click", async () => {
  if (!activeRagasJobId || !requireAuth(ragasEvalResult)) return;
  const jobId = activeRagasJobId;
  const authorization = await authHeadersOrPrompt(ragasEvalResult);
  if (!authorization.headers) return;
  const actionStartedAt = new Date();
  cancelledRagasJobs.add(jobId);
  activeRagasJobId = null;
  stopRagasButton.disabled = true;
  ragasEvalResult.textContent = appendActionTiming(
    ragasEvalResult.textContent || `job_id: ${jobId}`,
    actionStartedAt,
    null,
    "Stop RAGAS test timing",
  );
  try {
    const payload = await cancelRagasEvaluationJob(jobId, authorization);
    renderRagasRecord(payload);
    ragasEvalResult.textContent = appendActionTiming(
      ragasEvalResult.textContent,
      actionStartedAt,
      new Date(),
      "Stop RAGAS test timing",
    );
    setStatus(payload.status === "cancelled" ? "RAGAS test stopped." : "RAGAS stop requested.", "ok");
  } catch (error) {
    cancelledRagasJobs.delete(jobId);
    stopRagasButton.disabled = false;
    ragasEvalResult.textContent = appendActionTiming(
      error.message,
      actionStartedAt,
      new Date(),
      "Stop RAGAS test timing",
    );
    setStatus(error.message || "Unable to stop RAGAS evaluation.", "error");
  }
});

refreshRagJobsButton.addEventListener("click", async () => {
  if (!requireAuth(ragEvalResult)) {
    return;
  }
  const authorization = await authHeadersOrPrompt(ragEvalResult);
  if (!authorization.headers) {
    return;
  }

  const jobs = getStoredRagEvaluationJobs();
  if (!jobs.length) {
    setStatus("No saved evaluations to refresh.", "error");
    return;
  }

  refreshRagJobsButton.disabled = true;
  setStatus("Refreshing saved evaluations...");
  try {
    for (const job of jobs.slice(0, 5)) {
      const payload = await fetchRagEvaluationJob(job.job_id, authorization);
      updateStoredRagEvaluationJob(job.job_id, {
        status: payload.status,
        updated_at: payload.updated_at || new Date().toISOString(),
        summary: payload.summary || null,
        diagnostics: payload.diagnostics || null,
        error: payload.error || null,
      });
    }
    setStatus("Saved evaluations refreshed.", "ok");
  } catch (error) {
    setStatus(error.message || "Unable to refresh saved evaluations.", "error");
  } finally {
    refreshRagJobsButton.disabled = false;
  }
});

clearRagJobsButton.addEventListener("click", () => {
  clearStoredRagEvaluationJobs();
  clearStoredRagasEvaluationJobs();
  ragEvalResult.textContent = "";
  ragEvalResult.className = "meta";
  ragasEvalResult.textContent = "";
  ragasEvalResult.className = "meta";
  setRagEvaluationProgress(false);
  setStatus("Saved RAG evaluation data cleared.", "ok");
});

async function sendChatMessage(text, options = {}) {
  const addUserMessage = options.addUserMessage ?? true;
  if (!requireAuth()) {
    addMessage(signInRequiredMessage, "assistant", "auth required");
    return;
  }
  const authorization = await authHeadersOrPrompt(null);
  if (!authorization.headers) {
    addMessage(authorization.message, "assistant", "auth required");
    return;
  }
  if (!text) {
    return;
  }
  const requestKey = `${currentSessionId}:${text}`;
  if (activeChatRequestKey === requestKey) {
    setStatus("That message is already being sent.", "error");
    return;
  }
  activeChatRequestKey = requestKey;
  const traceId = crypto.randomUUID();
  console.info("chat_request_started", {
    traceId,
    sessionId: currentSessionId,
    messagePreview: text.slice(0, 80),
  });

  if (addUserMessage) {
    addMessage(text, "user");
  }
  const pendingMessage = addThinkingMessage();
  chatButton.disabled = true;
  setStatus("Thinking...");

  try {
    let response;
    let payload = {};

    for (let attempt = 0; attempt <= automaticRetryableAttempts; attempt += 1) {
      response = await fetch("chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-trace-id": traceId,
          ...authorization.headers,
        },
        body: JSON.stringify({
          message: text,
          session_id: currentSessionId,
          top_k: 25,
        }),
      });
      payload = await response.json().catch(() => ({}));
      if (response.ok) {
        break;
      }

      if (response.status === 401) {
        clearTokens();
        updateAuthState();
        throw new Error(signInAgainMessage());
      }

      const error = new Error(chatErrorMessage(response.status, payload));
      error.retryable = isRetryableChatFailure(response.status, error.message);

      if (!error.retryable || attempt >= automaticRetryableAttempts) {
        throw error;
      }

      await sleepMilliseconds(automaticRetryDelayMilliseconds);
    }

    replaceMessage(
      pendingMessage,
      payload.answer,
      "assistant",
      buildChatMeta(payload),
      buildChatScoreMeta(payload),
      buildChatDebugMeta(payload),
    );
    console.info("chat_request_finished", {
      traceId: payload.trace_id || traceId,
      sessionId: payload.session_id || currentSessionId,
      messageHash: payload.message_hash || "unknown",
      retrievedChunkLabels: payload.retrieved_chunk_labels || [],
    });
    setStatus("Ready.", "ok");
  } catch (error) {
    replaceMessage(pendingMessage, error.message, "assistant", "error");
    if (error.retryable) {
      addRetryButton(pendingMessage, text);
    }
    setStatus("Chat failed.", "error");
  } finally {
    activeChatRequestKey = null;
    chatButton.disabled = false;
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = message.value.trim();
  if (!text) {
    return;
  }

  message.value = "";
  await sendChatMessage(text);
});

signInButton.addEventListener("click", async () => {
  if (!authConfig?.cognito_hosted_ui_base_url || !authConfig?.cognito_client_id) {
    setStatus("Cognito login is not configured.", "error");
    return;
  }

  const verifier = generateCodeVerifier();
  const state = crypto.randomUUID();
  sessionStorage.setItem(pkceVerifierStorageKey, verifier);
  sessionStorage.setItem(pkceStateStorageKey, state);

  const loginUrl = new URL(cognitoUrl("/login"));
  loginUrl.searchParams.set("client_id", authConfig.cognito_client_id);
  loginUrl.searchParams.set("response_type", "code");
  loginUrl.searchParams.set("scope", "openid email profile");
  loginUrl.searchParams.set("redirect_uri", authRedirectUri());
  loginUrl.searchParams.set("code_challenge", await codeChallenge(verifier));
  loginUrl.searchParams.set("code_challenge_method", "S256");
  loginUrl.searchParams.set("state", state);
  window.location.href = loginUrl.toString();
});

signOutButton.addEventListener("click", () => {
  clearTokens();
  updateAuthState();
  if (authConfig?.cognito_hosted_ui_base_url && authConfig?.cognito_client_id) {
    const logoutUrl = new URL(cognitoUrl("/logout"));
    logoutUrl.searchParams.set("client_id", authConfig.cognito_client_id);
    logoutUrl.searchParams.set("logout_uri", authRedirectUri());
    window.location.href = logoutUrl.toString();
    return;
  }
  setStatus("Signed out.", "ok");
});

newChatButton.addEventListener("click", async () => {
  newChatButton.disabled = true;
  try {
    await startNewChat();
  } finally {
    newChatButton.disabled = false;
  }
});

updateSessionIdDisplay();
setRagEvaluationProgress(false);
renderRagEvaluationJobs();
loadAuthConfig()
  .then(async () => {
    await restoreStoredRagasEvaluationJobs();
    setStatus("Ready.", "ok");
  })
  .catch(() => setStatus("Auth configuration failed.", "error"));
