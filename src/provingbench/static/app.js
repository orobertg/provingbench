"use strict";
// One page, one hash route per view. All text goes in as text nodes: model output is untrusted.

const POLL_MS = 2000;
const MAX_ROWS = 300;
const CLOUD_WARNING = "This sends the conversation and the agent's notes to the provider.";

function el(tag, props, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "class") node.className = value;
    else if (key in node && key !== "list") node[key] = value;
    else if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  node.append(...children.flat().filter((c) => c !== null && c !== undefined));
  return node;
}

class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

async function api(path, method = "GET", body) {
  const options = { method };
  if (body !== undefined) {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(body);
  }
  const resp = await fetch(`/api/v1${path}`, options);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new ApiError(resp.status, (data.error && data.error.message) || `HTTP ${resp.status}`);
  return data;
}

const put = (path, body) => api(path, "PUT", body);
const post = (path, body) => api(path, "POST", body);

function time(iso) { return new Date(iso).toLocaleTimeString(); }
function seconds(value) { return value === null || value === undefined ? "" : `${value.toFixed(1)} s`; }
function usd(value) { return value === null || value === undefined ? "" : `$${value.toFixed(value && value < 0.01 ? 4 : 2)}`; }
function pretty(value) { return JSON.stringify(value, null, 2); }
function flagChips(flags) {
  return flags.map((f) => el("span", { class: "flag", title: f.detail || "" }, f.detail ? `${f.name}: ${f.detail}` : f.name));
}
function failure(error) {
  if (error.status === 401) {
    return el("div", { class: "empty" }, el("h1", {}, "Sign in"),
      el("p", {}, "Run ", el("code", {}, "provingbench dashboard"), " and open the address it prints."));
  }
  return el("p", { class: "note" }, `Couldn't load this: ${error.message}`);
}

// Live log

function callDetail(call) {
  const parts = [];
  if (call.response.error) parts.push(el("h2", {}, "Error"), el("pre", {}, call.response.error));
  parts.push(el("h2", {}, "Reply"), el("pre", {}, call.response.text || "(nothing)"));
  if (call.tool_calls.length) parts.push(el("h2", {}, "Tool calls"), el("pre", {}, pretty(call.tool_calls)));
  if (call.response.raw_tokens) parts.push(el("h2", {}, "What the model actually wrote"), el("pre", {}, call.response.raw_tokens));
  parts.push(el("h2", {}, "Messages sent"), el("pre", {}, pretty(call.request.messages)));
  return parts;
}

function callRows(call, fresh) {
  const tools = call.tool_calls.map((t) => t.name).join(", ");
  const cells = el("tr", { class: `open-row${fresh ? " fresh" : ""}`, tabIndex: 0 },
    el("td", {}, time(call.at)), el("td", {}, call.agent_id), el("td", {}, call.model),
    el("td", { class: call.state === "ok" ? "" : "warn" }, call.state),
    el("td", {}, flagChips(call.flags)), el("td", {}, tools),
    el("td", { class: "num" }, seconds(call.ttft_s)), el("td", { class: "num" }, seconds(call.total_s)),
    el("td", { class: "num" }, call.prompt_tokens === null ? "" : `${call.prompt_tokens}/${call.completion_tokens}`),
    el("td", { class: "num" }, usd(call.cost_usd)));
  let detail = null;
  const toggle = () => {
    if (detail) { detail.remove(); detail = null; return; }
    detail = el("tr", { class: "detail" }, el("td", { colSpan: 10 }, callDetail(call)));
    cells.after(detail);
  };
  cells.addEventListener("click", toggle);
  cells.addEventListener("keydown", (e) => { if (e.key === "Enter") toggle(); });
  return cells;
}

const FLAG_NAMES = ["empty_reply", "unknown_tool", "dropped_tool_call", "bad_arguments", "promised_no_action",
  "upstream_error", "slow", "rounds_exhausted"];

async function liveView(root, signal) {
  const [{ agents }, { models }] = await Promise.all([api("/agents"), api("/models")]);
  const options = (names, all) => [el("option", { value: "" }, all), ...names.map((n) => el("option", { value: n }, n))];
  const agent = el("select", { onchange: reset }, options(agents.map((a) => a.id), "All agents"));
  const model = el("select", { onchange: reset }, options(models.map((m) => m.id), "All models"));
  const flag = el("select", { onchange: reset }, options(FLAG_NAMES, "Any or no flag"));
  const body = el("tbody");
  const empty = el("p", { class: "empty" },
    "No calls yet. Point an agent's base URL at ", el("code", {}, `${location.origin}/a/<agent-id>/v1`), " and they appear here.");
  let last = 0;

  function reset() {
    last = 0;
    body.replaceChildren();
    poll();
  }

  async function poll() {
    const q = new URLSearchParams({ after: String(last), limit: "100" });
    if (agent.value) q.set("agent", agent.value);
    if (model.value) q.set("model", model.value);
    if (flag.value) q.set("flag", flag.value);
    const asked = last;
    const { calls } = await api(`/calls?${q}`);
    if (signal.aborted || asked !== last) return;
    for (const call of calls) {
      body.prepend(callRows(call, asked !== 0));
      last = Math.max(last, call.id);
    }
    while (body.children.length > MAX_ROWS) body.lastChild.remove();
    empty.hidden = body.children.length > 0;
  }

  root.replaceChildren(el("h1", {}, "Live log"),
    el("div", { class: "filters" }, el("label", {}, "Agent", agent), el("label", {}, "Model", model), el("label", {}, "Flag", flag)),
    empty,
    el("div", { class: "scroll" }, el("table", {},
      el("thead", {}, el("tr", {}, ["Time", "Agent", "Model", "State", "Flags", "Tools"].map((t) => el("th", {}, t)),
        ["First token", "Total", "Tokens in/out", "Cost"].map((t) => el("th", { class: "num" }, t)))),
      body)));
  await poll();
  const timer = setInterval(() => poll().catch(() => {}), POLL_MS);
  signal.addEventListener("abort", () => clearInterval(timer));
}

// Models

function priceText(m) {
  if (m.price_prompt === null || m.price_completion === null) return "unknown";
  if (!m.price_prompt && !m.price_completion) return "free";
  return `$${(m.price_prompt * 1e6).toFixed(2)} in, $${(m.price_completion * 1e6).toFixed(2)} out`;
}

function agentRow(agent, models) {
  const cloud = new Set(models.filter((m) => m.cloud).map((m) => m.id));
  const choice = el("input", { value: agent.model || "", list: "model-ids", placeholder: "Agent's own choice", "aria-label": `Model for ${agent.id}`, size: 28 });
  const warning = el("p", { class: "note", hidden: !cloud.has(agent.model) }, CLOUD_WARNING);
  const status = el("span", { class: "dim", role: "status" });
  choice.addEventListener("input", () => { warning.hidden = !cloud.has(choice.value); });
  const allowed = el("input", { value: agent.allowed_models.join(", "), "aria-label": `Paid models ${agent.id} may use`, placeholder: "None", size: 28 });
  const allowedStatus = el("span", { class: "dim", role: "status" });
  const say = async (place, action, done) => {
    try { await action(); place.textContent = done; }
    catch (e) { place.textContent = e.message; }
  };
  return el("tr", {},
    el("td", {}, agent.id),
    el("td", {}, el("div", { class: "row" }, choice,
      el("button", { class: "primary", onclick: () => say(status, () => put(`/agents/${agent.id}/model`, { model: choice.value || null }),
        choice.value ? `Switched to ${choice.value}` : "Using the agent's own choice") }, "Switch"), status), warning),
    el("td", {}, el("div", { class: "row" }, allowed,
      el("button", { onclick: () => say(allowedStatus, () => put(`/agents/${agent.id}/allowed-models`, {
        models: allowed.value.split(",").map((s) => s.trim()).filter(Boolean) }), "Saved") }, "Save"), allowedStatus)));
}

async function modelsView(root) {
  const [{ agents }, { models }] = await Promise.all([api("/agents"), api("/models")]);
  const body = el("tbody");
  const search = el("input", { type: "search", placeholder: "Filter by name", "aria-label": "Filter models" });
  const toolsOnly = el("input", { type: "checkbox" });
  const draw = () => {
    const words = search.value.toLowerCase();
    body.replaceChildren(...models
      .filter((m) => m.id.toLowerCase().includes(words) && (!toolsOnly.checked || m.supports_tools !== false))
      .map((m) => el("tr", {}, el("td", {}, el("a", { href: `#/model/${encodeURIComponent(m.id)}` }, m.id)), el("td", {}, priceText(m)),
        el("td", { class: "num" }, m.context_length ? m.context_length.toLocaleString() : ""),
        el("td", {}, { true: "Yes", false: "No", null: "Unknown" }[m.supports_tools]))));
  };
  search.addEventListener("input", draw);
  toolsOnly.addEventListener("change", draw);
  draw();
  root.replaceChildren(el("h1", {}, "Models"),
    el("h2", {}, "Which agent uses which"),
    agents.length
      ? el("div", { class: "scroll" }, el("table", {},
          el("thead", {}, el("tr", {}, ["Agent", "Model", "Paid models it may use"].map((t) => el("th", {}, t)))),
          el("tbody", {}, agents.map((a) => agentRow(a, models)))))
      : el("p", { class: "empty" }, "No agents yet. Point one at ", el("code", {}, `${location.origin}/a/<agent-id>/v1`), "."),
    el("datalist", { id: "model-ids" }, models.map((m) => el("option", { value: m.id }))),
    el("h2", {}, "Available models"),
    ...(models.length ? [] : [el("p", { class: "note" }, "No models found. Is Ollama running? Run provingbench doctor to check.")]),
    el("div", { class: "filters" }, search, el("label", { class: "row" }, toolsOnly, "Can use tools")),
    el("div", { class: "scroll" }, el("table", {},
      el("thead", {}, el("tr", {}, el("th", {}, "Model"), el("th", {}, "Price per million tokens"), el("th", { class: "num" }, "Context"), el("th", {}, "Tools"))),
      body)));
}

// Scoreboard

function segments(cell) {
  const blocks = [];
  const add = (count, kind, label) => { for (let i = 0; i < count; i++) blocks.push(el("span", { class: `seg ${kind}`, title: label })); };
  add(cell.passed, "seg-pass", "passed");
  add(cell.failed, "seg-fail", "failed");
  add(cell.errors, "seg-error", "couldn't be graded");
  return el("span", { class: "segs", "aria-hidden": "true" }, blocks);
}

function reading(cell) {
  const graded = cell.passed + cell.failed;
  const kind = cell.rate === null ? "dim" : cell.rate === 1 ? "pass" : cell.rate === 0 ? "fail" : "warn";
  return el("a", { class: "reading", href: `#/run/${cell.run_id}` },
    el("span", { class: `value ${kind}` }, graded ? `${cell.passed}/${graded}` : "n/a"),
    segments(cell),
    cell.errors ? el("span", { class: "small dim" }, `${cell.errors} ungraded`) : null,
    cell.stale ? el("span", { class: "small warn" }, "task changed since") : null);
}

async function scoreboardView(root) {
  const board = await api("/scoreboard");
  if (!board.cells.length) {
    root.replaceChildren(el("h1", {}, "Scoreboard"), el("p", { class: "empty" }, "No results yet. ",
      el("a", { href: "#/run" }, "Start a run"), " to see how your models do on your tasks."));
    return;
  }
  const tags = [...new Set(board.tasks.flatMap((t) => t.tags))].sort();
  const tag = el("select", { onchange: draw }, el("option", { value: "" }, "All tags"), tags.map((t) => el("option", { value: t }, t)));
  const model = el("select", { onchange: draw }, el("option", { value: "" }, "All models"), board.models.map((m) => el("option", { value: m }, m)));
  const table = el("table", { class: "board" });
  const cells = new Map(board.cells.map((c) => [`${c.model}\n${c.task_id}`, c]));

  function draw() {
    const models = board.models.filter((m) => !model.value || m === model.value);
    const tasks = board.tasks.filter((t) => !tag.value || t.tags.includes(tag.value));
    const overall = (m) => {
      const mine = tasks.map((t) => cells.get(`${m}\n${t.id}`)).filter(Boolean);
      const passed = mine.reduce((sum, c) => sum + c.passed, 0);
      const graded = passed + mine.reduce((sum, c) => sum + c.failed, 0);
      return graded ? `${Math.round((100 * passed) / graded)}%` : "n/a";
    };
    table.replaceChildren(
      el("thead", {}, el("tr", {}, el("th", {}, "Task"), models.map((m) =>
        el("th", {}, el("a", { href: `#/model/${encodeURIComponent(m)}` }, m), el("div", { class: "overall" }, overall(m)))))),
      el("tbody", {}, tasks.map((t) => el("tr", {},
        el("th", { scope: "row" }, t.id, t.tags.length ? el("div", { class: "small dim" }, t.tags.join(", ")) : null),
        models.map((m) => { const c = cells.get(`${m}\n${t.id}`); return el("td", {}, c ? reading(c) : el("span", { class: "dim" }, "Not run")); })))));
  }
  draw();
  root.replaceChildren(el("h1", {}, "Scoreboard"),
    el("div", { class: "filters" }, el("label", {}, "Tag", tag), el("label", {}, "Model", model)),
    el("div", { class: "scroll" }, table));
}

// Runs

const STATE_TEXT = { running: "Running", done: "Finished", stopped: "Stopped", stopped_cost: "Stopped at its cost limit",
  interrupted: "Interrupted", failed: "Failed" };

function runRow(run) {
  return el("tr", {}, el("td", {}, el("a", { href: `#/run/${run.id}` }, `Run ${run.id}`)), el("td", {}, new Date(run.at).toLocaleString()),
    el("td", {}, run.target), el("td", {}, run.suite), el("td", {}, run.models.join(", ")), el("td", {}, run.label), el("td", {}, STATE_TEXT[run.state] || run.state));
}

async function startRunView(root, signal) {
  const [{ targets }, { suites }, { models }, { runs }] = await Promise.all([api("/targets"), api("/suites"), api("/models"), api("/runs")]);
  const recent = runs.length ? [el("h2", {}, "Recent runs"), el("div", { class: "scroll" }, el("table", {},
    el("thead", {}, el("tr", {}, ["Run", "Started", "Target", "Suite", "Models", "Label", "State"].map((t) => el("th", {}, t)))),
    el("tbody", {}, runs.map(runRow))))] : [];
  if (!targets.length || !suites.length) {
    root.replaceChildren(el("h1", {}, "Start a run"), el("div", { class: "empty" },
      targets.length ? null : el("p", {}, "No targets yet. Add a target file to the targets folder in ProvingBench's data folder. docs/connecting-agents.md shows how."),
      suites.length ? null : el("p", {}, "No suites yet. Name a folder of task files under [suites] in ProvingBench's config.")), ...recent);
    return;
  }
  const target = el("select", { onchange: estimate }, targets.map((t) => el("option", { value: t.id }, t.id)));
  const suite = el("select", { onchange: estimate }, suites.map((s) => el("option", { value: s.name, disabled: Boolean(s.error) }, `${s.name} (${s.error ? "has a mistake" : `${s.tasks} tasks`})`)));
  const problems = suites.filter((s) => s.error).map((s) => el("p", { class: "note" }, s.error));
  const chosen = [];
  const chips = el("div", { class: "row" });
  const adding = el("input", { list: "model-ids", placeholder: "Model name", "aria-label": "Model to add", size: 32 });
  const repeats = el("input", { type: "number", min: 1, placeholder: "3", oninput: estimate, style: "width:6em" });
  const limit = el("input", { type: "number", min: 0, step: "0.5", value: "1", oninput: estimate, style: "width:6em" });
  const label = el("input", { placeholder: "A version or commit", size: 24 });
  const estimateText = el("p", { class: "dim", role: "status" });
  const status = el("p", { class: "warn", role: "status" });

  function body() {
    const out = { target: target.value, suite: suite.value, models: chosen, label: label.value };
    if (repeats.value) out.repeats = Number(repeats.value);
    if (limit.value !== "") out.max_cost_usd = Number(limit.value);
    return out;
  }
  async function estimate() {
    if (!chosen.length) { estimateText.textContent = "Add a model to see the estimated cost."; return; }
    try {
      const { estimate_usd: usdValue } = await post("/estimate", body());
      estimateText.textContent = usdValue === null ? "Estimated cost: unknown, because a model's price isn't known." : `Estimated cost: about ${usd(usdValue)}`;
    } catch (e) { estimateText.textContent = e.message; }
  }
  function drawChips() {
    chips.replaceChildren(...chosen.map((m) => el("button", { type: "button", title: `Remove ${m}`, onclick: () => {
      chosen.splice(chosen.indexOf(m), 1); drawChips(); estimate();
    } }, `${m} ×`)));
  }
  function addModel() {
    const name = adding.value.trim();
    if (name && !chosen.includes(name)) chosen.push(name);
    adding.value = "";
    drawChips();
    estimate();
  }
  async function start(event) {
    event.preventDefault();
    if (adding.value.trim()) addModel();
    try {
      const { id } = await post("/runs", body());
      location.hash = `#/run/${id}`;
    } catch (e) { status.textContent = e.message; }
  }
  adding.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); addModel(); } });
  estimate();
  root.replaceChildren(el("h1", {}, "Start a run"), ...problems,
    el("form", { class: "stack", onsubmit: start },
      el("div", { class: "filters" }, el("label", {}, "Target", target), el("label", {}, "Suite", suite)),
      el("label", {}, "Models", el("div", { class: "row" }, adding, el("button", { type: "button", onclick: addModel }, "Add"))), chips,
      el("div", { class: "filters" }, el("label", {}, "Repeats per task", repeats), el("label", {}, "Cost limit in dollars", limit), el("label", {}, "Label", label)),
      estimateText, status, el("div", {}, el("button", { class: "primary", type: "submit" }, "Start run"))),
    el("datalist", { id: "model-ids" }, models.map((m) => el("option", { value: m.id }))), ...recent);
}

function firstProblem(result) {
  if (result.error && !result.checks.length) return result.error;
  const failed = result.checks.find((c) => c.status === "fail");
  return failed ? failed.reason : "";
}

function outcomeCell(outcome) {
  const text = { pass: "Passed", fail: "Failed", error: "Couldn't grade" }[outcome] || outcome;
  return el("td", { class: { pass: "pass", fail: "fail", error: "warn" }[outcome] || "" }, text);
}

async function runDetailView(root, signal, id) {
  const heading = el("h1", {}, `Run ${id}`);
  const summary = el("p", { class: "value-line", role: "status" });
  const actions = el("div", { class: "row" });
  const message = el("p", { class: "warn", role: "status" });
  const body = el("tbody");
  root.replaceChildren(heading, summary, actions, message, el("div", { class: "scroll" }, el("table", {},
    el("thead", {}, el("tr", {}, ["Model", "Task", "Attempt", "Result", "Why", "Time", "Cost"].map((t) => el("th", {}, t)))), body)));
  let shownState = null;
  let timer = null;

  function drawActions(run) {
    const limit = el("input", { type: "number", min: 0, step: "0.5", value: String(Math.max(run.max_cost_usd * 2, 1)), "aria-label": "New cost limit in dollars", style: "width:6em" });
    const act = async (path, payload) => {
      try { await post(path, payload); await poll(); } catch (e) { message.textContent = e.message; }
    };
    if (run.state === "running") {
      actions.replaceChildren(el("button", { onclick: () => act(`/runs/${id}/stop`, {}) }, "Stop run"));
    } else if (run.state === "stopped_cost") {
      actions.replaceChildren(el("label", {}, "New cost limit in dollars", limit),
        el("button", { class: "primary", onclick: () => act(`/runs/${id}/resume`, { max_cost_usd: Number(limit.value) }) }, "Raise limit and continue"));
    } else if (["stopped", "interrupted", "failed"].includes(run.state)) {
      actions.replaceChildren(el("button", { onclick: () => act(`/runs/${id}/resume`, {}) }, "Continue run"));
    } else actions.replaceChildren();
  }

  async function poll() {
    const run = await api(`/runs/${id}`);
    if (signal.aborted) return;
    summary.textContent = `${STATE_TEXT[run.state] || run.state}. ${run.results.length} of ${run.total} attempts done. Spent ${usd(run.spent_usd)} of ${usd(run.max_cost_usd)}.`;
    if (run.error) message.textContent = run.error;
    if (run.state !== shownState) { shownState = run.state; drawActions(run); }
    body.replaceChildren(...run.results.map((r) => el("tr", {},
      el("td", {}, el("a", { href: `#/model/${encodeURIComponent(r.model)}` }, r.model)), el("td", {}, r.task_id),
      el("td", {}, el("a", { href: `#/result/${r.id}` }, `Attempt ${r.repeat + 1}`)), outcomeCell(r.override ? r.override.outcome : r.outcome),
      el("td", {}, r.override ? `You overruled this: ${r.override.note}` : firstProblem(r)), el("td", {}, seconds(r.seconds)), el("td", {}, usd(r.cost_usd)))));
    if (run.state === "running" && !timer) {
      timer = setInterval(() => poll().catch(() => {}), POLL_MS);
      signal.addEventListener("abort", () => clearInterval(timer));
    } else if (run.state !== "running" && timer) { clearInterval(timer); timer = null; }
  }
  await poll();
}

function runView(root, signal, id) { return id ? runDetailView(root, signal, id) : startRunView(root, signal); }

// One attempt

function answeredText(tool) {
  return { real: "ran in the agent", stubbed: "answered with the task's canned result", blocked: "blocked" }[tool.answered] || tool.answered;
}

function transcript(calls) {
  const blocks = [];
  if (calls.length) {
    const first = calls[0].request.messages.filter((m) => m.role !== "system" && m.role !== "tool" && !m.tool_calls);
    blocks.push(el("h2", {}, "The conversation"), el("pre", {}, first.map((m) => `${m.role}: ${typeof m.content === "string" ? m.content : pretty(m.content)}`).join("\n\n")));
  }
  calls.forEach((call, i) => {
    blocks.push(el("h2", {}, `Model call ${i + 1}`), el("p", { class: "dim" }, [call.model, seconds(call.total_s), usd(call.cost_usd)].filter(Boolean).join(", ")));
    if (call.flags.length) blocks.push(el("p", {}, flagChips(call.flags)));
    if (call.tool_calls.length) blocks.push(el("ul", {}, call.tool_calls.map((t) => el("li", {}, el("code", {}, `${t.name}(${t.arguments})`), `: ${answeredText(t)}`))));
    if (call.response.text) blocks.push(el("pre", {}, call.response.text));
    if (call.response.error) blocks.push(el("pre", {}, call.response.error));
    if (call.response.raw_tokens && !call.response.text) blocks.push(el("h2", {}, "What the model actually wrote"), el("pre", {}, call.response.raw_tokens));
  });
  return blocks;
}

function overrideControl(r, redraw) {
  const flipped = r.outcome === "pass" ? "fail" : "pass";
  const note = el("input", { size: 50, "aria-label": "Why you are overruling this grade", placeholder: "Why this grade is wrong" });
  const status = el("span", { class: "warn", role: "status" });
  const save = async () => {
    try { await post(`/results/${r.id}/override`, { outcome: flipped, note: note.value }); redraw(); }
    catch (e) { status.textContent = e.message; }
  };
  return el("div", { class: "row" }, note, el("button", { onclick: save }, `Count this as ${flipped === "pass" ? "passed" : "failed"}`), status);
}

async function resultView(root, signal, id) {
  const r = await api(`/results/${id}`);
  const redraw = () => resultView(root, signal, id).catch((e) => root.replaceChildren(failure(e)));
  const ruling = r.override
    ? el("p", { class: "note" }, `You counted this as ${r.override.outcome === "pass" ? "passed" : "failed"}: ${r.override.note}`)
    : "";
  root.replaceChildren(
    el("h1", {}, r.task_id),
    el("p", {}, el("a", { href: `#/model/${encodeURIComponent(r.model)}` }, r.model), `, attempt ${r.repeat + 1} of `, el("a", { href: `#/run/${r.run_id}` }, `run ${r.run_id}`)),
    el("table", { class: "plain" }, el("tbody", {}, el("tr", {}, el("th", {}, "Result"), outcomeCell(r.outcome)),
      el("tr", {}, el("th", {}, "Time"), el("td", {}, seconds(r.seconds))), el("tr", {}, el("th", {}, "Cost"), el("td", {}, usd(r.cost_usd))))),
    ruling,
    r.outcome === "error" ? "" : overrideControl(r, redraw),
    el("h2", {}, "Checks"),
    r.checks.length
      ? el("table", {}, el("tbody", {}, r.checks.map((c) => el("tr", {}, el("td", { class: { pass: "pass", fail: "fail" }[c.status] || "dim" }, { pass: "Passed", fail: "Failed", skipped: "Skipped" }[c.status]),
          el("td", {}, c.type), el("td", {}, c.reason)))))
      : el("p", { class: "dim" }, "No checks ran, because the agent failed before it could be graded."),
    r.error ? el("div", {}, el("h2", {}, "The agent's error"), el("pre", {}, r.error)) : "",
    el("h2", {}, "Final reply"), el("pre", {}, r.reply || "(nothing)"),
    ...transcript(r.calls));
}

// One model

function modelPath(id) { return `/models/${id.split("/").map(encodeURIComponent).join("/")}`; }

async function modelView(root, signal, id) {
  const p = await api(`${modelPath(id)}/profile`);
  const tags = Object.entries(p.by_tag).sort((a, b) => a[0].localeCompare(b[0]));
  root.replaceChildren(
    el("h1", {}, p.model),
    p.last_run
      ? el("p", { class: "dim" }, `Last tested ${new Date(p.last_run).toLocaleString()}.`)
      : el("p", { class: "empty" }, "Not tested yet. ", el("a", { href: "#/run" }, "Start a run"), " with this model."),
    el("h2", {}, "How it does by tag"),
    tags.length
      ? el("table", {}, el("tbody", {}, tags.map(([tag, rate]) => el("tr", {}, el("th", {}, tag),
          el("td", {}, rate === null ? "n/a" : `${Math.round(rate * 100)}%`), el("td", { class: "wide" }, el("span", { class: "gauge" }, el("span", { class: "gauge-fill", style: `width:${(rate || 0) * 100}%` })))))))
      : el("p", { class: "dim" }, "No tagged tasks have run."),
    el("h2", {}, "Tasks it fails"),
    p.failed_tasks.length ? el("ul", {}, p.failed_tasks.map((t) => el("li", {}, t))) : el("p", { class: "dim" }, "None."),
    el("h2", {}, "Problems seen in its calls"),
    p.flags.length ? el("ul", {}, p.flags.map((f) => el("li", {}, `${f.name}: ${f.count}`))) : el("p", { class: "dim" }, "None."),
    el("h2", {}, "Speed and cost"),
    el("p", {}, `Median time to first token: ${p.median_ttft_s === null ? "unknown" : seconds(p.median_ttft_s)}. Cost per attempt: ${p.cost_per_task === null ? "unknown" : usd(p.cost_per_task)}.`),
    el("h2", {}, "Notes"), noteEditor(id, p.notes));
}

function noteEditor(id, text) {
  const box = el("textarea", { rows: 4, cols: 60, value: text, "aria-label": "Notes on this model" });
  const status = el("span", { class: "dim", role: "status" });
  const save = async () => {
    try { await put(`${modelPath(id)}/notes`, { text: box.value }); status.textContent = "Saved"; }
    catch (e) { status.textContent = e.message; }
  };
  return el("div", { class: "stack" }, box, el("div", { class: "row" }, el("button", { onclick: save }, "Save notes"), status));
}

// Router

const NAV = [["scoreboard", "Scoreboard"], ["run", "Runs"], ["models", "Models"], ["live", "Live log"]];
const VIEWS = { scoreboard: scoreboardView, run: runView, result: resultView, model: modelView, live: liveView, models: modelsView };
const DEFAULT_VIEW = "scoreboard";
const NAV_FOR = { result: "run", model: "models" };

let current = null;

async function route() {
  if (current) current.abort();
  current = new AbortController();
  const [name, ...args] = location.hash.replace(/^#\/?/, "").split("/");
  const view = VIEWS[name] ? name : DEFAULT_VIEW;
  document.getElementById("nav").replaceChildren(...NAV.map(([id, text]) =>
    el("a", { href: `#/${id}`, "aria-current": id === (NAV_FOR[view] || view) ? "page" : null }, text)));
  const root = document.getElementById("view");
  root.replaceChildren();
  try {
    await VIEWS[view](root, current.signal, ...args.map(decodeURIComponent));
  } catch (e) {
    root.replaceChildren(failure(e));
  }
}

async function watchStatus() {
  const banner = document.getElementById("banner");
  try {
    const { log_errors: errors } = await api("/status");
    banner.hidden = !errors;
    banner.textContent = errors ? `${errors} log write${errors === 1 ? "" : "s"} failed. Agents weren't affected, but this log has gaps.` : "";
  } catch { /* the view reports it */ }
}

addEventListener("hashchange", route);
route();
watchStatus();
setInterval(watchStatus, POLL_MS * 5);
