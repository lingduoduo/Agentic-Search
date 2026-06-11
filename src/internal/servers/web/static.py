"""Static assets for the lightweight search/agent web experience."""

APP_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Agentic Search</title>
    <link rel="stylesheet" href="/assets/app.css" />
  </head>
  <body>
    <main class="shell">
      <section class="workspace" aria-label="Agentic search workspace">
        <header class="topbar">
          <div>
            <h1>Agentic Search</h1>
            <p>Ask, retrieve, and inspect grounded answers from your local search stack.</p>
          </div>
          <div class="status" id="status">Ready</div>
        </header>

        <section class="composer" aria-label="Search composer">
          <textarea id="query" rows="3" placeholder="Ask a question about your indexed content"></textarea>
          <div class="controls">
            <label>
              Retrieval URL
              <input id="search-url" value="http://localhost:8000/retrieve" />
            </label>
            <label>
              Top K
              <input id="top-k" type="number" value="5" min="1" max="20" />
            </label>
            <button id="submit" type="button">Search</button>
          </div>
        </section>

        <section class="answer-pane" aria-live="polite">
          <h2>Answer</h2>
          <div id="answer" class="answer empty">Results will appear here.</div>
        </section>

        <section>
          <div class="section-title">
            <h2>Sources</h2>
            <span id="source-count">0</span>
          </div>
          <div id="sources" class="sources"></div>
        </section>

        <section>
          <div class="section-title">
            <h2>Session</h2>
            <button id="new-session" type="button" class="secondary">New</button>
          </div>
          <div id="messages" class="messages"></div>
        </section>
      </section>
    </main>
    <script src="/assets/app.js"></script>
  </body>
</html>
"""

APP_CSS = """
:root {
  color-scheme: light;
  --bg: #f7f8f3;
  --panel: #ffffff;
  --ink: #1d2428;
  --muted: #647072;
  --line: #d9dfdc;
  --accent: #0f766e;
  --accent-dark: #115e59;
  --wash: #e9f3ef;
  --mark: #fff4bf;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family:
    Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
    sans-serif;
  background: var(--bg);
  color: var(--ink);
}

button,
input,
textarea {
  font: inherit;
}

.shell {
  min-height: 100vh;
  padding: 24px;
}

.workspace {
  max-width: 1120px;
  margin: 0 auto;
  display: grid;
  gap: 18px;
}

.topbar,
.composer,
.answer-pane,
section {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}

.topbar {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 24px;
  padding: 22px;
}

h1,
h2,
p {
  margin: 0;
}

h1 {
  font-size: 30px;
  line-height: 1.15;
}

h2 {
  font-size: 16px;
}

.topbar p {
  margin-top: 6px;
  color: var(--muted);
}

.status {
  min-width: 84px;
  border-radius: 999px;
  background: var(--wash);
  color: var(--accent-dark);
  padding: 7px 12px;
  text-align: center;
  font-size: 13px;
  font-weight: 650;
}

.composer {
  padding: 16px;
}

textarea {
  width: 100%;
  min-height: 108px;
  resize: vertical;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  color: var(--ink);
}

.controls {
  display: grid;
  grid-template-columns: minmax(220px, 1fr) 110px auto;
  gap: 12px;
  align-items: end;
  margin-top: 12px;
}

label {
  display: grid;
  gap: 6px;
  color: var(--muted);
  font-size: 13px;
  font-weight: 650;
}

input {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
  color: var(--ink);
}

button {
  border: 0;
  border-radius: 8px;
  min-height: 42px;
  padding: 0 16px;
  background: var(--accent);
  color: white;
  cursor: pointer;
  font-weight: 700;
}

button:hover {
  background: var(--accent-dark);
}

button.secondary {
  background: transparent;
  border: 1px solid var(--line);
  color: var(--ink);
}

.answer-pane,
section {
  padding: 16px;
}

.answer {
  margin-top: 12px;
  line-height: 1.65;
  white-space: pre-wrap;
}

.answer.empty {
  color: var(--muted);
}

.section-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}

#source-count {
  color: var(--muted);
  font-size: 13px;
}

.sources {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px;
  margin-top: 12px;
}

.source-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  min-height: 132px;
  display: grid;
  gap: 8px;
}

.source-card a {
  color: var(--accent-dark);
  font-weight: 750;
  text-decoration: none;
}

.source-card p {
  color: var(--muted);
  font-size: 14px;
  line-height: 1.45;
}

.badge {
  width: fit-content;
  border-radius: 999px;
  background: var(--mark);
  padding: 4px 8px;
  font-size: 12px;
  font-weight: 750;
}

.messages {
  display: grid;
  gap: 10px;
  margin-top: 12px;
}

.message {
  border-left: 3px solid var(--line);
  padding-left: 10px;
  color: var(--muted);
}

.message strong {
  color: var(--ink);
}

@media (max-width: 720px) {
  .shell {
    padding: 12px;
  }

  .topbar,
  .controls {
    grid-template-columns: 1fr;
  }

  .topbar {
    display: grid;
  }
}
"""

APP_JS = """
const state = {
  sessionId: null,
};

const queryEl = document.getElementById("query");
const searchUrlEl = document.getElementById("search-url");
const topKEl = document.getElementById("top-k");
const submitEl = document.getElementById("submit");
const statusEl = document.getElementById("status");
const answerEl = document.getElementById("answer");
const sourcesEl = document.getElementById("sources");
const sourceCountEl = document.getElementById("source-count");
const messagesEl = document.getElementById("messages");
const newSessionEl = document.getElementById("new-session");

function setStatus(text) {
  statusEl.textContent = text;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderSources(documents) {
  sourceCountEl.textContent = String(documents.length);
  sourcesEl.innerHTML = documents
    .map((doc) => {
      const title = escapeHtml(doc.title || doc.id);
      const content = escapeHtml(doc.content || "").slice(0, 360);
      const score = Number.isFinite(doc.score) ? doc.score.toFixed(3) : "0.000";
      const heading = doc.url
        ? `<a href="${escapeHtml(doc.url)}" target="_blank" rel="noreferrer">${title}</a>`
        : `<strong>${title}</strong>`;
      return `<article class="source-card">
        <span class="badge">${escapeHtml(doc.citation)} score ${score}</span>
        ${heading}
        <p>${content}</p>
      </article>`;
    })
    .join("");
}

function renderMessages(messages) {
  messagesEl.innerHTML = messages
    .map(
      (message) =>
        `<div class="message"><strong>${escapeHtml(message.role)}</strong><br />${escapeHtml(
          message.content
        )}</div>`
    )
    .join("");
}

async function createSession() {
  const response = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "Search session" }),
  });
  if (!response.ok) {
    throw new Error("Could not create session");
  }
  const data = await response.json();
  state.sessionId = data.id;
  renderMessages([]);
}

async function submitQuery() {
  const query = queryEl.value.trim();
  if (!query) {
    queryEl.focus();
    return;
  }
  submitEl.disabled = true;
  setStatus("Searching");
  answerEl.classList.remove("empty");
  answerEl.textContent = "";
  try {
    if (!state.sessionId) {
      await createSession();
    }
    const response = await fetch("/api/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        session_id: state.sessionId,
        search_url: searchUrlEl.value,
        top_k: Number(topKEl.value || 5),
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Search failed");
    }
    state.sessionId = data.session_id;
    answerEl.textContent = data.answer;
    renderSources(data.documents || []);
    renderMessages(data.messages || []);
    setStatus("Ready");
  } catch (error) {
    answerEl.textContent = error.message;
    renderSources([]);
    setStatus("Error");
  } finally {
    submitEl.disabled = false;
  }
}

submitEl.addEventListener("click", submitQuery);
queryEl.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    submitQuery();
  }
});
newSessionEl.addEventListener("click", async () => {
  setStatus("New");
  await createSession();
  answerEl.textContent = "Results will appear here.";
  answerEl.classList.add("empty");
  renderSources([]);
  setStatus("Ready");
});

createSession().catch(() => setStatus("Error"));
"""
