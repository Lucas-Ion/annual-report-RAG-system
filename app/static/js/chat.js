/* Streaming chat which
 * posts a question, read server-sent events off the response as they arrive
 * a hundred lines by hand and nothing else on the page needs JavaScript at all.
 */

const thread = document.getElementById("thread");
const form = document.getElementById("composer");
const box = document.getElementById("question");
const send = document.getElementById("send");

let conversationId = thread.dataset.conversation || null;
let busy = false;
let lastSources = [];

const escapeHtml = (text) =>
  text.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function render(text, citations) {
  const byQuote = new Map((citations || []).map((c) => [c.quote, c]));
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[(\d+)(?::\s*(?:&quot;|")([^&"]*)(?:&quot;|"))?\]/g, (_, n, quote) => {
      const found = quote ? byQuote.get(quote.trim()) : null;
      const ok = found ? found.verified : false;
      const source = lastSources[Number(n) - 1]
        ?? (found ? { document_id: found.document_id } : null);
      const page = found ? found.page : source ? source.page_start : null;
      const title = quote ? `p${page ?? "?"} — ${quote}` : `source ${n}`;
      if (source && page) {
        return `<a class="cite ${ok ? "" : "unverified"}" title="${title}" target="_blank"
                   rel="noopener" href="/documents/${source.document_id}/pdf#page=${page}">${n}</a>`;
      }
      return `<sup class="cite ${ok ? "" : "unverified"}" title="${title}">${n}</sup>`;
    });
}


function sourceRow(source, index) {
  const page = source.page_start ?? source.page;
  const link = source.document_id
    ? `<a class="page-link" target="_blank" rel="noopener"
         href="/documents/${source.document_id}/pdf#page=${page}">p${page}</a>`
    : `<span class="page">p${page}</span>`;
  const body = source.text ?? source.quote ?? "";
  return `
    <li class="source">
      <div class="source-head">
        <span class="source-n">${index}</span>
        ${source.logo ? `<span class="logo xs"><img src="${source.logo}" alt=""></span>` : ""}
        <strong>${escapeHtml(source.company ?? "")}</strong>
        ${link}
        ${source.section ? `<span class="source-section">${escapeHtml(source.section)}</span>` : ""}
      </div>
      ${body ? `<p class="snippet">${escapeHtml(body)}</p>` : ""}
    </li>`;
}

function sourcesBlock(sources, label = "sources used", seconds = null) {
  if (!sources || !sources.length) return "";
  const items = sources.map((s, i) => sourceRow(s, s.n ?? i + 1)).join("");
  const took = seconds === null ? "" : ` · ${seconds.toFixed(1)}s`;
  return `<details class="sources">
            <summary>${sources.length} ${label}${took}</summary>
            <ul>${items}</ul>
          </details>`;
}


function createLoader(label) {
  const element = document.createElement("span");
  element.className = "loading";
  element.setAttribute("role", "status");
  element.setAttribute("aria-live", "polite");
  element.innerHTML = `
    <span class="pixels" aria-hidden="true">${"<i></i>".repeat(9)}</span>
    <span class="loading-label">${escapeHtml(label)}</span>
    <span class="loading-time">0.0s</span>`;

  const time = element.querySelector(".loading-time");
  const started = performance.now();
  const tick = () => { time.textContent = format(performance.now() - started); };
  const timer = setInterval(tick, 100);

  return {
    element,
    setLabel(next) {
      element.querySelector(".loading-label").textContent = next;
    },

    stop() {
      clearInterval(timer);
      return (performance.now() - started) / 1000;
    },
  };
}

function format(ms) {
  const seconds = ms / 1000;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(1)}s`;
}

function addTurn(role, html) {
  const el = document.createElement("article");
  el.className = `turn ${role}`;
  el.innerHTML = `<div class="bubble">${html}</div>`;
  thread.appendChild(el);
  thread.scrollTop = thread.scrollHeight;
  return el;
}

async function ask(question) {
  if (busy || !question.trim()) return;
  busy = true;
  send.disabled = true;
  document.querySelector(".welcome")?.remove();

  addTurn("user", escapeHtml(question));
  const answer = addTurn("assistant", "");
  const bubble = answer.querySelector(".bubble");

  const loader = createLoader("Searching the reports");
  bubble.appendChild(loader.element);

  let text = "";
  let sources = [];
  let elapsed = null;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        conversation_id: conversationId ? Number(conversationId) : null,
      }),
    });

    if (!response.ok) {
      loader.stop();
      const detail = await response.json().catch(() => ({}));
      bubble.innerHTML = `<div class="alert" data-variant="destructive">${escapeHtml(detail.detail || response.statusText)}</div>`;
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";


    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";

      for (const frame of frames) {
        const name = frame.match(/^event: (.+)$/m)?.[1];
        const raw = frame.match(/^data: ([\s\S]*)$/m)?.[1];
        if (!name || raw === undefined) continue;
        const data = JSON.parse(raw);

        if (name === "meta") {
          conversationId = data.conversation_id;
          thread.dataset.conversation = conversationId;
          sources = data.sources;
          lastSources = data.sources;

          loader.setLabel(`Reading ${sources.length} excerpts`);
        } else if (name === "token") {

          if (elapsed === null) elapsed = loader.stop();
          text += data;
          bubble.innerHTML = render(text, []) + '<span class="caret"></span>';
          thread.scrollTop = thread.scrollHeight;
        } else if (name === "citations") {
          if (elapsed === null) elapsed = loader.stop();
          bubble.innerHTML =
            render(text, data) + sourcesBlock(sources, "sources used", elapsed);
        } else if (name === "error") {
          loader.stop();
          bubble.innerHTML = `<div class="alert" data-variant="destructive">${escapeHtml(data.message)}</div>`;
        }
      }
    }
    if (bubble.querySelector(".caret")) {
      bubble.innerHTML =
        render(text, []) + sourcesBlock(sources, "sources used", elapsed);
    }
  } catch (error) {
    bubble.innerHTML = `<div class="alert" data-variant="destructive">${escapeHtml(String(error))}</div>`;
  } finally {

    loader.stop();
    busy = false;
    send.disabled = false;
    box.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = box.value;
  box.value = "";
  box.style.height = "auto";
  ask(question);
});


box.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

box.addEventListener("input", () => {
  box.style.height = "auto";
  box.style.height = Math.min(box.scrollHeight, 144) + "px";
});

document.querySelectorAll(".suggest").forEach((button) =>
  button.addEventListener("click", () => ask(button.textContent.trim())));


async function restore(id) {
  const response = await fetch(`/api/conversations/${id}`);
  if (!response.ok) return;
  const data = await response.json();

  for (const message of data.messages) {
    if (message.role === "user") {
      addTurn("user", escapeHtml(message.content));
      continue;
    }
    const citations = message.citations ?? [];
    const turn = addTurn("assistant", render(message.content, citations));

    const verified = citations.filter((c) => c.verified);
    if (verified.length) {
      turn.querySelector(".bubble").innerHTML +=
        sourcesBlock(verified, "sources cited");
    }
  }
  thread.scrollTop = thread.scrollHeight;
}

if (conversationId) restore(conversationId);

thread.scrollTop = thread.scrollHeight;
box.focus();


function startRename(item) {
  const link = item.querySelector("a");
  if (item.querySelector("input")) return;

  const input = document.createElement("input");
  input.className = "input rename-field";
  input.value = link.textContent.trim();
  link.hidden = true;
  item.insertBefore(input, link);
  input.focus();
  input.select();

  const finish = async (save) => {
    const title = input.value.trim();
    input.remove();
    link.hidden = false;
    if (!save || !title || title === link.textContent.trim()) return;
    const response = await fetch(`/api/conversations/${item.dataset.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
    if (response.ok) link.textContent = title;
  };

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") finish(true);
    if (event.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => finish(true));
}

document.querySelectorAll(".thread .rename").forEach((button) =>
  button.addEventListener("click", (event) => {
    event.preventDefault();
    startRename(button.closest(".thread"));
  }));


async function removeThread(button) {
  const item = button.closest(".thread");
  const title = button.dataset.title || "this conversation";

  const confirmed = await confirmAction({
    title: `Delete "${title}"?`,
    body: "The questions, the answers and their citations are removed. This "
        + "cannot be undone.",
  });
  if (!confirmed) return;

  button.disabled = true;
  const response = await fetch(`/api/conversations/${item.dataset.id}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    button.disabled = false;
    return;
  }

  if (String(conversationId) === item.dataset.id) {
    window.location.href = "/chat";
    return;
  }
  item.remove();
}

document.querySelectorAll(".thread .remove-thread").forEach((button) =>
  button.addEventListener("click", (event) => {
    event.preventDefault();
    removeThread(button);
  }));
