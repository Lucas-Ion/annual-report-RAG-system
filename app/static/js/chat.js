/* Streaming chat.
 *
 * No framework. The whole job is: post a question, read server-sent events off
 * the response as they arrive, and render three kinds of thing. That is about
 * a hundred lines by hand and nothing else on the page needs JavaScript at all.
 *
 * Read with fetch and a ReadableStream rather than EventSource, because
 * EventSource can only issue GET requests and the question belongs in a body.
 */

const thread = document.getElementById("thread");
const form = document.getElementById("composer");
const box = document.getElementById("question");
const send = document.getElementById("send");
const scope = document.getElementById("scope");

let conversationId = thread.dataset.conversation || null;
let busy = false;
/* Kept between events so the citation renderer can map a marker number back to
   the source it came from, and therefore to a page in a PDF. */
let lastSources = [];

const escapeHtml = (text) =>
  text.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

/* Answers arrive as plain text with citation markers in them. Rendering is
 * deliberately minimal: **bold**, and markers turned into badges. Anything
 * more is a markdown parser, which is a dependency and an XSS surface for very
 * little gain on text this short. */
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
      /* A verified citation links straight to the page it was checked
       * against, so a reader can confirm the quotation rather than trust it. */
      if (source && page) {
        return `<a class="cite ${ok ? "" : "unverified"}" title="${title}" target="_blank"
                   rel="noopener" href="/documents/${source.document_id}/pdf#page=${page}">${n}</a>`;
      }
      return `<sup class="cite ${ok ? "" : "unverified"}" title="${title}">${n}</sup>`;
    });
}

/* One row per source. Kept tight on purpose: the excerpts are whole chunks and
 * printing them in full turns a two paragraph answer into a wall. The snippet
 * is clamped to three lines by CSS, which fades rather than cutting mid-glyph. */
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

function sourcesBlock(sources, label = "sources used") {
  if (!sources || !sources.length) return "";
  const items = sources.map((s, i) => sourceRow(s, s.n ?? i + 1)).join("");
  return `<details class="sources">
            <summary>${sources.length} ${label}</summary>
            <ul>${items}</ul>
          </details>`;
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
  const answer = addTurn("assistant", '<span class="caret"></span>');
  const bubble = answer.querySelector(".bubble");

  let text = "";
  let sources = [];

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        conversation_id: conversationId ? Number(conversationId) : null,
        document_id: scope.value ? Number(scope.value) : null,
      }),
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      bubble.innerHTML = `<div class="alert" data-variant="destructive">${escapeHtml(detail.detail || response.statusText)}</div>`;
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    /* SSE frames are separated by a blank line and can be split across reads,
     * so whatever is left after the last complete frame stays in the buffer. */
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
        } else if (name === "token") {
          text += data;
          bubble.innerHTML = render(text, []) + '<span class="caret"></span>';
          thread.scrollTop = thread.scrollHeight;
        } else if (name === "citations") {
          bubble.innerHTML = render(text, data) + sourcesBlock(sources);
        } else if (name === "error") {
          bubble.innerHTML = `<div class="alert" data-variant="destructive">${escapeHtml(data.message)}</div>`;
        }
      }
    }
    if (bubble.querySelector(".caret")) {
      bubble.innerHTML = render(text, []) + sourcesBlock(sources);
    }
  } catch (error) {
    bubble.innerHTML = `<div class="alert" data-variant="destructive">${escapeHtml(String(error))}</div>`;
  } finally {
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

/* Enter sends, shift+enter breaks the line. */
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

/* Reopening a conversation.
 *
 * The stored message is the answer exactly as written, markers and all, so it
 * has to go through the same renderer a live answer does. Rendering it raw was
 * the reason a reopened thread showed bare "[3: ...]" text and no sources. */
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
    /* Only the citations survive in the database, not every excerpt that was
     * retrieved, so a reopened thread lists what the answer actually cited. */
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


/* Renaming a conversation.
 *
 * Inline rather than a prompt dialog: the title is right there and swapping it
 * for an input is less disruptive than a modal for a three word edit. */
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
