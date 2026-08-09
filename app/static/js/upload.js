/* Uploading a report, and watching it being ingested.
 *
 * The server does the work on a background thread and records its progress in
 * the database as it goes, so this only has to poll one endpoint. That also
 * means a reload does not lose the display: the progress is a fact about the
 * database rather than about this page. */

/* The upload form only exists on the reports page; the remove buttons are on
   both. Everything below therefore checks before it reaches for an element. */
const form = document.getElementById("upload");
const status = document.getElementById("upload-status");
const button = document.getElementById("upload-go");

const STAGES = ["parse", "chunk", "embed", "extract"];
const LABELS = { parse: "Reading the PDF", chunk: "Splitting into chunks",
                 embed: "Building the index", extract: "Extracting datapoints" };

const escapeHtml = (text) =>
  String(text).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function show(html, variant) {
  if (!status) return;
  status.hidden = false;
  status.className = variant ? `alert` : "";
  if (variant) status.dataset.variant = variant; else delete status.dataset.variant;
  status.innerHTML = html;
}

function render(data) {
  const rows = STAGES.map((stage) => {
    const state = data.stages?.[stage]?.status ?? "pending";
    const mark = state === "running"
      ? '<span class="dots"><i></i><i></i><i></i></span>'
      : ({ done: "✓", failed: "✕" }[state] ?? "·");
    const detail = stage === "parse" && state === "running" && data.pages
      ? ` ${data.pages_parsed}/${data.pages} pages`
      : "";
    return `<li class="step ${state}"><span class="mark">${mark}</span>
              ${LABELS[stage]}${detail}</li>`;
  }).join("");

  const bar = data.percent
    ? `<div class="bar"><span style="width:${data.percent}%"></span></div>` : "";

  if (data.error) {
    show(`<strong>${escapeHtml(data.company)}</strong> failed.
          <br><small>${escapeHtml(data.error)}</small>`, "destructive");
    return true;
  }
  if (data.done) {
    show(`<strong>${escapeHtml(data.company)}</strong> is ready:
          ${data.chunks} chunks, ${data.facts} datapoints.
          <a href="/documents/${data.id}">Open it</a>, or reload this page.`);
    return true;
  }
  show(`<strong>Ingesting ${escapeHtml(data.company)}</strong>${bar}
        <ul class="steps">${rows}</ul>`);
  return false;
}

async function poll(id) {
  /* Every two seconds. Parsing takes minutes and the numbers move slowly, so
   * anything faster is load for no extra information. */
  while (true) {
    const response = await fetch(`/api/documents/${id}/progress`);
    if (!response.ok) {
      show("Lost track of that ingest. Reload to see where it got to.", "destructive");
      return;
    }
    if (render(await response.json())) return;
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}

form?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = document.getElementById("pdf").files[0];
  if (!file) return;

  button.disabled = true;
  show(`Uploading ${escapeHtml(file.name)}…`);

  const body = new FormData();
  body.append("file", file);
  const company = document.getElementById("company").value.trim();
  const year = document.getElementById("year").value.trim();
  if (company) body.append("company", company);
  if (year) body.append("year", year);

  try {
    const response = await fetch("/api/documents", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) {
      show(escapeHtml(data.detail || response.statusText), "destructive");
      button.disabled = false;
      return;
    }
    if (!data.started) {
      show(`<strong>${escapeHtml(data.company)}</strong> is already indexed.
            <a href="/documents/${data.id}">Open it</a>.`);
      button.disabled = false;
      return;
    }
    await poll(data.id);
  } catch (error) {
    show(escapeHtml(String(error)), "destructive");
  } finally {
    button.disabled = false;
  }
});


/* Removing a report from the index.
 *
 * Lives here rather than in its own file because it shares the page and the
 * feedback element with uploading, and the two are the same job seen from
 * opposite ends. */

async function remove(button) {
  const { id, company, redirect } = button.dataset;
  const confirmed = window.confirm(
    `Remove ${company} from the index?\n\n` +
    `Its blocks, chunks, embeddings and extracted datapoints are deleted. ` +
    `The source PDF stays on disk, so you can add it again later.`
  );
  if (!confirmed) return;

  button.disabled = true;
  try {
    const response = await fetch(`/api/documents/${id}`, { method: "DELETE" });
    const data = await response.json();
    if (!response.ok) {
      if (status) show(escapeHtml(data.detail || response.statusText), "destructive");
      button.disabled = false;
      return;
    }
    if (redirect) {
      window.location.href = redirect;
      return;
    }
    /* Take the card out rather than reloading, so the page does not jump. */
    button.closest(".card")?.remove();
    if (status) {
      const { chunks, facts } = data.removed;
      show(`Removed <strong>${escapeHtml(data.company)}</strong>:
            ${chunks} chunks and ${facts} datapoints. The PDF is still on disk.`);
    }
  } catch (error) {
    if (status) show(escapeHtml(String(error)), "destructive");
    button.disabled = false;
  }
}

document.querySelectorAll(".remove").forEach((button) =>
  button.addEventListener("click", () => remove(button)));
