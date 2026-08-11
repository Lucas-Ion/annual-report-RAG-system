/* A confirmation dialog used for deletion*/

function confirmAction({ title, body, confirmLabel = "Delete" }) {
  const dialog = document.getElementById("confirm-dialog");
  if (!dialog) return Promise.resolve(window.confirm(`${title}\n\n${body}`));

  dialog.querySelector("[data-title]").textContent = title;
  dialog.querySelector("[data-description]").textContent = body;
  const go = dialog.querySelector('[value="confirm"]');
  go.textContent = confirmLabel;

  return new Promise((resolve) => {
    const finish = (answer) => {
      dialog.close();
      dialog.removeEventListener("close", dismissed);
      go.removeEventListener("click", confirmed);
      resolve(answer);
    };
    const dismissed = () => finish(false);
    const confirmed = () => finish(true);

    dialog.addEventListener("close", dismissed, { once: true });
    go.addEventListener("click", confirmed, { once: true });
    dialog.showModal();
    go.focus();
  });
}
