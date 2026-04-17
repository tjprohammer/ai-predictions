/**
 * Non-board pages: show a bottom banner and poll GET /api/update-jobs/:id while a
 * pipeline job is queued or running. Uses sessionStorage (written from index.html)
 * and falls back to GET /api/update-jobs/active.
 */
(function updateJobFollower() {
  const STORAGE_KEY = "mlb_predictor_update_job_follow";
  const POLL_MS = 1200;

  function detectAppBase() {
    const pathname = window.location.pathname || "/";
    if (pathname === "/") return "/";
    const segments = pathname.split("/").filter(Boolean);
    if (segments.length <= 1) return "/";
    return `/${segments.slice(0, -1).join("/")}/`;
  }

  function appUrl(path) {
    const base = detectAppBase();
    const normalizedBase = base.endsWith("/") ? base : `${base}/`;
    return new URL(path, `${window.location.origin}${normalizedBase}`).toString();
  }

  const path = window.location.pathname || "/";
  if (path === "/" || path === "") {
    return;
  }

  let bannerEl = null;
  let pollTimer = null;

  function clearPoll() {
    if (pollTimer != null) {
      window.clearTimeout(pollTimer);
      pollTimer = null;
    }
  }

  function removeBanner() {
    if (bannerEl && bannerEl.parentNode) {
      bannerEl.parentNode.removeChild(bannerEl);
    }
    bannerEl = null;
  }

  function statusLine(job) {
    if (!job) return "";
    const total = job.total_steps || 0;
    const done = job.completed_steps || 0;
    const step = job.current_step ? String(job.current_step) : "";
    if (job.status === "queued" || job.status === "running") {
      return step
        ? `${job.label || "Update"} · ${step} (${done}/${total})`
        : `${job.label || "Update"} · ${done}/${total} steps`;
    }
    return `${job.label || "Update"} · ${job.status || "done"}`;
  }

  function showBanner({ title, detail, tone, showBoardLink, onDismiss }) {
    removeBanner();
    const wrap = document.createElement("div");
    wrap.setAttribute("role", "status");
    const bg =
      tone === "warn"
        ? "#fff4e6"
        : tone === "done"
          ? "#e8f5f0"
          : "#e8f0ff";
    wrap.style.cssText = [
      "position:fixed",
      "bottom:0",
      "left:0",
      "right:0",
      "z-index:99999",
      "padding:12px 16px",
      "font:14px/1.45 system-ui,-apple-system,sans-serif",
      "display:flex",
      "align-items:center",
      "justify-content:space-between",
      "gap:12px",
      "flex-wrap:wrap",
      "box-shadow:0 -4px 24px rgba(0,0,0,.12)",
      `background:${bg}`,
      "border-top:1px solid rgba(0,0,0,.1)",
    ].join(";");
    const text = document.createElement("div");
    text.style.cssText = "flex:1;min-width:200px";
    const t = document.createElement("div");
    t.style.fontWeight = "600";
    t.textContent = title;
    text.appendChild(t);
    if (detail) {
      const d = document.createElement("div");
      d.style.cssText = "opacity:.85;font-size:13px;margin-top:4px";
      d.textContent = detail;
      text.appendChild(d);
    }
    wrap.appendChild(text);
    const actions = document.createElement("div");
    actions.style.cssText = "display:flex;gap:10px;align-items:center";
    if (showBoardLink) {
      const a = document.createElement("a");
      a.href = appUrl("");
      a.textContent = "Open board";
      a.style.cssText =
        "color:#0b5;font-weight:600;text-decoration:underline;white-space:nowrap";
      actions.appendChild(a);
    }
    if (onDismiss) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = "Dismiss";
      b.style.cssText =
        "cursor:pointer;border:1px solid rgba(0,0,0,.2);background:#fff;border-radius:8px;padding:6px 12px;font:inherit";
      b.addEventListener("click", onDismiss);
      actions.appendChild(b);
    }
    wrap.appendChild(actions);
    document.body.appendChild(wrap);
    bannerEl = wrap;
  }

  async function fetchJson(url) {
    const r = await fetch(url, { credentials: "same-origin" });
    if (!r.ok) {
      throw new Error(String(r.status));
    }
    return r.json();
  }

  function readStoredJobId() {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (!raw) return null;
      const o = JSON.parse(raw);
      return o && o.job_id ? String(o.job_id) : null;
    } catch {
      return null;
    }
  }

  async function resolveInitialJobId() {
    const fromStore = readStoredJobId();
    if (fromStore) {
      return fromStore;
    }
    const payload = await fetchJson(appUrl("api/update-jobs/active"));
    const job = payload.job;
    if (job && (job.status === "queued" || job.status === "running")) {
      try {
        sessionStorage.setItem(
          STORAGE_KEY,
          JSON.stringify({
            job_id: job.job_id,
            label: job.label || "Pipeline",
            target_date: job.target_date || "",
            status: job.status,
          }),
        );
      } catch {
        /* ignore quota */
      }
      return String(job.job_id);
    }
    return null;
  }

  function finishSuccess(job) {
    try {
      sessionStorage.removeItem(STORAGE_KEY);
    } catch {
      /* ignore */
    }
    showBanner({
      title: "Pipeline finished",
      detail: job
        ? `${job.label || "Update"} completed for ${job.target_date || "target date"}.`
        : "The update job finished.",
      tone: "done",
      showBoardLink: true,
      onDismiss: () => {
        clearPoll();
        removeBanner();
      },
    });
    window.setTimeout(() => {
      removeBanner();
    }, 5000);
  }

  async function pollLoop(jobId) {
    clearPoll();
    let job;
    try {
      const payload = await fetchJson(
        appUrl(`api/update-jobs/${encodeURIComponent(jobId)}`),
      );
      job = payload.job;
    } catch {
      showBanner({
        title: "Update status unavailable",
        detail:
          "Could not reach the server. Reopen the board to check the pipeline.",
        tone: "warn",
        showBoardLink: true,
        onDismiss: () => {
          clearPoll();
          removeBanner();
        },
      });
      return;
    }

    if (!job) {
      try {
        sessionStorage.removeItem(STORAGE_KEY);
      } catch {
        /* ignore */
      }
      removeBanner();
      return;
    }

    if (job.status === "queued" || job.status === "running") {
      try {
        sessionStorage.setItem(
          STORAGE_KEY,
          JSON.stringify({
            job_id: job.job_id,
            label: job.label || "Pipeline",
            target_date: job.target_date || "",
            status: job.status,
          }),
        );
      } catch {
        /* ignore */
      }
      showBanner({
        title: "Pipeline running",
        detail: statusLine(job),
        tone: "run",
        showBoardLink: true,
        onDismiss: null,
      });
      pollTimer = window.setTimeout(() => pollLoop(jobId), POLL_MS);
      return;
    }

    finishSuccess(job);
  }

  function start() {
    resolveInitialJobId()
      .then((jobId) => {
        if (!jobId) return;
        pollLoop(jobId);
      })
      .catch(() => {
        /* no active job */
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
