// Minimal client glue: surface HTMX request failures in a tiny banner.
(function () {
  "use strict";

  function flash(message) {
    var banner = document.getElementById("status-banner");
    if (!banner) return;
    banner.textContent = message;
    banner.hidden = false;
    window.setTimeout(function () {
      banner.hidden = true;
    }, 5000);
  }

  document.body.addEventListener("htmx:responseError", function (evt) {
    var status = (evt.detail && evt.detail.xhr && evt.detail.xhr.status) || "?";
    flash("Request failed (HTTP " + status + ").");
  });

  document.body.addEventListener("htmx:sendError", function () {
    flash("Network error — could not reach the server.");
  });
})();

// WA4: recent-questions history on /ask. Activates only when #ask-history is
// present (no-ops on every other page). History is stored client-side in
// localStorage, most-recent-first, deduplicated, bounded to 20 entries.
(function () {
  "use strict";

  var KEY = "lit_ask_history";
  var MAX = 20;

  var list = document.getElementById("ask-history");
  var form = list && list.closest("section").querySelector("form");
  // sl-textarea is a custom element; select by tag name + attribute, not the
  // native textarea tag (the Shoelace migration replaced the native element).
  var textarea = form && form.querySelector('sl-textarea[name="question"]');
  if (!list || !form || !textarea) return; // not the /ask page

  function load() {
    try {
      var raw = window.localStorage.getItem(KEY);
      var arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr : [];
    } catch (e) {
      return []; // corrupt or storage blocked — start fresh
    }
  }

  function save(arr) {
    try {
      window.localStorage.setItem(KEY, JSON.stringify(arr));
    } catch (e) {
      /* storage full or blocked — render still reflects in-memory state */
    }
  }

  function record(question) {
    var q = (question || "").trim();
    if (!q) return load();
    var arr = load().filter(function (item) {
      return item !== q; // dedup: drop any existing copy, re-add at front
    });
    arr.unshift(q);
    arr = arr.slice(0, MAX); // bound
    save(arr);
    return arr;
  }

  function render(arr) {
    if (!arr || arr.length === 0) {
      // Show a muted empty-state line that matches the server-rendered default.
      list.innerHTML = '<p class="field-note">No questions asked yet.</p>';
      return;
    }
    list.textContent = "";
    arr.forEach(function (q) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn btn-action btn-sm";
      btn.textContent = q;
      btn.addEventListener("click", function () {
        // sl-textarea exposes .value as a property (Shoelace mirrors the attribute).
        textarea.value = q;
        // Re-run through the existing HTMX flow.
        if (window.htmx) window.htmx.trigger(form, "submit");
        else form.requestSubmit();
      });
      list.appendChild(btn);
    });
  }

  render(load());

  // Hook the existing HTMX lifecycle, matching site.js's body-listener style.
  // Only record on a successful POST to /ask/answer.
  document.body.addEventListener("htmx:afterRequest", function (evt) {
    var detail = evt.detail || {};
    var path = (detail.pathInfo && detail.pathInfo.requestPath) || "";
    var ok = detail.successful || (detail.xhr && detail.xhr.status === 200);
    if (path.indexOf("/ask/answer") === -1 || !ok) return;
    render(record(textarea.value));
  });
})();

// CU-2: Chart.js init for the /insights charts. Reads a JSON data island next
// to each <canvas data-chart="..."> and builds a chart, pulling colors from the
// site.css palette tokens (no color literals here). Graceful degradation: if the
// Chart.js CDN is blocked/offline (window.Chart absent) this no-ops and the data
// tables remain. Re-runs on htmx:afterSettle so the lazily-loaded timeline (and
// its toggle re-swaps) re-initialise; any prior chart on a canvas is destroyed
// first so re-swaps never leak or duplicate.
//
// P4 (Shoelace redesign) — CHART-IN-COLLAPSED-DETAILS GOTCHA: a Chart.js canvas
// inside a COLLAPSED <sl-details> has 0 width when first laid out, so a chart
// created at DOMContentLoaded draws broken/empty and never recovers on expand.
// Fix: charts that live inside a `[data-chart-details]` <sl-details> are NOT
// built eagerly — they are built lazily on Shoelace's `sl-after-show` (fired when
// the details finishes opening, at which point the canvas has real width). A
// per-canvas registry (`canvas._chart`) means each is inited at most once; every
// subsequent show just calls `chart.resize()` (cheap, and re-flows after any
// viewport change while collapsed). Charts NOT inside such a details (none today,
// but kept general) still init eagerly on load.
(function () {
  "use strict";

  // Read the six categorical palette tokens + the grid color ONCE per init.
  // getComputedStyle is a forced style read, so we resolve the CSS :root tokens
  // a single time and index into the result (cycled by slot for multi-series).
  function readPalette() {
    var cs = getComputedStyle(document.documentElement);
    var get = function (name) {
      return cs.getPropertyValue(name).trim();
    };
    var fallback = get("--chart-1") || get("--accent");
    var colors = [];
    for (var i = 1; i <= 6; i++) colors.push(get("--chart-" + i) || fallback);
    return { colors: colors, grid: get("--chart-grid") || get("--border") };
  }

  // Read the JSON island for a given canvas. Islands live in the same document
  // (sibling or anywhere under root) keyed by `<name>-data`. Returns null if the
  // island is missing or unparseable (chart is simply skipped — table remains).
  function readData(name, root) {
    var sel = 'script[type="application/json"][data-chart="' + name + '-data"]';
    var node = (root && root.querySelector && root.querySelector(sel)) || document.querySelector(sel);
    if (!node) return null;
    try {
      return JSON.parse(node.textContent || "null");
    } catch (e) {
      return null;
    }
  }

  // Build the Chart.js config for a single canvas given its parsed data island
  // and the once-per-init palette ({colors, grid}).
  function buildConfig(name, data, palette) {
    if (!data) return null;
    var color = function (i) {
      return palette.colors[i % palette.colors.length];
    };
    var stacked = !!data.stacked;
    var datasets;
    if (data.datasets) {
      // Multi-series (stacked timeline): one dataset per category, palette-cycled.
      datasets = data.datasets.map(function (ds, i) {
        return { label: ds.label, data: ds.data, backgroundColor: color(i) };
      });
    } else {
      // Single-series bar: one color per bar so categories read distinctly.
      var colors = (data.data || []).map(function (_, i) {
        return color(i);
      });
      datasets = [{ label: name, data: data.data || [], backgroundColor: colors }];
    }
    var yMax = typeof data.max === "number" ? data.max : undefined;
    return {
      type: "bar",
      data: { labels: data.labels || [], datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: !!data.datasets } },
        scales: {
          x: { stacked: stacked, grid: { color: palette.grid } },
          y: {
            stacked: stacked,
            beginAtZero: true,
            max: yMax,
            grid: { color: palette.grid },
          },
        },
      },
    };
  }

  // Build (or rebuild) the chart for a single canvas. `destroyFirst` controls
  // re-swap safety: on an HTMX swap the old chart must be destroyed before a new
  // one is created on the same canvas; on a first lazy build there is nothing to
  // destroy. Returns true if a chart now exists on the canvas.
  function _initCanvas(canvas, scope, palette, destroyFirst) {
    var name = canvas.getAttribute("data-chart");
    var data = readData(name, scope);
    var cfg = buildConfig(name, data, palette);
    if (!cfg) return false;
    if (destroyFirst && canvas._chart) canvas._chart.destroy();
    canvas._chart = new window.Chart(canvas, cfg);
    return true;
  }

  // Eager init for a subtree (load + HTMX swaps). Canvases that live inside a
  // collapsed-by-default `[data-chart-details]` are SKIPPED here — building them
  // now (0 width) would draw broken; they are built lazily on `sl-after-show`
  // instead. Canvases swapped in by HTMX (e.g. the timeline fragment, which only
  // loads once its details is already open) are not inside a collapsed details at
  // swap time, so they init/redraw correctly on the htmx:afterSettle path.
  function _initInsightsCharts(root) {
    if (!window.Chart) return; // CDN blocked/offline → tables remain, no error.
    var scope = root && root.querySelectorAll ? root : document;
    var canvases = scope.querySelectorAll("canvas[data-chart]");
    if (!canvases.length) return; // not an insights page / nothing swapped in.
    var palette = readPalette(); // resolve CSS tokens once per init.
    canvases.forEach(function (canvas) {
      // Defer charts inside a collapsed chart-details to sl-after-show.
      if (canvas.closest && canvas.closest("[data-chart-details]")) return;
      _initCanvas(canvas, scope, palette, true); // re-swap safe.
    });
  }

  // sl-after-show handler for the chart-bearing <sl-details>. The first time a
  // details is shown each canvas inside it is inited (now that it has real
  // width); on every subsequent show the existing chart is just resized so it
  // re-flows to the current container width. Registry: canvas._chart.
  function _showChartsIn(details) {
    if (!window.Chart || !details || !details.querySelectorAll) return;
    var palette = readPalette();
    details.querySelectorAll("canvas[data-chart]").forEach(function (canvas) {
      if (canvas._chart) {
        canvas._chart.resize(); // already inited → re-flow to real width.
      } else {
        _initCanvas(canvas, details, palette, false); // first show → build now.
      }
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    _initInsightsCharts(document);
  });
  // The timeline fragment loads lazily and re-swaps on toggle; re-init its
  // swapped subtree (e.detail.target is the swapped node).
  document.body.addEventListener("htmx:afterSettle", function (evt) {
    var target = (evt.detail && evt.detail.target) || evt.target;
    _initInsightsCharts(target);
  });
  // P4: lazily build/resize charts inside a collapsed <sl-details> when it opens.
  // sl-after-show fires after the panel is fully revealed (canvas has real width).
  // Scope to chart-details so unrelated sl-details (Recent events, lazy fragments)
  // don't trigger a needless palette read.
  document.body.addEventListener("sl-after-show", function (evt) {
    var details = evt.target;
    if (details && details.matches && details.matches("[data-chart-details]")) {
      _showChartsIn(details);
    }
  });
})();

// Theme toggle: dark default; flips data-theme + Shoelace's sl-theme-dark class,
// persisted in localStorage. The <head> bootstrap reads the same key on load.
(function () {
  "use strict";
  var KEY = "lit_theme";
  function currentTheme() {
    return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
  }
  function applyTheme(next) {
    if (next === "light") {
      document.documentElement.setAttribute("data-theme", "light");
      document.documentElement.classList.remove("sl-theme-dark");
    } else {
      document.documentElement.removeAttribute("data-theme");
      document.documentElement.classList.add("sl-theme-dark");
    }
    try { window.localStorage.setItem(KEY, next); } catch (e) {}
  }
  function paintGlyph(btn) {
    if (btn) btn.setAttribute("name", currentTheme() === "light" ? "sun" : "moon");
  }
  function init() {
    var btn = document.getElementById("theme-toggle");
    if (!btn) return;
    paintGlyph(btn);
    btn.addEventListener("click", function () {
      applyTheme(currentTheme() === "light" ? "dark" : "light");
      paintGlyph(btn);
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();

// Brain-build collection switcher: restore the true collection name on submit.
//
// Shoelace forbids spaces in <sl-option value="…"> and SILENTLY rewrites them to
// underscores ("Molecular Simulations" -> DOM value "Molecular_Simulations").
// The form would therefore submit the underscored token, and the backend would
// persist that corrupted name to paths.yaml — so the active collection would no
// longer match the real Zotero collection (the user-reported "switch doesn't
// work" bug, 2026-06-13).
//
// Fix: on submit of #collection-switch-form, if the dropdown rendered as an
// sl-select, overwrite the outgoing collection_name with the SELECTED OPTION'S
// LABEL (textContent), which preserves the original spaces. The fallback
// sl-input path needs no fixup (free text, no space corruption) and is left
// untouched. Using htmx:configRequest keeps this a pure parameter rewrite — no
// hidden field, no dependence on Shoelace form-association timing.
(function () {
  "use strict";
  document.body.addEventListener("htmx:configRequest", function (evt) {
    var form = evt.target;
    if (!form || form.id !== "collection-switch-form") return;
    var select = form.querySelector("sl-select");
    if (!select) return; // fallback sl-input path: nothing to repair
    var opt = select.querySelector(
      'sl-option[value="' + (select.value || "") + '"]'
    );
    // textContent is the true name (with spaces); value is the underscored token.
    var trueName = opt ? opt.textContent.trim() : "";
    if (trueName) evt.detail.parameters.collection_name = trueName;
  });
})();

// Run-gated live-progress pane (brain-build + discovery dashboards).
//
// The log pane (#live-progress-wrap) should be visible only while a run is
// active. The controls fragment above it re-renders itself every 5s and carries
// the running state as data-running on its <section>. On each controls swap (and
// the initial hx-trigger=load swap), read that flag and toggle `hidden` on the
// wrapper. The wrapper itself is a stable element in index.html — it is NEVER a
// swap target — so its sse-connect child keeps its EventSource alive across the
// 5s poll (no reconnect storm, no lost log lines). Generic across both pages: a
// page only has one of the two controls sections.
(function () {
  "use strict";
  function syncFromControls(section) {
    if (!section) return;
    if (!section.classList) return;
    if (
      !section.classList.contains("brain-build-controls") &&
      !section.classList.contains("discovery-controls")
    ) {
      return; // not a controls section — ignore (e.g. other afterSettle swaps)
    }
    var wrap = document.getElementById("live-progress-wrap");
    if (!wrap) return; // not a dashboard with a log pane
    var running = section.getAttribute("data-running") === "true";
    if (running) wrap.removeAttribute("hidden");
    else wrap.setAttribute("hidden", "");
  }
  // The controls fragment loads via hx-trigger=load and re-polls every 5s; each
  // settle gives us the freshest data-running. afterSettle fires after the DOM
  // is in place, so getAttribute reads the swapped-in markup.
  document.body.addEventListener("htmx:afterSettle", function (evt) {
    var target = (evt.detail && evt.detail.target) || evt.target;
    syncFromControls(target);
  });
  // Initial sync in case the controls section is already in the DOM at load
  // (e.g. server-rendered or already-settled before this listener attached).
  function initSync() {
    syncFromControls(document.querySelector(".brain-build-controls"));
    syncFromControls(document.querySelector(".discovery-controls"));
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initSync);
  } else {
    initSync();
  }
})();

// App-shell: mobile sidebar toggle + global Ask drawer open.
(function () {
  "use strict";
  function init() {
    var shell = document.querySelector(".app-shell");
    var sb = document.querySelector(".app-sidebar-toggle");
    if (sb && shell) sb.addEventListener("click", function () { shell.classList.toggle("sidebar-open"); });
    var ask = document.querySelector(".app-ask-btn");
    var drawer = document.getElementById("ask-drawer");
    if (ask && drawer) ask.addEventListener("click", function () { drawer.show(); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();

// RR7: Reset & Rebuild console page wiring.
//
// Activates only when #reset-rebuild-page is present (no-ops on every other page).
// Three responsibilities:
//   1. On load, fetch /api/reset/status and populate status lines + enrichment gate.
//   2. Wire SSE rebuild panes: reveal on Rebuild click, stream /api/rebuild/<c>/stream,
//      refresh status on done.
//   3. Wire the "Everything" typed-phrase gate: enable the confirm button only when
//      the sl-input value === "reset all".
//
// Uses plain EventSource (vanilla), consistent with the rest of site.js.
// All element lookups are null-guarded; the entire block no-ops when the page
// root element is absent.
(function () {
  "use strict";

  // Page guard — only run on the Reset & Rebuild page.
  if (!document.getElementById("reset-rebuild-page")) return;

  var COMPONENTS = ["vectors", "graph", "notes"];

  // --- 1. Status fetch & population -------------------------------------------

  function humanSize(bytes) {
    if (!bytes) return "0 B";
    var units = ["B", "KB", "MB", "GB"];
    var i = 0;
    var n = bytes;
    while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
    return n.toFixed(i > 0 ? 1 : 0) + " " + units[i];
  }

  function setStatusLine(comp, info) {
    var el = document.getElementById("rr-status-" + comp);
    if (!el) return;
    if (!info) { el.textContent = "Status unavailable."; return; }
    if (info.present) {
      el.innerHTML =
        '<span class="status-dot ok" title="Present" aria-label="Present"></span> ' +
        info.count + " " + info.unit + (info.count !== 1 ? "s" : "") +
        " · " + humanSize(info.size_bytes);
    } else {
      el.innerHTML =
        '<span class="status-dot pending" title="Not present" aria-label="Not present"></span> ' +
        "Not present";
    }
  }

  function applyEnrichmentGate(enrichment) {
    var sw = document.getElementById("rr-enrich-switch");
    var note = document.getElementById("rr-enrich-note");
    if (!sw || !note) return;
    var hasNlp = enrichment && enrichment.nlp;
    var hasKey = enrichment && enrichment.ollama_key;
    var capable = hasNlp && hasKey;
    if (capable) {
      sw.removeAttribute("disabled");
      note.hidden = true;
      note.textContent = "";
    } else {
      sw.setAttribute("disabled", "");
      note.hidden = false;
      var reasons = [];
      if (!hasNlp) reasons.push("Needs the [nlp] extra (transformers)");
      if (!hasKey) reasons.push("Needs OLLAMA_API_KEY");
      note.textContent = reasons.join(" · ");
    }
  }

  function setActionsDisabled(disabled) {
    COMPONENTS.concat(["everything"]).forEach(function (comp) {
      var btn = document.getElementById("rr-reset-btn-" + comp);
      if (btn) {
        if (disabled) btn.setAttribute("disabled", "");
        else btn.removeAttribute("disabled");
      }
      var rbtn = document.getElementById("rr-rebuild-btn-" + comp);
      if (rbtn) {
        if (disabled) rbtn.setAttribute("disabled", "");
        else rbtn.removeAttribute("disabled");
      }
    });
  }

  function fetchStatus() {
    fetch("/api/reset/status")
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) {
        if (!data) return;
        // Show/hide the not-configured banner.
        var banner = document.getElementById("rr-not-configured");
        if (banner) banner.hidden = !!data.configured;
        // Disable action buttons when not configured.
        setActionsDisabled(!data.configured);
        // Populate per-component status lines.
        COMPONENTS.forEach(function (comp) {
          setStatusLine(comp, data.components && data.components[comp]);
        });
        // Wire enrichment gate for the graph card.
        applyEnrichmentGate(data.enrichment);
      })
      .catch(function () { /* network error — leave status lines as-is */ });
  }

  // --- 2. SSE rebuild panes ---------------------------------------------------

  // Track open EventSources so we can close them if needed.
  var _sources = {};

  function openRebuildLog(comp, enrich) {
    var wrap = document.getElementById("rebuild-" + comp + "-log");
    var lines = document.getElementById("rebuild-" + comp + "-lines");
    var hint = document.getElementById("rebuild-" + comp + "-hint");
    if (!wrap || !lines) return;

    // Reveal the log pane.
    wrap.hidden = false;
    if (hint) hint.hidden = false;

    // Clear any lines from a previous run so a re-run never interleaves stale
    // output with fresh output.
    lines.innerHTML = "";

    // POST to start the rebuild.
    var body = new URLSearchParams();
    if (enrich) body.set("enrich", "true");

    fetch("/api/rebuild/" + comp, { method: "POST", body: body })
      .then(function (r) {
        if (!r.ok) {
          return r.json().then(function (d) {
            var msg = document.createElement("div");
            msg.className = "log-line";
            msg.textContent = "Error: " + (d.error || r.status);
            lines.appendChild(msg);
          });
        }
        // Open SSE stream.
        if (_sources[comp]) { try { _sources[comp].close(); } catch (e) {} }
        var es = new EventSource("/api/rebuild/" + comp + "/stream");
        _sources[comp] = es;

        es.addEventListener("progress", function (evt) {
          if (!evt.data) return;
          var div = document.createElement("div");
          div.innerHTML = evt.data; // data is already an HTML <div class="log-line">…</div>
          var child = div.firstChild;
          if (child) lines.appendChild(child);
          // Auto-scroll to bottom.
          lines.scrollTop = lines.scrollHeight;
        });

        es.addEventListener("done", function () {
          es.close();
          delete _sources[comp];
          // Refresh status lines after rebuild completes.
          fetchStatus();
        });

        es.onerror = function () {
          es.close();
          delete _sources[comp];
        };
      })
      .catch(function (err) {
        var msg = document.createElement("div");
        msg.className = "log-line";
        msg.textContent = "Network error: " + err;
        lines.appendChild(msg);
      });
  }

  // Wire Rebuild buttons.
  COMPONENTS.forEach(function (comp) {
    var btn = document.getElementById("rr-rebuild-btn-" + comp);
    if (!btn) return;
    btn.addEventListener("click", function () {
      var enrich = false;
      if (comp === "graph") {
        var sw = document.getElementById("rr-enrich-switch");
        enrich = sw && sw.checked;
      }
      openRebuildLog(comp, enrich);
    });
  });

  // --- 3. Reset confirm dialogs ----------------------------------------------

  function wireResetDialog(comp) {
    var openBtn = document.getElementById("rr-reset-btn-" + comp);
    var dialog = document.getElementById("rr-dialog-" + comp);
    var cancelBtn = document.getElementById("rr-dialog-" + comp + "-cancel");
    var confirmBtn = document.getElementById("rr-dialog-" + comp + "-confirm");
    if (!openBtn || !dialog || !cancelBtn || !confirmBtn) return;

    openBtn.addEventListener("click", function () { dialog.show(); });
    cancelBtn.addEventListener("click", function () { dialog.hide(); });

    // After a successful HTMX reset POST, close dialog and refresh status.
    confirmBtn.addEventListener("htmx:afterRequest", function (evt) {
      var ok = evt.detail && (evt.detail.successful || (evt.detail.xhr && evt.detail.xhr.status === 200));
      if (ok) {
        dialog.hide();
        fetchStatus();
      }
    });
  }

  ["vectors", "graph", "notes"].forEach(wireResetDialog);

  // "Everything" dialog — typed-phrase gate.
  (function () {
    var openBtn = document.getElementById("rr-reset-btn-everything");
    var dialog = document.getElementById("rr-dialog-everything");
    var cancelBtn = document.getElementById("rr-dialog-everything-cancel");
    var confirmBtn = document.getElementById("rr-dialog-everything-confirm");
    var phraseInput = document.getElementById("rr-everything-phrase");
    if (!openBtn || !dialog || !cancelBtn || !confirmBtn || !phraseInput) return;

    var PHRASE = "reset all";

    function updateConfirmState() {
      var val = (phraseInput.value || "").trim();
      if (val === PHRASE) {
        confirmBtn.removeAttribute("disabled");
      } else {
        confirmBtn.setAttribute("disabled", "");
      }
    }

    openBtn.addEventListener("click", function () {
      // Clear the input each time the dialog opens so the state is fresh.
      phraseInput.value = "";
      confirmBtn.setAttribute("disabled", "");
      dialog.show();
    });
    cancelBtn.addEventListener("click", function () { dialog.hide(); });

    // sl-input fires sl-input on value changes.
    phraseInput.addEventListener("sl-input", updateConfirmState);
    // Also wire native input in case the browser fires it before Shoelace.
    phraseInput.addEventListener("input", updateConfirmState);

    // Also update the hidden form field on the HTMX POST so the server
    // receives confirm=<typed value>.  The confirmBtn carries hx-post;
    // HTMX will serialize closest form or the element's own attrs — we
    // inject the param via htmx:configRequest.
    document.body.addEventListener("htmx:configRequest", function (evt) {
      if (evt.target !== confirmBtn) return;
      evt.detail.parameters.confirm = (phraseInput.value || "").trim();
    });

    confirmBtn.addEventListener("htmx:afterRequest", function (evt) {
      var ok = evt.detail && (evt.detail.successful || (evt.detail.xhr && evt.detail.xhr.status === 200));
      if (ok) {
        dialog.hide();
        phraseInput.value = "";
        fetchStatus();
      }
    });
  })();

  // --- Initial status load ---------------------------------------------------
  fetchStatus();
})();

// RR6: entry-guard for the Reset & Rebuild sidebar link. Intercept clicks on
// a[href="/setup/reset"], show a confirm dialog, and navigate only on "Continue".
// Uses delegated click on document.body so it works regardless of when the
// sidebar is rendered (HTMX swaps, etc.). Guard for null elements defensively.
(function () {
  "use strict";

  function init() {
    var dialog = document.getElementById("reset-entry-guard");
    var continueBtn = document.getElementById("reset-entry-continue");
    var cancelBtn = document.getElementById("reset-entry-cancel");
    if (!dialog || !continueBtn || !cancelBtn) return; // dialog not in DOM (shouldn't happen)

    // Delegated click: intercept any click on the Reset & Rebuild nav link.
    document.body.addEventListener("click", function (evt) {
      var anchor = evt.target && evt.target.closest('a[href="/setup/reset"]');
      if (!anchor) return;
      evt.preventDefault();
      dialog.show();
    });

    // "Continue" navigates to the reset page and closes the dialog.
    continueBtn.addEventListener("click", function () {
      dialog.hide();
      window.location.href = "/setup/reset";
    });

    // "Cancel" just closes the dialog.
    cancelBtn.addEventListener("click", function () {
      dialog.hide();
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
