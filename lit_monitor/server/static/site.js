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

  function _initInsightsCharts(root) {
    if (!window.Chart) return; // CDN blocked/offline → tables remain, no error.
    var scope = root && root.querySelectorAll ? root : document;
    var canvases = scope.querySelectorAll("canvas[data-chart]");
    if (!canvases.length) return; // not an insights page / nothing swapped in.
    var palette = readPalette(); // resolve CSS tokens once per init.
    canvases.forEach(function (canvas) {
      var name = canvas.getAttribute("data-chart");
      var data = readData(name, scope);
      var cfg = buildConfig(name, data, palette);
      if (!cfg) return;
      if (canvas._chart) canvas._chart.destroy(); // re-swap safety: no leak/dupe.
      canvas._chart = new window.Chart(canvas, cfg);
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
