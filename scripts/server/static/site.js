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
  var textarea = form && form.querySelector('textarea[name="question"]');
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
    list.textContent = "";
    arr.forEach(function (q) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn btn-action btn-sm";
      btn.textContent = q;
      btn.addEventListener("click", function () {
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

  // Pull the six categorical palette tokens once per init; falls back to a
  // single token if a slot is undefined (cycled by index for multi-series).
  function paletteColor(idx) {
    var root = document.documentElement;
    var name = "--chart-" + ((idx % 6) + 1);
    var v = getComputedStyle(root).getPropertyValue(name).trim();
    return v || getComputedStyle(root).getPropertyValue("--chart-1").trim();
  }

  function gridColor() {
    return (
      getComputedStyle(document.documentElement)
        .getPropertyValue("--chart-grid")
        .trim() || getComputedStyle(document.documentElement).getPropertyValue("--border").trim()
    );
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

  // Build the Chart.js config for a single canvas given its parsed data island.
  function buildConfig(name, data) {
    if (!data) return null;
    var stacked = !!data.stacked;
    var datasets;
    if (data.datasets) {
      // Multi-series (stacked timeline): one dataset per category, palette-cycled.
      datasets = data.datasets.map(function (ds, i) {
        return { label: ds.label, data: ds.data, backgroundColor: paletteColor(i) };
      });
    } else {
      // Single-series bar: one color per bar so categories read distinctly.
      var colors = (data.data || []).map(function (_, i) {
        return paletteColor(i);
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
          x: { stacked: stacked, grid: { color: gridColor() } },
          y: {
            stacked: stacked,
            beginAtZero: true,
            max: yMax,
            grid: { color: gridColor() },
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
    canvases.forEach(function (canvas) {
      var name = canvas.getAttribute("data-chart");
      var data = readData(name, scope);
      var cfg = buildConfig(name, data);
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
