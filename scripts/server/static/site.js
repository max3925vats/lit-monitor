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
