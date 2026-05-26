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
