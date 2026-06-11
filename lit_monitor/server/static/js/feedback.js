/**
 * Bundle H: minimal client-side helpers for the feedback/rating widget.
 *
 * Responsibilities:
 *  - Show a brief confirmation ("Saved", "Dismissed", etc.) next to the
 *    button that triggered the feedback POST, without requiring HTMX swap.
 *  - Close the <details class="rate-widget"> after a rating is submitted.
 *
 * Requires HTMX 2.x (already loaded by base.html). No other dependencies.
 */
(function () {
  'use strict';

  /**
   * After a successful /api/feedback POST from a .feedback-row button,
   * briefly flash a small status label next to the clicked element.
   */
  function handleFeedbackAfterRequest(evt) {
    if (!evt.detail.successful) return;

    var elt = evt.detail.elt;
    if (!elt || !elt.closest('.feedback-row')) return;

    /* Resolve a human-readable label from the hx-vals JSON. */
    var valsAttr = elt.getAttribute('hx-vals') || '{}';
    var signalType = '';
    try {
      signalType = JSON.parse(valsAttr).signal_type || '';
    } catch (_) { /* ignore */ }

    var labelMap = {
      saved: 'Saved!',
      dismissed: 'Dismissed',
      thumbs_up: 'Thumbs up!',
      thumbs_down: 'Thumbs down',
      rated: 'Rated!',
    };
    var msg = labelMap[signalType] || 'Done';

    /* Insert or reuse a .feedback-flash span after the element. */
    var flash = elt.parentNode.querySelector('.feedback-flash');
    if (!flash) {
      flash = document.createElement('span');
      flash.className = 'feedback-flash field-note';
      elt.parentNode.appendChild(flash);
    }
    flash.textContent = msg;
    flash.style.opacity = '1';

    /* Fade out after 2 s. */
    clearTimeout(flash._fadeTimer);
    flash._fadeTimer = setTimeout(function () {
      flash.style.transition = 'opacity 0.5s';
      flash.style.opacity = '0';
    }, 2000);

    /* Close the <details> rate widget if it was the submission source. */
    var rateWidget = elt.closest('details.rate-widget');
    if (rateWidget) {
      rateWidget.removeAttribute('open');
    }
  }

  document.addEventListener('htmx:afterRequest', handleFeedbackAfterRequest);
})();
