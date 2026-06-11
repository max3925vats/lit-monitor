"""Bundle K-a (v1.0): per-cluster atrophy feedback weights — pure computation.

A research theme you *stop dismissing* should not silently vanish. This module
assigns each cluster a ``feedback_weight`` in ``[floor, 1.0]`` that decays
**only** on active dismissal within that cluster and recovers as those
dismissals age out. Engagement (saves / thumbs-up / high ratings) and the
never-touched baseline both keep a cluster at full weight.

Design notes
------------
- **Pure function.** ``compute_per_cluster_feedback_weights`` takes raw
  ``feedback_events`` dicts plus an injected ``doi -> cluster_id`` mapping, so
  it is unit-testable without a DB. A thin DB-facing convenience
  (``compute_cluster_feedback_weights_from_db``) wires it to a ``StateDB``.
- **Reuses the Rocchio signal vocabulary.** What counts as a "dismissal" vs.
  "engagement" is decided by ``rocchio._classify_signal`` (the source of truth:
  dismissed / thumbs_down / rated≤2 are negative; saved / thumbs_up / rated≥4
  are positive; opened / rated==3 are neutral and skipped). Time-decay reuses
  ``rocchio.time_decay_weight`` so old dismissals fade on the same curve as the
  global interest vector.
- **Finite by construction.** No NaN/inf may ever reach a downstream consumer;
  every event is guarded and the result is clamped to ``[floor, 1.0]``.

Formula (DEFENSIBLE, TUNABLE — see ``_cluster_weight``)
-------------------------------------------------------
For each cluster we accumulate two time-decayed masses over its events:

  - ``dismissal_mass`` = Σ decay(age) over negative events,
  - ``positive_mass``  = Σ decay(age) over positive events,

then, with a constant *engagement prior* ``ENGAGEMENT_PRIOR``::

    dismissal_fraction = dismissal_mass
                         / (dismissal_mass + positive_mass + ENGAGEMENT_PRIOR)
    weight = max(floor, 1.0 - DECAY_STRENGTH * dismissal_fraction)

This is a *dismissal-fraction* model with a baseline-engagement prior. The
prior is what makes the floor a genuine atrophy floor rather than a hard
"any dismissal → floor" cliff, and it is what lets a cluster RECOVER:

- **0 events / only engagement** → dismissal_fraction 0 → weight 1.0.
- **A few recent dismissals** → small numerator vs. the prior → weight stays
  near 1.0 (one bad paper doesn't kill a theme).
- **Many *recent* dismissals** → numerator swamps the prior → fraction → 1 →
  weight pulled toward ``floor``.
- **Old dismissals (age ≫ half-life)** → their decayed mass shrinks below the
  prior → fraction → 0 → weight recovers toward 1.0. (This is the behaviour the
  fraction-only model lacked: without the prior, an all-dismissal cluster's
  fraction stays ~1 forever because decay cancels in numerator and denominator.)
- **Engagement counteracts dismissal** by growing the denominator, so saving
  papers in a cluster offsets prior dismissals.

``DECAY_STRENGTH``, ``ENGAGEMENT_PRIOR`` and the half-life are empirical
constants tuned for one user generating ~10–50 events/week; the floor is
config-driven.

NOTE: this bundle only COMPUTES + EXPOSES the weight. Nothing in the ranker or
discovery search consumes it yet (see the K-a pooled note in the bundle report);
consumption is wired in a later bundle (K-b exploration budget).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta

from scripts.learning.rocchio import (
    HALF_LIFE_DAYS,
    _classify_signal,
    _event_age_days,
    time_decay_weight,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Bundle K-b: under-engaged cluster detection.
# --------------------------------------------------------------------------- #

#: Default "quiet" window: a cluster that has surfaced no paper in a discovery
#: run within this many weeks is a candidate for exploration. Mirrors the
#: ``feedback.cluster_quiet_weeks`` config default.
DEFAULT_QUIET_WEEKS: int = 4

#: How far above the atrophy floor a cluster's K-a ``feedback_weight`` must sit
#: before exploration will probe it. Clusters at/near the floor are ones the
#: user is *actively dismissing*; spending exploration budget on them would
#: fight the user's own signal, so they are excluded. A small epsilon above the
#: floor is enough to separate "merely under-engaged" from "actively rejected".
DEFAULT_EXPLORE_WEIGHT_EPSILON: float = 0.05

# --------------------------------------------------------------------------- #
# Tunable constants (empirical — appropriate for one user, ~10-50 events/week).
# --------------------------------------------------------------------------- #

#: Default floor: a cluster's weight never decays below this (config-driven at
#: the CLI; the pure function takes ``floor=`` explicitly). Mirrors the plan's
#: ``feedback.minimum_cluster_floor`` default.
DEFAULT_FLOOR: float = 0.1

#: How hard a fully-dismissed cluster is pulled DOWN from 1.0. With
#: ``DECAY_STRENGTH = 1.0 - DEFAULT_FLOOR`` a cluster whose dismissal fraction
#: → 1.0 lands at the floor; anything less stays above it. Tunable: smaller ⇒
#: gentler atrophy.
DECAY_STRENGTH: float = 1.0 - DEFAULT_FLOOR

#: Constant "engagement baseline" added to the denominator of the dismissal
#: fraction. This is the keystone of the model: it (a) keeps a *few* dismissals
#: from tanking a cluster, and (b) lets old dismissals RECOVER — once their
#: decayed mass falls below this prior, the fraction collapses toward 0 and the
#: weight returns to 1.0. Roughly, it takes on the order of ``ENGAGEMENT_PRIOR``
#: full-weight recent dismissals before the cluster is meaningfully atrophied.
#: Tunable: larger ⇒ more inertia (slower atrophy, faster recovery).
ENGAGEMENT_PRIOR: float = 3.0

#: Numerical guard so a cluster with zero decayed mass yields fraction 0.0
#: (weight 1.0) instead of 0/0 even if the prior were ever set to 0.
_EPS: float = 1e-9


def _cluster_weight(
    dismissal_mass: float,
    positive_mass: float,
    *,
    floor: float,
    decay_strength: float = DECAY_STRENGTH,
    engagement_prior: float = ENGAGEMENT_PRIOR,
) -> float:
    """Dismissal-fraction → feedback_weight in ``[floor, 1.0]`` (finite).

    ``weight = max(floor, 1.0 - decay_strength * dismissal_fraction)`` where the
    dismissal fraction is
    ``dismissal_mass / (dismissal_mass + positive_mass + engagement_prior)``.
    Non-finite or negative masses are coerced to 0.0 so the result is always
    finite and clamped into the valid range.
    """
    d = dismissal_mass if math.isfinite(dismissal_mass) and dismissal_mass > 0 else 0.0
    p = positive_mass if math.isfinite(positive_mass) and positive_mass > 0 else 0.0

    fraction = d / (d + p + float(engagement_prior) + _EPS)
    raw = 1.0 - float(decay_strength) * fraction

    # Clamp into [floor, 1.0]; floor itself is clamped into a sane band.
    lo = min(max(float(floor), 0.0), 1.0)
    weight = max(lo, min(1.0, raw))
    if not math.isfinite(weight):
        return lo
    return weight


def compute_per_cluster_feedback_weights(
    events: list[dict],
    doi_to_cluster: dict[str, int],
    *,
    now: datetime,
    floor: float = DEFAULT_FLOOR,
    half_life_days: float = HALF_LIFE_DAYS,
) -> dict[int, float]:
    """Per-cluster ``feedback_weight`` from feedback events + a DOI→cluster map.

    For every cluster that appears in ``doi_to_cluster`` the result contains a
    weight in ``[floor, 1.0]``:

    - A cluster with no negative feedback (no events, or only engagement) keeps
      weight ``1.0`` — so a 0-event cluster is ``1.0`` (``≥ floor``), satisfying
      the plan's atrophy floor.
    - A cluster dominated by *recent* dismissals decays toward ``floor`` but
      never below it.
    - Dismissals far older than ``half_life_days`` fade, so weight recovers
      toward ``1.0`` over time.

    Args:
        events:         ``feedback_events`` row dicts (keys used: ``doi``,
                        ``signal_type``, ``rating``, ``created_at``).
        doi_to_cluster: ``{doi: cluster_id}`` mapping (injected — testable
                        without a DB). Every cluster id here gets a weight.
        now:            Reference ``datetime`` for time-decay.
        floor:          Lower clamp; never returns below this for any mapped
                        cluster.
        half_life_days: Days after which a dismissal's contribution halves.

    Returns:
        ``{cluster_id: feedback_weight}`` for every cluster in
        ``doi_to_cluster``. Always finite. Empty mapping → ``{}``.
    """
    # Seed every assigned cluster at zero mass so a 0-event cluster still gets a
    # weight (baseline 1.0). Clusters that never appear in the mapping are not
    # represented (we have no assignment to attribute feedback to).
    dismissal_mass: dict[int, float] = {cid: 0.0 for cid in doi_to_cluster.values()}
    positive_mass: dict[int, float] = {cid: 0.0 for cid in doi_to_cluster.values()}

    for ev in events:
        doi = ev.get("doi")
        if not doi:
            continue
        cluster_id = doi_to_cluster.get(doi)
        if cluster_id is None:
            # Event for a paper with no current cluster assignment — skip it
            # (cannot attribute the signal to any cluster).
            continue

        signal_type = ev.get("signal_type")
        if not signal_type:
            continue
        sign = _classify_signal(signal_type, ev.get("rating"))
        if sign == 0:
            continue  # opened / rated==3 / unrecognized → neutral, ignore

        age_days = _event_age_days(ev.get("created_at"), now)
        decay = time_decay_weight(age_days, half_life_days=half_life_days)
        if not math.isfinite(decay) or decay <= 0.0:
            continue

        if sign < 0:
            dismissal_mass[cluster_id] = dismissal_mass.get(cluster_id, 0.0) + decay
        else:
            positive_mass[cluster_id] = positive_mass.get(cluster_id, 0.0) + decay

    return {
        cid: _cluster_weight(
            dismissal_mass.get(cid, 0.0),
            positive_mass.get(cid, 0.0),
            floor=floor,
        )
        for cid in dismissal_mass
    }


# --------------------------------------------------------------------------- #
# DB-facing convenience (keeps the pure function the testable core)
# --------------------------------------------------------------------------- #

#: Generous cap on feedback events pulled for an atrophy compute — mirrors the
#: CLI's ``_LEARNING_MAX_EVENTS``. Atrophy is a coarse per-cluster signal, so
#: this is headroom, not a true limit.
_ATROPHY_MAX_EVENTS: int = 100_000


def _build_doi_to_cluster(state_db) -> dict[str, int]:  # noqa: ANN001 — StateDB duck-typed
    """Build a ``{doi: cluster_id}`` map over all ACTIVE clusters.

    A paper may have assignments to multiple clusters historically; we keep the
    nearest active one (smallest ``distance_to_centroid``) — the same rule
    ``StateDB.get_paper_cluster`` uses — so each DOI maps to a single cluster.
    """
    mapping: dict[str, int] = {}
    best_dist: dict[str, float] = {}
    for cluster in state_db.list_active_clusters():
        cid = int(cluster["id"])
        for row in state_db.get_cluster_assignments(cid):
            doi = row.get("doi")
            if not doi:
                continue
            dist = row.get("distance_to_centroid")
            dist_val = float(dist) if dist is not None else float("inf")
            if doi not in best_dist or dist_val < best_dist[doi]:
                best_dist[doi] = dist_val
                mapping[doi] = cid
    return mapping


def compute_cluster_feedback_weights_from_db(
    state_db,  # noqa: ANN001 — StateDB duck-typed to avoid a hard import cycle
    *,
    now: datetime | None = None,
    floor: float = DEFAULT_FLOOR,
    half_life_days: float = HALF_LIFE_DAYS,
    max_events: int = _ATROPHY_MAX_EVENTS,
) -> dict[int, float]:
    """Fetch feedback events + active cluster assignments from ``state_db`` and
    compute per-cluster feedback weights via the pure function above.

    Args:
        state_db:       A ``StateDB`` exposing ``list_feedback_events``,
                        ``list_active_clusters`` and ``get_cluster_assignments``.
        now:            Reference time (defaults to ``datetime.now()``).
        floor:          Lower clamp (config ``feedback.minimum_cluster_floor``).
        half_life_days: Dismissal half-life.
        max_events:     Cap on feedback events pulled.

    Returns:
        ``{cluster_id: feedback_weight}`` for every active, assigned cluster.
    """
    ref_now = now or datetime.now()
    events = state_db.list_feedback_events(limit=max_events)
    doi_to_cluster = _build_doi_to_cluster(state_db)
    return compute_per_cluster_feedback_weights(
        events,
        doi_to_cluster,
        now=ref_now,
        floor=floor,
        half_life_days=half_life_days,
    )


# --------------------------------------------------------------------------- #
# Bundle K-b: under-engaged cluster detection (pure, injectable)
# --------------------------------------------------------------------------- #


def _parse_iso_dt(value: str | None) -> datetime | None:
    """Best-effort parse of an ISO-ish timestamp (``YYYY-MM-DD[ HH:MM:SS]``).

    SQLite ``datetime('now')`` stores ``"YYYY-MM-DD HH:MM:SS"``; ``date.today()``
    stores ``"YYYY-MM-DD"``. ``datetime.fromisoformat`` handles both once a space
    is normalised to ``T``. Returns ``None`` on any parse failure so a malformed
    timestamp degrades to "never surfaced" rather than crashing detection.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).strip().replace(" ", "T"))
    except (ValueError, TypeError):
        return None


def find_under_engaged_clusters(
    last_surfaced: dict[int, str | None],
    feedback_weights: dict[int, float],
    *,
    now: datetime,
    quiet_weeks: int = DEFAULT_QUIET_WEEKS,
    floor: float = DEFAULT_FLOOR,
    min_explore_weight: float | None = None,
    explore_weight_epsilon: float = DEFAULT_EXPLORE_WEIGHT_EPSILON,
) -> list[int]:
    """Pure: pick clusters worth probing with exploration queries.

    A cluster is **under-engaged** (and thus a target for exploration) when BOTH:

    1. It has not surfaced a paper in a discovery run within the last
       ``quiet_weeks`` weeks. A cluster that has *never* surfaced a paper
       (``last_surfaced`` value ``None`` or unparseable) always qualifies on
       this leg.
    2. Its K-a ``feedback_weight`` is above the exclusion threshold
       ``min_explore_weight`` (defaults to ``floor + explore_weight_epsilon``).
       Clusters at/near the atrophy floor are ones the user is *actively
       dismissing* — exploring them would fight the user's own signal, so they
       are excluded. A cluster with no weight entry defaults to ``1.0`` (a
       never-touched, full-weight cluster is eligible).

    Both inputs are injected (a ``{cluster_id: iso_date|None}`` map and a
    ``{cluster_id: weight}`` map), so this is unit-testable without a DB. The
    universe of clusters considered is the key set of ``last_surfaced``.

    Args:
        last_surfaced:   ``{cluster_id: last_surfaced_iso | None}`` for the
                         clusters under consideration (typically every active
                         cluster). ``None`` → never surfaced.
        feedback_weights: ``{cluster_id: feedback_weight}`` from K-a. Missing
                         entries default to ``1.0``.
        now:             Reference time for the quiet-window cutoff.
        quiet_weeks:     Weeks of silence before a cluster is under-engaged.
        floor:           Atrophy floor (config ``minimum_cluster_floor``); only
                         used to derive the default exclusion threshold.
        min_explore_weight: Explicit exclusion threshold. When ``None``,
                         ``floor + explore_weight_epsilon`` is used.
        explore_weight_epsilon: Margin above the floor for the default
                         threshold.

    Returns:
        Sorted list of under-engaged cluster ids (ascending), each of which is
        BOTH quiet for ``quiet_weeks`` AND above the dismissal-exclusion
        threshold.
    """
    threshold = (
        float(min_explore_weight)
        if min_explore_weight is not None
        else float(floor) + float(explore_weight_epsilon)
    )
    cutoff = now - timedelta(weeks=max(0, int(quiet_weeks)))

    under_engaged: list[int] = []
    for cid, last_iso in last_surfaced.items():
        # Leg 1: quiet for the window (never surfaced always counts).
        last_dt = _parse_iso_dt(last_iso)
        is_quiet = last_dt is None or last_dt < cutoff
        if not is_quiet:
            continue
        # Leg 2: not actively dismissed — weight strictly above the threshold.
        weight = float(feedback_weights.get(cid, 1.0))
        if weight <= threshold:
            continue
        under_engaged.append(int(cid))

    return sorted(under_engaged)
