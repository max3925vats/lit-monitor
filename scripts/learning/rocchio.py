"""Rocchio interest-vector core (Bundle J1) — pure computation.

This module turns captured feedback into a single "interest direction": a
Rocchio centroid update that nudges a prior (the library centroid) TOWARD
papers the user liked and AWAY from ones they dismissed. A later bundle (J2)
uses that direction to re-rank discoveries.

Design notes
------------
- **Pure functions only.** Everything here takes vectors + event metadata and
  returns vectors/floats. No DB access, no embedding I/O. The one place that
  needs embeddings (``build_interest_inputs``) receives an injected
  ``embed_lookup`` callable, so the whole module is unit-testable with fakes.
- **Inert below the cold-start floor.** ``compute_interest_vector`` always
  builds *a* direction. What actually makes the loop inert until enough data
  exists is the rank-time soft-gate (``soft_gate``), which returns ``0.0`` when
  ``n_events < COLD_START_FLOOR``. J1 only computes + stores; the gate is
  applied where the vector is consumed (J2).
- **Finite by construction.** No NaN/inf may ever reach a downstream score, so
  every code path here is guarded with ``np.isfinite`` / zero-norm checks in
  the same discipline as ``scripts/llm/ranker.py``.

Signal classification source-of-truth: ``StateDB._FEEDBACK_WEIGHT_MAP`` /
``_VALID_SIGNAL_TYPES`` in ``scripts/core/state_db.py``. The closed vocabulary
is: opened | saved | dismissed | rated | thumbs_up | thumbs_down.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Tunable constants. All locked-decision defaults per the Bundle J brief.
# --------------------------------------------------------------------------- #

#: Rocchio positive coefficient — how hard liked papers pull the centroid.
#: Asymmetric (β > γ) per locked decision #J to damp echo-chamber drift.
BETA: float = 0.75

#: Rocchio negative coefficient — how hard dismissed papers push it away.
GAMMA: float = 0.15

#: Time-decay half-life (days): a signal this old counts half as much.
HALF_LIFE_DAYS: float = 90.0

#: Below this many contributing events the soft-gate returns 0.0 (fully off),
#: so the interest vector is never applied on a handful of noisy clicks.
COLD_START_FLOOR: int = 10

#: Smoothing constant in the soft-gate ``n / (n + k)``: larger k = slower ramp.
SOFT_GATE_K: int = 30

#: Implicit Zotero-save events are a weaker signal than an explicit save/thumb,
#: so their weight is scaled by this factor (locked decision: down-weight 0.5).
IMPLICIT_SAVE_WEIGHT: float = 0.5

#: Implicit-save source tag (mirrors ``StateDB.IMPLICIT_SAVE_SOURCE``). Kept as
#: a local constant so this pure module needs no StateDB import.
_IMPLICIT_SAVE_SOURCE: str = "implicit_zotero_save"

#: Rating thresholds for sign classification (1-5 scale, 3 = neutral midpoint).
_RATING_POSITIVE_MIN: int = 4
_RATING_NEGATIVE_MAX: int = 2

#: Signal types that are unconditionally positive / negative (rating handled
#: separately because its sign depends on the rating value).
_POSITIVE_SIGNALS: frozenset[str] = frozenset({"saved", "thumbs_up"})
_NEGATIVE_SIGNALS: frozenset[str] = frozenset({"dismissed", "thumbs_down"})


# --------------------------------------------------------------------------- #
# Gating + decay (pure scalars)
# --------------------------------------------------------------------------- #

def soft_gate(
    n_events: int,
    *,
    cold_start_floor: int = COLD_START_FLOOR,
    k: int = SOFT_GATE_K,
) -> float:
    """Confidence multiplier for the interest vector, in ``[0, 1)``.

    Below ``cold_start_floor`` events the loop is fully off (returns ``0.0``);
    above it, the gate ramps smoothly as ``n_events / (n_events + k)`` so
    personalization strengthens as evidence accumulates instead of switching
    on abruptly at the floor.

    Args:
        n_events:         Number of contributing feedback events.
        cold_start_floor: Hard floor below which the gate is 0.0.
        k:                Smoothing constant (larger ⇒ slower ramp).

    Returns:
        ``0.0`` if ``n_events < cold_start_floor``, else ``n / (n + k)``.
    """
    if n_events < cold_start_floor:
        return 0.0
    return float(n_events) / (float(n_events) + float(k))


def time_decay_weight(
    age_days: float,
    *,
    half_life_days: float = HALF_LIFE_DAYS,
) -> float:
    """Exponential time-decay multiplier: ``0.5 ** (age_days / half_life_days)``.

    A fresh signal (age 0) counts fully (1.0); one ``half_life_days`` old counts
    half; two half-lives old counts a quarter, and so on.

    Args:
        age_days:       Age of the event in days (clamped at 0 for the future).
        half_life_days: Days after which a signal's weight halves.

    Returns:
        Decay multiplier in ``(0, 1]``.
    """
    age = max(0.0, float(age_days))
    return float(0.5 ** (age / float(half_life_days)))


# --------------------------------------------------------------------------- #
# Rocchio centroid update (pure vector math)
# --------------------------------------------------------------------------- #

def _weighted_mean(
    items: list[tuple[np.ndarray, float]],
    dim: int,
) -> np.ndarray:
    """Weighted mean of (vector, weight) pairs, returning a finite ``(dim,)``.

    Non-finite vectors and non-finite/non-positive weights are skipped. An
    empty (or fully-degenerate) input yields a zero vector, which is the
    correct no-op for a Rocchio leg.
    """
    acc = np.zeros(dim, dtype=np.float64)
    total_w = 0.0
    for vec, weight in items:
        arr = np.asarray(vec, dtype=np.float64).ravel()
        if arr.shape[0] != dim or not np.isfinite(arr).all():
            continue
        w = float(weight)
        if not np.isfinite(w) or w <= 0.0:
            continue
        acc += w * arr
        total_w += w
    if total_w <= 0.0:
        return np.zeros(dim, dtype=np.float64)
    return acc / total_w


def _safe_normalize(vec: np.ndarray, dim: int) -> np.ndarray:
    """L2-normalize to a finite unit direction; zero/degenerate ⇒ zero vector.

    Mirrors ranker.py's discipline: a NaN norm slips past a naive ``> 0`` check,
    so the finite test comes first.
    """
    arr = np.asarray(vec, dtype=np.float64).ravel()
    if arr.shape[0] != dim or not np.isfinite(arr).all():
        return np.zeros(dim, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9 or not np.isfinite(norm):
        return np.zeros(dim, dtype=np.float64)
    out = arr / norm
    if not np.isfinite(out).all():
        return np.zeros(dim, dtype=np.float64)
    return out


def compute_interest_vector(
    library_centroid: np.ndarray,
    positive: list[tuple[np.ndarray, float]],
    negative: list[tuple[np.ndarray, float]],
    *,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> np.ndarray:
    """Rocchio interest direction from a prior + liked/dismissed papers.

    Computes::

        interest = library_centroid
                   + beta  * weighted_mean(positive)
                   - gamma * weighted_mean(negative)

    then L2-normalizes the result so it is a pure *direction* (cosine-ready).
    ``positive`` / ``negative`` are lists of ``(vector, weight)`` tuples whose
    weights already fold in time-decay and the implicit-source down-weight
    (see ``build_interest_inputs``).

    The cold-start soft-gate is NOT applied here — that happens at rank time
    (Bundle J2). This function always builds a direction.

    Edge cases (all yield a finite result, never NaN):
        - empty ``positive`` AND empty ``negative`` ⇒ normalized
          ``library_centroid`` unchanged.
        - degenerate / zero / non-finite ``library_centroid`` ⇒ falls back to
          the (normalized) Rocchio delta if that is finite, else a zero vector.
        - any non-finite intermediate ⇒ guarded so the RESULT is finite.

    Args:
        library_centroid: Prior direction (e.g. the library centroid).
        positive:         ``(vector, weight)`` pairs for liked papers.
        negative:         ``(vector, weight)`` pairs for dismissed papers.
        beta:             Positive Rocchio coefficient.
        gamma:            Negative Rocchio coefficient.

    Returns:
        A finite L2-unit ``(dim,)`` float32 direction (or a zero vector if
        every input was degenerate).
    """
    prior = np.asarray(library_centroid, dtype=np.float64).ravel()
    dim = int(prior.shape[0])
    if dim == 0:
        return np.zeros(0, dtype=np.float32)

    # A non-finite prior must not poison the sum; treat it as a zero prior and
    # let the Rocchio delta (if any) carry the direction.
    prior_finite = prior if np.isfinite(prior).all() else np.zeros(dim, dtype=np.float64)

    no_feedback = not positive and not negative
    if no_feedback:
        return _safe_normalize(prior_finite, dim).astype(np.float32)

    pos_mean = _weighted_mean(positive, dim)
    neg_mean = _weighted_mean(negative, dim)
    interest = prior_finite + float(beta) * pos_mean - float(gamma) * neg_mean

    normalized = _safe_normalize(interest, dim)
    if not normalized.any():
        # Whole thing degenerated (e.g. NaN prior + no usable feedback). Fall
        # back to the prior direction so we never return a meaningless zero when
        # a usable prior exists.
        logger.warning(
            "compute_interest_vector: Rocchio result degenerate; "
            "falling back to library-centroid direction."
        )
        normalized = _safe_normalize(prior_finite, dim)
    return normalized.astype(np.float32)


# --------------------------------------------------------------------------- #
# Orchestrator: raw feedback events -> Rocchio inputs
# --------------------------------------------------------------------------- #

def _event_age_days(created_at: str | None, now) -> float:
    """Age in days of ``created_at`` relative to ``now``; 0.0 if unparseable."""
    if not created_at:
        return 0.0
    from datetime import datetime

    text = str(created_at).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            ts = datetime.strptime(text[: len(fmt) + 2], fmt)
            return max(0.0, (now - ts).total_seconds() / 86400.0)
        except (ValueError, TypeError):
            continue
    logger.warning("build_interest_inputs: unparseable created_at %r; age=0", created_at)
    return 0.0


def _classify_signal(signal_type: str, rating: int | None) -> int:
    """Return +1 (positive), -1 (negative), or 0 (neutral / skip).

    Source-of-truth for the strings: ``StateDB._FEEDBACK_WEIGHT_MAP``.
    ``opened`` is treated as NEUTRAL (skipped) here — see module / pooled note.
    ``rated`` is positive at ≥4, negative at ≤2, neutral at 3.
    """
    if signal_type in _POSITIVE_SIGNALS:
        return 1
    if signal_type in _NEGATIVE_SIGNALS:
        return -1
    if signal_type == "rated" and rating is not None:
        if int(rating) >= _RATING_POSITIVE_MIN:
            return 1
        if int(rating) <= _RATING_NEGATIVE_MAX:
            return -1
        return 0  # rating == 3, neutral midpoint
    return 0  # 'opened' and anything unrecognized


def build_interest_inputs(
    events: list[dict],
    embed_lookup: Callable[[str], np.ndarray | None],
    now,
    *,
    implicit_weight: float = IMPLICIT_SAVE_WEIGHT,
) -> tuple[list[tuple[np.ndarray, float]], list[tuple[np.ndarray, float]], int]:
    """Map raw ``feedback_events`` rows into Rocchio positive/negative inputs.

    For each event:
        1. classify it as positive / negative / neutral (``_classify_signal``);
           neutral (``opened`` or ``rated==3``) is skipped;
        2. look up its DOI's embedding via the injected ``embed_lookup``; events
           whose DOI has no embedding are skipped;
        3. compute its magnitude as
           ``|base_signal_weight| × time_decay_weight(age) × source_factor``
           where ``source_factor = implicit_weight`` for an
           ``implicit_zotero_save`` event, else ``1.0``.

    The base signal magnitudes mirror ``StateDB._FEEDBACK_WEIGHT_MAP`` (and its
    ``(rating-3)*0.5`` scaling for ``rated``) so the writer and this reader can't
    drift. Sign is carried by which list the event lands in, so weights stored in
    the tuples are non-negative magnitudes.

    Args:
        events:          ``feedback_events`` row dicts (keys: doi, signal_type,
                         weight, rating, source, created_at).
        embed_lookup:    ``doi -> np.ndarray | None`` embedding provider.
        now:             ``datetime`` reference for time-decay.
        implicit_weight: Down-weight factor for implicit-save events.

    Returns:
        ``(positive, negative, n_events)`` where each list holds
        ``(vector, weight)`` tuples and ``n_events`` counts contributors.
    """
    positive: list[tuple[np.ndarray, float]] = []
    negative: list[tuple[np.ndarray, float]] = []
    n_events = 0

    for ev in events:
        signal_type = ev.get("signal_type")
        if not signal_type:
            continue
        rating = ev.get("rating")
        sign = _classify_signal(signal_type, rating)
        if sign == 0:
            continue

        doi = ev.get("doi")
        if not doi:
            continue
        vec = embed_lookup(doi)
        if vec is None:
            continue
        arr = np.asarray(vec, dtype=np.float32).ravel()
        if arr.size == 0 or not np.isfinite(arr).all():
            continue

        # Base magnitude: prefer the pre-computed stored weight (folds in the
        # rated (rating-3)*0.5 scaling), falling back to nothing if absent.
        stored = ev.get("weight")
        base = abs(float(stored)) if stored is not None and np.isfinite(float(stored)) else 0.0
        if base <= 0.0:
            continue

        age_days = _event_age_days(ev.get("created_at"), now)
        decay = time_decay_weight(age_days)
        source_factor = (
            float(implicit_weight)
            if ev.get("source") == _IMPLICIT_SAVE_SOURCE
            else 1.0
        )
        weight = base * decay * source_factor
        if weight <= 0.0 or not np.isfinite(weight):
            continue

        if sign > 0:
            positive.append((arr, weight))
        else:
            negative.append((arr, weight))
        n_events += 1

    return positive, negative, n_events
