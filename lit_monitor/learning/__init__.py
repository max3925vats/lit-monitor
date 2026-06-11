"""Active-learning core (Bundle J).

Pure-computation + storage layer for the Rocchio interest-vector loop. This
package builds an "interest direction" from captured feedback events; the
rank-time wiring that *uses* it (soft-gated re-ranking) lands in Bundle J2.
"""

from __future__ import annotations

from lit_monitor.learning.rocchio import (
    BETA,
    COLD_START_FLOOR,
    GAMMA,
    HALF_LIFE_DAYS,
    IMPLICIT_SAVE_WEIGHT,
    SOFT_GATE_K,
    build_interest_inputs,
    compute_interest_vector,
    soft_gate,
    time_decay_weight,
)

__all__ = [
    "BETA",
    "GAMMA",
    "HALF_LIFE_DAYS",
    "COLD_START_FLOOR",
    "SOFT_GATE_K",
    "IMPLICIT_SAVE_WEIGHT",
    "soft_gate",
    "time_decay_weight",
    "compute_interest_vector",
    "build_interest_inputs",
]
