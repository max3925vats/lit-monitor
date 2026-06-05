"""Bundle M (P5 closeout): the combined active-learning loop, end to end.

This is the single "P5 works end to end" proof. It ties together the pieces that
the per-bundle tests cover in isolation:

  * ``tests/unit/test_feedback_loop.py``  → J alone (Rocchio + rank lift)
  * ``tests/integration/test_exploration_budget.py`` → K-b alone (discovery wiring)

Here we drive the FULL loop against a REAL ``StateDB`` (J1 recompute →
J2 rank lift → K-a per-cluster atrophy → K-b under-engaged selection), composing
J and K so the seams between them are exercised together.

Fully OFFLINE and DETERMINISTIC: embeddings are hand-built synthetic vectors
delivered through a fake ``embed_lookup``; the ranker's vector store / LLM are
``MagicMock``s; no Ollama, no PubMed, no ChromaDB, no KuzuDB, no network. The
only real component is the on-disk SQLite ``StateDB`` (a ``tmp_path`` file).

The K-b leg uses the SAME pure selector discovery calls
(``find_under_engaged_clusters``) fed by the real DB-backed helpers
(``get_cluster_last_surfaced`` + ``compute_cluster_feedback_weights_from_db``),
so it proves the selection logic without needing a live graph or search backend
(those are covered by ``test_exploration_budget.py``).
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from scripts.clustering.atrophy import (
    compute_cluster_feedback_weights_from_db,
    find_under_engaged_clusters,
)
from scripts.core.state_db import StateDB
from scripts.learning.rocchio import (
    build_interest_inputs,
    compute_interest_vector,
    soft_gate,
)
from scripts.llm.ranker import rank_papers

# Three orthogonal "taste directions" in a tiny embedding space.
_LIKED_DIR = np.array([1.0, 0.0, 0.0], dtype=np.float32)     # saved toward this
_DISLIKED_DIR = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # dismissed toward this
_NEUTRAL_DIR = np.array([0.0, 0.0, 1.0], dtype=np.float32)   # library filler

#: ``feedback_events.source`` tag for an implicit Zotero-save (mirrors
#: ``StateDB.IMPLICIT_SAVE_SOURCE`` / ``rocchio.IMPLICIT_SAVE_WEIGHT``).
_IMPLICIT_SOURCE = StateDB.IMPLICIT_SAVE_SOURCE


@pytest.fixture
def db(tmp_path) -> StateDB:
    return StateDB(tmp_path / "state.db")


def _make_vector_store(score: float = 0.5) -> MagicMock:
    """A fake ChromaDB-like store: every text query returns one neighbour."""
    store = MagicMock()
    store.find_similar_to_text.return_value = [
        {"score": score, "id": "y", "document": "", "metadata": {}}
    ]
    return store


def _make_llm() -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = "{}"
    return llm


def _blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def test_p5_active_learning_loop_end_to_end(db: StateDB):
    """One cohesive proof that J + K compose into a working learning loop.

    Topology seeded into a real StateDB:

      * cluster SAVE   — papers near _LIKED_DIR; user SAVED them (incl. one
                         implicit Zotero-save). Never surfaced in discovery.
      * cluster DISMISS— papers near _DISLIKED_DIR; user heavily DISMISSED them.
                         Never surfaced in discovery.
      * cluster RECENT — healthy papers near _NEUTRAL_DIR; no negative feedback,
                         but surfaced in a RECENT discovery run.

    Assertions span the whole loop:
      (a) J1: the recomputed interest vector pulls toward the SAVED direction;
      (b) P4/J: the implicit save contributes at HALF weight vs an explicit save;
      (c) J2: with ranking.weights.feedback > 0 a saved-aligned candidate out-ranks
          a dismissed-aligned one (and the lift is attributed to feedback);
      (d) K-a: the heavily-dismissed cluster's atrophy weight is BELOW the
          never-dismissed clusters';
      (e) K-b: the under-engaged (never-surfaced, healthy-weight) SAVE cluster is
          selected by find_under_engaged_clusters while the recently-surfaced
          RECENT cluster is NOT (and the dismissed cluster is excluded too).
    """
    # ----------------------------------------------------------------- #
    # 1. Seed a real StateDB: papers, clusters, assignments, feedback.
    # ----------------------------------------------------------------- #
    embeddings: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(42)

    def _seed_paper(doi: str, base_dir: np.ndarray) -> None:
        jitter = rng.normal(0.0, 0.02, size=3).astype(np.float32)
        embeddings[doi] = (base_dir + jitter).astype(np.float32)
        db.upsert_paper({"doi": doi, "title": doi, "status": "in_library"})

    # -- cluster SAVE: liked papers, saved by the user (never surfaced). --
    cid_save = db.insert_cluster("Saved theme", 12, 0.6, _blob(_LIKED_DIR))
    save_dois = [f"10.1/like{i}" for i in range(11)]
    for doi in save_dois:
        _seed_paper(doi, _LIKED_DIR)
        db.upsert_cluster_assignment(doi, cid_save, distance_to_centroid=0.1)
        db.record_feedback_event(doi, "saved", source="discovery")
    # One IMPLICIT Zotero-save (P4 Part C): saved-after-surfaced, half weight.
    implicit_doi = "10.1/like-implicit"
    _seed_paper(implicit_doi, _LIKED_DIR)
    db.upsert_cluster_assignment(implicit_doi, cid_save, distance_to_centroid=0.1)
    db.record_feedback_event(implicit_doi, "saved", source=_IMPLICIT_SOURCE)

    # -- cluster DISMISS: disliked papers, heavily dismissed (never surfaced). --
    cid_dismiss = db.insert_cluster("Dismissed theme", 3, 0.4, _blob(_DISLIKED_DIR))
    dismiss_dois = [f"10.1/dis{i}" for i in range(3)]
    for doi in dismiss_dois:
        _seed_paper(doi, _DISLIKED_DIR)
        db.upsert_cluster_assignment(doi, cid_dismiss, distance_to_centroid=0.1)
        # Pile on recent dismissals so the K-a weight is driven below the K-b
        # exploration-exclusion threshold (floor + epsilon = 0.15).
        for _ in range(25):
            db.record_feedback_event(doi, "dismissed", source="discovery")

    # -- cluster RECENT: healthy neutral papers, surfaced in a recent run. --
    cid_recent = db.insert_cluster("Recent theme", 2, 0.5, _blob(_NEUTRAL_DIR))
    recent_dois = ["10.1/rec0", "10.1/rec1"]
    for doi in recent_dois:
        _seed_paper(doi, _NEUTRAL_DIR)
        db.upsert_cluster_assignment(doi, cid_recent, distance_to_centroid=0.1)
    # A discovery run that surfaced rec0 → cluster RECENT is NOT quiet.
    run_id = db.start_discovery_run({"rag_mode": "vector"})
    db.add_discovery_paper(
        run_id=run_id, doi="10.1/rec0", title="recent",
        score=0.9, rationale="", ingested=False,
    )
    db.finish_discovery_run(run_id, "success", total_found=1, total_ingested=0)

    # A few neutral library papers so the library centroid is not degenerate.
    for i in range(5):
        embeddings[f"10.1/fill{i}"] = _NEUTRAL_DIR.copy()

    def embed_lookup(doi: str) -> np.ndarray | None:
        return embeddings.get(doi)

    # Single reference time captured AFTER all writes so time-decay is stable for
    # every leg (events were written at ~now via SQLite datetime('now')).
    now = datetime.now()

    # ----------------------------------------------------------------- #
    # 2. J1 recompute: build_interest_inputs → compute_interest_vector → store.
    # ----------------------------------------------------------------- #
    events = db.list_feedback_events(limit=100_000)
    library_centroid = np.mean(
        np.stack(list(embeddings.values())), axis=0
    ).astype(np.float32)

    positive, negative, n_events = build_interest_inputs(
        events, embed_lookup, now=now
    )
    interest_vec = compute_interest_vector(library_centroid, positive, negative)
    db.store_interest_vector("global", interest_vec, n_events)

    # Round-trips through the real state table (interest_vectors).
    loaded = db.get_interest_vector("global")
    assert loaded is not None
    stored_vec, stored_n = loaded
    np.testing.assert_allclose(stored_vec, interest_vec, rtol=0, atol=1e-6)
    assert stored_n == n_events

    # (a) The learned direction leans toward LIKED and away from DISLIKED, and
    #     enough events accumulated to open the cold-start soft-gate.
    assert n_events >= 10
    assert soft_gate(n_events) > 0.0
    assert np.isfinite(interest_vec).all()
    assert float(np.dot(interest_vec, _LIKED_DIR)) > float(
        np.dot(interest_vec, _DISLIKED_DIR)
    )

    # ----------------------------------------------------------------- #
    # 3. (b) The implicit Zotero-save contributed at HALF the weight of an
    #        otherwise-identical explicit save (same signal, same age, same dir).
    # ----------------------------------------------------------------- #
    # Re-derive the per-event positive weights from build_interest_inputs by
    # mapping each positive (vector, weight) back to its DOI. Because every saved
    # paper shares ~the same embedding direction and age, the ONLY thing that can
    # halve a save's weight is the implicit-source down-weight (P4/J contract).
    explicit_save_event = next(
        e for e in events
        if e["signal_type"] == "saved" and e["source"] == "discovery"
    )
    implicit_save_event = next(
        e for e in events
        if e["signal_type"] == "saved" and e["source"] == _IMPLICIT_SOURCE
    )

    def _single_weight(ev: dict) -> float:
        pos, _neg, n = build_interest_inputs([ev], embed_lookup, now=now)
        assert n == 1 and len(pos) == 1
        return float(pos[0][1])

    explicit_w = _single_weight(explicit_save_event)
    implicit_w = _single_weight(implicit_save_event)
    # Implicit save is exactly half an explicit save (IMPLICIT_SAVE_WEIGHT=0.5).
    assert implicit_w == pytest.approx(0.5 * explicit_w, rel=1e-6)

    # ----------------------------------------------------------------- #
    # 4. (c) J2 rank lift: with feedback weight ON, the saved-aligned candidate
    #        out-ranks the dismissed-aligned one; with it OFF, no lift.
    # ----------------------------------------------------------------- #
    liked_candidate = {
        "doi": "10.1/cand-liked",
        "title": "Candidate aligned with what you SAVED",
        "abstract": "",
        "_embedding": _LIKED_DIR.copy(),
    }
    disliked_candidate = {
        "doi": "10.1/cand-disliked",
        "title": "Candidate aligned with what you DISMISSED",
        "abstract": "",
        "_embedding": _DISLIKED_DIR.copy(),
    }

    ranked = rank_papers(
        [disliked_candidate, liked_candidate],  # disliked passed FIRST on purpose
        _make_vector_store(0.5),
        _make_llm(),
        interest_vec=stored_vec,
        interest_vec_n_events=stored_n,
        interest_weight=0.5,  # ranking.weights.feedback > 0 → opt-in ON
    )
    by_doi = {p["doi"]: p for p in ranked}
    assert (
        by_doi["10.1/cand-liked"]["similarity_score"]
        > by_doi["10.1/cand-disliked"]["similarity_score"]
    )
    assert ranked[0]["doi"] == "10.1/cand-liked"
    assert by_doi["10.1/cand-liked"]["score_breakdown"]["feedback"] > 0.0

    # Inert contract: the SAME learned vector, feedback weight 0 → no lift.
    ranked_off = rank_papers(
        [liked_candidate],
        _make_vector_store(0.5),
        _make_llm(),
        interest_vec=stored_vec,
        interest_vec_n_events=stored_n,
        interest_weight=0.0,  # OFF
    )
    assert ranked_off[0]["similarity_score"] == 0.5
    assert ranked_off[0]["score_breakdown"]["feedback"] == 0.0

    # ----------------------------------------------------------------- #
    # 5. (d) K-a per-cluster atrophy: the heavily-dismissed cluster's weight is
    #        below the two never-dismissed clusters' (computed from the real DB).
    # ----------------------------------------------------------------- #
    floor = 0.1
    weights = compute_cluster_feedback_weights_from_db(db, now=now, floor=floor)
    assert weights[cid_dismiss] < weights[cid_save]
    assert weights[cid_dismiss] < weights[cid_recent]
    # Heavily-dismissed cluster is atrophied well toward the floor (the weight
    # approaches but never reaches it); never-dismissed clusters stay at 1.0.
    assert floor <= weights[cid_dismiss] < 0.3
    assert weights[cid_save] == pytest.approx(1.0, abs=1e-6)
    assert weights[cid_recent] == pytest.approx(1.0, abs=1e-6)

    # ----------------------------------------------------------------- #
    # 6. (e) K-b under-engaged selection: same DB-backed inputs discovery uses,
    #        fed into the pure selector. SAVE (never-surfaced, healthy) is picked;
    #        RECENT (just surfaced) is not; DISMISS (near floor) is excluded.
    # ----------------------------------------------------------------- #
    last_surfaced = db.get_cluster_last_surfaced()
    assert last_surfaced[cid_save] is None       # never surfaced
    assert last_surfaced[cid_recent] is not None  # surfaced in the recent run

    under_engaged = find_under_engaged_clusters(
        last_surfaced,
        weights,
        now=now,
        quiet_weeks=4,
        floor=floor,
    )
    assert cid_save in under_engaged       # under-engaged → probe it
    assert cid_recent not in under_engaged  # recently surfaced → leave it
    assert cid_dismiss not in under_engaged  # actively dismissed → don't fight it
