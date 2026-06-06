"""FI-1: unit tests for scripts/api/insights.py assemblers.

All fakes — no live embeddings, no LLM, no real DB. Each test injects a duck-typed
fake state_db / embeddings_db exposing only the methods the assembler calls.
"""
from __future__ import annotations

import numpy as np

from scripts.api.insights import get_cluster_weights, get_learning_state


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _FakeStateDB:
    """Minimal duck-typed StateDB for get_learning_state."""

    def __init__(
        self,
        *,
        interest: tuple[np.ndarray, int] | None,
        computed_at: str = "2026-06-06 12:00:00",
    ) -> None:
        self._interest = interest
        self._computed_at = computed_at

    def get_interest_vector(self, scope: str = "global"):  # noqa: ANN001
        return self._interest

    def get_paper(self, doi: str):  # noqa: ANN001
        return {"title": f"Title for {doi}"}

    # Mimic the raw computed_at read path used by the CLI.
    class _Conn:
        def __init__(self, computed_at: str) -> None:
            self._computed_at = computed_at

        def __enter__(self):
            return self

        def __exit__(self, *exc):  # noqa: ANN002
            return False

        def execute(self, *_a, **_k):  # noqa: ANN002
            outer = self

            class _Cur:
                def fetchone(self_inner):  # noqa: ANN001
                    return {"computed_at": outer._computed_at}

            return _Cur()

    def _connect(self):
        return _FakeStateDB._Conn(self._computed_at)


class _FakeEmbeddingsDB:
    """Minimal duck-typed EmbeddingsDB exposing get_paper_embeddings()."""

    def __init__(self, embeddings: dict[str, np.ndarray]) -> None:
        self._embeddings = embeddings

    def get_paper_embeddings(self) -> dict[str, np.ndarray]:
        return self._embeddings


class _FakeClusterDB:
    """Duck-typed StateDB for get_cluster_weights."""

    def __init__(self, clusters: list[dict]) -> None:
        self._clusters = clusters

    def list_active_clusters(self) -> list[dict]:
        return self._clusters


# --------------------------------------------------------------------------- #
# get_learning_state
# --------------------------------------------------------------------------- #
def test_get_learning_state_summary(monkeypatch) -> None:
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    fake_db = _FakeStateDB(interest=(vec, 42))
    fake_emb = _FakeEmbeddingsDB(
        {
            "10.1/a": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "10.1/b": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }
    )
    st = get_learning_state(fake_db, fake_emb, top_k=5)
    assert st["available"] is True and st["n_events"] == 42
    assert st["inert"] is False and 0.0 < st["soft_gate"] < 1.0 and st["dim"] > 0
    assert st["top_papers"]  # ranking produced something
    assert st["top_papers"][0]["doi"] == "10.1/a"  # perfectly aligned ranks first
    assert {"doi", "title", "cosine"} <= set(st["top_papers"][0])


def test_get_learning_state_inert_below_floor() -> None:
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    fake_db = _FakeStateDB(interest=(vec, 5))  # 5 < COLD_START_FLOOR(10)
    st = get_learning_state(fake_db, None)
    assert st["available"] is True
    assert st["inert"] is True and st["soft_gate"] == 0.0


def test_get_learning_state_absent() -> None:
    fake_db = _FakeStateDB(interest=None)
    assert get_learning_state(fake_db, None)["available"] is False


# --------------------------------------------------------------------------- #
# get_cluster_weights
# --------------------------------------------------------------------------- #
def test_get_cluster_weights_sorted(monkeypatch) -> None:
    import scripts.api.insights as insights_mod

    monkeypatch.setattr(
        insights_mod,
        "compute_cluster_feedback_weights_from_db",
        lambda state_db, *, floor: {1: 0.42, 2: 0.10},
    )
    fake_db = _FakeClusterDB(
        [
            {"id": 1, "display_name": "Alpha"},
            {"id": 2, "display_name": "Beta"},
        ]
    )
    out = get_cluster_weights(fake_db, floor=0.1)
    assert out["floor"] == 0.1
    weights = [c["weight"] for c in out["clusters"]]
    assert weights == sorted(weights)  # ascending
    assert out["clusters"][0]["at_floor"] is True  # weight <= floor


def test_get_cluster_weights_empty(monkeypatch) -> None:
    import scripts.api.insights as insights_mod

    monkeypatch.setattr(
        insights_mod,
        "compute_cluster_feedback_weights_from_db",
        lambda state_db, *, floor: {},
    )
    fake_db = _FakeClusterDB([])
    assert get_cluster_weights(fake_db, floor=0.1)["clusters"] == []
