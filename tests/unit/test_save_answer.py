"""WA3: save_ask_answer — write the Q&A to the Obsidian Connections folder.

Mirrors the persist-zone-safe pattern of
scripts/obsidian_tools/synthesize.py::_write_synthesis_note: write
``Ask_{slug}.md`` under ``vault_path / connections_folder`` and preserve any
user-edited ``## Notes`` section on re-save. No live LLM / graph / network.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def _cfg(tmp_path):
    return SimpleNamespace(
        obsidian=SimpleNamespace(
            vault_path=str(tmp_path / "vault"),
            connections_folder="Literature/Connections",
        )
    )


def test_save_ask_answer_writes_connections_note(tmp_path):
    from scripts.obsidian_tools.save_answer import save_ask_answer

    p = save_ask_answer(
        question="Which papers cite Smith 2021?",
        prose="Three papers cite it.",
        rows=[{"title": "Jones 2022"}],
        cypher="MATCH (p)-[:CITES]->(q) RETURN p",
        config=_cfg(tmp_path),
    )
    text = Path(p).read_text(encoding="utf-8")
    assert "Literature/Connections" in p and p.endswith(".md")
    assert "Which papers cite Smith 2021?" in text  # question
    assert "Three papers cite it." in text  # prose
    assert "Jones 2022" in text  # table
    assert "MATCH (p)-[:CITES]->(q)" in text  # cypher fenced
    assert "tags:" in text and "ask" in text  # front-matter tag


def test_save_ask_answer_preserves_user_notes(tmp_path):
    from scripts.obsidian_tools.save_answer import save_ask_answer

    cfg = _cfg(tmp_path)
    p = save_ask_answer("Q", "A1", [], None, cfg)
    Path(p).write_text(Path(p).read_text() + "\n## Notes\nmy hand edit\n")
    p2 = save_ask_answer("Q", "A2", [], None, cfg)  # same slug → update
    assert p2 == p
    text2 = Path(p2).read_text()
    assert "my hand edit" in text2
    # Idempotent: exactly ONE ## Notes header, never doubled on re-save.
    assert text2.count("## Notes") == 1
    # A THIRD save still keeps a single header + the edit (stable).
    save_ask_answer("Q", "A3", [], None, cfg)
    text3 = Path(p).read_text()
    assert text3.count("## Notes") == 1
    assert "my hand edit" in text3
