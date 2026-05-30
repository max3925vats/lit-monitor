"""
scripts.graph — KuzuDB-backed knowledge graph for lit-monitor (Phase 1 + 2).

Public API
----------
GraphDB      : KuzuDB wrapper; import is always safe. Instantiation raises
               ImportError if kuzu is not installed (``uv sync --extra graph``).
BiobertNER   : BioBERT NER wrapper (N1, Phase 2). Import is always safe;
               ``.extract()`` raises ImportError if [nlp] not installed
               (``uv sync --extra nlp``).
NerSpan      : Frozen dataclass for a single NER-extracted span.
safe_graph_db: Context-manager helper around GraphDB.
"""
from __future__ import annotations

# Re-export GraphDB so callers can write ``from scripts.graph import GraphDB``
# without knowing the internal module layout.  The import is always available
# regardless of whether kuzu is installed; the lazy kuzu import inside GraphDB
# ensures that non-graph users see no errors at import time.
from scripts.graph.db import GraphDB
from scripts.graph.import_citations import safe_graph_db

# N1 (Phase 2): BioBERT NER — lazy transformers import, safe to import always.
from scripts.graph.ner import BiobertNER, NerSpan

__all__ = ["GraphDB", "safe_graph_db", "BiobertNER", "NerSpan"]
