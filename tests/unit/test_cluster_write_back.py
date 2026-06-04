"""
Unit tests for Bundle C: Zotero write-back.

Tests cover:
- push_tags_to_zotero: strictly additive, dry_run respected, no delete_* calls
- push_collections_to_zotero: parent created if absent, sub-collections added,
  duplicate-membership when use_existing_priors enabled
- Zotero client is mocked throughout — no real API calls
"""
from __future__ import annotations

from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state_db_with_clusters(cluster_data: list[dict], assignment_data: list[dict]):
    """Return a mock StateDB that yields the given clusters and assignments."""
    sdb = MagicMock()
    sdb.list_active_clusters.return_value = cluster_data
    sdb.get_cluster_assignments.return_value = assignment_data
    return sdb


def _make_zot_client():
    """Return a fresh mock Zotero client with sensible defaults."""
    client = MagicMock()
    # get_all_collections returns empty by default
    client.get_all_collections.return_value = []
    # _zot is the underlying pyzotero instance
    client._zot = MagicMock()
    client._zot.create_collections.return_value = [{"key": "NEWKEY"}]
    return client


# ---------------------------------------------------------------------------
# push_tags_to_zotero
# ---------------------------------------------------------------------------

class TestPushTagsToZotero:
    def test_dry_run_makes_no_api_calls(self):
        """dry_run=True must not call any write methods on the Zotero client."""
        from scripts.clustering.write_back import push_tags_to_zotero

        clusters = [{"id": 1, "display_name": "Test Theme", "n_papers": 2}]
        assignments = [
            {"doi": "10.1000/a", "cluster_id": 1},
            {"doi": "10.1000/b", "cluster_id": 1},
        ]
        sdb = _make_state_db_with_clusters(clusters, assignments)

        # Mock the state_db to return Zotero keys for DOIs
        sdb.get_paper.side_effect = lambda doi: {
            "10.1000/a": {"zotero_key": "KEYA"},
            "10.1000/b": {"zotero_key": "KEYB"},
        }.get(doi)

        zot = _make_zot_client()

        report = push_tags_to_zotero(sdb, zot, dry_run=True)

        # No write calls
        zot._zot.update_item.assert_not_called()
        zot._zot.create_items.assert_not_called()

        # Report describes what would be written
        assert report is not None
        assert isinstance(report, dict)

    def test_confirm_adds_tags_additively(self):
        """dry_run=False actually calls update/tag methods — additive only."""
        from scripts.clustering.write_back import push_tags_to_zotero

        clusters = [{"id": 1, "display_name": "CEX Chromatography", "n_papers": 1}]
        assignments = [{"doi": "10.1000/a", "cluster_id": 1}]

        sdb = _make_state_db_with_clusters(clusters, assignments)
        sdb.get_paper.side_effect = lambda doi: {"zotero_key": "KEYA"} if doi == "10.1000/a" else None

        # Simulate: existing item has no tags
        zot = _make_zot_client()
        zot._zot.item.return_value = {"data": {"tags": []}}
        zot._zot.update_item.return_value = {}

        push_tags_to_zotero(sdb, zot, dry_run=False)

        # update_item must have been called (not create_items — tags use update)
        assert zot._zot.update_item.called

        # Verify the tag was in the lm/ namespace
        call_args = zot._zot.update_item.call_args
        updated_item = call_args[0][0]
        tag_values = [t["tag"] for t in updated_item.get("data", {}).get("tags", [])]
        assert any("lm/" in tv for tv in tag_values)

    def test_no_delete_methods_called(self):
        """Under no circumstances should delete_* methods be called."""
        from scripts.clustering.write_back import push_tags_to_zotero

        clusters = [{"id": 1, "display_name": "Test", "n_papers": 1}]
        assignments = [{"doi": "10.1000/a", "cluster_id": 1}]
        sdb = _make_state_db_with_clusters(clusters, assignments)
        sdb.get_paper.return_value = {"zotero_key": "KEY1"}

        zot = _make_zot_client()
        zot._zot.item.return_value = {"data": {"tags": []}}
        zot._zot.update_item.return_value = {}

        push_tags_to_zotero(sdb, zot, dry_run=False)

        # write_back is strictly additive (module docstring). Assert that none of
        # the real pyzotero destructive methods were invoked. The previous
        # `for attr in dir(zot._zot)` loop was vacuous: dir() on a MagicMock lists
        # only already-materialised children, so the "delete*" filter matched
        # nothing and asserted nothing. These named methods exist on the real
        # pyzotero.Zotero client (verified against the installed version).
        zot._zot.delete_item.assert_not_called()
        zot._zot.delete_collection.assert_not_called()
        zot._zot.delete_tags.assert_not_called()
        zot._zot.deletefrom_collection.assert_not_called()

    def test_skips_papers_without_zotero_key(self):
        """Papers with no zotero_key in state_db are silently skipped."""
        from scripts.clustering.write_back import push_tags_to_zotero

        clusters = [{"id": 1, "display_name": "Theme", "n_papers": 1}]
        assignments = [{"doi": "10.unknown/x", "cluster_id": 1}]
        sdb = _make_state_db_with_clusters(clusters, assignments)
        sdb.get_paper.return_value = None  # not in state_db

        zot = _make_zot_client()

        # Should not raise, and no updates should happen
        push_tags_to_zotero(sdb, zot, dry_run=False)
        zot._zot.update_item.assert_not_called()

    def test_tag_uses_lm_namespace(self):
        """Tags must be in the form lm/<theme-name>."""
        from scripts.clustering.write_back import push_tags_to_zotero

        clusters = [{"id": 1, "display_name": "Membrane Filtration", "n_papers": 1}]
        assignments = [{"doi": "10.1000/a", "cluster_id": 1}]
        sdb = _make_state_db_with_clusters(clusters, assignments)
        sdb.get_paper.return_value = {"zotero_key": "KEYA"}

        zot = _make_zot_client()
        zot._zot.item.return_value = {"data": {"tags": []}}
        zot._zot.update_item.return_value = {}

        push_tags_to_zotero(sdb, zot, dry_run=False)

        call_args = zot._zot.update_item.call_args[0][0]
        tags = [t["tag"] for t in call_args["data"]["tags"]]
        assert "lm/Membrane Filtration" in tags

    def test_report_splits_success_and_failure(self):
        """Q3.3: a failed Zotero op must NOT be counted as written.

        One of two items raises on update. The report must show exactly one
        success (tags_added=1, papers_processed=1) and one failure
        (tags_failed=1) — never all-success.
        """
        from scripts.clustering.write_back import push_tags_to_zotero

        clusters = [{"id": 1, "display_name": "Theme", "n_papers": 2}]
        assignments = [
            {"doi": "10.1000/ok", "cluster_id": 1},
            {"doi": "10.1000/bad", "cluster_id": 1},
        ]
        sdb = _make_state_db_with_clusters(clusters, assignments)
        sdb.get_paper.side_effect = lambda doi: {
            "10.1000/ok": {"zotero_key": "KEYOK"},
            "10.1000/bad": {"zotero_key": "KEYBAD"},
        }.get(doi)

        zot = _make_zot_client()
        # Fresh item dict per call (avoid a shared mutable return_value that
        # would make the second item appear already-tagged and skip the write).
        zot._zot.item.side_effect = lambda zkey: {"data": {"tags": []}}

        # First update_item call (KEYOK) succeeds, second (KEYBAD) raises.
        zot._zot.update_item.side_effect = [{}, RuntimeError("zotero write failed")]

        report = push_tags_to_zotero(sdb, zot, dry_run=False)

        assert report["tags_added"] == 1
        assert report["papers_processed"] == 1
        assert report["tags_failed"] == 1

    def test_dry_run_calls_zotero_noop(self):
        """Audit-2 F2: dry-run must not touch any Zotero mutating method.

        Migrated here from tests/integration/test_v09_e2e.py, where it was
        permanently skipped by a broken ``pytest.importorskip(
        "scripts.clustering.writeback")`` (wrong module — the real module
        is ``write_back``) plus a call to a non-existent ``write_back_tags``.
        It is a pure unit test (mocked state_db + zotero_client, no live
        services), so it lives in the unit suite and runs unconditionally.

        The live path of push_tags_to_zotero reads each item via
        ``zotero_client._zot.item`` and writes via ``_zot.update_item``.
        With dry_run=True neither must be invoked, and the report must
        carry ``dry_run=True``.
        """
        from scripts.clustering.write_back import push_tags_to_zotero

        clusters = [{"id": 0, "display_name": "Protein A & CEX", "n_papers": 2}]
        assignments = [
            {"doi": "10.0/a", "cluster_id": 0},
            {"doi": "10.0/b", "cluster_id": 0},
        ]
        sdb = _make_state_db_with_clusters(clusters, assignments)
        sdb.get_paper.side_effect = lambda doi: {
            "10.0/a": {"zotero_key": "KEYA"},
            "10.0/b": {"zotero_key": "KEYB"},
        }.get(doi)

        zot = _make_zot_client()

        report = push_tags_to_zotero(sdb, zot, dry_run=True)

        # In dry-run mode the Zotero read+write methods used by the live
        # path must be called zero times.
        zot._zot.item.assert_not_called()
        zot._zot.update_item.assert_not_called()

        # And the report explicitly flags itself as a dry run.
        assert report["dry_run"] is True


# ---------------------------------------------------------------------------
# push_collections_to_zotero
# ---------------------------------------------------------------------------

class TestPushCollectionsToZotero:
    def test_dry_run_no_collection_creates(self):
        """dry_run=True must not create collections or move items."""
        from scripts.clustering.write_back import push_collections_to_zotero

        clusters = [{"id": 1, "display_name": "CEX", "n_papers": 1}]
        assignments = [{"doi": "10.1000/a", "cluster_id": 1}]
        sdb = _make_state_db_with_clusters(clusters, assignments)
        sdb.get_paper.return_value = {"zotero_key": "KEYA"}

        zot = _make_zot_client()

        report = push_collections_to_zotero(sdb, zot, dry_run=True)

        zot._zot.create_collections.assert_not_called()
        assert report is not None

    def test_creates_parent_collection_if_absent(self):
        """If the parent collection doesn't exist, it must be created."""
        from scripts.clustering.write_back import push_collections_to_zotero

        clusters = [{"id": 1, "display_name": "CEX", "n_papers": 1}]
        assignments = [{"doi": "10.1000/a", "cluster_id": 1}]
        sdb = _make_state_db_with_clusters(clusters, assignments)
        sdb.get_paper.return_value = {"zotero_key": "KEYA"}

        zot = _make_zot_client()
        # No existing collections
        zot.get_all_collections.return_value = []
        # create_collections returns a list of created-item dicts
        zot._zot.create_collections.return_value = [{"key": "PARENTKEY"}]
        zot._zot.addto_collection.return_value = {}

        push_collections_to_zotero(sdb, zot, dry_run=False, parent_name="lit-monitor")

        # Should have been called to create the parent
        zot._zot.create_collections.assert_called()

    def test_no_delete_methods_called_collections(self):
        """Collections write-back must never call delete methods."""
        from scripts.clustering.write_back import push_collections_to_zotero

        clusters = [{"id": 1, "display_name": "SEP", "n_papers": 1}]
        assignments = [{"doi": "10.1000/a", "cluster_id": 1}]
        sdb = _make_state_db_with_clusters(clusters, assignments)
        sdb.get_paper.return_value = {"zotero_key": "KEYA"}

        zot = _make_zot_client()
        zot._zot.create_collections.return_value = [{"key": "X"}]
        zot._zot.addto_collection.return_value = {}

        push_collections_to_zotero(sdb, zot, dry_run=False)

        # Strictly additive: no destructive pyzotero method may fire (the old
        # dir()-based loop was vacuous — see test_no_delete_methods_called).
        zot._zot.delete_item.assert_not_called()
        zot._zot.delete_collection.assert_not_called()
        zot._zot.delete_tags.assert_not_called()
        zot._zot.deletefrom_collection.assert_not_called()

    def test_dry_run_report_lists_operations(self):
        """dry_run report must describe planned operations, not an empty dict."""
        from scripts.clustering.write_back import push_collections_to_zotero

        clusters = [
            {"id": 1, "display_name": "Theme A", "n_papers": 2},
            {"id": 2, "display_name": "Theme B", "n_papers": 1},
        ]
        assignments = [
            {"doi": "10.1000/a", "cluster_id": 1},
            {"doi": "10.1000/b", "cluster_id": 1},
            {"doi": "10.1000/c", "cluster_id": 2},
        ]
        sdb = _make_state_db_with_clusters(clusters, assignments)
        sdb.get_paper.side_effect = lambda doi: {"zotero_key": doi.split("/")[-1]}

        zot = _make_zot_client()

        report = push_collections_to_zotero(sdb, zot, dry_run=True)

        # Should describe at least 2 sub-collections to create
        total_ops = report.get("collections_to_create", 0) + report.get("items_to_add", 0)
        assert total_ops > 0
