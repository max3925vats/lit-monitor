"""
Tests for Phase 12: CLI and setup check modules.
All tests mock external I/O — no real Zotero/Ollama/filesystem access.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from scripts.cli import main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def runner():
    return CliRunner()
# ---------------------------------------------------------------------------
# check_configured tests
# ---------------------------------------------------------------------------
class TestCheckConfigured:
    def test_missing_secrets_file(self, tmp_path):
        from scripts.setup.check_configured import check_configured
        with patch("scripts.setup.check_configured._SECRETS_PATH", tmp_path / "no.toml"):
            results = check_configured()
        assert results["secrets_file"][0] is False
        assert "Missing" in results["secrets_file"][1]
    def test_valid_secrets_file(self, tmp_path):
        from scripts.setup.check_configured import check_configured
        secrets = tmp_path / "config.toml"
        secrets.write_text(
            '[zotero]\napi_key = "abc123"\nlibrary_id = "99999"\n'
            '[wos]\napi_key = "woskey"\n'
        )
        with patch("scripts.setup.check_configured._SECRETS_PATH", secrets):
            results = check_configured()
        assert results["secrets_file"][0] is True
        assert results["zotero.api_key"][0] is True
        assert results["zotero.library_id"][0] is True
    def test_missing_required_key(self, tmp_path):
        from scripts.setup.check_configured import check_configured
        secrets = tmp_path / "config.toml"
        secrets.write_text('[zotero]\nlibrary_id = "99999"\n')  # no api_key
        with patch("scripts.setup.check_configured._SECRETS_PATH", secrets):
            results = check_configured()
        assert results["zotero.api_key"][0] is False
    def test_optional_key_missing_still_passes(self, tmp_path):
        from scripts.setup.check_configured import check_configured
        secrets = tmp_path / "config.toml"
        secrets.write_text('[zotero]\napi_key = "x"\nlibrary_id = "1"\n')
        with patch("scripts.setup.check_configured._SECRETS_PATH", secrets):
            results = check_configured()

        # wos.api_key removed (WoS not supported by findpapers 0.6.x); only scopus remains optional
        assert "wos.api_key" not in results
        assert results["scopus.api_key"][0] is True

    def test_check_configured_absent_scopus_returns_warn_severity(self, tmp_path):
        """Optional-absent key must be ok=True severity='warn' (Bug D fix).

        Before the 3-state model, optional-absent rendered green ✓ because the
        boolean alone couldn't distinguish "configured" from "skipped". Now
        the severity attr surfaces it as yellow ⚠.
        """
        from scripts.setup.check_configured import check_configured
        secrets = tmp_path / "config.toml"
        secrets.write_text(
            '[zotero]\napi_key = "x"\nlibrary_id = "1"\n'
            '[pubmed]\nemail = "a@b.com"\n'
        )
        with patch("scripts.setup.check_configured._SECRETS_PATH", secrets):
            results = check_configured()
        scopus = results["scopus.api_key"]
        assert scopus.ok is True
        assert scopus.severity == "warn"
        # Required configured keys stay severity='ok'.
        assert results["zotero.api_key"].severity == "ok"
# ---------------------------------------------------------------------------
# check_ollama tests
# ---------------------------------------------------------------------------
class TestCheckOllama:
    def test_ollama_not_running(self):
        import requests

        from scripts.setup.check_ollama import check_ollama
        with patch("scripts.setup.check_ollama.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("refused")
            results = check_ollama()
        assert results["ollama_running"][0] is False
        assert "ollama serve" in results["ollama_running"][1]
    def test_ollama_running_no_model_check(self):
        from scripts.setup.check_ollama import check_ollama
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "mistral:7b"}]}
        mock_resp.raise_for_status.return_value = None
        with patch("scripts.setup.check_ollama.requests.get", return_value=mock_resp):
            results = check_ollama()
        assert results["ollama_running"][0] is True
        assert "model_available" not in results
    def test_model_found(self):
        from scripts.setup.check_ollama import check_ollama
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "mistral:7b"}]}
        mock_resp.raise_for_status.return_value = None
        with patch("scripts.setup.check_ollama.requests.get", return_value=mock_resp):
            results = check_ollama(model="mistral:7b")
        assert results["model_available"][0] is True
    def test_model_not_found(self):
        from scripts.setup.check_ollama import check_ollama
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "phi3:mini"}]}
        mock_resp.raise_for_status.return_value = None
        with patch("scripts.setup.check_ollama.requests.get", return_value=mock_resp):
            results = check_ollama(model="mistral:7b")
        assert results["model_available"][0] is False
        assert "ollama pull" in results["model_available"][1]
# ---------------------------------------------------------------------------
# check_zotero tests
# ---------------------------------------------------------------------------
class TestCheckZotero:
    def test_missing_secrets(self, tmp_path):
        from scripts.setup.check_zotero import check_zotero
        with patch("scripts.setup.check_zotero._SECRETS_PATH", tmp_path / "no.toml"):
            results = check_zotero()
        assert results["zotero_credentials"][0] is False
    def test_api_success(self, tmp_path):
        from scripts.setup.check_zotero import check_zotero
        secrets = tmp_path / "config.toml"
        secrets.write_text('[zotero]\napi_key = "abc"\nlibrary_id = "123"\n')
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with (
            patch("scripts.setup.check_zotero._SECRETS_PATH", secrets),
            patch("scripts.setup.check_zotero.requests.get", return_value=mock_resp),
        ):
            results = check_zotero()
        assert results["zotero_credentials"][0] is True
        assert results["zotero_api"][0] is True
    def test_api_forbidden(self, tmp_path):
        from scripts.setup.check_zotero import check_zotero
        secrets = tmp_path / "config.toml"
        secrets.write_text('[zotero]\napi_key = "bad"\nlibrary_id = "123"\n')
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        with (
            patch("scripts.setup.check_zotero._SECRETS_PATH", secrets),
            patch("scripts.setup.check_zotero.requests.get", return_value=mock_resp),
        ):
            results = check_zotero()
        assert results["zotero_api"][0] is False
        assert "403" in results["zotero_api"][1]
# ---------------------------------------------------------------------------
# CLI command tests
# ---------------------------------------------------------------------------
class TestCliStatus:
    def test_status_shows_counts(self, runner, tmp_path):
        mock_config = MagicMock()
        mock_config.state_db.path = str(tmp_path / "state.db")
        mock_state_db = MagicMock()
        mock_state_db.count_by_status.return_value = {"extraction_complete": 5, "no_markdown": 2}
        mock_state_db.get_all_by_source_type.side_effect = lambda t: (
            [{"doi": f"d{i}", "embeddings_indexed": 1} for i in range(5)] if t == "paper"
            else []
        )
        with (
            patch("scripts.cli._make_config", return_value=mock_config),
            patch("scripts.cli._make_state_db", return_value=mock_state_db),
        ):
            result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "Papers" in result.output
        assert "extraction_complete" in result.output

    def test_status_includes_graph_line_when_extra_installed(self, runner, tmp_path):
        """G13: when [graph] is importable, lit-monitor status includes Graph: line."""
        mock_config = MagicMock()
        mock_config.state_db.path = str(tmp_path / "state.db")
        mock_state_db = MagicMock()
        mock_state_db.count_by_status.return_value = {}
        mock_state_db.get_all_by_source_type.return_value = []

        # Mock graph DB returning paper_count=3, entity_count=10
        mock_conn = MagicMock()

        def _exec(sql: str) -> MagicMock:
            res = MagicMock()
            res.has_next.return_value = True
            if "Paper" in sql:
                res.get_next.return_value = [3]
            elif "Entity" in sql:
                res.get_next.return_value = [10]
            else:
                res.get_next.return_value = [0]
            return res

        mock_conn.execute.side_effect = _exec
        mock_graph_db = MagicMock()
        mock_graph_db._conn = mock_conn

        # state_db._connect() returns a context-manager conn with fetchall results
        mock_inner_conn = MagicMock()
        mock_inner_conn.execute.return_value.fetchall.side_effect = [
            [(2,)],   # graph_indexed=1 count  -> indexed=2
            [(5,)],   # total papers            -> total=5
        ]
        mock_state_db._connect.return_value.__enter__ = lambda s: mock_inner_conn
        mock_state_db._connect.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch("scripts.cli._make_config", return_value=mock_config),
            patch("scripts.cli._make_state_db", return_value=mock_state_db),
            patch("scripts.graph.safe_graph_db", return_value=mock_graph_db),
        ):
            result = runner.invoke(main, ["status"])

        assert result.exit_code == 0
        assert "Graph:" in result.output
        assert "indexed=" in result.output
        assert "entities=" in result.output

    def test_status_omits_graph_line_when_extra_missing(self, runner, tmp_path):
        """G13: ImportError on [graph] import -> status doesn't crash, omits Graph: line."""
        mock_config = MagicMock()
        mock_config.state_db.path = str(tmp_path / "state.db")
        mock_state_db = MagicMock()
        mock_state_db.count_by_status.return_value = {}
        mock_state_db.get_all_by_source_type.return_value = []

        with (
            patch("scripts.cli._make_config", return_value=mock_config),
            patch("scripts.cli._make_state_db", return_value=mock_state_db),
            patch("scripts.graph.safe_graph_db", side_effect=ImportError("no graph")),
        ):
            result = runner.invoke(main, ["status"])

        assert result.exit_code == 0
        assert "Graph:" not in result.output


class TestCliCheck:
    def test_check_all_pass(self, runner):
        from scripts.setup.check_configured import CheckResult
        all_ok = {"key": CheckResult(True, "all good", "ok")}
        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._load_secrets", return_value={}),
            patch("scripts.setup.check_configured.check_configured", return_value=all_ok),
            patch("scripts.setup.check_ollama.check_ollama", return_value=all_ok),
            patch("scripts.setup.check_zotero.check_zotero", return_value=all_ok),
        ):
            result = runner.invoke(main, ["check"])
        assert result.exit_code == 0
        assert "Tesseract" not in result.output
    def test_check_fails_on_bad_config(self, runner):
        from scripts.setup.check_configured import CheckResult
        bad = {"zotero.api_key": CheckResult(False, "Missing", "fail")}
        ok = {"x": CheckResult(True, "ok", "ok")}
        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._load_secrets", return_value={}),
            patch("scripts.setup.check_configured.check_configured", return_value=bad),
            patch("scripts.setup.check_ollama.check_ollama", return_value=ok),
            patch("scripts.setup.check_zotero.check_zotero", return_value=ok),
        ):
            result = runner.invoke(main, ["check"])
        assert result.exit_code == 1
    def test_check_shows_api_key_from_config_toml(self, runner):
        from scripts.setup.check_configured import CheckResult
        all_ok = {"key": CheckResult(True, "all good", "ok")}
        mock_cfg = MagicMock()
        mock_cfg.brain_build.ollama_host = "https://ollama.com"
        secrets = {"ollama": {"api_key": "secret-from-toml"}}
        env = {"OLLAMA_API_KEY": ""}  # env var absent
        with (
            patch("scripts.cli._make_config", return_value=mock_cfg),
            patch("scripts.cli._load_secrets", return_value=secrets),
            patch("scripts.setup.check_configured.check_configured", return_value=all_ok),
            patch("scripts.setup.check_ollama.check_ollama", return_value=all_ok),
            patch("scripts.setup.check_zotero.check_zotero", return_value=all_ok),
            patch.dict("os.environ", env, clear=False),
        ):
            result = runner.invoke(main, ["check"])
        assert "config.toml" in result.output
    def test_check_shows_api_key_from_env_var(self, runner):
        from scripts.setup.check_configured import CheckResult
        all_ok = {"key": CheckResult(True, "all good", "ok")}
        mock_cfg = MagicMock()
        mock_cfg.brain_build.ollama_host = "https://ollama.com"
        env = {"OLLAMA_API_KEY": "env-key-value"}
        with (
            patch("scripts.cli._make_config", return_value=mock_cfg),
            patch("scripts.cli._load_secrets", return_value={}),
            patch("scripts.setup.check_configured.check_configured", return_value=all_ok),
            patch("scripts.setup.check_ollama.check_ollama", return_value=all_ok),
            patch("scripts.setup.check_zotero.check_zotero", return_value=all_ok),
            patch.dict("os.environ", env, clear=False),
        ):
            result = runner.invoke(main, ["check"])
        assert "env var" in result.output
    def test_check_local_ollama_no_key_is_ok(self, runner):
        from scripts.setup.check_configured import CheckResult
        all_ok = {"key": CheckResult(True, "all good", "ok")}
        mock_cfg = MagicMock()
        mock_cfg.brain_build.ollama_host = "http://localhost:11434"
        env = {}
        with (
            patch("scripts.cli._make_config", return_value=mock_cfg),
            patch("scripts.cli._load_secrets", return_value={}),
            patch("scripts.setup.check_configured.check_configured", return_value=all_ok),
            patch("scripts.setup.check_ollama.check_ollama", return_value=all_ok),
            patch("scripts.setup.check_zotero.check_zotero", return_value=all_ok),
            patch.dict("os.environ", env, clear=False),
        ):
            result = runner.invoke(main, ["check"])
        assert "local Ollama" in result.output
        assert result.exit_code == 0
class TestCliBrainBuild:
    def test_brain_build_invokes_pipeline(self, runner):
        from scripts.core.state_db import CURRENT_SCHEMA_VERSION
        mock_summary = MagicMock()
        mock_summary.papers_processed = 3
        mock_summary.papers_skipped = 1
        mock_summary.papers_failed = 0
        mock_summary.errors = []
        mock_state_db = MagicMock()
        mock_state_db.get_schema_version.return_value = CURRENT_SCHEMA_VERSION
        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._make_state_db", return_value=mock_state_db),
            patch("scripts.cli._make_embeddings_db", return_value=MagicMock()),
            patch("scripts.cli._make_zotero_client", return_value=MagicMock()),
            patch("scripts.cli._make_llm", return_value=MagicMock()),
            patch("scripts.cli._load_secrets", return_value={}),
            patch("scripts.output.embeddings.check_embed_model_change"),
            patch(
                "scripts.pipelines.brain_build.run_brain_build",
                return_value=mock_summary,
            ),
        ):
            result = runner.invoke(main, ["brain-build", "--batch-size", "10"])
        assert result.exit_code == 0
        assert "Papers processed" in result.output
        assert "3" in result.output

    def test_brain_build_rate_limit_exhausted_maps_to_exit_2(self, runner):
        """P4.5: the CLI catches RateLimitExhausted (raised by run_brain_build)
        and maps it to exit code 2, preserving the historical exit-code-2
        behaviour now that the library no longer raises SystemExit."""
        from scripts.core.state_db import CURRENT_SCHEMA_VERSION
        from scripts.pipelines.brain_build import RateLimitExhausted
        mock_state_db = MagicMock()
        mock_state_db.get_schema_version.return_value = CURRENT_SCHEMA_VERSION
        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._make_state_db", return_value=mock_state_db),
            patch("scripts.cli._make_embeddings_db", return_value=MagicMock()),
            patch("scripts.cli._make_zotero_client", return_value=MagicMock()),
            patch("scripts.cli._make_llm", return_value=MagicMock()),
            patch("scripts.cli._load_secrets", return_value={}),
            patch("scripts.output.embeddings.check_embed_model_change"),
            patch(
                "scripts.pipelines.brain_build.run_brain_build",
                side_effect=RateLimitExhausted("rate limit persists"),
            ),
        ):
            result = runner.invoke(main, ["brain-build"])
        assert result.exit_code == 2, (
            f"Expected exit code 2 from RateLimitExhausted; got {result.exit_code}"
        )

    def test_brain_build_all_library_flag_is_passed_through(self, runner):
        """N22: ``brain-build --all-library`` must reach run_brain_build with
        ``all_library=True``.  Closes the CLI-plumbing gap surfaced in Audit R27.
        """
        from scripts.core.state_db import CURRENT_SCHEMA_VERSION
        mock_summary = MagicMock()
        mock_summary.papers_processed = 0
        mock_summary.papers_skipped = 0
        mock_summary.papers_failed = 0
        mock_summary.errors = []
        mock_state_db = MagicMock()
        mock_state_db.get_schema_version.return_value = CURRENT_SCHEMA_VERSION

        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._make_state_db", return_value=mock_state_db),
            patch("scripts.cli._make_embeddings_db", return_value=MagicMock()),
            patch("scripts.cli._make_zotero_client", return_value=MagicMock()),
            patch("scripts.cli._make_llm", return_value=MagicMock()),
            patch("scripts.cli._load_secrets", return_value={}),
            patch("scripts.output.embeddings.check_embed_model_change"),
            patch(
                "scripts.pipelines.brain_build.run_brain_build",
                return_value=mock_summary,
            ) as mock_run,
        ):
            result = runner.invoke(main, ["brain-build", "--all-library"])

        assert result.exit_code == 0
        assert mock_run.call_args.kwargs.get("all_library") is True, (
            "CLI did not propagate --all-library through to run_brain_build"
        )

    def test_brain_build_default_does_not_set_all_library(self, runner):
        """N22 inverse: without --all-library, run_brain_build is called with
        ``all_library=False`` (or the default).
        """
        from scripts.core.state_db import CURRENT_SCHEMA_VERSION
        mock_summary = MagicMock()
        mock_summary.papers_processed = 0
        mock_summary.papers_skipped = 0
        mock_summary.papers_failed = 0
        mock_summary.errors = []
        mock_state_db = MagicMock()
        mock_state_db.get_schema_version.return_value = CURRENT_SCHEMA_VERSION

        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._make_state_db", return_value=mock_state_db),
            patch("scripts.cli._make_embeddings_db", return_value=MagicMock()),
            patch("scripts.cli._make_zotero_client", return_value=MagicMock()),
            patch("scripts.cli._make_llm", return_value=MagicMock()),
            patch("scripts.cli._load_secrets", return_value={}),
            patch("scripts.output.embeddings.check_embed_model_change"),
            patch(
                "scripts.pipelines.brain_build.run_brain_build",
                return_value=mock_summary,
            ) as mock_run,
        ):
            result = runner.invoke(main, ["brain-build"])

        assert result.exit_code == 0
        # Default may be omitted from kwargs entirely, or explicitly False.
        assert mock_run.call_args.kwargs.get("all_library", False) is False

class TestCliRun:

    def test_run_dry_run(self, runner):
        mock_summary = MagicMock()
        mock_summary.new_papers_found = 5
        mock_summary.papers_ingested = 0
        mock_summary.papers_failed = 0
        mock_summary.digest_path = "/vault/Digests/Discovery_2026-05-02.md"
        mock_summary.errors = []
        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._make_state_db", return_value=MagicMock()),
            patch("scripts.cli._make_embeddings_db", return_value=MagicMock()),
            patch("scripts.cli._make_zotero_client", return_value=MagicMock()),
            patch("scripts.cli._make_llm", return_value=MagicMock()),
            patch("scripts.cli._load_secrets", return_value={}),
            patch(
                "scripts.pipelines.discovery.run_discovery",
                return_value=mock_summary,
            ),
        ):
            result = runner.invoke(main, ["run", "--dry-run"])
        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert "5" in result.output
class TestCliObsidian:
    def test_retheme_command(self, runner):
        mock_stats = {"files_modified": 7, "wikilinks_rewritten": 14, "page_renamed": 1}
        mock_config = MagicMock()
        mock_config.obsidian.vault_path = Path("/vault")
        with (
            patch("scripts.cli._make_config", return_value=mock_config),
            patch("scripts.obsidian_tools.retheme.retheme", return_value=mock_stats),
        ):
            result = runner.invoke(main, ["obsidian", "retheme", "--old", "Old", "--new", "New"])
        assert result.exit_code == 0
        assert "7" in result.output
    def test_rerender_single_doi(self, runner):
        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._make_state_db", return_value=MagicMock()),
            patch(
                "scripts.obsidian_tools.rerender.rerender_note",
                return_value="/vault/Papers/Smith2020.md",
            ),
        ):
            result = runner.invoke(main, ["obsidian", "rerender", "--doi", "10.1/test"])
        assert result.exit_code == 0
        assert "Smith2020" in result.output
    def test_re_extract_doi_exits_zero(self, runner):
        """Regression: cli.py must call _load_secrets() not _load_api_secrets() (NameError guard)."""
        mock_result = {"core_finding": "flux decline observed"}
        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._make_state_db", return_value=MagicMock()),
            patch("scripts.cli._make_llm", return_value=MagicMock()),
            patch("scripts.cli._load_secrets", return_value={"zotero": {}}),
            patch("scripts.cli._make_zotero_client", return_value=MagicMock()),
            patch("scripts.obsidian_tools.re_extract.re_extract", return_value=mock_result),
        ):
            result = runner.invoke(main, ["obsidian", "re-extract", "--doi", "10.1/test"])
        assert result.exit_code == 0, result.output

    def test_synthesize_no_results(self, runner):
        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._make_state_db", return_value=MagicMock()),
            patch("scripts.cli._make_embeddings_db", return_value=MagicMock()),
            patch("scripts.cli._make_llm", return_value=MagicMock()),
            patch("scripts.obsidian_tools.synthesize.synthesize", return_value=""),
        ):
            result = runner.invoke(main, ["obsidian", "synthesize", "--topic", "TopicX"])

        assert result.exit_code == 0
        assert "No relevant" in result.output
# ---------------------------------------------------------------------------
# I2 — --resolve-no-doi interactive reprocessing
# ---------------------------------------------------------------------------

class TestCliResolvNoDoi:
    def test_resolve_no_doi_flag_reprocesses_with_user_doi(self, runner):
        """
        When --resolve-no-doi is passed and summary.no_doi_items is non-empty,
        the CLI must call _process_paper() with the user-supplied DOI.
        """
        from scripts.core.state_db import CURRENT_SCHEMA_VERSION
        from scripts.pipelines.brain_build import BuildSummary

        # A summary where one item has no DOI
        mock_summary = BuildSummary()
        mock_summary.papers_processed = 0
        mock_summary.papers_skipped = 0
        mock_summary.papers_failed = 0
        mock_summary.no_doi_items = [
            {
                "zotero_key": "NODOI01",
                "title": "Paper Without DOI",
                "authors": [{"lastName": "Smith"}],
                "year": 2022,
                "_item": {"key": "NODOI01", "data": {"title": "Paper Without DOI"}},
            }
        ]

        mock_state_db = MagicMock()
        mock_state_db.get_schema_version.return_value = CURRENT_SCHEMA_VERSION

        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._make_state_db", return_value=mock_state_db),
            patch("scripts.cli._make_embeddings_db", return_value=MagicMock()),
            patch("scripts.cli._make_zotero_client", return_value=MagicMock()),
            patch("scripts.cli._make_llm", return_value=MagicMock()),
            patch("scripts.cli._load_secrets", return_value={}),
            patch("scripts.output.embeddings.check_embed_model_change"),
            patch(
                "scripts.pipelines.brain_build.run_brain_build",
                return_value=mock_summary,
            ),
            patch("scripts.pipelines.brain_build.write_brain_build_report",
                  return_value=None),
            patch(
                "scripts.pipelines.brain_build._process_paper"
            ) as mock_process,
            patch("scripts.pipelines.brain_build.extract_paper"),
            patch("scripts.output.obsidian_writer.write_paper_note"),
        ):
            # Supply DOI via stdin — matches click.prompt(default="")
            result = runner.invoke(
                main,
                ["brain-build", "--resolve-no-doi"],
                input="10.1/manual-doi\n",
            )

        assert result.exit_code == 0, result.output
        # _process_paper must have been called exactly once with the user-supplied DOI
        mock_process.assert_called_once()
        call_kwargs = mock_process.call_args.kwargs
        assert call_kwargs.get("doi") == "10.1/manual-doi", (
            f"Expected doi='10.1/manual-doi' in _process_paper call; got: {call_kwargs}"
        )


class TestCliRebuildCitations:
    """M6 — obsidian rebuild-citations command."""

    def _graph_result(self, n_resolved: int = 2, n_unresolved: int = 1) -> MagicMock:
        r = MagicMock()
        r.n_resolved = n_resolved
        r.n_unresolved = n_unresolved
        r.s2_references_count = n_resolved + n_unresolved
        return r

    def test_scope_doi_calls_re_extract_then_graph(self, runner):
        """--scope doi must call re_extract with phases=['complex'], then build_citation_graph."""
        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._make_state_db", return_value=MagicMock()),
            patch("scripts.cli._make_embeddings_db", return_value=MagicMock()),
            patch("scripts.cli._make_zotero_client", return_value=MagicMock()),
            patch("scripts.cli._make_llm", return_value=MagicMock()),
            patch("scripts.cli._load_secrets", return_value={}),
            patch("scripts.obsidian_tools.re_extract.re_extract") as mock_re_extract,
            patch(
                "scripts.search.citation_graph.build_citation_graph",
                return_value=self._graph_result(),
            ) as mock_graph,
            patch("scripts.obsidian_tools.relink.relink_note"),
        ):
            result = runner.invoke(
                main,
                ["obsidian", "rebuild-citations", "--doi", "10.1/test", "--scope", "doi"],
            )

        assert result.exit_code == 0, result.output
        mock_re_extract.assert_called_once()
        call_kwargs = mock_re_extract.call_args.kwargs
        assert call_kwargs.get("doi") == "10.1/test"
        assert call_kwargs.get("phases") == ["complex"]
        mock_graph.assert_called_once()

    def test_scope_doi_requires_doi_flag(self, runner):
        """--scope doi without --doi must exit non-zero with a usage error."""
        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._make_state_db", return_value=MagicMock()),
            patch("scripts.cli._make_embeddings_db", return_value=MagicMock()),
            patch("scripts.cli._make_zotero_client", return_value=MagicMock()),
            patch("scripts.cli._make_llm", return_value=MagicMock()),
            patch("scripts.cli._load_secrets", return_value={}),
        ):
            result = runner.invoke(
                main,
                ["obsidian", "rebuild-citations", "--scope", "doi"],
            )
        assert result.exit_code != 0

    def test_scope_all_skips_re_extract(self, runner):
        """--scope all must NOT call re_extract — only resolve + relink."""
        mock_state_db = MagicMock()
        # Two papers, both have key_citations
        mock_state_db.get_all_by_source_type.side_effect = lambda t: (
            [{"doi": "10.1/p1", "extraction_json": '{"key_citations": ["ref1"]}'}]
            if t == "paper" else []
        )
        mock_state_db.get_paper.return_value = {"doi": "10.1/p1", "note_path": None}
        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._make_state_db", return_value=mock_state_db),
            patch("scripts.cli._make_embeddings_db", return_value=MagicMock()),
            patch("scripts.cli._make_zotero_client", return_value=MagicMock()),
            patch("scripts.cli._make_llm", return_value=MagicMock()),
            patch("scripts.cli._load_secrets", return_value={}),
            patch("scripts.obsidian_tools.re_extract.re_extract") as mock_re_extract,
            patch(
                "scripts.search.citation_graph.build_citation_graph",
                return_value=self._graph_result(),
            ),
            patch("scripts.obsidian_tools.relink.relink_note"),
        ):
            result = runner.invoke(
                main,
                ["obsidian", "rebuild-citations", "--scope", "all"],
            )

        assert result.exit_code == 0, result.output
        mock_re_extract.assert_not_called()

    def test_scope_failed_skips_papers_with_existing_edges(self, runner):
        """--scope failed must skip papers that already have resolved citation edges."""
        mock_state_db = MagicMock()
        mock_state_db.get_all_by_source_type.side_effect = lambda t: (
            [
                {"doi": "10.1/has_edges", "extraction_json": '{"key_citations": ["r1"]}'},
                {"doi": "10.1/no_edges", "extraction_json": '{"key_citations": ["r2"]}'},
            ]
            if t == "paper" else []
        )
        mock_state_db.get_paper.return_value = {"doi": "10.1/no_edges", "note_path": None}

        def get_edges(doi):
            # paper with existing edges returns a non-empty list
            return [{"target_doi": "10.1/cited"}] if doi == "10.1/has_edges" else []

        mock_state_db.get_citation_edges.side_effect = get_edges

        graph_call_dois: list[str] = []

        def capture_graph(doi, *args, **kwargs):
            graph_call_dois.append(doi)
            r = MagicMock()
            r.n_resolved = 1
            r.n_unresolved = 0
            r.s2_references_count = 1
            return r

        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._make_state_db", return_value=mock_state_db),
            patch("scripts.cli._make_embeddings_db", return_value=MagicMock()),
            patch("scripts.cli._make_zotero_client", return_value=MagicMock()),
            patch("scripts.cli._make_llm", return_value=MagicMock()),
            patch("scripts.cli._load_secrets", return_value={}),
            patch(
                "scripts.search.citation_graph.build_citation_graph",
                side_effect=capture_graph,
            ),
            patch("scripts.obsidian_tools.relink.relink_note"),
        ):
            result = runner.invoke(
                main,
                ["obsidian", "rebuild-citations", "--scope", "failed"],
            )

        assert result.exit_code == 0, result.output
        # Only the paper with no edges should have been processed
        assert "10.1/has_edges" not in graph_call_dois, (
            "Paper with existing edges should be skipped under --scope failed"
        )
        assert "10.1/no_edges" in graph_call_dois

    def test_no_rerender_skips_relink(self, runner):
        """--no-rerender must skip relink_note even when a note_path is set."""
        mock_state_db = MagicMock()
        mock_state_db.get_paper.return_value = {
            "doi": "10.1/test", "note_path": "/vault/test.md"
        }
        mock_state_db.get_all_by_source_type.side_effect = lambda t: (
            [{"doi": "10.1/test", "extraction_json": '{"key_citations": ["r1"]}'}]
            if t == "paper" else []
        )
        mock_state_db.get_citation_edges.return_value = []

        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._make_state_db", return_value=mock_state_db),
            patch("scripts.cli._make_embeddings_db", return_value=MagicMock()),
            patch("scripts.cli._make_zotero_client", return_value=MagicMock()),
            patch("scripts.cli._make_llm", return_value=MagicMock()),
            patch("scripts.cli._load_secrets", return_value={}),
            patch(
                "scripts.search.citation_graph.build_citation_graph",
                return_value=self._graph_result(),
            ),
            patch("scripts.obsidian_tools.relink.relink_note") as mock_relink,
        ):
            result = runner.invoke(
                main,
                ["obsidian", "rebuild-citations", "--scope", "all", "--no-rerender"],
            )

        assert result.exit_code == 0, result.output
        mock_relink.assert_not_called()


class TestMaybeSetOllamaKey:
    """Unit tests for _maybe_set_ollama_key (N2 — credential unification)."""

    def test_sets_key_from_config_when_env_absent(self):
        from scripts.cli import _maybe_set_ollama_key
        secrets = {"ollama": {"api_key": "toml-key-123"}}
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("OLLAMA_API_KEY", None)
            _maybe_set_ollama_key(secrets)
            assert os.environ.get("OLLAMA_API_KEY") == "toml-key-123"
        os.environ.pop("OLLAMA_API_KEY", None)

    def test_env_var_wins_over_config(self):
        from scripts.cli import _maybe_set_ollama_key
        secrets = {"ollama": {"api_key": "toml-key"}}
        with patch.dict("os.environ", {"OLLAMA_API_KEY": "env-key"}, clear=False):
            _maybe_set_ollama_key(secrets)
            assert os.environ["OLLAMA_API_KEY"] == "env-key"

    def test_no_key_no_side_effect(self):
        from scripts.cli import _maybe_set_ollama_key
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("OLLAMA_API_KEY", None)
            _maybe_set_ollama_key({})
            assert "OLLAMA_API_KEY" not in os.environ


class TestMaybeSetS2Key:
    """Unit tests for _maybe_set_s2_key (M3 — S2 credential parity with OLLAMA)."""

    def test_sets_key_from_config_when_env_absent(self):
        from scripts.cli import _maybe_set_s2_key
        secrets = {"semantic_scholar": {"api_key": "toml-s2-key-123"}}
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("S2_API_KEY", None)
            _maybe_set_s2_key(secrets)
            assert os.environ.get("S2_API_KEY") == "toml-s2-key-123"
        os.environ.pop("S2_API_KEY", None)

    def test_env_var_wins_over_config(self):
        from scripts.cli import _maybe_set_s2_key
        secrets = {"semantic_scholar": {"api_key": "toml-key"}}
        with patch.dict("os.environ", {"S2_API_KEY": "env-key"}, clear=False):
            _maybe_set_s2_key(secrets)
            assert os.environ["S2_API_KEY"] == "env-key"

    def test_no_key_no_side_effect(self):
        from scripts.cli import _maybe_set_s2_key
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("S2_API_KEY", None)
            _maybe_set_s2_key({})
            assert "S2_API_KEY" not in os.environ

    def test_missing_section_no_crash(self):
        """Helper must tolerate config.toml without a [semantic_scholar] section."""
        from scripts.cli import _maybe_set_s2_key
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("S2_API_KEY", None)
            # Realistic config.toml has zotero/pubmed/ollama but no semantic_scholar.
            _maybe_set_s2_key({"zotero": {"api_key": "z"}, "ollama": {"api_key": "o"}})
            assert "S2_API_KEY" not in os.environ

    def test_obsidian_build_citation_graph_sets_s2_key_from_secrets(self, runner):
        """build-citation-graph must call _maybe_set_s2_key with loaded secrets."""
        mock_state_db = MagicMock()
        mock_state_db.get_all_by_source_type.return_value = []
        with (
            patch("scripts.cli._load_secrets", return_value={"semantic_scholar": {"api_key": "k"}}),
            patch("scripts.cli._maybe_set_s2_key") as mock_set_s2,
            patch("scripts.core.config.Config", return_value=MagicMock()),
            patch("scripts.core.state_db.StateDB", return_value=mock_state_db),
            patch("scripts.search.citation_graph.build_citation_graph"),
        ):
            result = runner.invoke(
                main,
                ["obsidian", "build-citation-graph", "--scope", "all"],
            )
        assert result.exit_code == 0, result.output
        mock_set_s2.assert_called_once()
        # Confirm it was called with the secrets dict, not an empty fallback.
        args, _kwargs = mock_set_s2.call_args
        assert args[0] == {"semantic_scholar": {"api_key": "k"}}


class TestCliSetupLogging:
    def test_jsonl_handler_emits_valid_json(self, tmp_path):
        import logging

        from scripts.cli import _JsonlFileHandler
        log_file = tmp_path / "test.jsonl"
        handler = _JsonlFileHandler(str(log_file))
        logger = logging.getLogger("test_jsonl")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info("hello world")
        handler.close()
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["level"] == "INFO"
        assert entry["msg"] == "hello world"
        assert "ts" in entry

    def test_setup_logging_deduplicates_jsonl_handler(self, tmp_path):
        """N14.1: calling _setup_logging twice for the same log file must produce
        exactly one _JsonlFileHandler on the root logger, not two."""
        import logging

        from scripts.cli import _JsonlFileHandler, _setup_logging
        root = logging.getLogger()
        # Remove any _JsonlFileHandler that may exist from prior test runs.
        for h in list(root.handlers):
            if isinstance(h, _JsonlFileHandler):
                root.removeHandler(h)
                h.close()
        before = len([h for h in root.handlers if isinstance(h, _JsonlFileHandler)])
        _setup_logging("test_dedup", log_dir=tmp_path, verbose=False)
        _setup_logging("test_dedup", log_dir=tmp_path, verbose=False)
        after = len([h for h in root.handlers if isinstance(h, _JsonlFileHandler)])
        assert after == before + 1, (
            f"Expected exactly one new _JsonlFileHandler after two calls, "
            f"got {after - before} extra(s)"
        )
        # Cleanup
        for h in list(root.handlers):
            if isinstance(h, _JsonlFileHandler):
                root.removeHandler(h)
                h.close()

    def test_setup_logging_ts_is_timezone_aware(self, tmp_path):
        """N14.2: JSONL 'ts' field must use a timezone-aware format (no utcnow deprecation)."""
        import json as _json
        import logging

        from scripts.cli import _JsonlFileHandler, _setup_logging
        root = logging.getLogger()
        _setup_logging("test_tz", log_dir=tmp_path, verbose=True)
        test_logger = logging.getLogger("_test_tz_emit")
        test_logger.warning("tz test message")
        # Find the JSONL file and read the last entry.
        log_files = list(tmp_path.glob("*.jsonl"))
        assert log_files, "No JSONL log file created"
        lines = log_files[0].read_text().strip().splitlines()
        assert lines, "No log lines written"
        entry = _json.loads(lines[-1])
        ts = entry["ts"]
        # datetime.now(timezone.utc).isoformat() produces '+00:00' suffix;
        # the old datetime.utcnow() produced no timezone info.
        assert "+" in ts or ts.endswith("Z"), (
            f"Expected timezone-aware timestamp, got: {ts!r}"
        )
        # Cleanup
        for h in list(root.handlers):
            if isinstance(h, _JsonlFileHandler):
                root.removeHandler(h)
                h.close()
# ---------------------------------------------------------------------------
# N20 — End-of-run reminder after brain-build
# ---------------------------------------------------------------------------
class TestN20Reminder:
    def test_reminder_printed_when_papers_processed(self, runner):
        """N20: brain-build prints next-steps reminder when >= 1 paper was processed."""
        from scripts.core.state_db import CURRENT_SCHEMA_VERSION
        mock_summary = MagicMock()
        mock_summary.papers_processed = 2
        mock_summary.papers_skipped = 0
        mock_summary.papers_failed = 0
        mock_summary.errors = []
        mock_state_db = MagicMock()
        mock_state_db.get_schema_version.return_value = CURRENT_SCHEMA_VERSION
        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._make_state_db", return_value=mock_state_db),
            patch("scripts.cli._make_embeddings_db", return_value=MagicMock()),
            patch("scripts.cli._make_zotero_client", return_value=MagicMock()),
            patch("scripts.cli._make_llm", return_value=MagicMock()),
            patch("scripts.cli._load_secrets", return_value={}),
            patch("scripts.output.embeddings.check_embed_model_change"),
            patch("scripts.pipelines.brain_build.run_brain_build", return_value=mock_summary),
        ):
            result = runner.invoke(main, ["brain-build"])
        assert result.exit_code == 0
        assert "relink" in result.output
        assert "rebuild-citations" in result.output

    def test_reminder_suppressed_when_no_papers_processed(self, runner):
        """N20: reminder block is suppressed for no-op runs (papers_processed == 0)."""
        from scripts.core.state_db import CURRENT_SCHEMA_VERSION
        mock_summary = MagicMock()
        mock_summary.papers_processed = 0
        mock_summary.papers_skipped = 5
        mock_summary.papers_failed = 0
        mock_summary.errors = []
        mock_state_db = MagicMock()
        mock_state_db.get_schema_version.return_value = CURRENT_SCHEMA_VERSION
        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._make_state_db", return_value=mock_state_db),
            patch("scripts.cli._make_embeddings_db", return_value=MagicMock()),
            patch("scripts.cli._make_zotero_client", return_value=MagicMock()),
            patch("scripts.cli._make_llm", return_value=MagicMock()),
            patch("scripts.cli._load_secrets", return_value={}),
            patch("scripts.output.embeddings.check_embed_model_change"),
            patch("scripts.pipelines.brain_build.run_brain_build", return_value=mock_summary),
        ):
            result = runner.invoke(main, ["brain-build"])
        assert result.exit_code == 0
        assert "rebuild-citations" not in result.output


# ---------------------------------------------------------------------------
# N21 — synthesize --topics-file
# ---------------------------------------------------------------------------
class TestN21TopicsFile:
    def test_topics_file_iterates_over_topics(self, runner, tmp_path):
        """N21: --topics-file generates one note per topic."""
        import yaml
        topics_file = tmp_path / "topics.yaml"
        topics_file.write_text(yaml.dump({"topics": ["topic A", "topic B"]}))

        call_log: list[str] = []

        def _fake_synthesize(topic, **kwargs):
            call_log.append(topic)
            return f"/vault/Connections/Synthesis_{topic}.md"

        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._make_state_db", return_value=MagicMock()),
            patch("scripts.cli._make_embeddings_db", return_value=MagicMock()),
            patch("scripts.cli._make_llm", return_value=MagicMock()),
            patch("scripts.obsidian_tools.synthesize.synthesize", side_effect=_fake_synthesize),
        ):
            result = runner.invoke(
                main,
                ["obsidian", "synthesize", "--topics-file", str(topics_file)],
            )
        assert result.exit_code == 0
        assert call_log == ["topic A", "topic B"]
        assert "topic A" in result.output
        assert "topic B" in result.output

    def test_topic_and_topics_file_mutually_exclusive(self, runner, tmp_path):
        """N21: --topic and --topics-file together must return a UsageError."""
        topics_file = tmp_path / "topics.yaml"
        topics_file.write_text("topics:\n  - foo\n")
        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._make_state_db", return_value=MagicMock()),
            patch("scripts.cli._make_embeddings_db", return_value=MagicMock()),
            patch("scripts.cli._make_llm", return_value=MagicMock()),
        ):
            result = runner.invoke(
                main,
                ["obsidian", "synthesize", "--topic", "foo", "--topics-file", str(topics_file)],
            )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()

    def test_no_topic_no_file_exits_with_error(self, runner):
        """N21: calling synthesize with neither --topic nor --topics-file exits non-zero."""
        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._make_state_db", return_value=MagicMock()),
            patch("scripts.cli._make_embeddings_db", return_value=MagicMock()),
            patch("scripts.cli._make_llm", return_value=MagicMock()),
        ):
            result = runner.invoke(main, ["obsidian", "synthesize"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# N18c — rechunk-all command
# ---------------------------------------------------------------------------
class TestRechunkAll:
    """
    `_make_zotero_client` is patched with ``autospec=True`` in every test
    so the mock enforces the real signature ``(config, secrets)``.  If the
    CLI code drops the ``secrets`` argument again (Round-27 regression),
    these tests will hard-fail rather than silently passing.
    """

    def test_rechunk_single_doi(self, runner):
        """--doi rechunks exactly one paper and prints Done."""
        mock_state_db = MagicMock()
        mock_state_db.get_paper.return_value = {"doi": "10.1/x", "zotero_key": "ABC123"}
        mock_zotero = MagicMock()
        mock_zotero.get_markdown_attachment.return_value = "# Intro\n\nSome content."
        mock_embed_db = MagicMock()

        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._make_state_db", return_value=mock_state_db),
            patch("scripts.cli._make_embeddings_db", return_value=mock_embed_db),
            patch("scripts.cli._make_zotero_client", autospec=True) as mock_zfactory,
            patch("scripts.cli._load_secrets", return_value={}),
            patch("scripts.cli._maybe_set_ollama_key"),
            patch("scripts.core.markdown_processor.strip_end_matter", side_effect=lambda t: t),
        ):
            mock_zfactory.return_value = mock_zotero
            result = runner.invoke(main, ["obsidian", "rechunk-all", "--doi", "10.1/x"])

        assert result.exit_code == 0, result.output
        assert "Done" in result.output
        mock_embed_db.add_chunks.assert_called_once()
        # Autospec catches signature drift — confirm both args were passed.
        mock_zfactory.assert_called_once()
        args, _kwargs = mock_zfactory.call_args
        assert len(args) == 2, "rechunk-all must call _make_zotero_client(config, secrets)"

    def test_rechunk_all_walks_db(self, runner):
        """--all iterates over all paper + review records and rechunks each."""
        records = [
            {"doi": "10.1/a", "zotero_key": "ZK1"},
            {"doi": "10.1/b", "zotero_key": "ZK2"},
        ]
        mock_state_db = MagicMock()
        mock_state_db.get_all_by_source_type.side_effect = lambda st: (
            records if st == "paper" else []
        )
        mock_state_db.get_paper.side_effect = lambda d: next(
            (r for r in records if r["doi"] == d), None
        )
        mock_zotero = MagicMock()
        mock_zotero.get_markdown_attachment.return_value = "# Body\n\nContent."
        mock_embed_db = MagicMock()

        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._make_state_db", return_value=mock_state_db),
            patch("scripts.cli._make_embeddings_db", return_value=mock_embed_db),
            patch("scripts.cli._make_zotero_client", autospec=True) as mock_zfactory,
            patch("scripts.cli._load_secrets", return_value={}),
            patch("scripts.cli._maybe_set_ollama_key"),
            patch("scripts.core.markdown_processor.strip_end_matter", side_effect=lambda t: t),
        ):
            mock_zfactory.return_value = mock_zotero
            result = runner.invoke(main, ["obsidian", "rechunk-all", "--all"])

        assert result.exit_code == 0, result.output
        assert mock_embed_db.add_chunks.call_count == 2
        assert "2 indexed" in result.output

    def test_rechunk_no_flags_exits_nonzero(self, runner):
        """Neither --doi nor --all: must exit with code 1 and an error message."""
        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._make_state_db", return_value=MagicMock()),
            patch("scripts.cli._make_embeddings_db", return_value=MagicMock()),
            patch("scripts.cli._make_zotero_client", autospec=True),
            patch("scripts.cli._load_secrets", return_value={}),
            patch("scripts.cli._maybe_set_ollama_key"),
        ):
            result = runner.invoke(main, ["obsidian", "rechunk-all"])

        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Audit-29 — strict mode CLI integration tests
# ---------------------------------------------------------------------------

class TestStrictModeFlag:
    """Tests for the --strict / -S global flag wired into main()."""

    def test_strict_flag_sets_strict_mode(self, runner) -> None:
        """--strict must activate is_strict() during the subcommand run."""
        import scripts.core.strict_mode as _sm
        from scripts.setup.check_configured import CheckResult

        captured: list[bool] = []

        def _fake_check_configured():
            captured.append(_sm.is_strict())
            return {"key": CheckResult(True, "ok", "ok")}

        # check_ollama / check_zotero return plain 2-tuples (their real
        # contract); only check_configured returns CheckResult.
        ollama_ok = {"key": (True, "ok")}
        zotero_ok = {"key": (True, "ok")}
        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._load_secrets", return_value={}),
            patch("scripts.setup.check_configured.check_configured", side_effect=_fake_check_configured),
            patch("scripts.setup.check_ollama.check_ollama", return_value=ollama_ok),
            patch("scripts.setup.check_zotero.check_zotero", return_value=zotero_ok),
        ):
            result = runner.invoke(main, ["--strict", "check"])

        assert result.exit_code == 0, result.output
        # strict mode was active when the subcommand ran
        assert captured == [True]
        # reset so other tests are not polluted
        _sm.set_strict(False)

    def test_verbose_and_strict_combine(self, runner) -> None:
        """--verbose --strict together must not conflict."""
        from scripts.setup.check_configured import CheckResult
        cfg_ok = {"key": CheckResult(True, "all good", "ok")}
        svc_ok = {"key": (True, "all good")}
        with (
            patch("scripts.cli._make_config", return_value=MagicMock()),
            patch("scripts.cli._load_secrets", return_value={}),
            patch("scripts.setup.check_configured.check_configured", return_value=cfg_ok),
            patch("scripts.setup.check_ollama.check_ollama", return_value=svc_ok),
            patch("scripts.setup.check_zotero.check_zotero", return_value=svc_ok),
        ):
            result = runner.invoke(main, ["--strict", "--verbose", "check"])
        assert result.exit_code == 0, result.output

        import scripts.core.strict_mode as _sm
        _sm.set_strict(False)


class TestLoadSecretsStrictMode:
    """Tests that _load_secrets respects strict mode."""

    def test_load_secrets_raises_in_strict_mode(self, tmp_path) -> None:
        """With strict mode on, a malformed TOML file must raise.

        L3: strict_fallback now preserves the original exception *type* when
        possible. TOMLDecodeError is 1-arg-constructible (it subclasses
        ValueError) so the strict-mode re-raise will be a TOMLDecodeError,
        not RuntimeError. Both are acceptable to callers — we assert on the
        base Exception plus the cause-chain message.
        """
        import scripts.core.strict_mode as _sm
        from scripts.cli import _load_secrets

        bad_toml = tmp_path / "config.toml"
        bad_toml.write_text("this is not valid toml !!!\n")

        _sm.set_strict(True)
        try:
            with patch("scripts.cli._SECRETS_PATH", bad_toml):
                with pytest.raises((ValueError, RuntimeError), match="Could not parse secrets file"):
                    _load_secrets()
        finally:
            _sm.set_strict(False)

    def test_load_secrets_falls_back_in_default_mode(self, tmp_path, caplog) -> None:
        """With strict off, a malformed TOML file logs a WARNING and returns {}."""
        import logging

        import scripts.core.strict_mode as _sm
        from scripts.cli import _load_secrets

        bad_toml = tmp_path / "config.toml"
        bad_toml.write_text("this is not valid toml !!!\n")

        _sm.set_strict(False)
        with patch("scripts.cli._SECRETS_PATH", bad_toml):
            with caplog.at_level(logging.WARNING, logger="scripts.cli"):
                result = _load_secrets()

        assert result == {}
        assert "Could not parse secrets file" in caplog.text


class TestDiagnoseCommand:
    """Tests for the `lit-monitor diagnose` subcommand."""

    def test_diagnose_returns_nonzero_when_loader_fails(self, runner, tmp_path) -> None:
        """diagnose exits 1 if any config file fails to load."""
        # Write an invalid YAML file for paths.yaml to trigger a FAIL
        (tmp_path / "paths.yaml").write_text("this: is: invalid: yaml: :::\n")

        with patch("scripts.core.config._CONFIG_DIR", tmp_path):
            result = runner.invoke(main, ["diagnose", "--config-only"])

        # Should exit non-zero because paths.yaml is malformed
        assert result.exit_code == 1
        assert "FAIL" in result.output

    def test_diagnose_config_only_skips_service_checks(self, runner, tmp_path) -> None:
        """--config-only must not invoke Ollama or Zotero checks."""
        # Write minimal valid file
        (tmp_path / "paths.yaml").write_text("key: value\n")

        with (
            patch("scripts.core.config._CONFIG_DIR", tmp_path),
            patch("scripts.setup.check_ollama.check_ollama") as mock_ollama,
            patch("scripts.setup.check_zotero.check_zotero") as mock_zotero,
        ):
            runner.invoke(main, ["diagnose", "--config-only"])

        mock_ollama.assert_not_called()
        mock_zotero.assert_not_called()
