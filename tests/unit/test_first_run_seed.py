"""first-run seeds the user config dir from packaged example configs."""
from lit_monitor.core.seed import seed_user_config


def test_seed_writes_example_yamls_into_target(tmp_path):
    target = tmp_path / "cfg"
    written = seed_user_config(target)
    assert (target / "paths.yaml").exists() and (target / "extraction.yaml").exists()
    assert any("paths.yaml" in str(p) for p in written)


def test_seed_does_not_clobber_existing(tmp_path):
    target = tmp_path / "cfg"
    target.mkdir()
    (target / "paths.yaml").write_text("KEEP ME")
    seed_user_config(target)
    assert (target / "paths.yaml").read_text() == "KEEP ME"
