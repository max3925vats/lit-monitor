"""Example configs must ship as package data, reachable via importlib.resources."""
from importlib.resources import files


def test_packaged_example_configs_exist():
    root = files("lit_monitor._data.config_examples")
    assert (root / "paths.example.yaml").is_file()
    assert (root / "examples" / "ml-research" / "topics.yaml").is_file()
