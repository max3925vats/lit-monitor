from pathlib import Path

VENDOR = Path("lit_monitor/server/static/vendor/shoelace")


def test_shoelace_autoloader_vendored():
    assert (VENDOR / "shoelace-autoloader.js").is_file(), "autoloader not vendored"


def test_shoelace_themes_vendored():
    assert (VENDOR / "themes" / "light.css").is_file()
    assert (VENDOR / "themes" / "dark.css").is_file()


def test_shoelace_chunks_present():
    chunks = VENDOR / "chunks"
    assert chunks.is_dir() and any(chunks.glob("*.js")), "component chunks missing"
