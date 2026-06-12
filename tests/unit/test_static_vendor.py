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


THEME = Path("lit_monitor/server/static/shoelace-theme.css")


def test_shoelace_theme_maps_brand_tokens():
    css = THEME.read_text()
    assert "--sl-color-primary-600" in css         # purple primary mapped
    assert "--sl-font-sans" in css                 # IBM Plex mapped
    assert ":not(:defined)" in css                 # FOUC hide present
