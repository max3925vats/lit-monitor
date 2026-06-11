import os
import logging
from typing import Optional
from lit_monitor._vendor.findpapers.tools.bibtex_generator_tool import generate_bibtex
from lit_monitor._vendor.findpapers.tools.search_runner_tool import search
from lit_monitor._vendor.findpapers.tools.refiner_tool import refine
from lit_monitor._vendor.findpapers.tools.downloader_tool import download

# Vendored: there is no installed distribution metadata for
# `lit_monitor._vendor.findpapers`, so the upstream
# `importlib.metadata.version(__name__)` lookup would raise PackageNotFoundError.
# Hard-code the vendored upstream version instead.
__version__ = "0.6.7"
