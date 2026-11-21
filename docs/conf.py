"""Sphinx configuration: ``python scripts/dev.py docs`` or ``sphinx-build docs docs/_build/html``."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lcms2  # noqa: E402

project = "python-lcms2"
author = "Josue N Rivera"
copyright = "2026, Josue N Rivera"
release = lcms2.__version__
version = ".".join(release.split(".")[:2])

DOCS_CHANNEL = os.environ.get("LCMS2_DOCS_CHANNEL", "dev")
DOCS_BASE_URL = "https://wzjoriv.github.io/python-lcms2"
REPO_URL = "https://github.com/wzjoriv/python-lcms2"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "signature"
autodoc_typehints_format = "short"
add_module_names = False
autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True

intersphinx_mapping = (
    {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable", None),
    }
    if os.environ.get("SPHINX_INTERSPHINX", "1") != "0"
    else {}
)

html_theme = "furo"
html_title = f"python-lcms2 {release}"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["custom.css"]
html_context = {
    "docs_channel": DOCS_CHANNEL,
    "docs_base_url": DOCS_BASE_URL,
    "docs_release": release,
}
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "versions.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
    ],
}
html_theme_options = {
    "navigation_with_keys": True,
    "source_repository": REPO_URL,
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#c2410c",
        "color-brand-content": "#9a3412",
    },
    "dark_css_variables": {
        "color-brand-primary": "#fb923c",
        "color-brand-content": "#fdba74",
    },
}
if DOCS_CHANNEL == "dev":
    html_theme_options["announcement"] = (
        f'Development docs for the next release. <a href="{DOCS_BASE_URL}/stable/">Stable docs</a>.'
    )

source_suffix = {".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "friendly"
pygments_dark_style = "monokai"
