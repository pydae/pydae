"""
Sphinx configuration for pydae-core documentation.
"""

import sys, pathlib
# Expose the src layout so autodoc can locate pydae.core and pydae.ssa
# even though pydae itself is a native namespace package (no __init__.py).
_src = pathlib.Path(__file__).parents[2] / "packages" / "pydae-core" / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# -- Project information -----------------------------------------------------
project = "pydae-core"
copyright = "2026, Juan Manuel Mauricio"
author = "Juan Manuel Mauricio"
release = "1.0.2"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_nb",                      # Markdown + executable notebooks
    "sphinx.ext.autodoc",           # API reference from docstrings
    "sphinx.ext.autosummary",       # Summary tables
    "sphinx.ext.napoleon",          # NumPy/Google docstring style
    "sphinx.ext.viewcode",          # "[source]" links
    "sphinx.ext.intersphinx",       # Cross-reference other docs
    "sphinx.ext.mathjax",           # LaTeX math
    "sphinx_copybutton",            # Copy button on code blocks
    "sphinx_design",                # Grids, cards, tabs
    "sphinxcontrib.mermaid",        # {mermaid} diagrams
]

# MyST extensions
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "linkify",
    "substitution",
    "tasklist",
]

# myst-nb notebook execution
nb_execution_mode = "off"   # set to "auto" or "cache" once examples are stable
nb_execution_timeout = 120

# Source file types. With myst-nb loaded, .md and .ipynb are both handled by
# the "myst-nb" parser. (Do NOT use "markdown" as the parser name here — it is
# not a registered Sphinx parser.)
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# Cross-references to external docs. The two sibling packages are commented
# out until their ReadTheDocs sites exist; uncomment once published.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "sympy": ("https://docs.sympy.org/latest", None),
    # "pydae-bps": ("https://pydae-bps.readthedocs.io/en/latest", None),
    # "pydae-uds": ("https://pydae-uds.readthedocs.io/en/latest", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
suppress_warnings = ["autosummary"]

# -- Autodoc ----------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
# Set to False for a pure "scaffold" build that doesn't require the package
# to be importable. Flip to True (and install the package in the build env)
# once you want an auto-generated API reference.
autosummary_generate = True

# Mock heavy/optional deps so autodoc can still import when building on a
# minimal environment.
autodoc_mock_imports = [
    "numpy",
    "scipy",
    "sympy",
    "matplotlib",
    "pandas",
    "cffi",
    "numba",
    "networkx",
    "hjson",
]

# -- HTML output ------------------------------------------------------------
html_theme = "furo"
html_title = "pydae-core"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/pydae/pydae",
    "source_branch": "main",
    "source_directory": "docs/pydae-core/",
}
