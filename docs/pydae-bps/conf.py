"""
Sphinx configuration for pydae-bps documentation.
"""

project = "pydae-bps"
copyright = "2026, Juan Manuel Mauricio"
author = "Juan Manuel Mauricio"
release = "0.10.0"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

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

nb_execution_mode = "off"
nb_execution_timeout = 120

# With myst-nb loaded, .md and .ipynb are both handled by the "myst-nb" parser.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# Sibling-package intersphinx commented out until those RTD sites exist.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "sympy": ("https://docs.sympy.org/latest", None),
    # "pydae-core": ("https://pydae-core.readthedocs.io/en/latest", None),
    # "pydae-uds": ("https://pydae-uds.readthedocs.io/en/latest", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autosummary_generate = False
autodoc_mock_imports = [
    "numpy",
    "scipy",
    "sympy",
    "matplotlib",
    "cffi",
    "numba",
    "networkx",
    "hjson",
    "pydae",
]

html_theme = "furo"
html_title = "pydae-bps"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/pydae/pydae",
    "source_branch": "main",
    "source_directory": "docs/pydae-bps/",
}
