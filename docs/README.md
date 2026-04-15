# pydae documentation

This folder hosts the documentation for the three packages in the pydae
monorepo. Each package has its own self-contained docs tree so it can be
published as an independent ReadTheDocs project.

```
docs/
├── pydae-core/        ← https://pydae-core.readthedocs.io  (proposed)
├── pydae-bps/         ← https://pydae-bps.readthedocs.io   (proposed)
└── pydae-uds/         ← https://pydae-uds.readthedocs.io   (proposed)
```

## Why MyST + Sphinx (and not classic Jupyter Book)

[Jupyter Book v1 is being superseded by MyST-MD](https://mystmd.org/guide) and
its maintainers now recommend using Sphinx directly with the `myst-parser`
and `myst-nb` extensions for new projects. You still get:

- Markdown (`.md`) source files with the MyST superset (admonitions, math,
  grids, tabs, cross-references).
- Executable Jupyter notebooks (`.ipynb`) rendered inline.
- The full Sphinx ecosystem: `autodoc` / `autosummary`, `intersphinx`
  cross-references (so pydae-bps docs can link into pydae-core), PDF /
  HTMLZip output on ReadTheDocs, full-text search, versioned builds.

Switching to plain Sphinx+MyST is also a no-op upgrade path if you later want
to migrate to MyST-MD (the native MyST CLI) — the content files don't change.

## Build locally

From the repository root:

```bash
# one-off: install the doc toolchain into the active env
pip install -r docs/pydae-core/requirements.txt

# build each package
sphinx-build -b html docs/pydae-core docs/pydae-core/_build/html
sphinx-build -b html docs/pydae-bps  docs/pydae-bps/_build/html
sphinx-build -b html docs/pydae-uds  docs/pydae-uds/_build/html
```

Open `docs/<pkg>/_build/html/index.html` in a browser.

Live-reloading during editing:

```bash
pip install sphinx-autobuild
sphinx-autobuild docs/pydae-core docs/pydae-core/_build/html
```

## Publish on ReadTheDocs

Each package is a **separate ReadTheDocs project**. Do this once per package:

1. Sign in to <https://readthedocs.org> and **Import a Project** pointing to
   the monorepo's GitHub URL.
2. Set the project **slug / name** (e.g. `pydae-core`, `pydae-bps`,
   `pydae-uds`). These determine the public URL
   `https://<slug>.readthedocs.io`.
3. Under **Admin → Advanced Settings → Configuration file**, set:

   | Project     | Configuration file path                  |
   |-------------|------------------------------------------|
   | pydae-core  | `docs/pydae-core/.readthedocs.yaml`      |
   | pydae-bps   | `docs/pydae-bps/.readthedocs.yaml`       |
   | pydae-uds   | `docs/pydae-uds/.readthedocs.yaml`       |

4. Trigger the first build from **Builds → Build version**.

Subsequent pushes to `main` will auto-build all three projects in parallel.
Each `.readthedocs.yaml` installs `pydae-core` first (because the other two
depend on it via the namespace package), then the package being built.

## Cross-references between packages

`intersphinx` is pre-configured so any of the three sites can link into any
other:

```markdown
See {external+pydae-core:doc}`overview` for the solver internals.
```

These links resolve automatically once all three projects are live on RTD.

## Auto-generated model pages (pydae-bps)

The `pydae-bps` docs include a **Models** section with one page per element
model, organised by family (synchronous machines, AVRs, governors, VSCs, …).
These pages are produced by a generator script:

```bash
python docs/pydae-bps/_scripts/generate_model_pages.py
```

For each `.py` file under `packages/pydae-bps/src/pydae/bps/<family>/`, the
script extracts:

- the module docstring (theory, equations; LaTeX is preserved verbatim), and
- the `descriptions()` function if defined — rendered as parameter / input /
  state / output tables.

Models without a `descriptions()` function still get a page with the module
docstring and a stub pointing to `milano2ord.py` as the reference
implementation. Re-run the script whenever you add, remove, or update a model.

## Adding content

Each docs tree currently contains:

- `index.md` — landing page.
- `overview.md` — conceptual explanation of the package.
- `getting_started.md` — hands-on tutorial.
- `api.md` — API-reference stub (flip `autosummary_generate = True` and add
  modules to the `.. autosummary::` block to auto-populate).
- `conf.py` — Sphinx configuration.
- `requirements.txt` — doc-tooling dependencies.
- `.readthedocs.yaml` — build configuration for ReadTheDocs.
- `_static/`, `_templates/` — for custom CSS/JS/HTML overrides.

Drop additional `.md` or `.ipynb` files anywhere in the tree and reference
them from a `{toctree}` block in `index.md`. Notebook execution is off by
default (`nb_execution_mode = "off"` in `conf.py`); flip it to `"cache"` once
you have curated, reproducible example notebooks you want executed at build
time.

## Alternative frameworks considered

If you ever want to revisit the choice:

- **Classic Jupyter Book (v1)** — still fine for mostly-notebook docs but the
  ecosystem is moving away from it. `_toc.yml` + `_config.yml` instead of
  `conf.py`.
- **MyST-MD (`mystmd` CLI)** — the actively developed successor to Jupyter
  Book. Fastest builds and the cleanest authoring story, but RTD support is
  still catching up and `intersphinx`-style cross-project linking is less
  mature. Reasonable choice for a single-site deployment.
- **MkDocs Material** — best-looking default theme, but weaker for
  auto-generated API reference from Python docstrings and no `intersphinx`
  support; better suited for user-guide-only projects.
