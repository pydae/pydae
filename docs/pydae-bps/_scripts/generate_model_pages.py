"""
Generate per-model documentation pages for pydae-bps.

Walks packages/pydae-bps/src/pydae/bps/<family>/*.py, extracts:
    - the module docstring (theory, equations)
    - the `descriptions()` function (if defined)
and emits one markdown page per model under docs/pydae-bps/models/<family>/,
plus one index page per family and one top-level models/index.md.

Run from the repository root:

    python docs/pydae-bps/_scripts/generate_model_pages.py

Re-run any time a model file is added or its docstring/descriptions() changes.
"""

from __future__ import annotations

import ast
import re
import sys
import types
from pathlib import Path
from textwrap import dedent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
BPS_ROOT = REPO_ROOT / "packages" / "pydae-bps" / "src" / "pydae" / "bps"
DOCS_ROOT = REPO_ROOT / "docs" / "pydae-bps" / "models"

# family -> (title, short description for index pages)
FAMILIES = {
    "syns":        ("Synchronous machines",
                    "Salient- and round-pole synchronous machine models of varying order."),
    "avrs":        ("Automatic voltage regulators",
                    "Excitation-system models that regulate the field voltage of synchronous machines."),
    "govs":        ("Turbine governors",
                    "Primary-frequency-control models for hydro, steam, and gas turbines."),
    "psss":        ("Power system stabilizers",
                    "Supplementary excitation controls that damp electromechanical oscillations."),
    "vsgs":        ("Virtual synchronous generators",
                    "Grid-forming converter controls that emulate synchronous-machine dynamics."),
    "vscs":        ("Voltage-source converters",
                    "Physical VSC models (grid-forming, grid-following, back-to-back, BESS)."),
    "vsc_ctrls":   ("VSC controls",
                    "Outer-loop control laws that pair with the physical VSC models."),
    "vsc_models":  ("VSC inner models",
                    "Low-level / internal VSC building blocks."),
    "wecs":        ("Wind energy conversion systems",
                    "Pitch, mechanical, and electrical-machine models for wind turbines."),
    "pvs":         ("PV plants",
                    "Photovoltaic plant models (dq current, string, steady-state, VRT)."),
    "loads":       ("Loads",
                    "Static and voltage-dependent load models (ZIP)."),
    "lines":       ("Lines",
                    "Transmission-line models, including dynamic thermal rating variants."),
    "pods":        ("Power oscillation dampers",
                    "Wide-area damping controls."),
    "sources":     ("Sources",
                    "Ideal sources and GENAPE models."),
    "pssdesigner": ("PSS designer",
                    "Design helpers for power-system stabilizers."),
}

# Always skip these files inside a family directory
SKIP_FILES = {"__init__.py"}
SKIP_PREFIXES = ("temp", "test_")
SKIP_CONTAINS = ("_back",)
SKIP_SUFFIXES = ("_dev.py", "_test.py")


def is_family_dispatcher(family: str, filename: str) -> bool:
    """The <family>.py file inside each family dir is the dispatcher; skip it."""
    return filename == f"{family}.py"


def should_skip(family: str, filename: str) -> bool:
    if filename in SKIP_FILES:
        return True
    if any(filename.startswith(p) for p in SKIP_PREFIXES):
        return True
    if any(s in filename for s in SKIP_CONTAINS):
        return True
    if any(filename.endswith(s) for s in SKIP_SUFFIXES):
        return True
    if is_family_dispatcher(family, filename):
        return True
    return False


# ---------------------------------------------------------------------------
# Docstring extraction (raw, to preserve LaTeX backslashes)
# ---------------------------------------------------------------------------

_DOCSTRING_RE = re.compile(
    r'\A(?:\s|#[^\n]*\n)*(?:r?"""(?P<d>.*?)"""|r?\'\'\'(?P<s>.*?)\'\'\')',
    re.DOTALL,
)


def extract_raw_module_docstring(source: str) -> str:
    """Return the raw module docstring exactly as it appears in source.

    Using the AST would evaluate string-literal escape sequences, mangling
    `\\frac`, `\\theta`, `\\right` etc. We preserve the raw text so MathJax
    renders it correctly.
    """
    m = _DOCSTRING_RE.match(source)
    if not m:
        return ""
    return (m.group("d") or m.group("s") or "").strip("\n")


# ---------------------------------------------------------------------------
# descriptions() extraction
# ---------------------------------------------------------------------------

def load_descriptions(source: str, tree: ast.Module):
    """Execute just the `descriptions` function from the file, if present.

    Returns the list it produces, or None.
    """
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "descriptions":
            lines = source.splitlines()
            fn_src = dedent("\n".join(lines[node.lineno - 1 : node.end_lineno]))
            namespace: dict = {}
            try:
                exec(fn_src, namespace)
                return namespace["descriptions"]()
            except Exception as exc:
                print(f"    ! descriptions() failed: {exc}", file=sys.stderr)
                return None
    return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

TYPE_ORDER = ["Parameter", "Input", "Dynamic State", "Algebraic State", "Output"]


def _md_escape(text: str) -> str:
    if text is None:
        return ""
    return str(text).replace("|", "\\|")


def render_descriptions_tables(items: list) -> str:
    """Group descriptions by type and render one Markdown table per type."""
    if not items:
        return ""

    by_type: dict[str, list[dict]] = {}
    for item in items:
        by_type.setdefault(item.get("type", "Other"), []).append(item)

    ordered_types = [t for t in TYPE_ORDER if t in by_type]
    ordered_types += [t for t in by_type if t not in TYPE_ORDER]

    out: list[str] = []
    for t in ordered_types:
        out.append(f"### {t}s\n")
        out.append("| Symbol | Variable | Default | Units | Description |")
        out.append("|---|---|---|---|---|")
        for item in by_type[t]:
            tex = item.get("tex", "")
            name = item.get("model") or item.get("data", "")
            default = item.get("default", "")
            units = item.get("units", "")
            desc = item.get("description", "")
            symbol = f"${tex}$" if tex else ""
            out.append(
                f"| {_md_escape(symbol)} | `{_md_escape(name)}` "
                f"| {_md_escape(default)} | {_md_escape(units)} "
                f"| {_md_escape(desc)} |"
            )
        out.append("")
    return "\n".join(out)


PAGE_TEMPLATE = """# {model}

*{family_title} — pydae-bps model.*

{docstring_block}

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `{model}` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

{tables_block}

## Source

- Module: `pydae.bps.{family}.{model}`
- File: [`packages/pydae-bps/src/pydae/bps/{family}/{model}.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/{family}/{model}.py)
"""


STUB_TABLES_TEMPLATE = """## Parameters, inputs, states, outputs

```{note}
This model does not yet define a `descriptions()` function. Add one to
`__REL_SRC__` and re-run `docs/pydae-bps/_scripts/generate_model_pages.py`
to populate this section automatically. See
`packages/pydae-bps/src/pydae/bps/syns/milano2ord.py` for a reference
implementation.
```
"""


def render_model_page(family: str, family_title: str, model: str,
                      docstring: str, items) -> str:
    if docstring:
        docstring_block = f"## Model description\n\n{docstring}"
    else:
        docstring_block = ("## Model description\n\n"
                           "*(No module docstring provided — add one to the "
                           "source file and re-run the generator.)*")

    if items:
        tables_block = ("## Parameters, inputs, states, outputs\n\n"
                        + render_descriptions_tables(items))
    else:
        rel_src = f"packages/pydae-bps/src/pydae/bps/{family}/{model}.py"
        tables_block = STUB_TABLES_TEMPLATE.replace("__REL_SRC__", rel_src)

    return PAGE_TEMPLATE.format(
        model=model,
        family=family,
        family_title=family_title,
        docstring_block=docstring_block,
        tables_block=tables_block,
    )


FAMILY_INDEX_TEMPLATE = """# {family_title}

{family_description}

```{{toctree}}
:maxdepth: 1

{entries}
```
"""

MODELS_INDEX_TEMPLATE = """# Models

The `pydae-bps` library ships a library of power-system component models.
Pages are auto-generated from each model file's module docstring and its
`descriptions()` function.

```{{toctree}}
:maxdepth: 2

{entries}
```

To regenerate these pages after adding a new model or updating an existing
one, run:

```bash
python docs/pydae-bps/_scripts/generate_model_pages.py
```
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not BPS_ROOT.exists():
        print(f"ERROR: {BPS_ROOT} does not exist", file=sys.stderr)
        return 1

    DOCS_ROOT.mkdir(parents=True, exist_ok=True)

    family_index_entries: list[str] = []

    for family, (family_title, family_description) in FAMILIES.items():
        family_dir = BPS_ROOT / family
        if not family_dir.is_dir():
            continue

        out_dir = DOCS_ROOT / family
        out_dir.mkdir(parents=True, exist_ok=True)

        entries: list[str] = []
        print(f"[{family}] {family_title}")

        for py in sorted(family_dir.glob("*.py")):
            if should_skip(family, py.name):
                continue
            model = py.stem
            src = py.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(src)
            except SyntaxError as exc:
                print(f"    ! skip {py.name}: syntax error: {exc}",
                      file=sys.stderr)
                continue

            docstring = extract_raw_module_docstring(src)
            items = load_descriptions(src, tree)

            page = render_model_page(family, family_title, model,
                                     docstring, items)
            (out_dir / f"{model}.md").write_text(page, encoding="utf-8")
            entries.append(model)
            flag = "T" if items else "-"   # T = has descriptions() tables
            print(f"    [{flag}] {model}")

        if entries:
            (out_dir / "index.md").write_text(
                FAMILY_INDEX_TEMPLATE.format(
                    family_title=family_title,
                    family_description=family_description,
                    entries="\n".join(entries),
                ),
                encoding="utf-8",
            )
            family_index_entries.append(f"{family}/index")

    # Top-level models/index.md
    (DOCS_ROOT / "index.md").write_text(
        MODELS_INDEX_TEMPLATE.format(
            entries="\n".join(family_index_entries),
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {DOCS_ROOT}/index.md with {len(family_index_entries)} families.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
