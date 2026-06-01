"""
Generate per-model documentation pages for pydae-uds.

Walks packages/pydae-uds/src/pydae/uds/<family>/*.py, extracts:
    - the module docstring (theory, equations)
    - the `descriptions()` function (if defined)
and emits one markdown page per model under docs/pydae-uds/models/<family>/,
plus one index page per family and one top-level models/index.md.

Run from the repository root:

    python docs/pydae-uds/_scripts/generate_model_pages.py

Re-run any time a model file is added or its docstring/descriptions() changes.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from textwrap import dedent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
UDS_ROOT = REPO_ROOT / "packages" / "pydae-uds" / "src" / "pydae" / "uds"
DOCS_ROOT = REPO_ROOT / "docs" / "pydae-uds" / "models"

# family -> (title, short description for index pages)
FAMILIES = {
    "sources":         ("Sources",
                        "Ideal voltage sources representing slack/infinite buses."),
    "transformers":    ("Transformers",
                        "Three-phase transformer connection models (Dyn11, Dyg11, …)."),
    "lines":           ("Lines",
                        "Three-phase / four-wire distribution-line models."),
    "loads":           ("Loads",
                        "ZIP and constant-power loads (AC three-phase + neutral, DC)."),
    "shunts":          ("Shunts",
                        "Shunt impedances connected between a single bus node and ground."),
    "vscs":            ("Voltage-source converters",
                        "AC-DC and AC-only VSC models (P/Q, V_dc/Q, 4-wire AC)."),
    "vsgs":            ("Virtual synchronous generators",
                        "Grid-forming VSC controls with virtual inertia and droop."),
    "vsc_ctrls":       ("VSC outer-loop controls",
                        "Outer-loop control laws that drive the VSC reference signals."),
    "ess":             ("Energy storage systems",
                        "Battery and DC-DC ESS models."),
    "fcs":             ("Fuel cell systems",
                        "PEM / SOFC fuel-cell models."),
    "pvs":             ("PV plants",
                        "Photovoltaic plant models."),
    "genapes":         ("GENAPE sources",
                        "Generic equivalent sources for benchmarking."),
    "miscellaneous":   ("Miscellaneous",
                        "Breakers and other utility components."),
}

# Always skip these files inside a family directory
SKIP_FILES = {"__init__.py"}
SKIP_PREFIXES = ("temp", "test_")
SKIP_CONTAINS = ("_back",)
SKIP_SUFFIXES = ("_dev.py", "_test.py")


def is_family_dispatcher(family: str, filename: str) -> bool:
    """The <family>.py file inside each family dir is the dispatcher; skip it
    *only* when other model files exist (e.g. loads/ has loads.py dispatcher
    plus load_ac.py / load_dc.py). For single-file families like shunts/ and
    lines/, the family file IS the model and must NOT be skipped — that case
    is handled by the caller of should_skip()."""
    return filename == f"{family}.py"


def should_skip(family: str, filename: str, has_siblings: bool) -> bool:
    if filename in SKIP_FILES:
        return True
    if any(filename.startswith(p) for p in SKIP_PREFIXES):
        return True
    if any(s in filename for s in SKIP_CONTAINS):
        return True
    if any(filename.endswith(s) for s in SKIP_SUFFIXES):
        return True
    if has_siblings and is_family_dispatcher(family, filename):
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

*{family_title} — pydae-uds model.*

{docstring_block}

## Usage

```python
from pydae.uds import UdsBuilder

grid = UdsBuilder("my_network.hjson")
grid.construct("my_system")
```

The `{model}` model is instantiated by including an entry in the relevant
section of the network HJSON (see [Overview](../../overview.md)).

{tables_block}

## Source

- Module: `pydae.uds.{family}.{model}`
- File: [`packages/pydae-uds/src/pydae/uds/{family}/{model}.py`](https://github.com/pydae/pydae/tree/master/packages/pydae-uds/src/pydae/uds/{family}/{model}.py)
"""


STUB_TABLES_TEMPLATE = """## Parameters, inputs, states, outputs

```{note}
This model does not yet define a `descriptions()` function. Add one to
`__REL_SRC__` and re-run `docs/pydae-uds/_scripts/generate_model_pages.py`
to populate this section automatically. See
`packages/pydae-bps/src/pydae/bps/syns/milano4ord.py` for a reference
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
        rel_src = f"packages/pydae-uds/src/pydae/uds/{family}/{model}.py"
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

The `pydae-uds` library ships a library of unbalanced-distribution component
models. Pages are auto-generated from each model file's module docstring and
its `descriptions()` function.

```{{toctree}}
:maxdepth: 2

{entries}
```

To regenerate these pages after adding a new model or updating an existing
one, run:

```bash
python docs/pydae-uds/_scripts/generate_model_pages.py
```
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not UDS_ROOT.exists():
        print(f"ERROR: {UDS_ROOT} does not exist", file=sys.stderr)
        return 1

    DOCS_ROOT.mkdir(parents=True, exist_ok=True)

    family_index_entries: list[str] = []

    for family, (family_title, family_description) in FAMILIES.items():
        family_dir = UDS_ROOT / family
        if not family_dir.is_dir():
            continue

        out_dir = DOCS_ROOT / family
        out_dir.mkdir(parents=True, exist_ok=True)

        entries: list[str] = []
        print(f"[{family}] {family_title}")

        # If the family dir contains models alongside a same-named dispatcher,
        # skip the dispatcher. If it only contains the same-named file, that
        # file IS the model — keep it.
        py_files = [p for p in sorted(family_dir.glob("*.py"))
                    if p.name not in SKIP_FILES
                    and not any(p.name.startswith(pre) for pre in SKIP_PREFIXES)
                    and not any(s in p.name for s in SKIP_CONTAINS)
                    and not any(p.name.endswith(s) for s in SKIP_SUFFIXES)]
        non_dispatcher_count = sum(1 for p in py_files if not is_family_dispatcher(family, p.name))
        has_siblings = non_dispatcher_count > 0

        for py in sorted(family_dir.glob("*.py")):
            if should_skip(family, py.name, has_siblings):
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
