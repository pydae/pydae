# -*- coding: utf-8 -*-
"""Render a component's ``descriptions()`` into Markdown tables appended to its
module docstring, grouped by type -- the format the pydae docs generators read."""


def _table(rows):
    headers = []
    for row in rows:
        headers += [k for k in row if k not in headers]
    width = {h: max(len(h), *(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    lines = ["| " + " | ".join(h.ljust(width[h]) for h in headers) + " |",
             "| " + " | ".join("-" * width[h] for h in headers) + " |"]
    lines += ["| " + " | ".join(str(r.get(h, "")).ljust(width[h]) for h in headers) + " |"
              for r in rows]
    return "\n".join(lines)


def descriptions_tables(descriptions):
    """Markdown: one table per type (Parameter, Input, Dynamic State, ...)."""
    out = ""
    for cat in ("Parameter", "Input", "Dynamic State", "Algebraic State", "Output"):
        rows = [{k: v for k, v in d.items() if k != "type"} for d in descriptions if d["type"] == cat]
        if rows:
            out += f"\n### {cat}s\n\n" + _table(rows) + "\n"
    return out
