# -*- coding: utf-8 -*-
"""Dispatcher for the ``ppcs`` (Plant Power Controllers) component family.

Plant-level controllers are declared as a separate top-level section in the
HJSON (``ppcs``), distinct from the ``weccs`` converter entries they command.
Each entry specifies which converter(s) it controls via the ``weccs`` key
(list of converter names / bus names).

Analogous to ``syns/syns.py`` for generators, but operating at plant level.
"""

from pydae.bps.ppcs.repc_a import repc_a as _repc_a
from pydae.bps.ppcs.repc_d import repc_d as _repc_d


def add_ppcs(grid):
    """Instantiate all ``ppcs`` entries declared in the HJSON data dict."""
    for idx, item in enumerate(grid.data['ppcs']):
        ppc_type = item.get('type', 'repc_a')
        name     = item.get('name', f"{ppc_type}{idx + 1}")

        if ppc_type == 'repc_a':
            _repc_a(grid.dae, item, name, grid.backend)
        elif ppc_type == 'repc_d':
            _repc_d(grid.dae, item, name, grid.backend)
        else:
            raise ValueError(f"Unknown ppcs type: {ppc_type!r}")
