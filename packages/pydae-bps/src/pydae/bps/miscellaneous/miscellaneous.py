# -*- coding: utf-8 -*-
"""
Top-level miscellaneous-component dispatch.

Dispatches per-entry blocks under the hjson keys ``faults`` and ``plls``
to the appropriate per-module builder using the canonical sexs-style
signature ``(dae, data, name, bus_name, backend)``.
"""

from pydae.bps.miscellaneous.fault import add_fault
from pydae.bps.miscellaneous.pll import pll
from pydae.bps.miscellaneous.sogi import sogi
from pydae.bps.miscellaneous.sogi_fll import sogi_fll
from pydae.bps.miscellaneous.sogi_pll import sogi_pll


def add_miscellaneous(grid):

    if 'faults' in grid.data:
        for item in grid.data['faults']:
            add_fault(grid, item)

    if 'plls' in grid.data:
        for item in grid.data['plls']:
            bus_name = item.get('bus', None)
            name = item.get('name', bus_name)
            t = item.get('type', 'pll')

            if t == 'sogi':
                sogi(grid.dae, item, name, bus_name, grid.backend)
            elif t == 'sogi_fll':
                sogi_fll(grid.dae, item, name, bus_name, grid.backend)
            elif t == 'sogi_pll':
                sogi_pll(grid.dae, item, name, bus_name, grid.backend)
            else:
                pll(grid.dae, item, name, bus_name, grid.backend)
