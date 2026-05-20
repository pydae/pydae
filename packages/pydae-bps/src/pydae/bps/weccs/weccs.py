# -*- coding: utf-8 -*-
"""Dispatcher for the ``weccs`` (WECC converter) component family.

Analogous to ``syns/syns.py``:

* :func:`~pydae.bps.weccs.regc_a.regc_a` is always called first (the
  grid-interface converter — equivalent to a syn model).
* If the HJSON entry contains a ``reec`` sub-dict,
  :func:`~pydae.bps.weccs.reec_b.reec_b` is called next (local electrical
  controls — equivalent to an AVR/governor nested inside a syn).

Plant-level control (REPC_A and future controllers) lives in ``bps/ppcs/``
and is handled by the separate ``add_ppcs`` dispatcher, called after
``add_weccs`` in BpsBuilder.
"""

from pydae.bps.weccs.regc_a   import regc_a
from pydae.bps.weccs.regc_b   import regc_b
from pydae.bps.weccs.regfm_a1 import regfm_a1
from pydae.bps.weccs.regfm_b1 import regfm_b1
from pydae.bps.weccs.reec_b   import reec_b
from pydae.bps.weccs.reec_e   import reec_e


def add_weccs(grid):
    """Instantiate all ``weccs`` entries declared in the HJSON data dict."""
    buses      = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]
    S_base     = grid.backend.symbols('S_base')

    for item in grid.data['weccs']:
        data_dict = item
        bus_name  = item['bus']
        name      = item.get('name', bus_name)

        for gen_id in range(100):
            if name not in grid.generators_id_list:
                grid.generators_id_list.append(name)
                break
            name = f"{name}_{gen_id}"
        item['name'] = name

        # ── base converter (REGC_A) — always required ────────────────────
        wecc_type = item.get('type', 'regc_a')
        if wecc_type == 'regc_a':
            p_W, q_var = regc_a(grid, name, bus_name, data_dict)
        elif wecc_type == 'regc_b':
            p_W, q_var = regc_b(grid, name, bus_name, data_dict)
        elif wecc_type == 'regfm_a1':
            p_W, q_var = regfm_a1(grid, name, bus_name, data_dict)
        elif wecc_type == 'regfm_b1':
            p_W, q_var = regfm_b1(grid, name, bus_name, data_dict)
        else:
            raise ValueError(f"Unknown weccs type: {wecc_type!r}")

        # ── bus power injection ──────────────────────────────────────────
        idx_bus = buses_list.index(bus_name)
        if 'idx_powers' not in buses[idx_bus]:
            buses[idx_bus]['idx_powers'] = 0
        buses[idx_bus]['idx_powers'] += 1

        grid.dae['g'][idx_bus * 2]     += -p_W   / S_base
        grid.dae['g'][idx_bus * 2 + 1] += -q_var / S_base

        # ── local electrical controls ────────────────────────────────────
        if 'reec' in item:
            reec_type = item['reec'].get('type', 'reec_b')
            if reec_type == 'reec_e':
                reec_e(grid.dae, item, name, bus_name, grid.backend)
            else:
                reec_b(grid.dae, item, name, bus_name, grid.backend)
