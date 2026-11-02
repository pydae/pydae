# -*- coding: utf-8 -*-
"""Dispatch every ``machines`` item to its model by ``type``."""

from pydae.emac.machines.pmsm_dq import pmsm_dq
from pydae.emac.machines.pmsm_lut import pmsm_lut

MACHINES = {"pmsm_dq": pmsm_dq, "pmsm_lut": pmsm_lut}


def add_machines(grid):
    for k, item in enumerate(grid.data.get("machines", [])):
        name = str(item.get("name", k + 1))
        if name in grid.machines:
            raise ValueError(f"two machines are named {name!r}")
        item["name"] = name
        kind = item.get("type", "pmsm_dq")
        if kind not in MACHINES:
            raise ValueError(f"machine {name}: unknown type {kind!r}; one of {sorted(MACHINES)}")
        MACHINES[kind](grid, name, item)
