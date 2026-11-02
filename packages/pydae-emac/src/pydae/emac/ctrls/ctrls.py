# -*- coding: utf-8 -*-
"""Dispatch every ``ctrls`` item to its controller by ``type``.  Controllers are
added after the machines, so they can drive the machines' inputs."""

from pydae.emac.ctrls.foc_pi import foc_pi

CTRLS = {"foc_pi": foc_pi}


def add_ctrls(grid):
    for k, item in enumerate(grid.data.get("ctrls", [])):
        name = str(item.get("name", item.get("machine", k + 1)))
        if name in grid.ctrls:
            raise ValueError(f"two controllers are named {name!r}")
        item["name"] = name
        kind = item.get("type", "foc_pi")
        if kind not in CTRLS:
            raise ValueError(f"controller {name}: unknown type {kind!r}; one of {sorted(CTRLS)}")
        CTRLS[kind](grid, name, item)
