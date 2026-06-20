"""Shared constants and plant-config helpers for the modbus test suite.

Everything here is config-driven from the plant JSON so the same drivers work
for any M×N plant (the inverter list, POI keys and API port all come from the
``ppc`` block / device list, not hardcoded).
"""
from __future__ import annotations

import glob
import json
import os

# Env interpreter used across the benchmark (see AGENTS.md).
PYTHON = "/home/jmmmacubu/miniforge3/envs/pydae_dev/bin/python3"

# PPC management API, reachable via the socat relay (same as run_step.sh).
PPC_URL = os.environ.get("PPC_URL", "http://localhost:8090")

# pydae ground-truth columns the recorder merges into the CSV (Watts / vars).
SIM_P_COL = "sim_p_line_POI_GRID"
SIM_Q_COL = "sim_q_line_POI_GRID"

# Repo layout: <modbus>/tests/lib/common.py  ->  <modbus>
_LIB_DIR    = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR   = os.path.dirname(_LIB_DIR)
MODBUS_DIR  = os.path.dirname(TESTS_DIR)


def _discover_plant_json() -> str:
    """The plant JSON is the one ``*.json`` in the plant dir carrying a ``ppc``
    block (name varies per plant, e.g. ``copv_2_5.json`` / ``copv_5_10.json``);
    ``*_xy_0.json`` seeds are skipped.  Keeps the suite name-agnostic so it can
    be scaffolded verbatim by ``pv_designer.write_suite`` for any M×N plant."""
    cands = sorted(glob.glob(os.path.join(MODBUS_DIR, "*.json")))
    for p in cands:
        try:
            with open(p) as fh:
                d = json.load(fh)
            if isinstance(d, dict) and "ppc" in d:
                return p
        except (json.JSONDecodeError, OSError):
            continue
    return cands[0] if cands else os.path.join(MODBUS_DIR, "plant.json")


PLANT_JSON = _discover_plant_json()


def load_ppc_cfg(path: str = PLANT_JSON) -> dict:
    """Return the ``ppc`` block of the plant JSON (gains, ids, limits)."""
    with open(path) as fh:
        return json.load(fh).get("ppc", {})


def inverter_ids(path: str = PLANT_JSON) -> list[str]:
    """Per-inverter emec ids, e.g. ['LV0101', ... 'LV0205']."""
    return list(load_ppc_cfg(path).get("pv_ids", []))


def baseline_gains(path: str = PLANT_JSON) -> dict:
    """The configured closed-loop gains, for restoring after a test."""
    c = load_ppc_cfg(path)
    return {
        "K_pp": float(c.get("K_pp", 0.1)), "K_ip": float(c.get("K_ip", 0.35)),
        "K_pq": float(c.get("K_pq", 0.1)), "K_iq": float(c.get("K_iq", 0.35)),
        "ctrl_dt_s": float(c.get("ctrl_dt_s", 0.1)),
    }
