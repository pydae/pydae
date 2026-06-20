"""Tiny PPC HTTP client — consolidates the urllib boilerplate that used to be
copy-pasted across every test script.

Endpoints (see ``colinker.modbus_ppc``):
  GET  /status      — telemetry snapshot
  POST /setpoints   — target_mw/mvar, K_pp/K_ip/K_pq/K_iq, ctrl_dt_s, feedforward
  GET/POST /sine    — sinusoidal modulation (p_on/p_amp/p_freq, q_*)
"""
from __future__ import annotations

import json
import urllib.request

from .common import PPC_URL


def _post(path: str, body: dict, base: str = PPC_URL, timeout: float = 2.0) -> dict:
    req = urllib.request.Request(
        f"{base.rstrip('/')}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(path: str, base: str = PPC_URL, timeout: float = 2.0) -> dict:
    with urllib.request.urlopen(f"{base.rstrip('/')}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def status(base: str = PPC_URL) -> dict:
    return _get("/status", base)


def set_setpoints(base: str = PPC_URL, **kw) -> dict:
    """POST any subset of target_mw, target_mvar, K_pp, K_ip, K_pq, K_iq,
    ctrl_dt_s, feedforward.  None values are dropped."""
    return _post("/setpoints", {k: v for k, v in kw.items() if v is not None}, base)


def set_sine(base: str = PPC_URL, **kw) -> dict:
    """POST any subset of p_on, p_amp, p_freq, q_on, q_amp, q_freq."""
    return _post("/sine", {k: v for k, v in kw.items() if v is not None}, base)


def get_sine(base: str = PPC_URL) -> dict:
    return _get("/sine", base)
