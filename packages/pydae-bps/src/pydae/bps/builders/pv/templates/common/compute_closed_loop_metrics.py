#!/usr/bin/env python3
"""Compute closed-loop step metrics (overshoot + 2% settling time) for every
protocol and write common/closed_loop_metrics.md.

Reads each ``<proto>/tests/closed_loop_step/csv/closed_loop_step.csv`` and
measures the pydae ground truth (sim_p_line_POI_GRID / sim_q_line_POI_GRID) at
the POI for both channels (P 2->3 MW, Q 0->1 Mvar).

    python3 common/compute_closed_loop_metrics.py
"""
from __future__ import annotations

import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROTOCOLS = ["local", "modbus", "iec104", "dnp3", "dds"]


def _metrics(rows, col, tgtcol, target, step_from):
    t  = [float(r["t_elapsed"]) for r in rows]
    y  = [float(r[col]) * 1e-6 for r in rows]
    tg = [float(r[tgtcol]) for r in rows]
    t0 = None
    for i in range(1, len(tg)):
        if tg[i - 1] <= step_from + 1e-6 and tg[i] >= target - 1e-6:
            t0 = t[i]; break
    if t0 is None:
        return None
    seg = [(tt, yy) for tt, yy in zip(t, y) if tt >= t0]
    step = target - step_from
    peak = max(yy for _, yy in seg) if step > 0 else min(yy for _, yy in seg)
    over = 100.0 * (peak - target) / abs(step) if step else 0.0
    band = 0.02 * abs(step)
    tset = None
    for k in range(len(seg)):
        if all(abs(yy - target) <= band for _, yy in seg[k:]):
            tset = seg[k][0] - t0; break
    return {"final": seg[-1][1], "over": over, "tset": tset}


def _plant_json(proto):
    """Discover the plant JSON in <ROOT>/<proto>/ (the one with a ppc block),
    name-agnostic so this works for any generated M×N case."""
    for p in sorted(glob.glob(os.path.join(ROOT, proto, "*.json"))):
        if p.endswith("xy_0.json"):
            continue
        try:
            if "ppc" in json.load(open(p)):
                return p
        except Exception:
            continue
    return None


def _ctrl_dt(proto):
    try:
        p = _plant_json(proto)
        return json.load(open(p))["ppc"].get("ctrl_dt_s") if p else None
    except Exception:
        return None


def main():
    out = [
        f"# Closed-loop step metrics — settling time & overshoot "
        f"({os.path.basename(ROOT)})",
        "",
        "Response metrics for the `closed_loop_step` test (P 2→3 MW, then "
        "Q 0→1 Mvar), measured on the pydae ground truth "
        "(`sim_p_line_POI_GRID` / `sim_q_line_POI_GRID`) at the POI.",
        "",
        "Controller condition: **feedforward ON**, with a feedforward-off "
        "pre-settle so the reactive integral starts from Q ≈ 0. Gains are "
        "identical across stacks; only `ctrl_dt_s` differs (comms speed).",
        "",
        "- **Overshoot** = `100 × (peak − target) / step` (step = 1.0 MW / 1.0 Mvar).",
        "- **Settling time** = time from step onset until the signal stays within "
        "**±2 %** of target.",
        "",
        "| Protocol | ctrl_dt_s | P overshoot | P settling (2 %) | Q overshoot | Q settling (2 %) |",
        "|----------|:---------:|:-----------:|:----------------:|:-----------:|:----------------:|",
    ]

    def fmt_t(v):
        return "—" if v is None else f"{v:.2f} s"

    def fmt_o(v):
        return "—" if v is None else f"{v:.1f} %"

    for proto in PROTOCOLS:
        path = os.path.join(ROOT, proto, "tests", "closed_loop_step",
                            "csv", "closed_loop_step.csv")
        dt = _ctrl_dt(proto)
        if not os.path.isfile(path):
            out.append(f"| {proto} | {dt} | — | — | — | — |  <!-- no CSV -->")
            continue
        rows = list(csv.DictReader(open(path)))
        P = _metrics(rows, "sim_p_line_POI_GRID", "target_mw", 3.0, 2.0) or {}
        Q = _metrics(rows, "sim_q_line_POI_GRID", "target_mvar", 1.0, 0.0) or {}
        out.append(
            f"| {proto} | {dt} | {fmt_o(P.get('over'))} | {fmt_t(P.get('tset'))} "
            f"| {fmt_o(Q.get('over'))} | {fmt_t(Q.get('tset'))} |"
        )
        print(f"{proto}: P over={P.get('over')} tset={P.get('tset')}  "
              f"Q over={Q.get('over')} tset={Q.get('tset')}")

    out.append("")
    md = os.path.join(HERE, "closed_loop_metrics.md")
    open(md, "w").write("\n".join(out))
    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
