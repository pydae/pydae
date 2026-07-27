#!/usr/bin/env python3
"""Live POI dashboard — read-only terminal monitor for a running cosim.

Polls the PPC management API (:8090/status) and the pydae ground truth
(:8080/cosim/measurements) a few times a second and shows, with rolling
sparklines:

  * P at the POI  — target vs PPC readback vs plant ground truth (+ the gap)
  * Q at the POI  — target vs PPC readback vs plant ground truth (+ the gap)
  * V_POI, feedforward, ctrl_dt, and the latest run_all_tests progress.

GET-only, so it's safe to run alongside run_all_tests.py (it never writes
setpoints). Handles the cosim being down between stacks (teardown/relaunch).

    python3 common/monitor.py            # ~2 Hz
    python3 common/monitor.py --hz 5
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PPC = "http://localhost:8090"
SIM = "http://localhost:8080"
SUMMARY = os.path.join(HERE, "run_logs", "summary.txt")
BARS = "▁▂▃▄▅▆▇█"


def _get(url: str, timeout: float = 1.5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def spark(vals, width=48):
    vals = [v for v in vals if v is not None][-width:]
    if not vals:
        return " " * width
    lo, hi = min(vals), max(vals)
    rng = hi - lo or 1.0
    out = "".join(BARS[min(len(BARS) - 1, int((v - lo) / rng * (len(BARS) - 1)))]
                  for v in vals)
    return out.rjust(width)


def summary_tail(n=6):
    try:
        lines = open(SUMMARY).read().splitlines()
        return lines[-n:]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hz", type=float, default=2.0, help="refresh rate (default 2)")
    ap.add_argument("--width", type=int, default=48, help="sparkline width")
    args = ap.parse_args()
    dt = 1.0 / args.hz

    hist = {"pt": [], "pr": [], "ps": [], "qt": [], "qr": [], "qs": []}
    maxlen = args.width * 2

    def push(k, v):
        hist[k].append(v)
        if len(hist[k]) > maxlen:
            hist[k].pop(0)

    print("\033[2J", end="")  # clear once
    try:
        while True:
            st = sim = None
            try:
                st = _get(f"{PPC}/status")
            except Exception:
                pass
            try:
                sim = _get(f"{SIM}/cosim/measurements")
            except Exception:
                pass

            tgt_p = st.get("target_mw") if st else None
            tgt_q = st.get("target_mvar") if st else None
            poi_p = st.get("poi_mw") if st else None
            poi_q = st.get("poi_mvar") if st else None
            ff = st.get("feedforward") if st else None
            cdt = st.get("ctrl_dt_s") if st else None
            sim_p = sim_q = v_poi = None
            if sim and "data" in sim:
                d = sim["data"]
                sim_p = (d.get("p_line_POI_GRID") or 0) * 1e-6
                sim_q = (d.get("q_line_POI_GRID") or 0) * 1e-6
                v_poi = d.get("V_POI")

            for k, v in (("pt", tgt_p), ("pr", poi_p), ("ps", sim_p),
                         ("qt", tgt_q), ("qr", poi_q), ("qs", sim_q)):
                push(k, v)

            def f(v, u=""):
                return f"{v:8.3f}{u}" if isinstance(v, (int, float)) else f"{'--':>8}{u}"

            gap_p = (poi_p - sim_p) if (poi_p is not None and sim_p is not None) else None
            gap_q = (poi_q - sim_q) if (poi_q is not None and sim_q is not None) else None

            up = st is not None and sim is not None
            status = "\033[32mLIVE\033[0m" if up else "\033[33mcosim down / transitioning\033[0m"

            lines = []
            lines.append(f"  POI live monitor   [{status}]   "
                         f"ff={ff}  ctrl_dt={cdt}  V_POI={f(v_poi)}        ")
            lines.append("")
            lines.append(f"  P (MW)   target {f(tgt_p)}   readback {f(poi_p)}   "
                         f"plant {f(sim_p)}   gap {f(gap_p)}   ")
            lines.append(f"    plant  |{spark(hist['ps'], args.width)}|   ")
            lines.append(f"    ref    |{spark(hist['pt'], args.width)}|   ")
            lines.append("")
            lines.append(f"  Q (Mvar) target {f(tgt_q)}   readback {f(poi_q)}   "
                         f"plant {f(sim_q)}   gap {f(gap_q)}   ")
            lines.append(f"    plant  |{spark(hist['qs'], args.width)}|   ")
            lines.append(f"    ref    |{spark(hist['qt'], args.width)}|   ")
            lines.append("")
            lines.append("  run_all_tests progress:                          ")
            for s in summary_tail():
                lines.append(f"    {s}      ")
            lines.append("")
            lines.append("  (read-only; Ctrl-C to quit)                      ")

            print("\033[H" + "\n".join(lines) + "\033[J", end="", flush=True)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[monitor] bye")


if __name__ == "__main__":
    main()
