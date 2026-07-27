#!/usr/bin/env python3
"""Open-loop P/Q step — plant + communications time response.

Reference feedforward is incremental (velocity form), so with the PI gains at
zero the *absolute* setpoint depends on the initial condition.  To get a clean
measurement we first settle **closed-loop** at the pre-step operating point
(P = 2 MW, Q = 0), then zero all PI gains (feedforward stays on) and apply the
step.  Each inverter setpoint is then driven open-loop by ΔT/N with no feedback
correction, so the POI trace is the raw plant + transport response and the
step amplitude reveals the channel gain (the reactive loop shows the network's
non-unity Q gain).

Steps P 2→3 MW then Q 0→1 Mvar, records PPC readback + pydae ground truth,
writes CSV + SVG + interactive HTML.  Restores the configured gains afterward.

Prereq: cosim running, PPC API at http://localhost:8090 (same as run_step.sh).
For wire-level arrival overlays, launch the cosim with
``--arrivals-dir tests/open_loop_step/arrivals``.

    python run.py
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # tests/  -> import lib
from lib import ppc_api, capture, plots, common      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="open_loop_step", help="output basename")
    args = ap.parse_args()

    csv_path = os.path.join(_HERE, "csv", f"{args.name}.csv")
    plot_base = os.path.join(_HERE, "plots", args.name)

    import time
    g = common.baseline_gains()
    try:
        # 1) Settle closed-loop at the pre-step IC (P=2, Q=0) so the
        #    incremental feedforward starts from the correct absolute setpoint.
        #    Settle with feedforward OFF: with ff on, the shared control law
        #    freezes the integral while |err| > ff_i_band (0.15 Mvar), so the
        #    reactive channel can't reach target=0 and parks at an arbitrary
        #    residual.  Pure PI (ff off) is never frozen and nulls Q to ~0.
        print(f"[open-loop] closed-loop pre-settle at P=2, Q=0 (gains {g}, ff OFF)…")
        ppc_api.set_setpoints(feedforward=False, **g)
        capture.pre_settle(2.0, 0.0)
        # pre_settle only blocks on the active channel; wait for Q to reach
        # target too before freezing the gains.
        deadline = time.monotonic() + 60.0
        q = float("nan")
        while time.monotonic() < deadline:
            q = ppc_api.status().get("poi_mvar", float("nan"))
            if q is not None and abs(q) <= 0.05:
                break
            time.sleep(0.5)
        print(f"[open-loop] reactive channel settled at poi_mvar={q:.3f}")

        # 2) Drop to open loop: zero PI gains, feedforward on (drives the step
        #    via ΔT/N with no feedback correction).
        print("[open-loop] gains -> 0 (feedforward only); recording PQ step …")
        ppc_api.set_setpoints(K_pp=0.0, K_ip=0.0, K_pq=0.0, K_iq=0.0,
                              feedforward=True)
        with capture.Recorder(csv_path, pre=3, post=10, trigger_mw=0.1):
            capture.drive_pq_step(p_initial=2.0, p_step=3.0, q_step=1.0, t_end=15)
    finally:
        print(f"[open-loop] restoring gains {g}")
        ppc_api.set_setpoints(**g)

    plots.step_figure(csv_path, plot_base,
                      "Open-loop step (gains=0, feedforward only) — plant response")
    print(f"[open-loop] wrote {csv_path}")
    print(f"[open-loop] wrote {plot_base}.svg / .html")


if __name__ == "__main__":
    main()
