#!/usr/bin/env python3
"""Closed-loop P/Q step — PI gain-design workspace.

Restores the configured PI gains from ``copv_2_5.json`` (feedforward toggled
via ``--feedforward/--no-feedforward``), pre-settles at P = 2 MW, steps
P 2→3 MW then Q 0→1 Mvar, records PPC readback + pydae ground truth, and writes
CSV + SVG + interactive HTML.

Override gains on the command line to sweep tunings; the ``--name`` is used for
the output filenames so runs don't clobber each other.

    python run.py                              # configured gains, feedforward on
    python run.py --name kip05 --K_ip 0.5      # try a gain
    python run.py --no-feedforward --name noff

Prereq: cosim running, PPC API at http://localhost:8090.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from lib import ppc_api, capture, plots, common      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="closed_loop_step",
                    help="output basename (default: closed_loop_step)")
    for k in ("K_pp", "K_ip", "K_pq", "K_iq", "ctrl_dt_s"):
        ap.add_argument(f"--{k}", type=float, default=None)
    ap.add_argument("--feedforward", dest="ff", action="store_true", default=True)
    ap.add_argument("--no-feedforward", dest="ff", action="store_false")
    args = ap.parse_args()

    csv_path  = os.path.join(_HERE, "csv", f"{args.name}.csv")
    plot_base = os.path.join(_HERE, "plots", args.name)

    gains = common.baseline_gains()
    for k in ("K_pp", "K_ip", "K_pq", "K_iq", "ctrl_dt_s"):
        v = getattr(args, k)
        if v is not None:
            gains[k] = v
    import time
    print(f"[closed-loop] gains {gains}  feedforward={args.ff}")
    # Pre-settle with feedforward OFF so the reactive integral nulls Q to its
    # target (0). With feedforward on, the integral is frozen while
    # |err_Q| > ff_i_band (0.15 Mvar), so on networked stacks Q parks at a
    # residual and the step can never track. Pure PI nulls cleanly; we then
    # re-enable feedforward (args.ff) for the recorded step so it still tracks
    # fast AND to target.
    ppc_api.set_setpoints(feedforward=False, **gains)
    print("[closed-loop] pre-settle at 2.0 MW (ff off; nulling Q) …")
    capture.pre_settle(2.0, 0.0)
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        q = ppc_api.status().get("poi_mvar", float("nan"))
        if q is not None and abs(q) <= 0.05:
            break
        time.sleep(0.5)
    ppc_api.set_setpoints(feedforward=args.ff)
    print(f"[closed-loop] feedforward={args.ff}; recording PQ step …")
    with capture.Recorder(csv_path, pre=3, post=10, trigger_mw=0.1):
        capture.drive_pq_step(p_initial=2.0, p_step=3.0, q_step=1.0, t_end=15)

    ff = "ff" if args.ff else "no-ff"
    plots.step_figure(csv_path, plot_base,
                      f"Closed-loop step  ({ff})  "
                      f"K_pp={gains['K_pp']} K_ip={gains['K_ip']} "
                      f"K_pq={gains['K_pq']} K_iq={gains['K_iq']}")
    print(f"[closed-loop] wrote {csv_path}")
    print(f"[closed-loop] wrote {plot_base}.svg / .html")


if __name__ == "__main__":
    main()
