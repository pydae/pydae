#!/usr/bin/env python3
"""Modbus man-in-the-middle attack capture.

The attack falsifies the POI active-power holding register so the PPC reads a
doctored value; the controller then drives the *real* plant to the wrong
operating point.  The attack is armed when the cosim is launched — it is NOT
controlled from here — so this driver only records a window and plots:

  * PPC readback  (poi_mw)            — what the controller believes
  * plant ground truth (sim_p_line)   — what the plant actually does
  * the gap between them               — the attack's physical effect

In the default **offset** mode the readback tracks target while the plant sits
~5% off; in **fixed** mode (`--mitm-value-mw`) the readback is pinned and the
PI rails to saturation.

Relaunch the cosim with the attack armed, then run this within the window::

    # Terminal A — offset (default) or fixed spoof
    sudo python3 run.py --no-cli --mitm --mitm-offset-frac 0.05 \\
         --mitm-start-after 30 --mitm-duration 60
    # Terminal B (this script), spanning the attack window
    python run.py --duration 90

Prereq: cosim running WITH --mitm, PPC API at http://localhost:8090.
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from lib import capture, plots                       # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="mitm_attack")
    ap.add_argument("--duration", type=float, default=90.0,
                    help="capture window in seconds (default: 90)")
    args = ap.parse_args()

    csv_path  = os.path.join(_HERE, "csv", f"{args.name}.csv")
    plot_base = os.path.join(_HERE, "plots", args.name)

    print(f"[mitm] recording {args.duration} s window …")
    with capture.Recorder(csv_path, pre=2, post=0, trigger_mw=1e9) as rec:
        rec.arm()                       # no step trigger — record immediately
        time.sleep(args.duration)

    plots.step_figure(csv_path, plot_base,
                      "Modbus MITM — PPC readback (falsified) vs plant ground truth")
    print(f"[mitm] wrote {csv_path}")
    print(f"[mitm] wrote {plot_base}.svg / .html")


if __name__ == "__main__":
    main()
