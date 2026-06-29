#!/usr/bin/env python3
"""Man-in-the-middle attack capture (protocol-agnostic).

The attack falsifies the POI active-power measurement so the PPC reads a
doctored value; the controller then drives the *real* plant to the wrong
operating point.  The attack is armed when the cosim is launched (the
protocol-specific ``colinker.attacks.*_mitm`` module wired by the benchmark's
own ``run.py``) — it is NOT controlled from here — so this driver only records
a window and plots:

  * PPC readback  (poi_mw)            — what the controller believes
  * plant ground truth (sim_p_line)   — what the plant actually does
  * the gap between them               — the attack's physical effect

Attack policies (set on the cosim's ``run.py``):
  * over-power offset (``--mitm-offset-frac`` negative, e.g. -0.0909): readback
    tracks target while the real plant is driven ~10 % ABOVE it — a quiet
    grid-code breach with no saturation or alarm.
  * under-power offset (positive frac): plant settles below target.
  * fixed (``--mitm-value-mw``): readback pinned, the PI rails to saturation.

Self-timing capture (default)
-----------------------------
Rather than blind-recording a fixed window (which has to be hand-aligned to the
attack), this driver records continuously and watches the live readback-vs-
ground-truth divergence: it detects when the attack ARMS, then keeps recording
``--record`` more seconds to span the attack steady state and the recovery after
it disarms.  So it captures a clean episode regardless of exactly when it is
launched, as long as it starts before the attack disarms.

Relaunch the cosim with the attack armed, then run this::

    # Terminal A — over-power offset, held long enough to record comfortably
    sudo python3 run.py --no-cli --mitm --mitm-offset-frac -0.0909 \\
         --mitm-start-after 30 --mitm-duration 120
    # Terminal B (this script) — auto-detects the window
    python tests/mitm_attack/run.py

Use ``--target-mw`` to park the plant at a specific operating point first (e.g.
to match a reference case); pass ``--blind SECONDS`` for the old fixed window.

Prereq: cosim running WITH --mitm, PPC API at http://localhost:8090.
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from lib import capture, plots, ppc_api               # noqa: E402


def _readback_mw():
    """PPC's (possibly falsified) POI active-power readback, in MW."""
    try:
        return float(ppc_api.status().get("poi_mw", float("nan")))
    except Exception:
        return float("nan")


def _real_poi_mw():
    """Plant ground-truth POI active power from pydae, in MW (wire is W)."""
    url = capture.SIM_URL.rstrip("/") + "/cosim/measurements"
    try:
        with urllib.request.urlopen(url, timeout=2.0) as r:
            data = json.loads(r.read()).get("data", {})
        raw = data.get("sim_p_line_POI_GRID", data.get("p_line_POI_GRID"))
        return float(raw) / 1e6
    except Exception:
        return float("nan")


def _capture_auto(*, settle_max, record_after, thresh, poll_dt):
    """Record continuously, detect the attack arming via readback-vs-truth
    divergence, then keep recording ``record_after`` s past detection.

    Returns the elapsed seconds spent recording."""
    t0 = time.perf_counter()
    armed_t = None
    print(f"[mitm] watching readback vs plant ground truth "
          f"(arm threshold {thresh:.0%} of target) …")
    while True:
        now = time.perf_counter() - t0
        rb, real = _readback_mw(), _real_poi_mw()
        try:
            target = abs(float(ppc_api.status().get("target_mw", 0.0))) or 1.0
        except Exception:
            target = 1.0
        gap = abs(rb - real) if (math.isfinite(rb) and math.isfinite(real)) else 0.0
        rel = gap / target

        if armed_t is None and rel >= thresh:
            armed_t = now
            print(f"[mitm]   t={now:5.1f}s ATTACK DETECTED — readback={rb:.3f} "
                  f"real={real:.3f} MW  gap={gap:+.3f} ({rel:+.1%})")
        if armed_t is not None and (now - armed_t) >= record_after:
            print(f"[mitm]   captured {now - armed_t:.0f}s past detection — done")
            return now
        if armed_t is None and now >= settle_max:
            print(f"[mitm]   no attack seen in {settle_max:.0f}s — stopping "
                  f"(is the cosim armed with --mitm?)")
            return now
        time.sleep(poll_dt)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="mitm_attack")
    ap.add_argument("--target-mw", type=float, default=None,
                    help="Set the PPC active-power setpoint (MW) before "
                         "recording, to park the plant at a specific operating "
                         "point. Default: leave the cosim's current setpoint.")
    ap.add_argument("--target-mvar", type=float, default=0.0,
                    help="Reactive setpoint (Mvar) to apply with --target-mw.")
    ap.add_argument("--blind", type=float, default=None,
                    help="Fixed-window mode: record this many seconds and stop "
                         "(no attack detection). Use when you time the window "
                         "by hand.")
    ap.add_argument("--record", type=float, default=150.0,
                    help="Auto mode: seconds to keep recording after the attack "
                         "is detected (spans the attack hold + recovery).")
    ap.add_argument("--settle-max", type=float, default=240.0,
                    help="Auto mode: give up if no attack is seen within this "
                         "many seconds.")
    ap.add_argument("--thresh", type=float, default=0.03,
                    help="Auto mode: |readback-truth|/target divergence that "
                         "counts as the attack being armed (default 3%%).")
    ap.add_argument("--poll-dt", type=float, default=0.5)
    args = ap.parse_args()

    csv_path  = os.path.join(_HERE, "csv", f"{args.name}.csv")
    plot_base = os.path.join(_HERE, "plots", args.name)

    if args.target_mw is not None:
        print(f"[mitm] setting operating point to {args.target_mw} MW / "
              f"{args.target_mvar} Mvar …")
        settled = capture.pre_settle(args.target_mw, args.target_mvar)
        print(f"[mitm]   settled poi_mw = {settled:.3f}")

    with capture.Recorder(csv_path, pre=2, post=0, trigger_mw=1e9) as rec:
        rec.arm()                       # start writing now; no step trigger
        if args.blind is not None:
            print(f"[mitm] blind recording {args.blind:.0f} s window …")
            time.sleep(args.blind)
        else:
            _capture_auto(settle_max=args.settle_max, record_after=args.record,
                          thresh=args.thresh, poll_dt=args.poll_dt)

    plots.step_figure(csv_path, plot_base,
                      "MITM — PPC readback (falsified) vs plant ground truth")
    print(f"[mitm] wrote {csv_path}")
    print(f"[mitm] wrote {plot_base}.svg / .html")


if __name__ == "__main__":
    main()
