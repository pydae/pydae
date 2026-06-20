#!/usr/bin/env python3
"""Sinusoidal frequency sweep → Bode of the reference→POI response.

Three modes (``--mode``):
  p     — sinusoid on P, Q held constant.
  q     — sinusoid on Q, P held constant.
  pq90  — sinusoid on both, Q lagging P by 90°.  The /sine API resets each
          loop's phase to 0 on arm, so we arm the Q sinusoid a quarter-period
          (1/4f) after P to obtain exactly 90° — no controller change needed.

For each frequency the driver arms the sinusoid(s), polls /status for
``--duration`` s logging the commanded reference vs the measured POI, then
disarms.  Outputs per mode (in csv/ and plots/):
  csv/<mode>_f*.csv      — t, ref_p, poi_p, ref_q, poi_q
  plots/<mode>_timeseries.{svg,html}
  plots/<mode>_bode_{p,q}.{svg,html}   (whichever loop(s) are excited)
  csv/<mode>_bode.csv

    python run.py --mode p
    python run.py --mode q
    python run.py --mode pq90

Prereq: cosim running, PPC API at http://localhost:8090.
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from lib import ppc_api, plots, fit, common          # noqa: E402


def _poll_sweep(f, mode, base_p, amp_p, base_q, amp_q, duration, dt):
    """Arm the sinusoid(s) for one frequency and return logged rows."""
    p_on = mode in ("p", "pq90")
    q_on = mode == "q"                       # pq90 arms Q later, mid-loop
    if p_on:
        ppc_api.set_sine(p_on=True, p_amp=amp_p, p_freq=f)
    if q_on:
        ppc_api.set_sine(q_on=True, q_amp=amp_q, q_freq=f)

    tq0 = 1.0 / (4.0 * f) if mode == "pq90" else None   # quarter-period delay
    armed_q = q_on
    t0 = time.perf_counter(); next_t = t0
    rows = []
    while True:
        now = time.perf_counter() - t0
        if mode == "pq90" and not armed_q and now >= tq0:
            ppc_api.set_sine(q_on=True, q_amp=amp_q, q_freq=f)
            armed_q = True
        ref_p = base_p + (amp_p * math.sin(2 * math.pi * f * now) if p_on else 0.0)
        if mode == "q":
            ref_q = base_q + amp_q * math.sin(2 * math.pi * f * now)
        elif mode == "pq90" and armed_q:
            ref_q = base_q + amp_q * math.sin(2 * math.pi * f * (now - tq0))
        else:
            ref_q = base_q
        try:
            s = ppc_api.status()
        except Exception:
            s = {}
        rows.append((now, ref_p, _f(s.get("poi_mw")), ref_q, _f(s.get("poi_mvar"))))
        if now >= duration:
            break
        next_t += dt
        sleep = next_t - time.perf_counter()
        time.sleep(sleep) if sleep > 0 else None
    ppc_api.set_sine(p_on=False, q_on=False)
    return rows


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("p", "q", "pq90"), default="p")
    ap.add_argument("--freqs", type=float, nargs="+",
                    default=[0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--base-mw", type=float, default=5.0)
    ap.add_argument("--amp-mw", type=float, default=2.0)
    ap.add_argument("--base-mvar", type=float, default=0.0)
    ap.add_argument("--amp-mvar", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--settle", type=float, default=4.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--skip", type=float, default=4.0, help="transient to drop before fit")
    args = ap.parse_args()

    csvd = os.path.join(_HERE, "csv"); plotd = os.path.join(_HERE, "plots")
    print(f"[freq] mode={args.mode} freqs={args.freqs} Hz")
    ppc_api.set_sine(p_on=False, q_on=False)

    excite_p = args.mode in ("p", "pq90")
    excite_q = args.mode in ("q", "pq90")
    ts_p, ts_q = [], []
    bode_p, bode_q = [], []

    for f in args.freqs:
        ppc_api.set_setpoints(target_mw=args.base_mw, target_mvar=args.base_mvar)
        ppc_api.set_sine(p_on=False, q_on=False)
        time.sleep(args.settle)
        rows = _poll_sweep(f, args.mode, args.base_mw, args.amp_mw,
                           args.base_mvar, args.amp_mvar, args.duration, args.dt)
        # save per-frequency CSV
        with open(os.path.join(csvd, f"{args.mode}_f{f:0.2f}.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["t", "ref_p", "poi_p", "ref_q", "poi_q"])
            for r in rows:
                w.writerow([f"{x:.6f}" for x in r])
        a = np.array(rows)
        t = a[:, 0]; mask = (t >= args.skip) & np.isfinite(a[:, 2])
        if excite_p:
            ts_p.append((f, t, a[:, 1], a[:, 2]))
            bode_p.append((f, *fit.gain_phase(t[mask], a[mask, 1], a[mask, 2], f)))
        if excite_q:
            mq = (t >= args.skip) & np.isfinite(a[:, 4])
            ts_q.append((f, t, a[:, 3], a[:, 4]))
            bode_q.append((f, *fit.gain_phase(t[mq], a[mq, 3], a[mq, 4], f)))
        print(f"[freq]   f={f:.2f} Hz  {len(rows)} samples")

    # time series + Bode per excited loop
    if excite_p:
        plots.sweep_timeseries_figure(ts_p, os.path.join(plotd, f"{args.mode}_timeseries_p"),
                                      f"{args.mode}: P reference vs POI", "P (MW)")
        _write_bode(bode_p, args.mode, "p", csvd, plotd)
    if excite_q:
        plots.sweep_timeseries_figure(ts_q, os.path.join(plotd, f"{args.mode}_timeseries_q"),
                                      f"{args.mode}: Q reference vs POI", "Q (Mvar)")
        _write_bode(bode_q, args.mode, "q", csvd, plotd)

    # restore P/Q references
    ppc_api.set_setpoints(target_mw=args.base_mw, target_mvar=args.base_mvar)
    print(f"[freq] done — outputs in {csvd} and {plotd}")


def _write_bode(bode, mode, loop, csvd, plotd):
    bode.sort()
    f = [b[0] for b in bode]; g = [b[1] for b in bode]; ph = [b[2] for b in bode]
    plots.bode_figure(f, g, ph, os.path.join(plotd, f"{mode}_bode_{loop}"),
                      f"{mode}: reference→POI {loop.upper()} frequency response")
    with open(os.path.join(csvd, f"{mode}_bode_{loop}.csv"), "w", newline="") as fh:
        import numpy as _np
        w = csv.writer(fh); w.writerow(["freq_hz", "gain", "gain_db", "phase_deg"])
        phu = _np.degrees(_np.unwrap(_np.radians(ph)))
        for i in range(len(f)):
            w.writerow([f"{f[i]:.3f}", f"{g[i]:.5f}",
                        f"{20*_np.log10(g[i]):.3f}", f"{phu[i]:.3f}"])
    print(f"[freq] {loop.upper()} bode:")
    for fi, gi, pi in zip(f, g, ph):
        print(f"        f={fi:.2f}  |G|={gi:.3f} ({20*np.log10(gi):+.2f} dB)  ph={pi:+.1f}°")


if __name__ == "__main__":
    main()
