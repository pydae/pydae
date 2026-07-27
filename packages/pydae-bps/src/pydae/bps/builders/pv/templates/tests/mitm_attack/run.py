#!/usr/bin/env python3
"""Man-in-the-middle over-power attack capture (protocol-agnostic).

Falsifies the POI active-power reading the PPC receives so the controller
drives the *real* plant off target.  This driver records three signals and
plots them:

  * PPC readback  (poi_mw)            — what the controller believes
  * plant ground truth (sim_p_line)   — what the plant actually does
  * the gap between them               — the attack's physical effect

**Over-power (serve mode, default).**  The cosim is launched idle with
``run.py --no-cli --mitm-serve`` (an in-path proxy that does nothing until
armed).  This driver parks the plant at its cap with **feedforward off** (so the
PI integral is free to wind), then arms→records→disarms via the control file
``/tmp/colinker_mitm_<proto>.ctl``.  The readback is biased LOW
(``offset_frac<0``, spoofed = real·(1+frac)); the controller drives the
falsified reading to the cap, so the **real plant is pushed to cap/(1+frac)** —
e.g. a 12 MW cap with ``-0.0909`` lands the plant at ~13.2 MW, +10 % over, with
no saturation or alarm.

    # Terminal A
    cd <proto>/ && sudo python3 run.py --no-cli --mitm-serve
    # Terminal B (this driver): arm → record → disarm
    python tests/mitm_attack/run.py

If no serve-mode cosim is detected it falls back to a passive record window
(the attack must then be armed at launch with ``run.py --mitm``).

The protocol is inferred from the folder (``<proto>/tests/mitm_attack/``), so
the same driver works for every stack.  Prereq: cosim up, PPC API at
http://localhost:8090.
"""
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from lib import capture, plots, ppc_api               # noqa: E402

# Protocol = the case-folder name two levels up (<proto>/tests/mitm_attack/).
_PROTO = os.path.basename(os.path.dirname(os.path.dirname(_HERE)))
_LABEL = {"modbus": "Modbus", "iec104": "IEC 104",
          "dnp3": "DNP3", "dds": "DDS"}.get(_PROTO, _PROTO.upper())
CTRL_FILE = f"/tmp/colinker_mitm_{_PROTO}.ctl"       # matches run.py --mitm-serve


def _write_ctl(path: str, **kw) -> None:
    with open(path, "w") as f:
        json.dump(kw, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="mitm_attack")
    ap.add_argument("--duration", type=float, default=90.0,
                    help="legacy passive mode: total capture window (s)")
    ap.add_argument("--control-file", default=CTRL_FILE,
                    help="serve-mode control file written by run.py --mitm-serve")
    ap.add_argument("--offset-frac", type=float, default=-0.090909,
                    help="serve mode: bias the POI reading by this fraction "
                         "(spoofed=real*(1+frac)). Negative under-reports, so the "
                         "PI drives the real plant OVER target: -0.0909 => +10%%.")
    ap.add_argument("--cap-mw", type=float, default=__MITM_CAP_MW__,
                    help="serve mode: operate at this plant cap (feedforward off) "
                         "before arming, so the attack drives the real plant "
                         "cap/(1+offset_frac).")
    ap.add_argument("--baseline", type=float, default=15.0,
                    help="serve mode: clean seconds before arming")
    ap.add_argument("--attack", type=float, default=30.0,
                    help="serve mode: seconds the attack is armed")
    ap.add_argument("--recovery", type=float, default=15.0,
                    help="serve mode: clean seconds after disarming")
    args = ap.parse_args()

    csv_path  = os.path.join(_HERE, "csv", f"{args.name}.csv")
    plot_base = os.path.join(_HERE, "plots", args.name)
    title = f"{_LABEL} MITM — PPC readback (falsified) vs plant ground truth"

    # Serve mode is active when the proxy has written its status file.
    serve = os.path.exists(args.control_file + ".status")

    if serve:
        print(f"[mitm] {_PROTO} serve mode — over-power via {args.control_file} "
              f"(cap={args.cap_mw} MW, offset_frac={args.offset_frac:+.4f} "
              f"=> ~{args.cap_mw / (1 + args.offset_frac):.2f} MW real)")
        # Operate at the cap with feedforward OFF so the integral is free to wind
        # (the ff_i_band freeze would otherwise defeat the over-power attack).
        ppc_api.set_setpoints(target_mw=args.cap_mw, target_mvar=0.0,
                              feedforward=False)
        capture.pre_settle(args.cap_mw, 0.0)
        _write_ctl(args.control_file, armed=False)          # ensure clean start
        with capture.Recorder(csv_path, pre=2, post=0, trigger_mw=1e9) as rec:
            rec.arm()
            time.sleep(args.baseline)                        # clean baseline
            print("[mitm] ARM"); _write_ctl(args.control_file,
                                            armed=True, offset_frac=args.offset_frac)
            time.sleep(args.attack)                          # attack window
            print("[mitm] DISARM"); _write_ctl(args.control_file, armed=False)
            time.sleep(args.recovery)                        # recovery
    else:
        print(f"[mitm] passive mode — recording {args.duration} s "
              "(attack must be armed at cosim launch) …")
        with capture.Recorder(csv_path, pre=2, post=0, trigger_mw=1e9) as rec:
            rec.arm()
            time.sleep(args.duration)

    plots.step_figure(csv_path, plot_base, title)
    print(f"[mitm] wrote {csv_path}")
    print(f"[mitm] wrote {plot_base}.svg / .html")


if __name__ == "__main__":
    main()
