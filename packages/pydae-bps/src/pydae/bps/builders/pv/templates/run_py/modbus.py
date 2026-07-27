#!/usr/bin/env python3
"""Launch the co-simulation (Mininet + pydae + proxies + PPC).

Reads the plant JSON sitting next to this file and delegates to
``colinker.mn.mininet_topo.setup()``. Requires root.

Usage::

    sudo python3 run.py [--no-cli] [--no-ppc]
    sudo python3 run.py --no-cli --arrivals-dir results_delays/arrivals
    sudo python3 run.py --no-cli --mitm                 # default: +5% offset
    sudo python3 run.py --no-cli --mitm --mitm-offset-frac 0.05 \
                        --arrivals-dir results_overpower/arrivals \
                        --pcap-dir     results_overpower/pcaps
    sudo python3 run.py --no-cli --mitm --mitm-value-mw 7.5   # fixed-spoof
"""
import argparse
import os
import sys


_HERE         = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_JSON = os.path.join(_HERE, "__JSON_FILENAME__")
_PYTHON       = "/home/jmmmacubu/miniforge3/envs/pydae_dev/bin/python3"
_COLINKER_SRC = "/home/jmmmacubu/mac_shared/colinker/src"

# Pcap default capture set: PPC, POI, MITM, plus one host per inverter.
_PV_HOSTS = __PV_HOSTS_LITERAL__


def _check_root() -> None:
    if os.geteuid() != 0:
        print("error: Mininet requires root. Re-run with:")
        print(f"  sudo python3 {sys.argv[0]}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the co-simulation (Modbus TCP stack)"
    )
    parser.add_argument("--json", "-j", default=_DEFAULT_JSON,
                        help=f"Plant JSON path (default: {_DEFAULT_JSON})")
    parser.add_argument("--no-cli", action="store_true",
                        help="Skip Mininet CLI; block on SIGINT (headless)")
    parser.add_argument("--no-ppc", action="store_true",
                        help="Skip auto-starting the PPC")
    parser.add_argument("--arrivals-dir", default=None,
                        help="If set, each pv_proxy writes a 50 ms-cadence CSV "
                             "of the wire-level setpoint values that arrived "
                             "at it (one file per inverter, named "
                             "<emec_id>_arrivals.csv).")
    parser.add_argument("--mitm", action="store_true",
                        help="Arm the Modbus MITM attack from the MITM host. "
                             "Default: spoof POI active-power reading.")
    parser.add_argument("--mitm-args", default=None,
                        help="Override the full argv passed to "
                             "colinker.attacks.modbus_mitm (when --mitm is set). "
                             "Implies --mitm.")
    parser.add_argument("--mitm-value-mw", type=float, default=None,
                        help="FIXED-spoof mode: pin the POI active-power "
                             "readback at this constant (MW) while armed. The "
                             "PI error never closes, so the plant drives to "
                             "saturation. Mutually exclusive with "
                             "--mitm-offset-frac / --mitm-offset-mw.")
    parser.add_argument("--mitm-offset-frac", type=float, default=0.05,
                        help="OFFSET-spoof mode (default): add this fraction of "
                             "the REAL POI reading before the PPC sees it "
                             "(0.05 = +5%%). The PPC settles at a small, "
                             "non-saturating steady-state error — the plant "
                             "ends up ~this fraction off target.")
    parser.add_argument("--mitm-offset-mw", type=float, default=0.0,
                        help="OFFSET-spoof mode: also add this absolute bias "
                             "(MW) to the real POI reading. Combined with "
                             "--mitm-offset-frac.")
    parser.add_argument("--mitm-start-after", type=float, default=30.0,
                        help="Seconds after launch before the attack arms.")
    parser.add_argument("--mitm-duration", type=float, default=60.0,
                        help="Seconds the attack holds, then disarms.")
    parser.add_argument("--mitm-serve", action="store_true",
                        help="Launch the MITM proxy IDLE, armed/disarmed at "
                             "runtime from a control file "
                             "(/tmp/colinker_mitm_modbus.ctl) — so a test driver "
                             "(tests/mitm_attack/run.py) can run the attack "
                             "against this same cosim, no relaunch. Uses the "
                             "deterministic explicit-redirect path.")
    parser.add_argument("--pcap-dir", default=None,
                        help="If set, run tcpdump in every host's netns "
                             "(PPC, POI, all PV hosts, MITM) capturing all "
                             "Modbus (tcp/5000 + tcp/502) and ARP traffic to "
                             "<PCAP_DIR>/<HOST>.pcap.")
    args = parser.parse_args()

    _check_root()

    if not os.path.exists(args.json):
        print(f"error: JSON not found: {args.json}")
        sys.exit(1)

    # sudo-rs blocks env-var forwarding even with -E, so set inline.
    os.environ.setdefault("COLINKER_PYTHON", _PYTHON)

    if args.arrivals_dir:
        os.makedirs(args.arrivals_dir, exist_ok=True)
        os.environ["COLINKER_ARRIVALS_DIR"] = os.path.abspath(args.arrivals_dir)
        print(f"[run] arrivals CSVs -> {os.environ['COLINKER_ARRIVALS_DIR']}/<name>_arrivals.csv")

    if args.pcap_dir:
        os.makedirs(args.pcap_dir, exist_ok=True)
        os.environ["COLINKER_PCAP_DIR"] = os.path.abspath(args.pcap_dir)
        # PPC/POI/MITM are the three vantage points needed to see the attack
        # signature (same Modbus tid, different payload bytes).  Per-inverter
        # captures are optional — drop them to save memory on small VMs.
        pcap_hosts = ["PPC", "POI", "MITM"]
        if os.environ.get("COLINKER_PCAP_INCLUDE_PV", "0") not in ("0", "", "false"):
            pcap_hosts += list(_PV_HOSTS)
        os.environ["COLINKER_PCAP_HOSTS"] = ",".join(pcap_hosts)
        print(f"[run] pcap captures -> {os.environ['COLINKER_PCAP_DIR']}/<HOST>.pcap")

    if args.mitm or args.mitm_args:
        # Default attack: falsify the POI active-power register
        # (p_line_POI_GRID, float32 CDAB, raw W) the PPC reads.
        #
        #   offset mode (default): add a small bias to the REAL reading
        #     (cur*(1+frac)+abs) → the PPC settles ~frac off target without
        #     saturating — stealthy, realistic falsification.
        #   fixed mode (--mitm-value-mw): pin the readback at a constant →
        #     the PI rails to saturation (the original over-power demo).
        #
        # --explicit-redirect mode: the PPC is told via --device-ip POI=<MITM>
        # to connect directly to MITM:5000; MITM forwards to the real POI.
        # Deterministic test path — the ARP-spoof + iptables NAT path is
        # documented but not used here (conntrack flakiness inside netns).
        MITM_IP          = "10.10.0.9"
        # Listen on the same port the POI exposes so the PPC's
        # `--device-ip POI=<MITM>` override (IP only, not port) reaches us.
        POI_PORT         = __POI_MODBUS_PORT__
        MITM_LISTEN_PORT = POI_PORT
        # POI register holding p_line_POI_GRID (offset 370 from reg_0=1000).
        POI_P_REGISTER   = __POI_P_REGISTER__
        if args.mitm_value_mw is not None:
            if args.mitm_offset_mw or args.mitm_offset_frac != 0.05:
                print("[run] WARNING: --mitm-value-mw (fixed) overrides "
                      "--mitm-offset-* (offset)")
            inject_arg = f"--value {args.mitm_value_mw * 1e6}"
        else:
            # offset mode: bias real reading by frac (+ abs MW → W)
            inject_arg = (f"--offset-frac {args.mitm_offset_frac} "
                          f"--offset {args.mitm_offset_mw * 1e6}")
        default_args = (
            "--iface mitm-eth0 "
            "--victim-ip 10.10.0.4   --victim-mac 00:00:00:00:00:04 "
            "--target-ip 10.10.0.5   --target-mac 00:00:00:00:00:05 "
            f"--target-port {POI_PORT}  --listen-port {MITM_LISTEN_PORT} "
            f"--register {POI_P_REGISTER}  --n-regs 2  --dtype float32  --fmt CDAB "
            f"{inject_arg} "
            f"--start-after {args.mitm_start_after}  --duration {args.mitm_duration} "
            "--explicit-redirect"
        )
        os.environ["COLINKER_MITM_ARGS"] = args.mitm_args or default_args
        if "--explicit-redirect" in os.environ["COLINKER_MITM_ARGS"]:
            os.environ["COLINKER_PPC_POI_IP_OVERRIDE"] = MITM_IP
            print(f"[run] PPC POI device-ip overridden to MITM ({MITM_IP})")
        print(f"[run] MITM attack ARMED — args:\n    {os.environ['COLINKER_MITM_ARGS']}")

    elif args.mitm_serve:
        # Idle serve mode: proxy in-path (explicit-redirect, deterministic) but
        # transparent until a test driver arms it via the control file. Same
        # POI-register spoof target as the armed path above.
        MITM_IP, POI_PORT, POI_P_REGISTER = "10.10.0.9", __POI_MODBUS_PORT__, __POI_P_REGISTER__
        CTRL = "/tmp/colinker_mitm_modbus.ctl"
        for _p in (CTRL, CTRL + ".status"):
            try:
                os.remove(_p)
            except OSError:
                pass
        os.environ["COLINKER_MITM_ARGS"] = (
            "--iface mitm-eth0 "
            "--victim-ip 10.10.0.4   --victim-mac 00:00:00:00:00:04 "
            "--target-ip 10.10.0.5   --target-mac 00:00:00:00:00:05 "
            f"--target-port {POI_PORT}  --listen-port {POI_PORT} "
            f"--register {POI_P_REGISTER}  --n-regs 2  --dtype float32  --fmt CDAB "
            "--offset-frac 0.0 "
            f"--control-file {CTRL}  --explicit-redirect"
        )
        os.environ["COLINKER_PPC_POI_IP_OVERRIDE"] = MITM_IP
        print(f"[run] PPC POI device-ip overridden to MITM ({MITM_IP})")
        print(f"[run] MITM SERVE mode — idle; control file {CTRL}\n"
              f"[run]   run tests/mitm_attack/run.py to arm→record→disarm")

    if _COLINKER_SRC not in sys.path:
        sys.path.insert(0, _COLINKER_SRC)

    from mininet.log import setLogLevel
    from colinker.mn.mininet_topo import setup

    setLogLevel("info")
    setup(args.json,
          interactive=not args.no_cli,
          ppc_auto=not args.no_ppc)


if __name__ == "__main__":
    main()
