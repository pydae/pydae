#!/usr/bin/env python3
# =============================================================================
# run_all_tests.py — run every structured test for every protocol + local.
#
#   sudo python3 common/run_all_tests.py                # all: modbus iec104 dnp3 dds local
#   sudo python3 common/run_all_tests.py iec104 dds     # only these
#
# For each MININET protocol (modbus/iec104/dnp3/dds) it: cleans up any prior
# cosim, launches `python3 run.py --no-cli`, waits for the PPC API + a LIVE
# ground-truth feed, runs open_loop_step + closed_loop_step + frequency_response
# (p/q/pq90), then tears the cosim down.  `local` uses run.sh (no Mininet) on
# the same ports.  Finally it regenerates the comparison plots and the
# closed-loop metrics table.
#
# Must run as root (Mininet needs it).  Test drivers + plots run as the
# invoking user so output files are NOT root-owned.
# =============================================================================
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

PY   = os.environ.get("SUITE_PY") or "__PY_INTERP__"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PPC  = "http://localhost:8090"
SIM  = "http://localhost:8080"
FREQ_MODES = ["p", "q", "pq90"]
ALL = ["modbus", "iec104", "dnp3", "dds", "local"]
MININET = {"modbus", "iec104", "dnp3", "dds"}

# MITM (stealthy over-power) runs only on the networked stacks (local has no
# wire to intercept). It uses a dedicated --mitm-serve cosim where the
# mitm_attack driver arms/records/disarms itself via a control file — no
# launch-time timing.
#
# Attack goal: the plant is operated at its cap (MITM_CAP_MW, ~12 MW). The MITM
# biases the POI readback LOW so the PI (feedforward OFF, so the integral is
# free to wind) quietly drives the REAL plant MITM_OVERPOWER above the cap.
# Spoofed = real*(1+offset_frac); the loop drives spoofed→target, so
# real = target/(1+offset_frac). For +10 %: offset_frac = -0.10/1.10 ≈ -0.0909,
# real ≈ 13.2 MW (within the 15 MW plant limit → no saturation, stays stealthy).
MITM_CAP_MW      = __MITM_CAP_MW__
MITM_OVERPOWER   = 0.10
MITM_OFFSET_FRAC = -MITM_OVERPOWER / (1.0 + MITM_OVERPOWER)   # ≈ -0.0909

RUN_USER = os.environ.get("SUDO_USER") or os.environ.get("USER") or "root"
LOGDIR   = os.path.join(ROOT, "common", "run_logs")
SUMMARY  = os.path.join(LOGDIR, "summary.txt")


# ---- helpers ---------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def note(msg: str) -> None:
    with open(SUMMARY, "a") as f:
        f.write(msg + "\n")


def as_user_cmd(args: list[str]) -> list[str]:
    """Wrap a command so it runs as the invoking user with PPC_URL/SIM_URL set."""
    return ["sudo", "-u", RUN_USER, "env", f"PPC_URL={PPC}", f"SIM_URL={SIM}", *args]


def run_driver(script: str, extra: list[str], logpath: str) -> bool:
    with open(logpath, "a") as fh:
        r = subprocess.run(as_user_cmd([PY, script, *extra]),
                           stdout=fh, stderr=subprocess.STDOUT)
    return r.returncode == 0


def set_ff(on: bool) -> None:
    try:
        req = urllib.request.Request(
            f"{PPC}/setpoints",
            data=json.dumps({"feedforward": on}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=3).read()
    except Exception:
        pass


def wait_ppc(timeout: float = 200.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{PPC}/status", timeout=2).read()
            return True
        except Exception:
            time.sleep(2)
    return False


def _post(d: dict) -> None:
    req = urllib.request.Request(f"{PPC}/setpoints", data=json.dumps(d).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=3).read()


def _simp() -> float:
    d = json.loads(urllib.request.urlopen(f"{SIM}/cosim/measurements", timeout=3).read())
    return d["data"]["p_line_POI_GRID"] * 1e-6


def feed_live() -> bool:
    """Drive P and confirm the ground-truth sim feed actually moves (dds-freeze guard)."""
    try:
        _post({"target_mw": 4.0, "target_mvar": 0.0, "feedforward": False})
        ok = False
        for _ in range(8):
            p = _simp()
            if abs(p - 3.0) > 0.05 and abs(p) > 0.1:
                ok = True
            time.sleep(0.8)
        return ok
    except Exception as e:
        log(f"feed check error: {e}")
        return False


def pkill(pattern: str) -> None:
    subprocess.run(["pkill", "-f", pattern], stderr=subprocess.DEVNULL)


def kill_cosim(proc: subprocess.Popen | None) -> None:
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(2)
    for pat in [r"colinker\.", "run.py --no-cli", "pydae_sim", "pydae_ppc"]:
        pkill(pat)
    subprocess.run(["mn", "-c"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)


def run_suite(proto: str) -> None:
    """Run all tests against whatever cosim is currently up."""
    tlog = os.path.join(LOGDIR, f"{proto}_tests.log")
    open(tlog, "w").close()
    base = os.path.join(ROOT, proto, "tests")

    log("  → open_loop_step")
    ok = run_driver(os.path.join(base, "open_loop_step", "run.py"), [], tlog)
    note(f"  {proto} open_loop_step   {'OK' if ok else 'FAIL'}")

    log("  → closed_loop_step")
    ok = run_driver(os.path.join(base, "closed_loop_step", "run.py"), [], tlog)
    note(f"  {proto} closed_loop_step {'OK' if ok else 'FAIL'}")

    set_ff(True)   # frequency response characterises the feedforward-on bandwidth
    for m in FREQ_MODES:
        log(f"  → frequency_response --mode {m}")
        ok = run_driver(os.path.join(base, "frequency_response", "run.py"),
                        ["--mode", m], tlog)
        note(f"  {proto} freq_response({m}) {'OK' if ok else 'FAIL'}")


# ---- per-protocol drivers --------------------------------------------------
def launch(cmd: list[str], cwd: str, logname: str, env: dict | None = None) -> subprocess.Popen:
    fh = open(os.path.join(LOGDIR, logname), "w")
    return subprocess.Popen(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT,
                            start_new_session=True, env=env)


def do_mininet(proto: str) -> None:
    log(f"=== {proto} : cleanup + launch cosim ===")
    kill_cosim(None)                       # clear any leftovers first
    proc = launch(["python3", "run.py", "--no-cli"],
                  os.path.join(ROOT, proto), f"{proto}_cosim.log")
    log(f"{proto} : waiting for PPC API {PPC} …")
    if not wait_ppc():
        note(f"{proto} COSIM_DID_NOT_START (see {proto}_cosim.log)"); kill_cosim(proc); return
    if not feed_live():
        note(f"{proto} FEED_FROZEN — skipped (restart cosim, re-run just {proto})")
        kill_cosim(proc); return
    log(f"{proto} : cosim up, feed live")
    run_suite(proto)
    log(f"=== {proto} : teardown ===")
    kill_cosim(proc)


def do_local() -> None:
    log("=== local : cleanup + launch (pydae_sim + pydae_ppc, no Mininet) ===")
    pkill("pydae_ppc"); pkill("pydae_sim"); time.sleep(2)
    env = dict(os.environ, PYDAE_PORT="8080", API_PORT="8090", PY=PY)
    proc = launch(["./run.sh"], os.path.join(ROOT, "local"), "local_cosim.log", env=env)
    log(f"local : waiting for PPC API {PPC} …")
    if not wait_ppc():
        note("local COSIM_DID_NOT_START"); kill_cosim(proc); return
    if not feed_live():
        note("local FEED_FROZEN — skipped"); kill_cosim(proc); return
    log("local : cosim up, feed live")
    run_suite("local")
    log("=== local : teardown ===")
    pkill("pydae_ppc"); pkill("pydae_sim"); time.sleep(2)


def do_mitm(proto: str) -> None:
    """MITM as a test call: relaunch the cosim with the idle serve-mode proxy
    (--mitm-serve), then the mitm_attack driver arms→records→disarms itself via
    the control file. A separate cosim from the clean suite so the transparent
    proxy hop doesn't perturb the functional/latency results."""
    log(f"=== {proto} : MITM serve-mode cosim (--mitm-serve) ===")
    kill_cosim(None)
    # Capture the wire traffic (PPC/POI/MITM netns) for the whole serve session
    # — baseline, armed attack, and recovery — so the pcaps show the falsified
    # payloads next to the clean ones. tcpdump runs as root inside the netns;
    # we chown the results back to the invoking user after teardown.
    pcap_dir = os.path.join(ROOT, proto, "tests", "mitm_attack", "pcap")
    proc = launch(["python3", "run.py", "--no-cli", "--mitm-serve",
                   "--pcap-dir", pcap_dir],
                  os.path.join(ROOT, proto), f"{proto}_mitm_cosim.log")
    log(f"{proto} : waiting for PPC API {PPC} (idle MITM) …")
    if not wait_ppc():
        note(f"{proto} MITM_COSIM_DID_NOT_START (see {proto}_mitm_cosim.log)")
        kill_cosim(proc); return
    if not feed_live():
        note(f"{proto} MITM FEED_FROZEN — skipped"); kill_cosim(proc); return
    # Operate at the plant cap with feedforward OFF (pure PI) so the integral is
    # free to wind — otherwise the ff_i_band freeze would defeat the over-power
    # attack. Settle to the cap before the driver records its clean baseline.
    try:
        _post({"target_mw": MITM_CAP_MW, "target_mvar": 0.0, "feedforward": False})
    except Exception:
        pass
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            if abs(_simp() - MITM_CAP_MW) <= 0.02 * MITM_CAP_MW:
                break
        except Exception:
            pass
        time.sleep(1)
    log(f"{proto} : at cap {MITM_CAP_MW} MW — running mitm_attack "
        f"(over-power {MITM_OVERPOWER:+.0%}, offset_frac={MITM_OFFSET_FRAC:.4f}) …")
    tlog = os.path.join(LOGDIR, f"{proto}_tests.log")
    ok = run_driver(os.path.join(ROOT, proto, "tests", "mitm_attack", "run.py"),
                    ["--offset-frac", f"{MITM_OFFSET_FRAC:.6f}"], tlog)
    note(f"  {proto} mitm_attack(overpower) {'OK' if ok else 'FAIL'}")
    log(f"=== {proto} : MITM teardown ===")
    kill_cosim(proc)
    # tcpdump wrote root-owned pcaps inside the netns; hand them to the user and
    # report where they landed.
    if os.path.isdir(pcap_dir):
        subprocess.run(["chown", "-R", f"{RUN_USER}:{RUN_USER}", pcap_dir],
                       stderr=subprocess.DEVNULL)
        caps = sorted(f for f in os.listdir(pcap_dir) if f.endswith(".pcap"))
        note(f"    {proto} pcaps: {', '.join(caps) or '(none)'} "
             f"in {os.path.relpath(pcap_dir, ROOT)}/")


# ---- main ------------------------------------------------------------------
def main() -> None:
    if os.geteuid() != 0:
        sys.exit("ERROR: run with sudo (Mininet needs root)")

    run_mitm  = "--no-mitm" not in sys.argv
    only_mitm = "--only-mitm" in sys.argv       # skip the functional suite
    targets = [a for a in sys.argv[1:] if not a.startswith("-")] or ALL
    os.makedirs(LOGDIR, exist_ok=True)
    open(SUMMARY, "w").close()

    if only_mitm:
        run_mitm = True                         # --only-mitm implies the phase runs
        log(f"MITM-only run   targets: {' '.join(targets)}   (user={RUN_USER})")
        for proto in targets:
            if proto in MININET:
                do_mitm(proto)
            elif proto == "local":
                log("local has no wire layer — no MITM, skipping")
            else:
                log(f"unknown protocol '{proto}' — skipping")
        print()
        log("DONE (MITM only).  Summary:")
        print(open(SUMMARY).read())
        log(f"Per-protocol logs in {LOGDIR}/")
        return

    log(f"targets: {' '.join(targets)}   mitm={'on' if run_mitm else 'off'}   (user={RUN_USER})")
    for proto in targets:
        if proto == "local":
            do_local()          # no wire layer -> no MITM
        elif proto in MININET:
            do_mininet(proto)
            if run_mitm:
                do_mitm(proto)
        else:
            log(f"unknown protocol '{proto}' — skipping")

    log("=== regenerating comparison plots + metrics ===")
    with open(SUMMARY, "a") as fh:
        for extra in (["--test", "open_loop_step"], ["--test", "closed_loop_step"]):
            subprocess.run(as_user_cmd([PY, os.path.join(ROOT, "common", "plot_pq_compare.py"), *extra]),
                           stdout=fh, stderr=subprocess.STDOUT)
        subprocess.run(as_user_cmd([PY, os.path.join(ROOT, "common", "compute_closed_loop_metrics.py")]),
                       stdout=fh, stderr=subprocess.STDOUT)

    print()
    log("DONE.  Summary:")
    print(open(SUMMARY).read())
    log(f"Per-protocol logs in {LOGDIR}/")


if __name__ == "__main__":
    main()
