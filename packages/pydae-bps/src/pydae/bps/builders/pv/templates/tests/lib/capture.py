"""Transient capture helpers — pre-settle the plant and record a window of
PPC + pydae telemetry to a merged CSV.

Launches ``colinker.recorder`` directly as a background subprocess and stops it
via its pidfile, deliberately bypassing ``colinker.record_ctl start`` whose
``proc.stdout.readline()`` banner loop (record_ctl.py:92) blocks when the
recorder emits fewer than 8 startup lines.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time

from . import ppc_api
from .common import PYTHON, PPC_URL

# pydae ground-truth source for the recorder. Defaults to the Mininet cosim's
# rtsim port (8080); in local (no-Mininet) mode export SIM_URL to point at the
# pydae_sim launched by local/run.sh (e.g. http://localhost:8011).
SIM_URL = os.environ.get("SIM_URL", "http://localhost:8080")


def pre_settle(target_mw: float, target_mvar: float = 0.0,
               tol: float = 0.03, timeout: float = 30.0) -> float:
    """Park the references and block until POI active power settles.

    Returns the settled ``poi_mw``.  Used so the recorder's auto-arm triggers
    on the *real* step rather than the jump to the pre-step operating point.
    """
    ppc_api.set_setpoints(target_mw=target_mw, target_mvar=target_mvar)
    deadline = time.monotonic() + timeout
    poi = float("nan")
    while time.monotonic() < deadline:
        poi = ppc_api.status().get("poi_mw", float("nan"))
        if poi is not None and abs(poi - target_mw) <= tol:
            return poi
        time.sleep(0.5)
    return poi


def drive_pq_step(p_initial: float = 2.0, p_step: float = 3.0,
                  q_step: float = 1.0, t_end: float = 15.0) -> str:
    """Drive a P then Q reference step via ``colinker.poi_step_test`` and return
    its captured stdout (the ref-vs-actual table)."""
    out = subprocess.run(
        [PYTHON, "-m", "colinker.poi_step_test", "--ppc", PPC_URL,
         "--p-initial", str(p_initial), "--p-step", str(p_step),
         "--q-step", str(q_step), "--t-end", str(t_end)],
        capture_output=True, text=True, check=True)
    return out.stdout


class Recorder:
    """Background ``colinker.recorder`` controlled via its pidfile.

    Usage::

        with Recorder(out_csv, pre=3, post=10, trigger_mw=0.1):
            ... drive the step ...
        # CSV flushed on exit
    """

    def __init__(self, out_csv: str, pre: float = 3.0, post: float = 10.0,
                 trigger_mw: float = 0.1, workdir: str | None = None):
        self.out_csv    = os.path.abspath(out_csv)
        self.workdir    = workdir or os.path.dirname(self.out_csv) or "."
        self.pidfile    = os.path.join(self.workdir, ".recorder.pid")
        self.logfile    = os.path.splitext(self.out_csv)[0] + "_recorder.log"
        self.pre, self.post, self.trigger_mw = pre, post, trigger_mw
        self._proc = None

    def __enter__(self) -> "Recorder":
        for p in (self.out_csv, self.pidfile, self.logfile):
            if os.path.exists(p):
                os.unlink(p)
        log = open(self.logfile, "w")
        self._proc = subprocess.Popen(
            [PYTHON, "-m", "colinker.recorder", "--pidfile", self.pidfile,
             "--ppc", PPC_URL, "--sim", SIM_URL,
             "--out", self.out_csv, "--trigger-mw", str(self.trigger_mw),
             "--pre", str(self.pre), "--post", str(self.post)],
            stdout=log, stderr=subprocess.STDOUT, cwd=self.workdir)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not os.path.exists(self.pidfile):
            time.sleep(0.2)
        return self

    def arm(self) -> None:
        """Force the recorder to start writing now (SIGUSR1), for captures with
        no target-step auto-trigger (e.g. the MITM attack window)."""
        try:
            pid = int(open(self.pidfile).read().strip())
            os.kill(pid, signal.SIGUSR1)
        except (FileNotFoundError, ValueError, ProcessLookupError):
            pass

    def __exit__(self, *exc):
        try:
            pid = int(open(self.pidfile).read().strip())
            os.kill(pid, signal.SIGTERM)
        except (FileNotFoundError, ValueError, ProcessLookupError):
            pass
        # Let it flush, then ensure the subprocess is reaped.
        if self._proc is not None:
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if os.path.exists(self.pidfile):
            try:
                os.unlink(self.pidfile)
            except OSError:
                pass
        return False
