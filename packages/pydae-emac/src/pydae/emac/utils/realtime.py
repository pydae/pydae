# -*- coding: utf-8 -*-
"""Pace a control loop against the wall clock -- or let it run free.

A loop that calls the plant once per control period ``T_s`` (simulated
time) can run in two ways:

* **free** -- as fast as the computer allows; simulated time and wall time
  are unrelated.  For studies.
* **real time** -- period k ends no earlier than ``t0 + k T_s`` on the wall
  clock, so the simulation keeps step with the world (hardware in the loop,
  a live dashboard, an operator).

Usage::

    pacer = Pacer(T_s, realtime=True)
    pacer.start()
    for k in range(n):
        ...                       # measure, compute, apply, model.run(...)
        pacer.wait(k + 1)         # end of period k
    print(pacer.report())

**How it waits.**  ``time.sleep`` is only accurate to a fraction of a
millisecond, far coarser than a 50 us period, so the pacer sleeps until
``spin`` seconds before the deadline and busy-waits on ``perf_counter`` for
the rest: microsecond accuracy, at the price of a CPU core while waiting.

**The schedule is absolute.**  A period that finishes late is counted as an
overrun and the next ones start at once, catching up, so simulated and wall
time stay aligned over the run instead of drifting by every delay.  This is
soft real time: Python, the garbage collector and the operating system can
all delay a period, and the report says how often and by how much.  The
margin (``slack``: time left before each deadline) says how much more work
a period could take.
"""
import time


class Pacer:
    """Wall-clock pacing of a loop with period ``period`` [s] (``realtime``),
    or none at all (``realtime=False``: ``wait`` returns at once)."""

    def __init__(self, period, realtime=True, spin=200e-6):
        self.period, self.realtime, self.spin = float(period), bool(realtime), float(spin)
        self.t0 = None
        self.n = 0                    # deadlines met or missed so far
        self.overruns = 0
        self.max_late = 0.0           # worst lateness [s]
        self.min_slack = float("inf") # least time left before a met deadline [s]
        self._slack_sum = 0.0

    def start(self):
        """Mark wall time zero (= simulated time at the loop's start)."""
        self.t0 = time.perf_counter()

    def wait(self, k):
        """Block until the end of period k-1, i.e. wall time t0 + k * period."""
        if self.t0 is None:
            self.start()
        self.n += 1
        if not self.realtime:
            return
        target = self.t0 + k * self.period
        now = time.perf_counter()
        slack = target - now
        if slack <= 0.0:
            self.overruns += 1
            self.max_late = max(self.max_late, -slack)
            return
        self.min_slack = min(self.min_slack, slack)
        self._slack_sum += slack
        if slack > self.spin:
            time.sleep(slack - self.spin)
        while time.perf_counter() < target:
            pass

    @property
    def elapsed(self):
        """Wall time since start() [s]."""
        return time.perf_counter() - self.t0

    def report(self):
        """One line: wall vs simulated time, overruns and margin."""
        sim = self.n * self.period
        line = "wall %.3f s for %.3f s simulated (%.2fx real time)" % (
            self.elapsed, sim, sim / max(self.elapsed, 1e-12))
        if self.realtime:
            met = self.n - self.overruns
            line += "; real time: %d of %d periods late (worst by %.0f us)" % (
                self.overruns, self.n, 1e6 * self.max_late)
            if met:
                line += ", margin min %.0f us / mean %.0f us of %.0f us" % (
                    1e6 * self.min_slack, 1e6 * self._slack_sum / met, 1e6 * self.period)
        else:
            line += "; free running"
        return line
