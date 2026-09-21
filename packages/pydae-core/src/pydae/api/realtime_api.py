"""Soft real-time DAE simulation with a FastAPI REST interface.

Architecture
------------
A single background daemon thread runs the IDAS integrator in fixed-size
*chunks* (default 50 ms of simulation time per chunk).  After each chunk the
thread sleeps until the absolute wall-clock target for the next chunk, so
accumulated timing error stays bounded rather than compounding.

The thread never appends to the model's ``Time`` / ``X`` / ``Y`` / ``Z``
lists.  Only the two "hot" NumPy arrays ``model.x`` and ``model.y_run`` are
updated in place, keeping memory consumption flat for arbitrarily long runs.

All shared state between the background thread and the FastAPI handlers is
protected by a single :class:`threading.Lock`.  Handlers acquire it only for
the minimum critical section — the integrator step itself, or a single array
read / write.

Requires the optional ``api`` extra::

    pip install pydae[api]

Usage
-----
::

    from pydae.api.realtime_api import app
    app.state.model    = my_casadi_model   # CasadiModel, already ini()-ed
    app.state.chunk_ms = 50.0             # optional, default 50 ms
    app.state.solver   = "trapezoidal"    # optional: "idas" (default) or "trapezoidal"

    # optional: run each model's discrete control part every chunk
    from pydae.bps.discrete import DiscreteOrchestrator
    app.state.discrete = DiscreteOrchestrator.from_model(my_casadi_model)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

REST endpoints
--------------
``GET  /measurements``
    All h_dict outputs (browser-friendly).
``POST /measurements``
    Selective h_dict outputs: ``{"names": ["E_p", "E_k"]}``
``GET  /inputs``
    All runtime input (u_run setpoint) values.
``GET  /cosim/measurements``
    Only the measurements declared in ``configs`` (co-simulation subset).
``POST /set_input``
    Write a single setpoint: ``{"name": "f_x", "value": 0.5}``
``POST /setpoints``
    Write multiple setpoints — envelope ``{"setpoints": {"f_x": 0.5}, "timestamp": 1.0}``
    or flat ``{"f_x": 0.5}``.
``POST /cosim/setpoints``
    Write setpoints restricted to those declared in ``configs``; rejects unknown keys.
``GET  /status``
    Returns ``{"is_running": bool, "t_sim": float}``
``GET  /faults``
    Declared and currently-active fault buses (needs ``app.state.faults``).
``POST /faults/apply``
    Apply a shunt fault: ``{"bus": "POI", "t_clear": 0.2}`` (``t_clear`` optional).
``POST /faults/clear``
    Clear a shunt fault: ``{"bus": "POI"}``
"""

import re
import time
import inspect
import threading
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator


def _resolve_emec_names(entry: dict, candidate_names: set[str]) -> list[str]:
    """Resolve one config entry to matching pydae variable names.

    Handles the three patterns used in the ``configs`` section of a pydae
    co-simulation JSON:

    ``emec_name``
        Exact variable name — returned as-is if present in *candidate_names*.
    ``emec_prefix``
        Prefix: returns all candidates matching ``{prefix}_{suffix}``.
    ``emec_template``
        Template with one or more ``<emec_id>`` tokens.  The first token
        becomes a regex capture group; subsequent tokens become backreferences
        so that only names where all tokens resolve to the *same* device id
        are matched (e.g. ``p_line_<emec_id>_<emec_id>LV`` matches
        ``p_line_SS1_SS1LV`` but not ``p_line_SS1_SS2LV``).
    """
    if "emec_name" in entry:
        name = entry["emec_name"]
        return [name] if name in candidate_names else []

    if "emec_prefix" in entry:
        prefix = entry["emec_prefix"] + "_"
        return sorted(n for n in candidate_names if n.startswith(prefix))

    if "emec_template" in entry:
        template = entry["emec_template"]
        parts = template.split("<emec_id>")
        n_tokens = len(parts) - 1
        if n_tokens == 0:
            return [template] if template in candidate_names else []
        # Build regex: first token → capture group, rest → backreference.
        pattern = re.escape(parts[0])
        for i in range(n_tokens):
            pattern += "([A-Za-z0-9_]+)" if i == 0 else r"\1"
            pattern += re.escape(parts[i + 1])
        rx = re.compile(pattern)
        return sorted(n for n in candidate_names if rx.fullmatch(n))

    return []


class RealTimeSimulator:
    r"""Drift-proof real-time wrapper around :class:`~pydae.core.model.casadi_model.CasadiModel`.

    The loop body is::

        acquire lock
          t_next = t_sim + chunk_sec
          run one IDAS step: x, y_run ← integrator(x, y_run, p)
          t_sim = t_next
        release lock
        sleep until wall_t0 + t_sim          ← absolute target, not relative

    Using an absolute target means a slow step causes a *one-shot* lag that is
    printed as an overrun warning, not a permanent offset that grows forever.

    Parameters
    ----------
    model:
        An initialised ``CasadiModel`` (``ini()`` already called).  The
        simulator reads and writes ``model.x``, ``model.y_run``,
        ``model.p_vals``, and ``model.u_run_vals`` directly.
    chunk_ms:
        Wall-clock chunk size in milliseconds.  Smaller values give finer
        time resolution but more OS scheduler overhead.
    solver:
        Per-chunk integrator: ``"idas"`` (default; adaptive SUNDIALS BDF, most
        robust) or ``"trapezoidal"`` (fixed-step implicit trapezoid).  The
        trapezoid reuses the model's cached fixed-step CasADi functions and is
        usually faster in real time because it avoids IDAS's per-chunk cold
        restart (``calc_ic`` + BDF ramp-up).
    trapezoidal_substeps:
        Number of implicit-trapezoid sub-steps per chunk when
        ``solver="trapezoidal"``.  ``None`` (default) picks
        ``round(chunk_sec / model.Dt)`` so accuracy matches the model's ``Dt``.
    trapezoidal_linsol:
        Linear-solver backend for the trapezoid's Newton step: ``"scipy"``
        (default, ``scipy.sparse.linalg.splu``) or any ``ca.Linsol`` plugin
        (``"qr"`` — recommended CasADi-native — ``"lapacklu"``, ``"csparse"``,
        ``"ma27"``, ...).  An unavailable plugin falls back to SciPy with a warning.
    discrete:
        Optional discrete-control orchestrator.  Any object exposing
        ``step(model)`` (typically a
        :class:`pydae.bps.discrete.DiscreteOrchestrator`).  It is called once
        per chunk, *before* the integration step, so the model's discrete
        control parts (e.g. a fault-ride-through relay) can update the runtime
        inputs the next step will integrate with.  Kept duck-typed so this
        module has no dependency on ``pydae-bps``.
    faults:
        Optional fault controller.  Any object exposing ``apply(bus)`` /
        ``clear(bus)`` and a ``buses`` property (typically a
        :class:`pydae.bps.utils.faults.FaultController`).  Its methods run a
        frozen-time continuation on the live model and are invoked under the
        integrator lock by :meth:`apply_fault` / :meth:`clear_fault`.  Kept
        duck-typed for the same bps-independence reason.
    """

    def __init__(self, model, chunk_ms: float = 50.0, ramp_chunks: int = 20,
                 discrete=None, faults=None, solver: str = "idas",
                 trapezoidal_substeps: int | None = None,
                 trapezoidal_linsol: str = "scipy"):
        self.model = model
        self.chunk_sec = chunk_ms / 1000.0
        self.t_sim: float = 0.0
        self._lock = threading.Lock()
        self.is_running: bool = False
        self._thread: threading.Thread | None = None
        # Per-chunk integrator: 'idas' (adaptive SUNDIALS, robust) or
        # 'trapezoidal' (fixed-step implicit trapezoid, typically faster in RT
        # because it avoids IDAS's per-chunk cold restart / calc_ic).
        self.solver = solver
        self._integrator = None          # IDAS integrator (built for 'idas')
        self._trap = None                # cached trapezoidal machinery
        self._trap_substeps = trapezoidal_substeps
        # Linear-solver backend for the trapezoid's Newton step: "scipy"
        # (scipy sparse LU) or any ca.Linsol plugin ("qr", "lapacklu", ...).
        self._trap_linsol = trapezoidal_linsol
        self._trap_tol = 1e-9
        self._trap_max_iter = 10
        # Modified-Newton factorization reuse: refresh the iteration matrix every
        # N sub-steps, or adaptively when a step needs > refactor_iters iterations
        # (this reuse is what makes the trapezoid fast). _trap_sc counts sub-steps.
        self._trap_refactor_every = 25
        self._trap_refactor_iters = 4
        self._trap_sc = 0
        self._ramp_chunks = ramp_chunks
        self._ramp_queue: list[dict[str, float]] = []
        # Discrete-control orchestrator (optional; falls back to a `discrete`
        # attribute the model itself may carry). None => no discrete step.
        self._discrete = discrete if discrete is not None else getattr(
            model, "discrete", None)
        # Latency-tier orchestrators gate on the sim clock via step(model, t).
        # Decide once whether this (duck-typed) hook accepts the time argument,
        # so a legacy step(model)-only hook is still called correctly.
        self._discrete_pass_t = self._hook_accepts_time(self._discrete)
        # Fault controller (optional; falls back to a `fault_controller` attribute
        # the model may carry). Duck-typed: apply(bus)/clear(bus)/buses. Kept
        # bps-agnostic so this module has no dependency on pydae-bps.
        self._faults = faults if faults is not None else getattr(
            model, "fault_controller", None)
        self._active_faults: set[str] = set()
        # Bus -> sim time at which an auto-clear is due (set by apply_fault's
        # t_clear). Serviced under the lock in the sim loop.
        self._scheduled_clears: dict[str, float] = {}
        # Wall-clock pacing baseline (set when the loop starts) and a one-shot
        # flag asking the loop to re-baseline after a blocking fault event.
        self._wall_t0: float = 0.0
        self._rebase_pacing: bool = False
        # Co-simulation variable lists — populated by load_cosim_config().
        self._cosim_meas_names: list[str] = []
        self._cosim_setp_names: list[str] = []

    # ── integrator setup ───────────────────────────────────────────────────

    def _build_integrator(self):
        """Build and cache the CasADi IDAS integrator sized to ``chunk_sec``.

        This is separate from ``model.integrator`` because the model's own
        integrator may have a different step size (``model.Dt``).  Building
        here avoids mutating the model object from the background thread.
        """
        import casadi as ca

        opts = {
            "calc_ic": True,        # let IDAS correct algebraic initial values
            "print_stats": False,
            "max_num_steps": 5000,
            "reltol": getattr(self.model, "_integrator_reltol", 1e-6),
            "abstol": getattr(self.model, "_integrator_abstol", 1e-5),
            "linear_solver": "csparse",
        }
        # The integrator is created for the fixed interval [0, chunk_sec].
        # Passing t0=0 and tf=chunk_sec means each call advances exactly one
        # chunk regardless of absolute simulation time.
        self._integrator = ca.integrator(
            "idas_rt", "idas", self.model.dae_dict, 0.0, self.chunk_sec, opts
        )

    def _build_trapezoidal(self):
        """Build and cache the fixed-step implicit-trapezoid machinery.

        Reuses the model's own CasADi function builders — ``_build_fixed_step_fns``
        (residual ``R``, iteration matrix ``J``, slope ``f`` for ``alpha=0.5``) and
        ``_make_lin_backend`` (the linear-solver backend) — so the DAE maths is
        never re-derived here; only the per-chunk in-place stepping lives in this
        module (symmetric to how ``_build_integrator`` inlines the IDAS call).
        """
        fns = self.model._build_fixed_step_fns(0.5)          # alpha=0.5 (trapezoid)
        backend = self.model._make_lin_backend(
            self._trap_linsol, fns["J"].sparsity_out(0))
        Dt = getattr(self.model, "Dt", 0.01) or 0.01
        n = self._trap_substeps or max(1, int(round(self.chunk_sec / Dt)))
        self._trap = {"fns": fns, "backend": backend,
                      "n": n, "dt": self.chunk_sec / n}

    def _build_stepper(self):
        """Build whichever per-chunk integrator ``self.solver`` selects."""
        if self.solver == "trapezoidal":
            self._build_trapezoidal()
        else:
            self._build_integrator()

    # ── single integration step ────────────────────────────────────────────

    def _step(self):
        """Advance ``model.x`` / ``model.y_run`` by one chunk with the selected
        solver. Nothing is appended to ``Time/X/Y/Z`` — memory stays O(1)."""
        if self.solver == "trapezoidal":
            self._step_trapezoidal()
        else:
            self._step_idas()

    def _step_idas(self):
        """One chunk via the cached IDAS integrator (adaptive, robust).

        CasADi's IDAS integrator expects the parameter vector to be the
        concatenation of physical parameters and runtime inputs, matching the
        ``dae_dict['p']`` symbol list built by ``CasadiBuilder``.
        """
        # CasADi convention: p = [p_vals | u_run_vals]
        p_run_vec = np.concatenate((self.model.p_vals, self.model.u_run_vals))
        res = self._integrator(x0=self.model.x, z0=self.model.y_run, p=p_run_vec)
        # Write results back in place; no copies kept.
        self.model.x = np.array(res["xf"]).flatten()
        self.model.y_run = np.array(res["zf"]).flatten()

    def _step_trapezoidal(self):
        """One chunk via the fixed-step implicit trapezoid, in place.

        Advances over ``n`` sub-steps of ``dt``; each sub-step is a modified-Newton
        solve of the coupled ``[differential; algebraic]`` residual (so algebraic
        consistency is maintained without a separate ``calc_ic``). The iteration
        matrix is factorized on a schedule and reused across steps (that reuse is
        the speed win); a step that fails to converge is retried once with a fresh
        factorization before raising.
        """
        m = self.model
        fns, backend = self._trap["fns"], self._trap["backend"]
        n, dt = self._trap["n"], self._trap["dt"]
        nx, _f, _R, _J = fns["nx"], fns["f"], fns["R"], fns["J"]
        p = np.concatenate((m.p_vals, m.u_run_vals))
        x = np.asarray(m.x, float).flatten()
        z = np.asarray(m.y_run, float).flatten()

        def newton(w, xf, fnv):
            for it in range(self._trap_max_iter):
                R = np.array(_R(w, xf, fnv, p, dt)).flatten()
                if np.linalg.norm(R) < self._trap_tol:
                    return it
                if backend.stale or (self._trap_sc % self._trap_refactor_every == 0 and it == 0):
                    backend.refresh(_J(w, xf, fnv, p, dt))
                w += backend.solve(-R)
            return None

        for _ in range(n):
            fnv = np.array(_f(x, z, p)).flatten()   # frozen slope f(x_n, z_n)
            w = np.concatenate((x, z))
            iters = newton(w, x, fnv)
            if iters is None:                       # stale/ill-conditioned -> fresh J, retry
                backend.mark_stale()
                w = np.concatenate((x, z))
                iters = newton(w, x, fnv)
                if iters is None:
                    raise RuntimeError(
                        f"RT trapezoidal did not converge at t_sim={self.t_sim:.4f} s "
                        "(try solver='idas', a smaller chunk_ms, or more substeps)")
            elif iters > self._trap_refactor_iters:  # hard step -> refresh next time
                backend.mark_stale()
            x, z = w[:nx], w[nx:]
            self._trap_sc += 1
        m.x, m.y_run = x, z

    # ── per-chunk helpers (ramp, discrete) ─────────────────────────────────

    def _apply_next_ramp_step(self):
        """Pop and apply one chunk's worth of ramped setpoint deltas.

        Must be called with ``self._lock`` already held.
        """
        if self._ramp_queue:
            deltas = self._ramp_queue.pop(0)
            for name, delta in deltas.items():
                u_run_names = self.model.u_run_names
                if name in u_run_names:
                    self.model.u_run_vals[u_run_names.index(name)] += delta
                else:
                    self.model.set_value(name, self.model.get_value(name) + delta)

    @staticmethod
    def _hook_accepts_time(discrete) -> bool:
        """Whether the discrete hook's ``step`` takes a time argument.

        Returns ``True`` if ``discrete.step`` accepts a second positional/keyword
        parameter (so it can be called as ``step(model, t_sim)``), ``False`` for
        a legacy ``step(model)``-only hook.  On any introspection failure it
        defaults to ``False`` (the safe, always-callable single-arg form).
        """
        if discrete is None:
            return False
        step = getattr(discrete, "step", None)
        if step is None:
            return False
        try:
            params = list(inspect.signature(step).parameters.values())
        except (ValueError, TypeError):
            return False
        # `step` is bound, so `self`/`model` is already peeled off. It can be
        # called as step(model, t_sim) if a *second* positional slot exists:
        # either >=2 positionally-fillable params, or a *args catch-all.
        positional = [p for p in params
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        has_varargs = any(p.kind == p.VAR_POSITIONAL for p in params)
        return has_varargs or len(positional) >= 2

    def _run_discrete(self):
        """Run the model's discrete control parts for this chunk.

        Delegates to the injected orchestrator's ``step(model, t_sim)`` (or
        ``step(model)`` for a legacy hook).  ``t_sim`` is the latency-tier clock,
        so tiered controllers fire at their own cadence.  It writes runtime
        inputs (``model.u_run_vals`` / via ``set_value``) that the upcoming
        :meth:`_step` integrates with.  A failure here is logged and swallowed so
        a single misbehaving controller cannot kill the real-time loop.  Must be
        called with ``self._lock`` already held.
        """
        if self._discrete is None:
            return
        try:
            if self._discrete_pass_t:
                self._discrete.step(self.model, self.t_sim)
            else:
                self._discrete.step(self.model)
        except Exception as exc:  # noqa: BLE001 - keep the RT loop alive
            print(f"[RT discrete step failed at t_sim={self.t_sim:.3f} s] {exc}")

    # ── fault application ──────────────────────────────────────────────────

    def _sync_discrete_after_event(self):
        """Re-seed the discrete orchestrator after an out-of-band model change.

        A fault apply/clear jumps the operating point (and the converters' LVRT
        inputs) via the continuation.  Re-seed the discrete controllers from the
        new state so the per-chunk relay does not fight it, and reset their tier
        timers so slow tiers re-evaluate immediately.  Must hold ``self._lock``.
        """
        # The fault continuation jumped the operating point, so a cached
        # trapezoidal factorization is stale — force a refresh on the next step.
        if self._trap is not None:
            self._trap["backend"].mark_stale()
        if self._discrete is None:
            return
        sync = getattr(self._discrete, "sync", None)
        if callable(sync):
            sync(self.model)
        reset = getattr(self._discrete, "reset_schedule", None)
        if callable(reset):
            reset()

    def _apply_fault_locked(self, bus, t_clear=None, **kw):
        """Apply a fault and (optionally) schedule its auto-clear. Lock held."""
        result = self._faults.apply(bus, **kw)
        self._sync_discrete_after_event()
        self._active_faults.add(bus)
        if t_clear is not None:
            self._scheduled_clears[bus] = self.t_sim + float(t_clear)
        else:
            self._scheduled_clears.pop(bus, None)
        self._rebase_pacing = True  # drop the wall-clock lag this just caused
        return result

    def _clear_fault_locked(self, bus, **kw):
        """Clear a fault and drop any pending auto-clear. Lock held."""
        result = self._faults.clear(bus, **kw)
        self._sync_discrete_after_event()
        self._active_faults.discard(bus)
        self._scheduled_clears.pop(bus, None)
        self._rebase_pacing = True  # drop the wall-clock lag this just caused
        return result

    def _run_due_fault_clears(self):
        """Auto-clear faults whose scheduled ``t_sim`` has arrived. Lock held."""
        if not self._scheduled_clears:
            return
        due = [bus for bus, t in self._scheduled_clears.items()
               if self.t_sim >= t - 1e-9]
        for bus in due:
            try:
                self._clear_fault_locked(bus)
            except Exception as exc:  # noqa: BLE001 - keep the RT loop alive
                print(f"[RT auto-clear failed for '{bus}' at "
                      f"t_sim={self.t_sim:.3f} s] {exc}")

    def apply_fault(self, bus, t_clear=None, **kw):
        """Apply a shunt fault at ``bus`` on the live model, thread-safely.

        Runs the fault controller's frozen-time continuation under the integrator
        lock (so it never races a chunk), then re-seeds the discrete layer.  With
        ``t_clear`` set, the fault self-clears after that many seconds of sim
        time; otherwise it stays until :meth:`clear_fault`.  Extra ``kw`` are
        forwarded to the controller — e.g. ``X_fault`` (override the fault
        reactance; larger = softer) and ``fault_steps`` (continuation steps;
        fewer = faster, so a soft fault causes a smaller under-lock pause), plus
        ``iq`` / ``v_lvrt``.
        """
        if self._faults is None:
            raise RuntimeError("No fault controller attached "
                               "(set app.state.faults).")
        with self._lock:
            return self._apply_fault_locked(bus, t_clear=t_clear, **kw)

    def clear_fault(self, bus, **kw):
        """Clear the shunt fault at ``bus`` on the live model, thread-safely."""
        if self._faults is None:
            raise RuntimeError("No fault controller attached "
                               "(set app.state.faults).")
        with self._lock:
            return self._clear_fault_locked(bus, **kw)

    def declared_fault_buses(self):
        """Buses that have a fault declared (empty if no controller attached)."""
        return list(getattr(self._faults, "buses", []) or [])

    def active_faults(self):
        """Buses currently faulted, sorted."""
        return sorted(self._active_faults)

    # ── background simulation loop ─────────────────────────────────────────

    def _sim_loop(self):
        """Main loop executed in the background daemon thread."""
        self._build_stepper()
        self._wall_t0 = time.perf_counter()

        while self.is_running:
            with self._lock:
                self._apply_next_ramp_step()
                self._run_due_fault_clears()
                self._run_discrete()
                t_next = self.t_sim + self.chunk_sec
                try:
                    self._step()
                    self.t_sim = t_next
                except RuntimeError as exc:
                    print(f"[RT step failed at t_sim={self.t_sim:.3f} s] {exc}")

            # A fault apply/clear runs a blocking continuation under the lock,
            # stalling the loop for much longer than a chunk. Rather than
            # fast-forwarding to reclaim that wall-clock lag (a burst of
            # [RT overrun] warnings that also compresses any short fault out of a
            # polling client's view), re-baseline the schedule so real time
            # simply resumes from here.
            if self._rebase_pacing:
                self._rebase_pacing = False
                self._wall_t0 = time.perf_counter() - self.t_sim

            target_wall_time = self._wall_t0 + self.t_sim
            sleep_time = target_wall_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                overrun_ms = -sleep_time * 1000.0
                # Ignore sub-millisecond overshoot (scheduler jitter and the
                # exact-zero at a pacing re-baseline) — only report real lag.
                if overrun_ms > 1.0:
                    print(
                        f"[RT overrun] {overrun_ms:.2f} ms behind schedule"
                        f" at t_sim={self.t_sim:.3f} s"
                    )

    # ── co-simulation config ───────────────────────────────────────────────

    def load_cosim_config(self, cosim_config: dict) -> None:
        """Parse a ``configs`` dict and build co-simulation variable lists.

        Call this before :meth:`start`.  The *cosim_config* is the value of
        the ``configs`` key in the co-simulation JSON (already parsed — pass
        ``data["configs"]``, not the file path).

        Each top-level key in *cosim_config* is a device type (e.g. ``"poi"``,
        ``"inv"``).  Each device type has ``measurements`` and ``setpoints``
        lists whose entries use ``emec_name``, ``emec_prefix``, or
        ``emec_template`` to reference pydae variables.

        Measurement names are resolved against ``h_dict``; setpoint names
        against ``u_run_names``.  Duplicates are removed preserving first-seen
        order so the response is stable across calls.
        """
        h_names: set[str] = set(
            getattr(self.model, "sys_dict", {}).get("h_dict", {}).keys()
        )
        u_names: set[str] = set(getattr(self.model, "u_run_names", []))

        seen_m: set[str] = set()
        seen_s: set[str] = set()
        meas: list[str] = []
        setp: list[str] = []

        for device_cfg in cosim_config.values():
            for entry in device_cfg.get("measurements", []):
                for name in _resolve_emec_names(entry, h_names):
                    if name not in seen_m:
                        seen_m.add(name)
                        meas.append(name)
            for entry in device_cfg.get("setpoints", []):
                for name in _resolve_emec_names(entry, u_names):
                    if name not in seen_s:
                        seen_s.add(name)
                        setp.append(name)

        self._cosim_meas_names = meas
        self._cosim_setp_names = setp

    def cosim_measurements(self) -> dict[str, float | None]:
        """Return current values for co-simulation measurements, thread-safely.

        Returns only the variables declared in the ``configs`` section of the
        co-simulation JSON (populated by :meth:`load_cosim_config`).  The
        values are raw pydae h_dict values — unit conversion (``emec_scale``)
        is the responsibility of the co-simulation client.
        """
        return self.measurements(self._cosim_meas_names)

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        """Spawn the background daemon thread and start simulating.

        The thread is a *daemon* so that it does not prevent the Python
        process from exiting if the main thread finishes or raises.
        """
        self.is_running = True
        self._thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the loop to exit and block until the thread joins.

        A 5-second timeout prevents deadlock if the integrator hangs; after
        the timeout the thread is abandoned (it is a daemon, so the process
        can still exit cleanly).
        """
        self.is_running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # ── thread-safe setpoint writers ──────────────────────────────────────

    def _write_one(self, name: str, value: float) -> None:
        """Write one value to ``u_run_vals`` or via ``model.set_value``.

        The caller **must** already hold ``self._lock``.  Keeping this as a
        separate helper allows both :meth:`set_input` and :meth:`setpoints`
        to reuse the same write logic without nested lock acquisitions.

        ``u_run_names`` is checked first because the ``in`` test on a plain
        list is O(n) but avoids an exception path for the common case where
        the name is a runtime input.
        """
        u_run_names = self.model.u_run_names
        if name in u_run_names:
            # Direct array index write — no Python overhead from set_value.
            self.model.u_run_vals[u_run_names.index(name)] = value
        else:
            # Delegate to the model for parameters, states, and algebraics.
            self.model.set_value(name, value)

    def _enqueue_ramp(self, targets: dict[str, float]) -> None:
        """Merge ``targets`` into the pending ramp.

        Must be called with ``self._lock`` already held.  Reads each
        variable's *current* value (including any in-progress ramp) and
        spreads the remaining delta evenly over ``_ramp_chunks`` chunks.

        Variables present in *targets* are re-ramped from their current
        value toward the new target.  Variables already mid-ramp but not
        in *targets* keep their remaining ramp intact — this lets
        multi-source co-simulation clients (e.g. one HTTP POST per
        device) compose ramps without clobbering each other.
        """
        def _current(name: str) -> float:
            u_run_names = self.model.u_run_names
            if name in u_run_names:
                return float(self.model.u_run_vals[u_run_names.index(name)])
            return float(self.model.get_value(name))

        n = self._ramp_chunks

        # Preserve in-flight ramps for variables NOT being re-targeted.
        preserved: dict[str, list[float]] = {}
        for chunk in self._ramp_queue:
            for name, delta in chunk.items():
                if name in targets:
                    continue
                preserved.setdefault(name, []).append(delta)

        new_chunks: list[dict[str, float]] = [
            {name: (target - _current(name)) / n for name, target in targets.items()}
            for _ in range(n)
        ]

        # Pad new_chunks with any preserved-variable deltas, position by position.
        for name, deltas in preserved.items():
            for i, d in enumerate(deltas):
                new_chunks[i][name] = d

        self._ramp_queue = new_chunks

    def set_input(self, name: str, value: float) -> None:
        """Write a single runtime input via a smooth ramp."""
        with self._lock:
            self._enqueue_ramp({name: value})

    def setpoints(self, setpoints_dict: dict[str, float]) -> None:
        """Write one or more runtime inputs via a smooth ramp."""
        with self._lock:
            self._enqueue_ramp(setpoints_dict)

    # ── thread-safe input reader ──────────────────────────────────────────

    def inputs(self, names: list[str] | None = None) -> dict[str, float | None]:
        """Return current runtime input (``u_run``) values, thread-safely.

        These are the writable setpoints (the entries of ``u_run_dict``), the
        counterpart to :meth:`setpoints`.  With ``names=None`` every input is
        returned; otherwise only the requested names, with ``None`` for any name
        that is not a runtime input.
        """
        with self._lock:
            u_names = list(self.model.u_run_names)
            u_vals = np.asarray(self.model.u_run_vals).flatten()
            index = {n: i for i, n in enumerate(u_names)}
            if names is None:
                names = u_names
            return {n: (float(u_vals[index[n]]) if n in index else None)
                    for n in names}

    # ── thread-safe measurement reader ────────────────────────────────────

    def measurements(self, names: list[str]) -> dict[str, float | None]:
        r"""Return current values of h_dict output variables, thread-safely.

        Only variables explicitly declared in ``sys_dict['h_dict']``
        (the output map :math:`z = h(x, y, u, p)`) are accessible here.
        States, algebraics, and inputs are intentionally excluded — if a
        variable must be observable from outside, declare it in ``h_dict``.

        ``_h_fn`` is evaluated **once** per call regardless of how many names
        are requested, so the CasADi function-call overhead is paid once even
        when reading a large number of outputs simultaneously.

        Parameters
        ----------
        names:
            Output variable names to read.  Names absent from ``h_dict``
            return ``None`` without raising.

        Returns
        -------
        dict
            ``{name: float}`` for names in ``h_dict``,
            ``{name: None}`` for names not found.
        """
        with self._lock:
            h_fn = getattr(self.model, "_h_fn", None)
            h_names: list[str] = []
            h_vals: np.ndarray | None = None

            if h_fn is not None and "h_dict" in getattr(self.model, "sys_dict", {}):
                h_names = list(self.model.sys_dict["h_dict"].keys())
                # Same concatenation convention as _step().
                p_run_vec = np.concatenate((self.model.p_vals, self.model.u_run_vals))
                h_vals = np.array(
                    h_fn(self.model.x, self.model.y_run, p_run_vec)
                ).flatten()

            return {
                name: (
                    float(h_vals[h_names.index(name)])
                    if h_vals is not None and name in h_names
                    else None
                )
                for name in names
            }


# ── Pydantic request / response schemas ────────────────────────────────────────

class InputPayload(BaseModel):
    """Payload for ``POST /set_input`` — a single name/value pair."""

    name: str
    value: float




class SetpointsPayload(BaseModel):
    """Payload for ``POST /setpoints`` — one or more name/value pairs.

    Two equivalent body shapes are accepted:

    - **envelope** — ``{"setpoints": {"f_x": 0.5}, "timestamp": 1.0}``
    - **flat** — ``{"f_x": 0.5}`` — the whole body is the setpoints mapping
      (convenience; no ``timestamp``).

    A body without a ``"setpoints"`` key is treated as the flat form, so an input
    literally named ``setpoints`` must use the envelope form.
    """

    setpoints: dict[str, float]
    timestamp: float | None = None  # optional wall-clock time from the sender

    @model_validator(mode="before")
    @classmethod
    def _accept_flat(cls, data):
        # Wrap a flat {name: value} body into the envelope so both shapes parse.
        if isinstance(data, dict) and "setpoints" not in data:
            return {"setpoints": data}
        return data


class MeasurementsRequest(BaseModel):
    """Request body for ``POST /measurements`` — a list of output names."""

    names: list[str]


class FaultApplyPayload(BaseModel):
    """Payload for ``POST /faults/apply``."""

    bus: str
    t_clear: float | None = None       # auto-clear after this many sim seconds
    X_fault: float | None = None       # override fault reactance (pu); larger = softer
    fault_steps: int | None = None     # continuation steps; fewer = faster (soft faults)


class FaultClearPayload(BaseModel):
    """Payload for ``POST /faults/clear``."""

    bus: str
    fault_steps: int | None = None     # continuation steps; fewer = faster


# ── Module-level simulator reference ──────────────────────────────────────────

# Populated by the lifespan context manager on server startup.
# FastAPI route handlers read this reference; it is never written from the
# background thread, so no lock is needed for the reference itself.
simulator: RealTimeSimulator | None = None


# ── Lifespan context manager ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the simulator on startup and cleanly stop it on shutdown.

    The model must be attached to ``app.state`` *before* the ASGI server
    starts (i.e., before this context manager runs), because ``uvicorn``
    calls the lifespan on the first request or immediately at boot, not
    lazily.
    """
    global simulator

    model = getattr(app.state, "model", None)
    if model is None:
        raise RuntimeError(
            "No model attached to app.state.  "
            "Set  app.state.model = <CasadiModel>  before starting the server."
        )

    chunk_ms: float = getattr(app.state, "chunk_ms", 50.0)
    ramp_chunks: int = getattr(app.state, "ramp_chunks", 20)
    solver: str = getattr(app.state, "solver", "idas")
    trapezoidal_substeps = getattr(app.state, "trapezoidal_substeps", None)
    trapezoidal_linsol = getattr(app.state, "trapezoidal_linsol", "scipy")
    # Optional discrete-control orchestrator (e.g. pydae.bps.discrete
    # .DiscreteOrchestrator). Duck-typed: any object with step(model). Falls
    # back to a `discrete` attribute on the model if app.state has none.
    discrete = getattr(app.state, "discrete", None)
    # Optional fault controller (e.g. pydae.bps.utils.faults.FaultController).
    # Duck-typed: any object with apply(bus)/clear(bus)/buses.
    faults = getattr(app.state, "faults", None)
    simulator = RealTimeSimulator(model, chunk_ms=chunk_ms,
                                  ramp_chunks=ramp_chunks, discrete=discrete,
                                  faults=faults, solver=solver,
                                  trapezoidal_substeps=trapezoidal_substeps,
                                  trapezoidal_linsol=trapezoidal_linsol)

    cosim_config = getattr(app.state, "cosim_config", None)
    if cosim_config:
        simulator.load_cosim_config(cosim_config)

    simulator.start()

    yield  # server is live here

    simulator.stop()


# ── FastAPI application ────────────────────────────────────────────────────────

app = FastAPI(
    title="pydae Real-Time API",
    version="1.0.0",
    description=(
        "Soft real-time DAE simulation backed by CasADi IDAS.  "
        "The simulator runs in a background thread; endpoints provide "
        "lock-safe access to setpoints and measurements."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/set_input", summary="Write a single setpoint")
def set_input(payload: InputPayload):
    """Write one runtime input or parameter by name.

    Use ``/setpoints`` when you need to update several inputs atomically.
    """
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not running")
    simulator.set_input(payload.name, payload.value)
    return {"status": "ok", "name": payload.name, "value": payload.value}


@app.post("/setpoints", summary="Write multiple setpoints atomically")
def setpoints(payload: SetpointsPayload):
    """Write several runtime inputs in a single lock acquisition.

    All values are applied before the next integration step sees them,
    so the solver never encounters a partially-updated input vector.
    """
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not running")
    simulator.setpoints(payload.setpoints)
    return {"status": "ok", "setpoints": payload.setpoints, "timestamp": payload.timestamp}


@app.post("/measurements", summary="Read h_dict output variables")
def measurements(request: MeasurementsRequest):
    """Return current values of output variables declared in ``h_dict``.

    Names absent from ``h_dict`` are returned as ``null``.
    The response also includes ``t_sim`` (simulation time in seconds).
    """
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not running")
    data = simulator.measurements(request.names)
    return {"data": data, "t_sim": simulator.t_sim}


@app.get("/measurements", summary="Read all h_dict output variables")
def measurements_get():
    """Return current values of all output variables declared in ``h_dict``.

    Equivalent to ``POST /measurements`` with all h_dict names, but callable
    from a browser or any GET client without a request body.
    """
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not running")
    h_names: list[str] = []
    if hasattr(simulator.model, "sys_dict") and "h_dict" in simulator.model.sys_dict:
        h_names = list(simulator.model.sys_dict["h_dict"].keys())
    data = simulator.measurements(h_names)
    return {"data": data, "t_sim": simulator.t_sim}


@app.get("/inputs", summary="Read current runtime input (setpoint) values")
def inputs_get():
    """Return the current value of every runtime input (``u_run``).

    These are the writable setpoints — the counterpart to ``/setpoints``.
    Callable from a browser or any GET client without a request body.
    """
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not running")
    data = simulator.inputs()
    return {"data": data, "t_sim": simulator.t_sim}


@app.get("/cosim/measurements", summary="Co-simulation measurements (configs subset)")
def cosim_measurements_get():
    """Return current values for the measurements declared in the co-simulation config.

    Only variables listed in the ``configs.*.measurements`` section of the
    co-simulation JSON are returned.  Values are raw pydae h_dict values;
    apply ``emec_scale`` on the client side if field-unit conversion is needed.

    Requires ``app.state.cosim_config`` to be set before server startup.
    Returns 501 if no co-simulation config was loaded.
    """
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not running")
    if not simulator._cosim_meas_names:
        raise HTTPException(status_code=501, detail="No co-simulation config loaded")
    data = simulator.cosim_measurements()
    return {"data": data, "t_sim": simulator.t_sim}


@app.post("/cosim/setpoints", summary="Co-simulation setpoints (configs subset)")
def cosim_setpoints(payload: SetpointsPayload):
    """Write setpoints restricted to variables declared in the co-simulation config.

    Only keys listed in ``configs.*.setpoints`` are accepted.  Unknown keys
    are rejected with 422 so the co-simulation client gets immediate feedback
    instead of silently writing into an unreachable variable.

    Values are in pydae units; apply ``emec_scale`` on the client side before
    sending if your source is in field units.
    """
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not running")
    if not simulator._cosim_setp_names:
        raise HTTPException(status_code=501, detail="No co-simulation config loaded")
    allowed = set(simulator._cosim_setp_names)
    unknown = [k for k in payload.setpoints if k not in allowed]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail={"unknown_setpoints": unknown, "allowed": sorted(allowed)},
        )
    simulator.setpoints(payload.setpoints)
    return {"status": "ok", "setpoints": payload.setpoints, "timestamp": payload.timestamp}


@app.get("/status", summary="Simulator health check")
def status():
    """Return whether the simulator is running and the current simulation time."""
    if simulator is None:
        return {"is_running": False, "t_sim": 0.0}
    return {"is_running": simulator.is_running, "t_sim": simulator.t_sim}

# ── Fault endpoints ────────────────────────────────────────────────────────────

def _require_fault_controller():
    """Common guards for the fault endpoints; returns the live simulator."""
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not running")
    if simulator._faults is None:
        raise HTTPException(
            status_code=501,
            detail="No fault controller attached (set app.state.faults)",
        )
    return simulator


@app.get("/faults", summary="List declared and active faults")
def faults_list():
    """Return the buses with a declared fault and those currently faulted."""
    sim = _require_fault_controller()
    return {
        "declared": sim.declared_fault_buses(),
        "active": sim.active_faults(),
        "t_sim": sim.t_sim,
    }


@app.post("/faults/apply", summary="Apply a shunt fault on the live model")
def faults_apply(payload: FaultApplyPayload):
    """Apply a shunt fault at ``bus``, optionally auto-clearing after ``t_clear``.

    The fault is switched on with a frozen-time continuation run under the
    integrator lock (robust for deep faults; it drives converter ride-through).
    Unknown buses are rejected with **422**.
    """
    sim = _require_fault_controller()
    try:
        result = sim.apply_fault(payload.bus, t_clear=payload.t_clear,
                                 X_fault=payload.X_fault,
                                 fault_steps=payload.fault_steps)
    except ValueError as exc:  # undeclared fault bus
        raise HTTPException(status_code=422, detail=str(exc))
    return {"status": "ok", "fault": result,
            "active": sim.active_faults(), "t_sim": sim.t_sim}


@app.post("/faults/clear", summary="Clear a shunt fault on the live model")
def faults_clear(payload: FaultClearPayload):
    """Clear the shunt fault at ``bus`` (frozen-time continuation under the lock)."""
    sim = _require_fault_controller()
    try:
        result = sim.clear_fault(payload.bus, fault_steps=payload.fault_steps)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"status": "ok", "fault": result,
            "active": sim.active_faults(), "t_sim": sim.t_sim}


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("pydae.api.realtime_api:app", host="0.0.0.0", port=8000, reload=False)
