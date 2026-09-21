# -*- coding: utf-8 -*-
r"""
Generic shunt-fault application/clearing for pydae power-system models.

A **fault** is a balanced shunt impedance $Z_f = R_{pu} + jX_{pu}$ switched onto
a bus (declared in a ``"faults"`` section of the network data, which
``BpsBuilder`` turns into an off shunt plus a target admittance in
``model.sys_dict['faults']``).

Design
------
:class:`FaultManager` is **generic** -- it knows how to ramp a bus's fault
admittance in and out *robustly* (adaptive continuation with snapshot/restore),
and nothing else. It has no notion of converters, grid codes or ride-through.

The *particular* response of the grid elements to the fault is injected as an
``on_step`` callback. At every continuation step ``FaultManager`` relaxes that
callback to a fixed point (states frozen), so the fault is applied *with* the
elements following their own law -- reached by the robust continuation rather
than by a time-domain ramp. This is what lets even deep faults converge.

* Systems with no fault-sensitive elements pass no callback -- just the
  admittance ramp.
* Grid-following converters (``pv_pq_ss`` / ``bess_pq_ss`` ...) pass an
  ``on_step`` that sets their ``lvrt_ext`` / ``i_sr_ref`` / ``i_sa_ref``
  references. :func:`lvrt_response` builds the standard low-voltage
  ride-through response (constant or grid-code ``iq(V)`` reactive current); a
  caller can supply any other callback.

The *same* control law is applied two ways by two owners: during the fault
switch the callback is relaxed to a fixed point by ``FaultManager``; during the
subsequent time-domain integration the caller applies it per step (e.g. a
rate-limited relay).
"""
import numpy as np
from typing import Any, Callable, Dict, Optional, Tuple


class FaultManager:
    """Applies and clears a shunt fault by robustly ramping the bus admittance.

    Generic and element-agnostic: pass an ``on_step(model) -> max_change``
    callback to ``apply``/``clear`` to let grid elements follow their own
    response as the fault is switched (see :func:`lvrt_response`).
    """
    def __init__(self, model: Any, bus_name: str, fault_steps: int = 20,
                 verbose: bool = True):
        # fault_steps is the dominant cost: the continuation does at least two
        # full algebraic solves per step and up to relax_iters+1 of them, so the
        # total is linear in this number. Measured on a 14-converter PV+BESS
        # plant, one apply+clear at the declared X_f = 0.1 pu costs 1401 solves
        # / 7.3 s at 100, against 401 solves / 3.1 s at 20 -- for the same
        # faulted operating point to within 4e-4 pu. The adaptive halving in
        # _continue is what provides robustness, not a fine nominal step, so 20
        # is the better default and the ceiling stays available per call.
        self.model = model
        self.bus_name = bus_name
        self.fault_steps = fault_steps
        self.verbose = verbose

        faults = getattr(model, 'sys_dict', {}).get('faults', {})
        if bus_name not in faults:
            raise ValueError(
                f"No fault declared at bus '{bus_name}'. Declared fault buses: "
                f"{sorted(faults.keys())}. Add a 'faults' section to the network data."
            )
        g_name = f'g_shunt_{bus_name}'
        b_name = f'b_shunt_{bus_name}'
        self.fault_on = {g_name: faults[bus_name]['g'], b_name: faults[bus_name]['b']}
        self.fault_off = {g_name: 0.0, b_name: -1e-6}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def apply(self, on_step: Optional[Callable] = None,
              target: Optional[Dict[str, float]] = None,
              fault_steps: Optional[int] = None, **cont_kw):
        """Apply the fault (frozen time) by ramping the admittance to Z_f.

        ``target`` overrides the declared fault admittance (e.g. a different
        reactance); ``fault_steps`` overrides the continuation resolution for
        this call (fewer steps = faster, enough for a soft fault).
        ``cont_kw`` forwards ``tol`` / ``relax_iters`` / ``settle`` to
        :meth:`_continue` -- the three knobs that set how much work each step
        does, as opposed to how many steps there are.
        """
        cont_kw.setdefault('tol', 1e-8)
        self._continue(target or self.fault_on, on_step=on_step,
                       fault_steps=fault_steps, **cont_kw)
        self._report('fault applied')

    def clear(self, on_step: Optional[Callable] = None,
              fault_steps: Optional[int] = None, **cont_kw):
        """Clear the fault (frozen time) by ramping the admittance back out.

        ``cont_kw`` is as in :meth:`apply`; the clear defaults to a looser
        ``tol`` because it ends at the unfaulted operating point, which is
        already known to be well conditioned.
        """
        cont_kw.setdefault('tol', 1e-5)
        self._continue(self.fault_off, on_step=on_step,
                       fault_steps=fault_steps, **cont_kw)
        self._report('fault cleared')

    # ------------------------------------------------------------------ #
    # Continuation machinery
    # ------------------------------------------------------------------ #
    def _continue(self, target: Dict[str, float], on_step: Optional[Callable] = None,
                  tol: float = 1e-8, relax_iters: int = 30, settle: float = 1e-4,
                  fault_steps: Optional[int] = None):
        # Adaptive continuation of the fault admittance from its current value to
        # ``target``. Each step is halved on failure and grown on success. When an
        # ``on_step`` callback is given it is relaxed to a fixed point inside every
        # step: the callback reads the operating point, nudges its references, and
        # returns the max change; the loop re-solves and repeats until it settles.
        #
        # ``settle`` is the second cost multiplier after ``fault_steps``: the
        # relaxation converges geometrically at rate (1 - gain), so the sweeps
        # per step go as log((initial error)/settle). It is a band on *reference*
        # values in pu, and 1e-4 pu of current reference is already far below
        # anything physical -- measured on a 14-converter plant, tightening it to
        # 1e-5 costs 40 % more algebraic solves and moves the faulted operating
        # point by nothing at all (V_POI 0.7618 either way).
        start = {name: self.model.get_value(name) for name in target}
        entry = self._snapshot()          # rollback point if this gives up

        def set_frac(f):
            for name, tgt in target.items():
                self.model.set_value(name, start[name] + f * (tgt - start[name]))

        steps = fault_steps if fault_steps else self.fault_steps
        progress, step_size, max_step = 0.0, 1.0 / steps, 1.0 / steps
        while progress < 1.0 - 1e-9:
            next_progress = min(1.0, progress + step_size)
            snapshot = self._snapshot()
            set_frac(next_progress)
            try:
                self.model.recalculate_algebraics(tol=tol)
                if on_step is not None:
                    for _ in range(relax_iters):
                        changed = on_step(self.model)
                        self.model.recalculate_algebraics(tol=tol)
                        if changed is None or changed < settle:
                            break
            except RuntimeError:
                self._restore(snapshot)
                step_size *= 0.5
                if step_size < 1e-7:
                    # Roll all the way back to the state we were handed, not to
                    # the last intermediate snapshot. A caller that does not own
                    # the time loop -- FaultController under the real-time API --
                    # otherwise resumes integrating a half-applied fault, which
                    # is an operating point nothing asked for and which stops
                    # the clock dead.
                    self._restore(entry)
                    set_frac(0.0)
                    raise RuntimeError(
                        f"Fault continuation at bus '{self.bus_name}' did not "
                        "converge even at minimum step size.")
                continue
            progress = next_progress
            step_size = min(max_step, step_size * 1.5)

    def _snapshot(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (self.model.x.copy(), self.model.y_run.copy(),
                self.model.p_vals.copy(), self.model.u_run_vals.copy())

    def _restore(self, snapshot):
        self.model.x[:], self.model.y_run[:], self.model.p_vals[:], self.model.u_run_vals[:] = (
            s.copy() for s in snapshot)

    def _report(self, msg: str):
        if self.verbose:
            print(f"FaultManager[{self.bus_name}]: {msg}")


# ---------------------------------------------------------------------- #
# Converter response callbacks (particular behaviour, injected into
# FaultManager.apply/clear). These live outside FaultManager on purpose.
# ---------------------------------------------------------------------- #
def lvrt_response(devices, v_lvrt: float = 0.85, iq=0.5, i_active: float = 0.0,
                  gain: float = 0.5) -> Callable:
    """Standard low-voltage ride-through response for grid-following converters.

    Returns an ``on_step(model) -> max_change`` callback for
    :meth:`FaultManager.apply` / :meth:`FaultManager.clear`. Each converter
    (any element exposing ``lvrt_ext_<name>`` / ``i_sr_ref_<name>`` /
    ``i_sa_ref_<name>``) enters LVRT -- bounded current injection -- when its own
    terminal voltage is at or below ``v_lvrt``, and returns to PQ above it. The
    reactive current reference is ``iq``: a scalar, or a callable grid-code curve
    ``iq(V)`` (e.g. Spanish REE PO 12.3). ``i_active`` is the active current in
    LVRT. ``gain`` damps the fixed-point relaxation FaultManager runs each step.

    Because the converter references are reached through FaultManager's
    continuation (not a time-domain ramp), this converges for any fault depth,
    including deep faults where the constant-power current would otherwise
    diverge.
    """
    def _iq_of(V):
        return iq(V) if callable(iq) else iq

    def on_step(model) -> float:
        max_change = 0.0
        for bus in devices:
            V = model.get_value(f'V_{bus}')
            mode_tgt = 1.0 if V <= v_lvrt else 0.0
            iq_tgt = _iq_of(V) if mode_tgt else 0.0
            le = model.get_value(f'lvrt_ext_{bus}')
            iqc = model.get_value(f'i_sr_ref_{bus}')
            new_le = le + gain * (mode_tgt - le)
            new_iq = iqc + gain * (iq_tgt - iqc)
            max_change = max(max_change, abs(new_le - le), abs(new_iq - iqc))
            model.set_value(f'lvrt_ext_{bus}', new_le)
            model.set_value(f'i_sr_ref_{bus}', new_iq)
            model.set_value(f'i_sa_ref_{bus}', i_active)
        return max_change

    return on_step


def ted749_response(devices, K_1=2.0, i_lim_up=0.9, i_lim_lo=-0.9,
                    V_min=0.85, V_max=1.15, i_max=1.0, gain=0.5) -> Callable:
    r"""Orden TED/749/2020 fast-current-injection response, as an ``on_step``.

    The continuation-side twin of
    :class:`~pydae.bps.utils.frt.Ted749Controller`: the same law of apartado
    2.3.1 / figura 8, relaxed to a fixed point by :meth:`FaultManager.apply` /
    :meth:`FaultManager.clear` instead of rate-limited over time steps. Use it
    in place of :func:`lvrt_response` whenever the time-domain relay is the
    TED/749 one, so the operating point the continuation reaches is the one the
    relay will then hold.

    Two properties of the order's law make it converge where the PO 12.3 curve
    stalls, both because it is a *continuous* control rather than a dead-band
    curve behind a latch:

    * There is no binary threshold for the relaxation to chatter across. A bare
      ``V <= v_lvrt`` test makes converters flip in and out of ride-through
      between relaxation sweeps, which stalls the continuation at the threshold.
    * The converter is never commanded to inject exactly zero current. At
      ``V == V_pre`` the law returns the pre-disturbance current, and the active
      current keeps the apparent-current headroom (apartado 3.3.1). A converter
      at exactly zero current has an unobservable terminal angle, i.e. a
      structurally singular ``Gy``.

    The pre-disturbance operating point (``U1i``, ``I1i`` and the active
    current) is captured on the first call, so build one callback per event,
    before ``apply()``, and reuse that same object for the matching ``clear()``.
    """
    pre: Dict[str, Tuple[float, float, float]] = {}

    def on_step(model) -> float:
        max_change = 0.0
        for bus in devices:
            V = model.get_value(f'V_{bus}')
            if bus not in pre:
                # First sweep: latch the operating point the model came in with.
                iq0 = model.get_value(f'q_s_{bus}') / V if V > 0.0 else 0.0
                ia0 = model.get_value(f'p_s_{bus}') / V if V > 0.0 else 0.0
                pre[bus] = (V, iq0, ia0)
            v_pre, iq_pre, ia_pre = pre[bus]

            # Arming is re-evaluated every sweep and is deliberately NOT
            # latched. Latching it stalls the continuation at the V_min
            # boundary: the relaxation then has to carry lvrt_ext from 0 to 1
            # with no way back, and the algebraic solve cannot cross that mode
            # switch in one step. Letting it fall back makes the boundary a
            # damped fixed point the continuation walks through.
            armed = (not (V_min <= V <= V_max)
                     or model.get_value(f'lvrt_ext_{bus}') > 0.5)
            if armed:
                iq_tgt = min(max(iq_pre + K_1 * (v_pre - V), i_lim_lo), i_lim_up)
                ia_tgt = min(ia_pre, max(i_max ** 2 - iq_tgt ** 2, 0.0) ** 0.5)
                mode_tgt = 1.0
            else:
                iq_tgt = ia_tgt = mode_tgt = 0.0

            le = model.get_value(f'lvrt_ext_{bus}')
            iqc = model.get_value(f'i_sr_ref_{bus}')
            iac = model.get_value(f'i_sa_ref_{bus}')
            new_le = le + gain * (mode_tgt - le)
            new_iq = iqc + gain * (iq_tgt - iqc)
            new_ia = iac + gain * (ia_tgt - iac)
            max_change = max(max_change, abs(new_le - le), abs(new_iq - iqc),
                             abs(new_ia - iac))
            model.set_value(f'lvrt_ext_{bus}', new_le)
            model.set_value(f'i_sr_ref_{bus}', new_iq)
            model.set_value(f'i_sa_ref_{bus}', new_ia)
        return max_change

    return on_step


def apply_fault(model, bus_name, t_clear, lvrt_iq=0.5, lvrt_id=0.0, v_lvrt=0.85,
                fault_steps=100, verbose=True, **_ignored):
    r"""Convenience wrapper: apply a shunt fault, hold it ``t_clear`` seconds and
    clear it, with the standard grid-following LVRT response (:func:`lvrt_response`).

    Applies at the current model time, holds (advancing ``model.run``), then
    clears; the caller resumes afterwards. ``lvrt_iq`` may be a scalar or a
    callable ``iq(V)`` curve. For finer control (custom response, caller-owned
    hold), use :class:`FaultManager` with ``apply(on_step=...)`` /
    ``clear(on_step=...)`` directly.
    """
    devices = [n[len('lvrt_ext_'):] for n in model.u_run_names
               if n.startswith('lvrt_ext_')]
    response = lvrt_response(devices, v_lvrt=v_lvrt, iq=lvrt_iq, i_active=lvrt_id)
    manager = FaultManager(model, bus_name, fault_steps=fault_steps, verbose=verbose)

    t_incept = model.t
    manager.apply(on_step=response)
    hold_cmd = {}
    for bus in devices:
        hold_cmd[f'lvrt_ext_{bus}'] = model.get_value(f'lvrt_ext_{bus}')
        hold_cmd[f'i_sr_ref_{bus}'] = model.get_value(f'i_sr_ref_{bus}')
        hold_cmd[f'i_sa_ref_{bus}'] = model.get_value(f'i_sa_ref_{bus}')
    model.run(t_incept + t_clear, hold_cmd)
    manager.clear(on_step=response)
    return hold_cmd


def fault_admittance(R_pu: float, X_pu: float, S_mva: float,
                     S_base: float) -> Tuple[float, float]:
    r"""System-base shunt admittance ``(g, b)`` of a fault :math:`Z_f = R_{pu} + jX_{pu}`.

    ``R_pu`` / ``X_pu`` are per unit on the fault's own ``S_mva`` base; ``S_base``
    is the system base (VA). Mirrors the computation ``BpsBuilder`` does for the
    declared fault, so an overridden reactance produces a consistent admittance.
    """
    S_line = 1e6 * S_mva
    R = R_pu * S_base / S_line
    X = X_pu * S_base / S_line
    d = R ** 2 + X ** 2
    return R / d, -X / d


class FaultController:
    r"""Live-model fault apply/clear, for event-driven and real-time simulation.

    Wraps :class:`FaultManager` + :func:`lvrt_response` so a fault can be switched
    on/off on a model *in place*, without owning the time loop -- unlike
    :func:`apply_fault`, which advances ``model.run`` itself. This is the object
    injected into the real-time API as ``app.state.faults``: its :meth:`apply` /
    :meth:`clear` are called between integrator chunks (under the API's lock),
    each running the robust frozen-time continuation on the current operating
    point and driving the grid-following converters' LVRT response.

    The converters are discovered once (any unit exposing ``lvrt_ext_<name>`` --
    pv_pq_ss inverters and bess_pq_ss VSCs). ``iq`` is the reactive-current
    reference reached through the continuation: a scalar, or a callable grid-code
    curve ``iq(V)`` (e.g. ``pydae.bps.pvs.pv_pq_ss.iq_po123``). One
    :class:`FaultManager` is cached per bus.

    ``law`` selects the grid code: ``"po123"`` (default, :func:`lvrt_response`)
    or ``"ted749"`` (:func:`ted749_response`, Orden TED/749/2020). Pair it with
    the matching time-domain relay --
    ``DiscreteOrchestrator.from_model(model, law="ted749")`` -- so the
    continuation and the relay follow the same law. ``law_kwargs`` are forwarded
    to the response builder (``K_1``, ``i_lim_up`` ...).

    This is duck-typed for the API (methods ``apply(bus)`` / ``clear(bus)`` and a
    ``buses`` property); it holds no time loop and no lock of its own -- the
    caller owns both.
    """
    def __init__(self, model, iq=0.5, v_lvrt: float = 0.85,
                 i_active: float = 0.0, gain: float = 0.5,
                 fault_steps: int = 20, verbose: bool = False,
                 law: str = "po123", **law_kwargs):
        self.model = model
        self.iq = iq
        self.v_lvrt = v_lvrt
        self.i_active = i_active
        self.gain = gain
        self.fault_steps = fault_steps
        self.verbose = verbose
        self.law = str(law).lower()
        self.law_kwargs = law_kwargs
        self.converters = [n[len('lvrt_ext_'):] for n in
                           getattr(model, 'u_run_names', [])
                           if n.startswith('lvrt_ext_')]
        self._managers: Dict[str, FaultManager] = {}
        # TED/749's callback latches the pre-disturbance operating point on its
        # first call, so the apply and its matching clear must share one object;
        # PO 12.3's is stateless and can be rebuilt per call.
        self._responses: Dict[str, Callable] = {}
        self.active: set = set()

    @property
    def buses(self):
        """Declared fault buses (from ``model.sys_dict['faults']``)."""
        return sorted(getattr(self.model, 'sys_dict', {}).get('faults', {}).keys())

    def _manager(self, bus: str) -> FaultManager:
        mgr = self._managers.get(bus)
        if mgr is None:
            # Raises ValueError for a bus with no declared fault.
            mgr = FaultManager(self.model, bus, fault_steps=self.fault_steps,
                               verbose=self.verbose)
            self._managers[bus] = mgr
        return mgr

    def _response(self, bus: str, iq=None, v_lvrt=None,
                  new: bool = True) -> Optional[Callable]:
        """The converter response for this event, per the configured ``law``.

        For ``"ted749"`` the callback is cached per bus so the clear reuses the
        object the apply latched its pre-disturbance operating point into --
        rebuilding it at clear time would latch the *faulted* voltage as
        ``U1i`` and drive the converters to the wrong reference.
        """
        if not self.converters:
            return None
        if self.law in ('ted749', 'ted', 'ted/749/2020', 'boe-a-2020-8965'):
            if new or bus not in self._responses:
                self._responses[bus] = ted749_response(self.converters,
                                                       **self.law_kwargs)
            return self._responses[bus]
        return lvrt_response(self.converters,
                             v_lvrt=self.v_lvrt if v_lvrt is None else v_lvrt,
                             iq=self.iq if iq is None else iq,
                             i_active=self.i_active, gain=self.gain)

    def _n_in_lvrt(self) -> int:
        # Plain int (not numpy) so callers can JSON-serialize the status dict.
        return int(sum(bool(self.model.get_value(f'lvrt_ext_{bus}') > 0.5)
                       for bus in self.converters))

    def _fault_target(self, bus: str, X_pu: float) -> Dict[str, float]:
        """Fault admittance handles for ``bus`` at an overridden reactance."""
        ft = getattr(self.model, 'sys_dict', {}).get('faults', {}).get(bus, {})
        R_pu = ft.get('R_pu', 0.0005)
        S_mva = ft.get('S_mva', 100.0)
        S_base = ft.get('S_base') or getattr(self.model, 'sys_dict', {}).get(
            'params_dict', {}).get('S_base', 100e6)
        g, b = fault_admittance(R_pu, X_pu, S_mva, S_base)
        return {f'g_shunt_{bus}': g, f'b_shunt_{bus}': b}

    def apply(self, bus: str, X_fault=None, fault_steps=None, iq=None,
              v_lvrt=None, tol=None, relax_iters=None, settle=None,
              **_ignored) -> dict:
        """Apply the fault at ``bus`` (frozen-time continuation) with LVRT.

        ``X_fault`` optionally overrides the declared fault reactance (pu on the
        fault's ``S_mva`` base) — a larger value is a softer fault. ``fault_steps``
        overrides the continuation resolution; a soft fault converges in fewer
        steps, so the under-lock pause (and any real-time overrun) shrinks.
        ``iq`` / ``v_lvrt`` optionally override the LVRT response.

        ``tol`` / ``relax_iters`` / ``settle`` tune how much work each
        continuation step does, as opposed to how many steps there are. They
        were previously reachable only by calling :class:`FaultManager`
        directly.
        """
        target = None if X_fault is None else self._fault_target(bus, X_fault)
        cont_kw = {k: v for k, v in (('tol', tol), ('relax_iters', relax_iters),
                                     ('settle', settle)) if v is not None}
        self._manager(bus).apply(
            on_step=self._response(bus, iq=iq, v_lvrt=v_lvrt, new=True),
            target=target, fault_steps=fault_steps, **cont_kw)
        self.active.add(bus)
        return {'bus': bus, 'converters_in_lvrt': self._n_in_lvrt(),
                'n_converters': len(self.converters),
                'X_fault': X_fault, 'fault_steps': fault_steps or self.fault_steps}

    def clear(self, bus: str, fault_steps=None, iq=None, v_lvrt=None,
              tol=None, relax_iters=None, settle=None, **_ignored) -> dict:
        """Clear the fault at ``bus`` (frozen-time continuation).

        ``fault_steps`` optionally overrides the continuation resolution;
        ``tol`` / ``relax_iters`` / ``settle`` are as in :meth:`apply`.
        """
        cont_kw = {k: v for k, v in (('tol', tol), ('relax_iters', relax_iters),
                                     ('settle', settle)) if v is not None}
        self._manager(bus).clear(
            on_step=self._response(bus, iq=iq, v_lvrt=v_lvrt, new=False),
            fault_steps=fault_steps, **cont_kw)
        self._responses.pop(bus, None)
        self.active.discard(bus)
        return {'bus': bus, 'converters_in_lvrt': self._n_in_lvrt(),
                'n_converters': len(self.converters)}
