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
    def __init__(self, model: Any, bus_name: str, fault_steps: int = 100,
                 verbose: bool = True):
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
    def apply(self, on_step: Optional[Callable] = None):
        """Apply the fault (frozen time) by ramping the admittance to Z_f."""
        self._continue(self.fault_on, on_step=on_step, tol=1e-8)
        self._report('fault applied')

    def clear(self, on_step: Optional[Callable] = None):
        """Clear the fault (frozen time) by ramping the admittance back out."""
        self._continue(self.fault_off, on_step=on_step, tol=1e-5)
        self._report('fault cleared')

    # ------------------------------------------------------------------ #
    # Continuation machinery
    # ------------------------------------------------------------------ #
    def _continue(self, target: Dict[str, float], on_step: Optional[Callable] = None,
                  tol: float = 1e-8, relax_iters: int = 30, settle: float = 1e-5):
        # Adaptive continuation of the fault admittance from its current value to
        # ``target``. Each step is halved on failure and grown on success. When an
        # ``on_step`` callback is given it is relaxed to a fixed point inside every
        # step: the callback reads the operating point, nudges its references, and
        # returns the max change; the loop re-solves and repeats until it settles.
        start = {name: self.model.get_value(name) for name in target}

        def set_frac(f):
            for name, tgt in target.items():
                self.model.set_value(name, start[name] + f * (tgt - start[name]))

        progress, step_size, max_step = 0.0, 1.0 / self.fault_steps, 1.0 / self.fault_steps
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
