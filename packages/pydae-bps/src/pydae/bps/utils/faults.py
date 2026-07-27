# -*- coding: utf-8 -*-
r"""
Shunt-fault application/clearing with grid-following converter ride-through.

A **fault** is a balanced shunt impedance $Z_f = R_{pu} + jX_{pu}$ switched onto
a bus (declared in a ``"faults"`` section of the network data, which
``BpsBuilder`` turns into an off shunt plus a target admittance in
``model.sys_dict['faults']``). This module drives the event and manages the
low-voltage ride-through (LVRT) of every grid-following converter that exposes
the ``lvrt_ext_<name>`` / ``i_sr_ref_<name>`` / ``i_sa_ref_<name>`` interface
(``pv_pq_ss`` inverters, ``bess_pq_ss`` VSCs, ...).

Two entry points:

* :class:`FaultManager` -- decoupled ``apply()`` / ``clear()`` for event-driven
  simulation, where the caller owns the hold integration between the two.
* :func:`apply_fault` -- a convenience wrapper that applies, holds for
  ``t_clear`` and clears in one call.

Why this is not a single Newton solve
-------------------------------------
Grid-following converters are **constant power** in normal (PQ) operation, so
their AC current $i_s \sim S/V$ diverges as the faulted bus voltage collapses
and the algebraic Jacobian goes singular. Applying a deep fault is therefore a
*continuation* (fault admittance ramped in, algebraic system re-solved at every
step, states frozen) during which the converters switch to *bounded current
injection* ($p_s = V\,i_{sa,ref}$, $q_s = V\,i_{sr,ref}$) via ``lvrt_ext``. Two
relay strategies decide each converter's ``lvrt_ext``:

* **hard** (default): a one-way, latched per-device relay keyed on ``v_lvrt``.
* **soft** (``lvrt_soft=True``): ``lvrt_ext`` follows a continuous ramp of the
  unit's own undervoltage, solved as a damped fixed point inside each step, so
  it crosses the intermediate fault-severity band where the hard relay's
  simultaneous (symmetric-plant) switch folds the solve.
"""
import numpy as np
from typing import Any, Callable, Dict, Optional, Tuple


class FaultManager:
    """
    Manages the discrete application and clearing of a shunt fault.
    Handles the low-voltage ride-through (LVRT) continuation solvers.
    """
    def __init__(self, model: Any, bus_name: str, lvrt_iq: float = 0.5,
                 lvrt_id: float = 0.0, v_lvrt: float = 0.5, lvrt_soft: bool = False,
                 lvrt_band: float = 0.2, fault_steps: int = 100, verbose: bool = True):
        self.model = model
        self.bus_name = bus_name
        self.lvrt_iq = lvrt_iq
        self.lvrt_id = lvrt_id
        self.v_lvrt = v_lvrt
        self.lvrt_soft = lvrt_soft
        self.lvrt_band = lvrt_band
        self.fault_steps = fault_steps
        self.verbose = verbose

        # 1. Validate and Setup Fault Targets
        faults = getattr(model, 'sys_dict', {}).get('faults', {})
        if bus_name not in faults:
            raise ValueError(
                f"No fault declared at bus '{bus_name}'. Declared fault buses: "
                f"{sorted(faults.keys())}. Add a 'faults' section to the network data."
            )

        self.g_fault = faults[bus_name]['g']
        self.b_fault = faults[bus_name]['b']

        g_name = f'g_shunt_{bus_name}'
        b_name = f'b_shunt_{bus_name}'
        self.fault_on = {g_name: self.g_fault, b_name: self.b_fault}
        self.fault_off = {g_name: 0.0, b_name: -1e-6}

        # 2. Discover Converters
        self.devices = [
            name[len('lvrt_ext_'):] for name in model.u_run_names
            if name.startswith('lvrt_ext_')
        ]

        # 3. Pre-compute LVRT commands for all devices
        self.lvrt_on = {}
        self.lvrt_off = {}
        for device in self.devices:
            self.lvrt_on.update(self._get_lvrt_command(device, 1.0))
            self.lvrt_off.update(self._get_lvrt_command(device, 0.0))

    # ------------------------------------------------------------------ #
    # Public API: Apply and Clear
    # ------------------------------------------------------------------ #
    def apply(self) -> Dict[str, float]:
        """
        Applies the fault instantaneously (frozen time) via continuation.
        Returns the command dictionary required to hold the units in LVRT during integration.
        """
        if self.lvrt_soft:
            return self._apply_soft()
        else:
            return self._apply_hard()

    def clear(self):
        """
        Clears the fault instantaneously (frozen time) and returns converters to PQ mode.
        """
        if self.lvrt_soft:
            self._clear_soft()
        else:
            self._clear_hard()

    # ------------------------------------------------------------------ #
    # Internal Split Logic: Hard Relay
    # ------------------------------------------------------------------ #
    def _apply_hard(self) -> Dict[str, float]:
        mode = self._apply_fault_hard_relay()
        hold_cmd = {}
        for device in self.devices:
            hold_cmd.update(self._get_lvrt_command(device, mode[device]))
        return hold_cmd

    def _clear_hard(self):
        self._clear_fault_hard_relay()

    # ------------------------------------------------------------------ #
    # Internal Split Logic: Soft Relay
    # ------------------------------------------------------------------ #
    def _apply_soft(self) -> Dict[str, float]:
        for device in self.devices:
            self.model.set_value(f'i_sr_ref_{device}', self.lvrt_iq)
            self.model.set_value(f'i_sa_ref_{device}', self.lvrt_id)

        self._soft_continue(self.fault_on, self.fault_steps)

        hold_cmd = {}
        mode = {}
        for device in self.devices:
            lvrt_val = self.model.get_value(f'lvrt_ext_{device}')
            mode[device] = lvrt_val
            hold_cmd[f'lvrt_ext_{device}'] = lvrt_val
            hold_cmd[f'i_sr_ref_{device}'] = self.lvrt_iq
            hold_cmd[f'i_sa_ref_{device}'] = self.lvrt_id

        active_count = sum(1 for val in mode.values() if val > 0.5)
        self._report(
            f'LVRT soft relay: {active_count}/{len(self.devices)} devices >50% into LVRT '
            f'(lvrt_ext range {min(mode.values()):.2f}-{max(mode.values()):.2f})'
        )
        return hold_cmd

    def _clear_soft(self):
        self._soft_continue(self.fault_off, self.fault_steps, tol=1e-5)
        self._continue_to(self.lvrt_off, nominal_steps=8)

    # ------------------------------------------------------------------ #
    # Mathematical Solvers & Continuation
    # ------------------------------------------------------------------ #
    def _apply_fractional_targets(self, targets: Dict[str, float], progress: float, start_values: Dict[str, float]):
        for name, target_val in targets.items():
            current_start = start_values[name]
            self.model.set_value(name, current_start + progress * (target_val - current_start))

    def _adaptive_continue(self, targets: Dict[str, float], nominal_steps: int,
                           tol: float = 1e-8, on_step: Optional[Callable] = None):
        start_values = {name: self.model.get_value(name) for name in targets}

        progress = 0.0
        step_size = 1.0 / nominal_steps
        max_step_size = step_size

        while progress < 1.0 - 1e-9:
            next_progress = min(1.0, progress + step_size)
            self._apply_fractional_targets(targets, next_progress, start_values)

            try:
                self.model.recalculate_algebraics(tol=tol)
            except RuntimeError:
                self._apply_fractional_targets(targets, progress, start_values)
                step_size *= 0.5
                if step_size < 1e-7:
                    raise RuntimeError("Solver failed to converge even at minimum step size.")
                continue

            progress = next_progress
            if on_step is not None:
                on_step()
            step_size = min(max_step_size, step_size * 1.5)

    def _continue_to(self, targets: Dict[str, float], nominal_steps: int, tol: float = 1e-8):
        self._adaptive_continue(targets, nominal_steps, tol)

    def _soft_continue(self, targets: Dict[str, float], nominal_steps: int, tol: float = 1e-8):
        start_values = {name: self.model.get_value(name) for name in targets}

        progress = 0.0
        step_size = 1.0 / nominal_steps
        max_step_size = step_size

        while progress < 1.0 - 1e-9:
            next_progress = min(1.0, progress + step_size)
            snapshot = self._snapshot_state()
            self._apply_fractional_targets(targets, next_progress, start_values)

            try:
                self.model.recalculate_algebraics(tol=tol)
                for _ in range(20):
                    if self._relax_lvrt(tol=tol) < 1e-5:
                        break
            except RuntimeError:
                self._restore_state(snapshot)
                step_size *= 0.5
                if step_size < 1e-7:
                    raise RuntimeError("Soft continuation solver failed.")
                continue

            progress = next_progress
            step_size = min(max_step_size, step_size * 1.5)

    # ------------------------------------------------------------------ #
    # Relay Logic (Hard / Soft)
    # ------------------------------------------------------------------ #
    def _apply_fault_hard_relay(self) -> Dict[str, float]:
        snapshot = self._snapshot_state()
        try:
            combined_targets = {**self.fault_on, **self.lvrt_on}
            self._adaptive_continue(combined_targets, self.fault_steps)
        except RuntimeError:
            self._restore_state(snapshot)
            self._adaptive_continue(self.fault_on, self.fault_steps)

        mode = {device: self.model.get_value(f'lvrt_ext_{device}') for device in self.devices}
        for _ in range(len(self.devices) + 1):
            demote_cmd = {}
            for device in self.devices:
                if mode[device] and self.model.get_value(f'V_{device}') > self.v_lvrt:
                    mode[device] = 0.0
                    demote_cmd.update(self._get_lvrt_command(device, 0.0))

            if not demote_cmd:
                break
            self._continue_to(demote_cmd, nominal_steps=12)

        active_lvrt = int(sum(mode.values()))
        self._report(f'LVRT relay: {active_lvrt}/{len(self.devices)} devices in LVRT, '
                     f'{len(self.devices) - active_lvrt} in PQ (threshold V = {self.v_lvrt})')
        return mode

    def _clear_fault_hard_relay(self):
        v_relax = self.v_lvrt - 0.1

        def relay_off():
            switch_cmd = {}
            for device in self.devices:
                if (self.model.get_value(f'lvrt_ext_{device}') and
                        self.model.get_value(f'V_{device}') > v_relax):
                    switch_cmd.update(self._get_lvrt_command(device, 0.0))
            if switch_cmd:
                self._continue_to(switch_cmd, nominal_steps=8)

        snapshot = self._snapshot_state()
        try:
            combined_targets = {**self.fault_off, **self.lvrt_off}
            self._adaptive_continue(combined_targets, self.fault_steps, tol=1e-5)
        except RuntimeError:
            self._restore_state(snapshot)
            self._adaptive_continue(self.fault_off, self.fault_steps, tol=1e-5, on_step=relay_off)
            self._continue_to(self.lvrt_off, nominal_steps=8)

    def _relax_lvrt(self, tol: float = 1e-8) -> float:
        max_change = 0.0
        for device in self.devices:
            V = self.model.get_value(f'V_{device}')
            target = min(1.0, max(0.0, (self.v_lvrt - V) / self.lvrt_band))
            current = self.model.get_value(f'lvrt_ext_{device}')

            new_val = current + 0.5 * (target - current)
            max_change = max(max_change, abs(new_val - current))
            self.model.set_value(f'lvrt_ext_{device}', new_val)

        self.model.recalculate_algebraics(tol=tol)
        return max_change

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def _get_lvrt_command(self, device_name: str, mode: float) -> Dict[str, float]:
        return {
            f'lvrt_ext_{device_name}': mode,
            f'i_sr_ref_{device_name}': self.lvrt_iq if mode else 0.0,
            f'i_sa_ref_{device_name}': self.lvrt_id if mode else 0.0,
        }

    def _snapshot_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.model.x.copy(),
            self.model.y_run.copy(),
            self.model.p_vals.copy(),
            self.model.u_run_vals.copy()
        )

    def _restore_state(self, snapshot: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        self.model.x[:], self.model.y_run[:], self.model.p_vals[:], self.model.u_run_vals[:] = (
            s.copy() for s in snapshot
        )

    def _report(self, msg: str):
        if self.verbose:
            print(msg)


def apply_fault(model, bus_name, t_clear,
                lvrt_iq=0.5, lvrt_id=0.0, v_lvrt=0.5,
                lvrt_soft=False, lvrt_band=0.2, fault_steps=100, verbose=True):
    r"""Apply, hold and clear a shunt fault in one call (convenience wrapper).

    Thin wrapper around :class:`FaultManager`: applies the fault at the current
    model time (frozen-time continuation), holds it for ``t_clear`` seconds
    (``model.run`` is advanced), then clears it. The caller resumes the
    post-fault simulation afterwards. For finer control over the hold interval,
    use :class:`FaultManager` and call ``apply()`` / ``clear()`` directly.

    Parameters
    ----------
    model : CasadiModel
        Initialised model built from data containing a ``"faults"`` section.
    bus_name : str
        Bus at which the fault is applied (must match a declared fault).
    t_clear : float
        Fault duration in seconds.
    lvrt_iq, lvrt_id, v_lvrt, lvrt_soft, lvrt_band, fault_steps, verbose
        Forwarded to :class:`FaultManager`.

    Returns
    -------
    dict
        The per-device hold command (``lvrt_ext`` / ``i_sr_ref`` / ``i_sa_ref``)
        applied during the fault.
    """
    manager = FaultManager(model, bus_name, lvrt_iq=lvrt_iq, lvrt_id=lvrt_id,
                           v_lvrt=v_lvrt, lvrt_soft=lvrt_soft, lvrt_band=lvrt_band,
                           fault_steps=fault_steps, verbose=verbose)
    t_incept = model.t
    hold_cmd = manager.apply()
    model.run(t_incept + t_clear, hold_cmd)
    manager.clear()
    return hold_cmd
