# -*- coding: utf-8 -*-
r"""
Shunt-fault application/clearing with grid-following converter ride-through.

``apply_fault`` drives a balanced shunt fault at a bus through to clearing,
managing the low-voltage ride-through (LVRT) of every grid-following converter
in the model (``pv_pq_ss`` inverters, ``bess_pq_ss`` VSCs, ... any unit that
exposes the ``lvrt_ext_<name>`` / ``i_sr_ref_<name>`` / ``i_sa_ref_<name>``
interface).

The fault itself is a shunt impedance ``Z_f = R_pu + j X_pu`` declared in the
network data under a ``"faults"`` section, which ``BpsBuilder`` turns into an
*off* shunt at the bus (exposing ``g_shunt_<bus>`` / ``b_shunt_<bus>``) plus a
target admittance recorded in ``model.sys_dict['faults']``.

Why this is not a single Newton solve
--------------------------------------
The converters are constant-power in normal (PQ) operation, so their AC current
``~ S/V`` diverges as the bus voltage collapses and the algebraic Jacobian goes
singular. To ride through, each unit can switch to *bounded current injection*
(``p_s = V i_sa_ref``, ``q_s = V i_sr_ref``) via ``lvrt_ext``. Applying the
fault is therefore a continuation in which the fault admittance is ramped in
while the converters move into LVRT, with the algebraic system re-solved at
every step (states frozen). Two relay strategies are provided:

* **hard** (default): a one-way, latched per-device relay. Every unit enters
  LVRT together with the fault; once a unit's terminal recovers above
  ``v_lvrt`` it latches back to PQ and never returns. Matches a hard
  undervoltage relay threshold.
* **soft** (``lvrt_soft=True``): each unit's ``lvrt_ext`` follows a continuous
  ramp of its own undervoltage, ``clip((v_lvrt - V)/lvrt_band, 0, 1)``, solved
  as a damped fixed point inside each step. No mode discontinuity, so it crosses
  the intermediate fault-severity band where the hard relay's simultaneous
  (symmetric-plant) switch folds the solve.
"""
import numpy as np


def apply_fault(model, bus_name, t_clear,
                lvrt_iq=0.5, lvrt_id=0.0, v_lvrt=0.5,
                lvrt_soft=False, lvrt_band=0.2, fault_steps=100,
                verbose=True):
    r"""Apply, hold and clear a shunt fault, managing converter LVRT.

    The model is assumed to be initialised and time-stepped up to fault
    inception (``model.run(t_fault, ...)`` already done). This function applies
    the fault instantaneously (algebraic continuation at frozen time), holds it
    for ``t_clear`` seconds (``model.run`` is advanced), then clears it
    instantaneously. The caller resumes the post-fault simulation afterwards.

    Parameters
    ----------
    model : CasadiModel
        Model built from data that contains a ``"faults"`` section; the fault
        target admittance is read from ``model.sys_dict['faults'][bus_name]``.
    bus_name : str
        Bus at which the fault is applied (must match a declared fault).
    t_clear : float
        Fault duration in seconds (the shunt is removed at ``t + t_clear``).
    lvrt_iq, lvrt_id : float, optional
        Reactive / active current references commanded in LVRT (pu). Default
        ``0.5`` reactive (voltage-supporting), ``0.0`` active.
    v_lvrt : float, optional
        Terminal-voltage threshold separating LVRT from PQ operation.
    lvrt_soft : bool, optional
        Use the continuous (soft) relay instead of the hard latched relay.
    lvrt_band : float, optional
        Voltage band of the soft relay (full LVRT at ``v_lvrt - lvrt_band``).
    fault_steps : int, optional
        Nominal number of continuation steps for the fault ramp (adaptive).
    verbose : bool, optional
        Print a one-line summary of the resolved per-device modes.

    Returns
    -------
    dict
        Resolved per-device ``lvrt_ext`` values at the fault hold.
    """
    faults = getattr(model, 'sys_dict', {}).get('faults', {})
    if bus_name not in faults:
        raise ValueError(
            f"No fault declared at bus '{bus_name}'. Declared fault buses: "
            f"{sorted(faults)}. Add a 'faults' section to the network data.")
    g_fault = faults[bus_name]['g']
    b_fault = faults[bus_name]['b']
    g_name = f'g_shunt_{bus_name}'
    b_name = f'b_shunt_{bus_name}'
    fault_on = {g_name: g_fault, b_name: b_fault}
    fault_off = {g_name: 0.0, b_name: -1e-6}

    # Discover the grid-following converters from the LVRT interface.
    devices = [n[len('lvrt_ext_'):] for n in model.u_run_names
               if n.startswith('lvrt_ext_')]

    def lvrt_cmd(bus, mode):
        # mode 1.0 -> LVRT (bounded current injection); 0.0 -> normal PQ.
        return {
            f'lvrt_ext_{bus}': mode,
            f'i_sr_ref_{bus}': lvrt_iq if mode else 0.0,
            f'i_sa_ref_{bus}': lvrt_id if mode else 0.0,
        }

    lvrt_on = {k: v for b in devices for k, v in lvrt_cmd(b, 1.0).items()}
    lvrt_off = {k: v for b in devices for k, v in lvrt_cmd(b, 0.0).items()}

    def report(msg):
        if verbose:
            print(msg)

    # ------------------------------------------------------------------ #
    # Continuation primitives
    # ------------------------------------------------------------------ #
    def adaptive_continue(targets, steps, tol=1e-8, on_step=None):
        # Walk {param: target} from current values to the targets, halving the
        # step on a failed solve and growing it again on success, so fine
        # resolution is spent only where the operating point is stiff.
        start = {name: model.get_value(name) for name in targets}

        def set_frac(f):
            for name, target in targets.items():
                model.set_value(name, start[name] + f*(target - start[name]))

        f, df, df_max = 0.0, 1.0/steps, 1.0/steps
        while f < 1.0 - 1e-9:
            nf = min(1.0, f + df)
            set_frac(nf)
            try:
                model.recalculate_algebraics(tol=tol)
            except RuntimeError:
                set_frac(f)
                df *= 0.5
                if df < 1e-7:
                    raise
                continue
            f = nf
            if on_step is not None:
                on_step()
            df = min(df_max, df*1.5)

    def continue_to(targets, steps, tol=1e-8):
        adaptive_continue(targets, steps, tol=tol)

    def snapshot():
        return (model.x.copy(), model.y_run.copy(),
                model.p_vals.copy(), model.u_run_vals.copy())

    def restore(snap):
        model.x[:], model.y_run[:], model.p_vals[:], model.u_run_vals[:] = (
            s.copy() for s in snap)

    # ------------------------------------------------------------------ #
    # Hard latched relay
    # ------------------------------------------------------------------ #
    def apply_fault_with_lvrt(steps):
        # Step 1: bring the plant into the faulted state. Try ramping the fault
        # in with all converters going into LVRT together (bounded current, good
        # conditioning for hard faults); if that folds -- milder faults, where
        # reactive injection at healthy terminals overvolts -- fall back to plain
        # PQ.
        snap = snapshot()
        try:
            adaptive_continue({**fault_on, **lvrt_on}, steps)
        except RuntimeError:
            restore(snap)
            adaptive_continue(fault_on, steps)

        # Step 2: demote-only per-device relay. Keep LVRT only while the unit's
        # own terminal is at/below v_lvrt; latch any recovered unit back to PQ.
        # Monotone (each unit switches at most once), so it always terminates.
        mode = {b: model.get_value(f'lvrt_ext_{b}') for b in devices}
        for _ in range(len(devices) + 1):
            demote = {}
            for b in devices:
                if mode[b] and model.get_value(f'V_{b}') > v_lvrt:
                    mode[b] = 0.0
                    demote.update(lvrt_cmd(b, 0.0))
            if not demote:
                break
            continue_to(demote, steps=12)
        n_lvrt = int(sum(mode.values()))
        report(f'LVRT relay: {n_lvrt}/{len(devices)} devices in LVRT, '
               f'{len(devices) - n_lvrt} in PQ (threshold V = {v_lvrt})')
        return mode

    def clear_fault_hard():
        v_relax = v_lvrt - 0.1

        def relay_off():
            switch = {}
            for b in devices:
                if (model.get_value(f'lvrt_ext_{b}')
                        and model.get_value(f'V_{b}') > v_relax):
                    switch.update(lvrt_cmd(b, 0.0))
            if switch:
                continue_to(switch, steps=8)

        # Symmetric single-mode plants clear via the proportional reverse
        # co-ramp; mixed-mode states need the latched relay. Try co-ramp first.
        snap = snapshot()
        try:
            adaptive_continue({**fault_off, **lvrt_off}, fault_steps, tol=1e-5)
        except RuntimeError:
            restore(snap)
            adaptive_continue(fault_off, fault_steps, tol=1e-5,
                              on_step=relay_off)
            continue_to(lvrt_off, steps=8)

    # ------------------------------------------------------------------ #
    # Soft continuous relay
    # ------------------------------------------------------------------ #
    def relax_lvrt(tol=1e-8):
        # One damped fixed-point sweep: lvrt_ext -> clip((v_lvrt - V)/band, 0, 1).
        changed = 0.0
        for b in devices:
            V = model.get_value(f'V_{b}')
            target = min(1.0, max(0.0, (v_lvrt - V) / lvrt_band))
            cur = model.get_value(f'lvrt_ext_{b}')
            new = cur + 0.5*(target - cur)
            changed = max(changed, abs(new - cur))
            model.set_value(f'lvrt_ext_{b}', new)
        model.recalculate_algebraics(tol=tol)
        return changed

    def soft_continue(targets, steps, tol=1e-8):
        # Adaptive continuation that relaxes the soft relay to its fixed point
        # within each step; the whole step is rolled back and halved if it fails.
        start = {name: model.get_value(name) for name in targets}

        def set_frac(f):
            for name, target in targets.items():
                model.set_value(name, start[name] + f*(target - start[name]))

        f, df, df_max = 0.0, 1.0/steps, 1.0/steps
        while f < 1.0 - 1e-9:
            nf = min(1.0, f + df)
            snap = snapshot()
            set_frac(nf)
            try:
                model.recalculate_algebraics(tol=tol)
                for _ in range(20):
                    if relax_lvrt(tol=tol) < 1e-5:
                        break
            except RuntimeError:
                restore(snap)
                df *= 0.5
                if df < 1e-7:
                    raise
                continue
            f = nf
            df = min(df_max, df*1.5)

    # ------------------------------------------------------------------ #
    # Drive the event: apply (frozen) -> hold (integrate t_clear) -> clear
    # ------------------------------------------------------------------ #
    t_incept = model.t

    if lvrt_soft:
        for b in devices:                       # arm the current references
            model.set_value(f'i_sr_ref_{b}', lvrt_iq)
            model.set_value(f'i_sa_ref_{b}', lvrt_id)
        soft_continue(fault_on, fault_steps)
        hold_cmd = {}
        mode = {}
        for b in devices:
            le = model.get_value(f'lvrt_ext_{b}')
            mode[b] = le
            hold_cmd[f'lvrt_ext_{b}'] = le
            hold_cmd[f'i_sr_ref_{b}'] = lvrt_iq
            hold_cmd[f'i_sa_ref_{b}'] = lvrt_id
        n_act = sum(1 for b in devices if mode[b] > 0.5)
        report(f'LVRT soft relay: {n_act}/{len(devices)} devices >50% into LVRT '
               f'(lvrt_ext range {min(mode.values()):.2f}-{max(mode.values()):.2f})')

        model.run(t_incept + t_clear, hold_cmd)

        soft_continue(fault_off, fault_steps, tol=1e-5)
        continue_to(lvrt_off, steps=8)
    else:
        mode = apply_fault_with_lvrt(fault_steps)
        hold_cmd = {k: v for b in devices for k, v in lvrt_cmd(b, mode[b]).items()}

        model.run(t_incept + t_clear, hold_cmd)

        clear_fault_hard()

    return mode
