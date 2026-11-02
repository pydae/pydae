# -*- coding: utf-8 -*-
r"""
Field-oriented current control: dq PI with decoupling (FOC_PI).

The inner loop of a vector-controlled drive, as a continuous mean-value
block: one PI per axis on the current error, plus the cross-coupling
feedforward, driving the voltages of the machine named by ``machine``.

$$v_d^* = \mathrm{PI}_d(i_d^{ref} - i_d) - \omega_e \lambda_q, \qquad
  v_q^* = \mathrm{PI}_q(i_q^{ref} - i_q) + \omega_e \lambda_d$$

The two PIs are :func:`pydae.control.continuos.pi.pi` blocks (output
saturation at $\pm V_{max}$, back-calculation anti-windup), so their states
are ``xi_pi_d_{name}``, ``xi_pi_q_{name}``.  The decoupling uses the
machine's true fluxes, as the switched drive's ``PIController`` does.

**Tuning.**  Give ``K_p``/``K_i`` per axis (``K_p_d``, ``K_i_d``,
``K_p_q``, ``K_i_q``), or a ``bandwidth`` [Hz] and the incremental
inductances: then $K_p = 2\pi f_{bw} L$ and $K_i = K_p R_s / L$, which
cancels the stator pole and leaves a first-order loop of time constant
$1/(2\pi f_{bw})$ on a linear machine.  ``L_d``, ``L_q`` and ``R_s`` default
to the machine's own when it is a ``pmsm_dq``; for a ``pmsm_lut`` give
them (:func:`pydae.emac.utils.analysis.incremental_inductances`).

**Inverter.**  The voltage asked for is applied.  ``T_delay`` > 0 [s] puts
a first-order lag (:func:`pydae.control.continuos.lpf1.lpf1`) between
command and machine, the usual mean-value stand-in for the DSP's
computation delay plus the PWM's half period: $1.5\,T_s$ for a
double-update drive.  The full effect -- sampling, carrier, edges -- is the
switched drive's (``pydae.emac.drive``).

``V_max`` [V] limits each axis (default $V_{dc}/\sqrt{3}$ when ``V_dc`` is
given, else unlimited): a square, not the inverter's hexagon or circle.
"""

import numpy as np
from pydae.emac.utils.docs import descriptions_tables


def descriptions():
    """Single source of truth for parameters, inputs, states and outputs."""
    d = []
    d += [{"type": "Parameter", "tex": "K_p", "data": "K_p_d, K_p_q", "model": "K_p_pi_d, K_p_pi_q", "default": "", "description": "Proportional gains (or from bandwidth)", "units": "V/A"}]
    d += [{"type": "Parameter", "tex": "K_i", "data": "K_i_d, K_i_q", "model": "K_i_pi_d, K_i_pi_q", "default": "", "description": "Integral gains (or from bandwidth)", "units": "V/(A.s)"}]
    d += [{"type": "Parameter", "tex": "f_{bw}", "data": "bandwidth", "model": "", "default": "", "description": "Current-loop bandwidth, sets K_p, K_i", "units": "Hz"}]
    d += [{"type": "Parameter", "tex": "V_{max}", "data": "V_max", "model": "z_max_pi_d, z_max_pi_q", "default": "V_dc/sqrt(3)", "description": "Voltage limit per axis", "units": "V"}]
    d += [{"type": "Parameter", "tex": "T_{delay}", "data": "T_delay", "model": "T_lpf1_vd, T_lpf1_vq", "default": 0.0, "description": "Inverter delay as a first-order lag (0: none)", "units": "s"}]
    d += [{"type": "Input", "tex": "i_d^{ref}", "data": "i_d_ref", "model": "i_d_ref", "default": 0.0, "description": "d-axis current reference (peak)", "units": "A"}]
    d += [{"type": "Input", "tex": "i_q^{ref}", "data": "i_q_ref", "model": "i_q_ref", "default": 0.0, "description": "q-axis current reference (peak)", "units": "A"}]
    d += [{"type": "Dynamic State", "tex": r"\xi_d, \xi_q", "data": "", "model": "xi_pi_d, xi_pi_q", "default": "", "description": "PI integrators", "units": "A.s"}]
    d += [{"type": "Algebraic State", "tex": "v_d^*, v_q^*", "data": "", "model": "v_d_cmd, v_q_cmd", "default": "", "description": "Voltage commands (= the machine's v_d, v_q when T_delay = 0)", "units": "V"}]
    return d


def gains(data_dict, machine):
    """(K_p, K_i) per axis, as arrays [d, q], from explicit gains or bandwidth."""
    if "bandwidth" in data_dict:
        L_dq = machine.get("L_dq")
        L = np.array([data_dict.get("L_d", L_dq and L_dq[0]), data_dict.get("L_q", L_dq and L_dq[1])],
                     dtype=float)
        R = float(data_dict.get("R_s", machine["R_s"]))
        if np.isnan(L).any():
            raise KeyError("foc_pi: bandwidth tuning needs L_d and L_q for a pmsm_lut machine")
        K_p = 2 * np.pi * float(data_dict["bandwidth"]) * L
        return K_p, K_p * R / L
    return (np.array([data_dict["K_p_d"], data_dict["K_p_q"]], float),
            np.array([data_dict["K_i_d"], data_dict["K_i_q"]], float))


def foc_pi(grid, name, data_dict):
    """Append a dq current controller ``name`` driving machine ``data_dict['machine']``."""
    from pydae.control.continuos.lpf1 import lpf1
    from pydae.control.continuos.pi import pi

    bk = grid.backend
    dae = grid.dae
    m = str(data_dict.get("machine", name))
    if m not in grid.machines:
        raise KeyError(f"foc_pi {name}: no machine named {m!r}")
    machine = grid.machines[m]
    K_p, K_i = gains(data_dict, machine)
    V_max = float(data_dict.get("V_max", float(data_dict["V_dc"]) / np.sqrt(3) if "V_dc" in data_dict
                                else 1e9))
    T_delay = float(data_dict.get("T_delay", 0.0))

    lam_d, lam_q = bk.symbols(f"lam_d_{m}"), bk.symbols(f"lam_q_{m}")
    omega_e = machine["omega_e"]
    feedforward = {"d": -omega_e * lam_q, "q": omega_e * lam_d}

    for j, ax in enumerate(("d", "q")):
        ref = f"i_{ax}_ref_{name}"
        pi(dae, f"{ax}_{name}", bk,
           {"K_p": K_p[j], "K_i": K_i[j], "K_aw": 1.0 / K_p[j], "z_min": -V_max, "z_max": V_max},
           input=[ref, f"i_{ax}_{m}"])                  # error = ref - measured
        dae["u_ini_dict"][ref] = float(data_dict.get(f"i_{ax}_ref", 0.0))
        dae["u_run_dict"][ref] = float(data_dict.get(f"i_{ax}_ref", 0.0))

        z = bk.symbols(f"z_pi_{ax}_{name}")
        v_m = bk.symbols(f"v_{ax}_{m}")                 # the machine's voltage input
        if T_delay > 0:
            v_cmd = bk.symbols(f"v_{ax}_cmd_{name}")
            dae["g_list"] += [v_cmd - (z + feedforward[ax])]
            dae["y_ini_list"] += [v_cmd]
            dae["y_run_list"] += [v_cmd]
            dae["xy_0_dict"][f"{v_cmd}"] = 0.0
            # lpf1 promotes the machine's v_{ax} from input to its filtered output
            lpf1(dae, f"v{ax}_{name}", bk, {"T": T_delay}, input=f"{v_cmd}", output=f"{v_m}")
        else:
            # the command IS the machine's voltage: promote it from input to algebraic
            dae["u_ini_dict"].pop(f"{v_m}")
            dae["u_run_dict"].pop(f"{v_m}")
            dae["g_list"] += [v_m - (z + feedforward[ax])]
            dae["y_ini_list"] += [v_m]
            dae["y_run_list"] += [v_m]
            dae["xy_0_dict"][f"{v_m}"] = 0.0
    grid.ctrls[name] = dict(type="foc_pi", machine=m, K_p=K_p, K_i=K_i, V_max=V_max, T_delay=T_delay)


__doc__ += descriptions_tables(descriptions())
