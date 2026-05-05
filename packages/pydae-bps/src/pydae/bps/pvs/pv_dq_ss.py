# -*- coding: utf-8 -*-
r"""
PV inverter model with L-filter coupling, PQ control, and simplified PV array
(steady-state / purely algebraic).

**Auxiliar equations**

$$v_{sD} = V \sin(\theta)$$
$$v_{sQ} = V \cos(\theta)$$
$$v_{sd} = v_{sD} \cos(\delta) - v_{sQ} \sin(\delta)$$
$$v_{sq} = v_{sD} \sin(\delta) + v_{sQ} \cos(\delta)$$
$$v_m = \sqrt{v_{sd}^2 + v_{sq}^2}$$
$$\text{lvrt} = \begin{cases} 0.0 & v_m \geq v_{\text{lvrt}} \\ 1.0 & v_m < v_{\text{lvrt}} \end{cases}$$
$$+ \text{lvrt}_{\text{ext}}$$
$$V_{oc,t} = N_{pv,s} V_{oc} \left(1 + \frac{K_{vt}}{100}(T - T_{stc})\right)$$
$$V_{mp,t} = N_{pv,s} V_{mp} \left(1 + \frac{K_{vt}}{100}(T - T_{stc})\right)$$
$$I_{mp,t} = N_{pv,p} I_{mp} \left(1 + \frac{K_{it}}{100}(T - T_{stc})\right)$$
$$I_{mp,i} = I_{mp,t} \frac{G}{1000}$$
$$v_{dc,v} = v_1 - \frac{(i_1 - i_{pv})(v_1 - v_2)}{i_1 - i_2}$$
$$p_{mp} = \frac{V_{mp,t} I_{mp,i}}{S_n}$$

**Dynamic equations** (optional state-space filter for p/q references)

$$\dot{x} = A_{pq} x + B_{pq} \begin{bmatrix} p_{s,ppc} \\ q_{s,ppc} \end{bmatrix}$$
$$\begin{bmatrix} p_{s,ppc,d} \\ q_{s,ppc,d} \end{bmatrix} =$$
$$C_{pq} x + D_{pq} \begin{bmatrix} p_{s,ppc} \\ q_{s,ppc} \end{bmatrix}$$

**Algebraic equations**

$$0 = v_{dc,v}/V_{dc,b} - v_{dc}$$
$$0 = -i_{sd,ref} + \text{sat}\left((1-\text{lvrt}) i_{sd,pq,ref} + \text{lvrt} \cdot i_{sd,ar,ref}\right)$$
$$0 = -i_{sq,ref} + \text{sat}\left((1-\text{lvrt}) i_{sq,pq,ref} + \text{lvrt} \cdot i_{sq,ar,ref}\right)$$
$$0 = v_{ti} - R_s i_{si} + X_s i_{sr} - v_{si}$$
$$0 = v_{tr} - R_s i_{sr} - X_s i_{si} - v_{sr}$$
$$0 = i_{si} v_{si} + i_{sr} v_{sr} - p_s$$
$$0 = i_{si} v_{sr} - i_{sr} v_{si} - q_s$$
"""
import io

import numpy as np
import sympy as sym  # noqa: F401 - used by symbolic_dev()


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.
    """
    descriptions_list = []

    # Parameters - Inverter
    descriptions_list += [
        {"type": "Parameter", "tex": "S_n", "data": "S_n", "model": "S_n",
         "default": 1e6, "description": "Nominal power", "units": "VA"}]
    descriptions_list += [
        {"type": "Parameter", "tex": "U_n", "data": "U_n", "model": "U_n",
         "default": 400.0,
         "description": "Nominal RMS line-to-line voltage", "units": "V"}]
    descriptions_list += [
        {"type": "Parameter", "tex": "F_n", "data": "F_n", "model": "F_n",
         "default": 50.0, "description": "Nominal frequency", "units": "Hz"}]
    descriptions_list += [
        {"type": "Parameter", "tex": "X_s", "data": "X_s", "model": "X_s",
         "default": 0.1,
         "description": "Coupling reactance (pu, S_n base)", "units": "pu"}]
    descriptions_list += [
        {"type": "Parameter", "tex": "R_s", "data": "R_s", "model": "R_s",
         "default": 0.01,
         "description": "Coupling resistance (pu, S_n base)", "units": "pu"}]

    # Parameters - PV module
    descriptions_list += [
        {"type": "Parameter", "tex": "I_{sc}", "data": "I_sc",
         "model": "I_sc", "default": 8.0,
         "description": "Short-circuit current at STC", "units": "A"}]
    descriptions_list += [
        {"type": "Parameter", "tex": "V_{oc}", "data": "V_oc",
         "model": "V_oc", "default": 42.1,
         "description": "Open-circuit voltage at STC", "units": "V"}]
    descriptions_list += [
        {"type": "Parameter", "tex": "I_{mp}", "data": "I_mp",
         "model": "I_mp", "default": 3.56,
         "description": "MPP current at STC", "units": "A"}]
    descriptions_list += [
        {"type": "Parameter", "tex": "V_{mp}", "data": "V_mp",
         "model": "V_mp", "default": 33.7,
         "description": "MPP voltage at STC", "units": "V"}]
    descriptions_list += [
        {"type": "Parameter", "tex": "K_{vt}", "data": "K_vt",
         "model": "K_vt", "default": -0.16,
         "description": "Voltage temperature coefficient", "units": "%/C"}]
    descriptions_list += [
        {"type": "Parameter", "tex": "K_{it}", "data": "K_it",
         "model": "K_it", "default": 0.065,
         "description": "Current temperature coefficient", "units": "%/C"}]
    descriptions_list += [
        {"type": "Parameter", "tex": "N_{pv,s}", "data": "N_pv_s",
         "model": "N_pv_s", "default": 25,
         "description": "Number of series PV cells", "units": "-"}]
    descriptions_list += [
        {"type": "Parameter", "tex": "N_{pv,p}", "data": "N_pv_p",
         "model": "N_pv_p", "default": 250,
         "description": "Number of parallel PV strings", "units": "-"}]

    # Parameters - LVRT
    descriptions_list += [
        {"type": "Parameter", "tex": "v_{\\text{lvrt}}", "data": "v_lvrt",
         "model": "v_lvrt", "default": 0.8,
         "description": "LVRT voltage threshold", "units": "pu"}]

    # Inputs
    descriptions_list += [
        {"type": "Input", "tex": "p_{s,ppc}", "data": "p_s_ppc",
         "model": "p_s_ppc", "default": 0.5,
         "description": "Active power reference from PPC", "units": "pu"}]
    descriptions_list += [
        {"type": "Input", "tex": "q_{s,ppc}", "data": "q_s_ppc",
         "model": "q_s_ppc", "default": 0.0,
         "description": "Reactive power reference from PPC", "units": "pu"}]
    descriptions_list += [
        {"type": "Input", "tex": "v_{dc}", "data": "v_dc",
         "model": "v_dc", "default": 1.5,
         "description": "DC-link voltage (pu)", "units": "pu"}]
    descriptions_list += [
        {"type": "Input", "tex": "i_{sa,ref}", "data": "i_sa_ref",
         "model": "i_sa_ref", "default": 0.0,
         "description": "Active current reference (arbitrary mode)",
         "units": "pu"}]
    descriptions_list += [
        {"type": "Input", "tex": "i_{sr,ref}", "data": "i_sr_ref",
         "model": "i_sr_ref", "default": 0.0,
         "description": "Reactive current reference (arbitrary mode)",
         "units": "pu"}]
    descriptions_list += [
        {"type": "Input", "tex": "\\text{lvrt}_{\\text{ext}}",
         "data": "lvrt_ext", "model": "lvrt_ext", "default": 0.0,
         "description": "External LVRT trigger", "units": "-"}]
    descriptions_list += [
        {"type": "Input", "tex": "T", "data": "temp_deg",
         "model": "temp_deg", "default": 25.0,
         "description": "Cell temperature", "units": "C"}]
    descriptions_list += [
        {"type": "Input", "tex": "G", "data": "irrad",
         "model": "irrad", "default": 1000.0,
         "description": "Irradiance", "units": "W/m2"}]

    # Dynamic States (from optional ss filter)
    descriptions_list += [
        {"type": "Dynamic State", "tex": "x_{pq}", "data": "",
         "model": "x_pq", "default": "",
         "description": "State-space filter states (when A_pq provided)",
         "units": "-"}]

    # Algebraic States
    descriptions_list += [
        {"type": "Algebraic State", "tex": "v_{dc}", "data": "",
         "model": "v_dc", "default": "",
         "description": "DC-link voltage (normalized)", "units": "pu"}]
    descriptions_list += [
        {"type": "Algebraic State", "tex": "i_{sd,ref}", "data": "",
         "model": "i_sd_ref", "default": "",
         "description": "d-axis current reference", "units": "pu"}]
    descriptions_list += [
        {"type": "Algebraic State", "tex": "i_{sq,ref}", "data": "",
         "model": "i_sq_ref", "default": "",
         "description": "q-axis current reference", "units": "pu"}]
    descriptions_list += [
        {"type": "Algebraic State", "tex": "i_{si}", "data": "",
         "model": "i_si", "default": "",
         "description": "Inverter active-axis current", "units": "pu"}]
    descriptions_list += [
        {"type": "Algebraic State", "tex": "i_{sr}", "data": "",
         "model": "i_sr", "default": "",
         "description": "Inverter reactive-axis current", "units": "pu"}]
    descriptions_list += [
        {"type": "Algebraic State", "tex": "p_s", "data": "",
         "model": "p_s", "default": "",
         "description": "Injected active power", "units": "pu"}]
    descriptions_list += [
        {"type": "Algebraic State", "tex": "q_s", "data": "",
         "model": "q_s", "default": "",
         "description": "Injected reactive power", "units": "pu"}]

    # Outputs
    descriptions_list += [
        {"type": "Output", "tex": "m_{ref}", "data": "",
         "model": "m_ref", "default": "",
         "description": "Modulation index reference", "units": "-"}]
    descriptions_list += [
        {"type": "Output", "tex": "\\theta_{t,ref}", "data": "",
         "model": "theta_t_ref", "default": "",
         "description": "Inverter voltage angle reference", "units": "rad"}]
    descriptions_list += [
        {"type": "Output", "tex": "v_{sd}", "data": "",
         "model": "v_sd", "default": "",
         "description": "d-axis bus voltage", "units": "pu"}]
    descriptions_list += [
        {"type": "Output", "tex": "v_{sq}", "data": "",
         "model": "v_sq", "default": "",
         "description": "q-axis bus voltage", "units": "pu"}]
    descriptions_list += [
        {"type": "Output", "tex": "\\text{lvrt}", "data": "",
         "model": "lvrt", "default": "",
         "description": "LVRT active flag", "units": "-"}]
    descriptions_list += [
        {"type": "Output", "tex": "p_{mp}", "data": "",
         "model": "p_mp", "default": "",
         "description": "Maximum power point (pu)", "units": "pu"}]
    descriptions_list += [
        {"type": "Output", "tex": "i_{pv}", "data": "",
         "model": "i_pv", "default": "",
         "description": "PV array current (pu)", "units": "pu"}]
    descriptions_list += [
        {"type": "Output", "tex": "v_{dc,v}", "data": "",
         "model": "v_dc_v", "default": "",
         "description": "PV array voltage (V)", "units": "V"}]
    descriptions_list += [
        {"type": "Output", "tex": "v_{ac,v}", "data": "",
         "model": "v_ac_v", "default": "",
         "description": "AC terminal voltage (V)", "units": "V"}]
    descriptions_list += [
        {"type": "Output", "tex": "i_s", "data": "",
         "model": "i_s", "default": "",
         "description": "Inverter current magnitude", "units": "pu"}]

    return descriptions_list


def pv_dq_ss(grid, name, bus_name, data_dict):
    """
    PV inverter with L-filter coupling and PQ control (steady-state).

    The model represents a grid-following PV inverter whose inner current
    loops are assumed ideal (algebraic).  Power references come from a
    plant-level controller (PPC) and can optionally pass through a
    state-space low-pass filter defined by A_pq/B_pq/C_pq/D_pq matrices.

    The PV array is modelled as a two-point linearised I-V curve that
    intersects the DC-link power balance to determine the operating
    voltage.  Temperature and irradiance shift the curve via standard
    coefficients.

    Parameters
    ----------
    grid : BpsBuilder
        Builder instance that accumulates DAE components.
    name : str
        Unique instance identifier (e.g. ``"1"``).
    bus_name : str
        Name of the bus the inverter is connected to.
    data_dict : dict
        User-supplied parameter values.

    Returns
    -------
    p_W : sympy/casadi expression
        Active power injection in watts.
    q_var : sympy/casadi expression
        Reactive power injection in vars.
    """
    backend = grid.backend
    sin = backend.sin
    cos = backend.cos

    # 1. Fetch metadata and defaults
    meta = descriptions()
    default_map = {
        item["data"]: item["default"]
        for item in meta if "data" in item and item["data"]
    }

    # 2. Bus-side inputs
    V_s = backend.symbols(f"V_{bus_name}")
    theta_s = backend.symbols(f"theta_{bus_name}")

    # 3. PPC / external inputs
    p_s_ppc = backend.symbols(f"p_s_ppc_{name}")
    q_s_ppc = backend.symbols(f"q_s_ppc_{name}")
    v_dc = backend.symbols(f"v_dc_{name}")
    i_sa_ref = backend.symbols(f"i_sa_ref_{name}")
    i_sr_ref = backend.symbols(f"i_sr_ref_{name}")
    lvrt_ext = backend.symbols(f"lvrt_ext_{name}")

    # 4. Environmental inputs
    temp_deg = backend.symbols(f"temp_deg_{name}")
    irrad = backend.symbols(f"irrad_{name}")

    # 5. Parameters - Inverter
    S_n = backend.symbols(f"S_n_{name}")
    U_n = backend.symbols(f"U_n_{name}")
    V_dc_b = U_n * np.sqrt(2)
    X_s = backend.symbols(f"X_s_{name}")
    R_s = backend.symbols(f"R_s_{name}")

    # 6. Parameters - PV module
    K_vt = backend.symbols(f"K_vt_{name}")
    K_it = backend.symbols(f"K_it_{name}")
    V_oc = backend.symbols(f"V_oc_{name}")
    V_mp = backend.symbols(f"V_mp_{name}")
    I_mp = backend.symbols(f"I_mp_{name}")
    N_pv_s = backend.symbols(f"N_pv_s_{name}")
    N_pv_p = backend.symbols(f"N_pv_p_{name}")

    # 7. Parameters - LVRT
    v_lvrt = backend.symbols(f"v_lvrt_{name}")

    # 8. Algebraic states (must be defined before use in equations)
    i_sd_ref = backend.symbols(f"i_sd_ref_{name}")
    i_sq_ref = backend.symbols(f"i_sq_ref_{name}")
    i_si = backend.symbols(f"i_si_{name}")
    i_sr = backend.symbols(f"i_sr_{name}")
    p_s = backend.symbols(f"p_s_{name}")
    q_s = backend.symbols(f"q_s_{name}")

    # 9. Temperature-dependent PV characteristics
    T_stc_deg = 25.0

    V_oc_t = N_pv_s * V_oc * (1 + K_vt / 100.0 * (temp_deg - T_stc_deg))
    V_mp_t = N_pv_s * V_mp * (1 + K_vt / 100.0 * (temp_deg - T_stc_deg))
    I_mp_t = N_pv_p * I_mp * (1 + K_it / 100.0 * (temp_deg - T_stc_deg))
    I_mp_i = I_mp_t * irrad / 1000.0

    v_1, i_1 = V_mp_t, I_mp_i
    v_2, i_2 = V_oc_t, 0

    # 10. PV I-V curve intersection with DC-link power balance
    i_pv = p_s * S_n / (v_dc * V_dc_b)
    p_mp = (V_mp_t * I_mp_i) / S_n
    v_dc_v = v_1 - (i_1 - i_pv) * (v_1 - v_2) / (i_1 - i_2)
    g_v_dc = -v_dc + v_dc_v / V_dc_b

    # 11. PLL-synchronous frame transformation (ideal PLL: delta = theta_s)
    delta = theta_s
    v_sD = V_s * sin(theta_s)
    v_sQ = V_s * cos(theta_s)
    v_sd = v_sD * cos(delta) - v_sQ * sin(delta)
    v_sq = v_sD * sin(delta) + v_sQ * cos(delta)

    v_m = backend.sqrt(v_sd**2 + v_sq**2)

    # 12. LVRT detection
    lvrt = backend.Piecewise(
        (0.0, v_m >= v_lvrt),
        (1.0, v_m < v_lvrt),
    ) + lvrt_ext

    # 13. Optional state-space filters for p/q references
    from pydae.bps.utils.ss_num2sym import ss_num2sym

    p_s_ppc_sym = sym.Symbol(f"p_s_ppc_{name}")
    q_s_ppc_sym = sym.Symbol(f"q_s_ppc_{name}")

    p_s_ppc_d = p_s_ppc_sym
    q_s_ppc_d = q_s_ppc_sym

    if "A_pq" in data_dict:
        # Coupled MIMO filter (2 inputs, N states)
        A_pq = np.array(data_dict["A_pq"])
        B_pq = np.array(data_dict["B_pq"])
        C_pq = np.array(data_dict["C_pq"])
        D_pq = np.array(data_dict["D_pq"])

        sys_pq = ss_num2sym(f"pq_{name}", A_pq, B_pq, C_pq, D_pq)
        sys_pq["dx"] = sys_pq["dx"].replace(sys_pq["u"][0, 0], p_s_ppc_sym)
        sys_pq["dx"] = sys_pq["dx"].replace(sys_pq["u"][1, 0], q_s_ppc_sym)
        sys_pq["z_evaluated"] = sys_pq["z_evaluated"].replace(
            sys_pq["u"][0, 0], p_s_ppc_sym
        )
        sys_pq["z_evaluated"] = sys_pq["z_evaluated"].replace(
            sys_pq["u"][1, 0], q_s_ppc_sym
        )

        p_s_ppc_d = sys_pq["z_evaluated"][0, 0]
        q_s_ppc_d = sys_pq["z_evaluated"][1, 0]

        grid.dae["f"] += list(sys_pq["dx"])
        grid.dae["x"] += list(sys_pq["x"])
        grid.dae["params_dict"].update(sys_pq["params_dict"])

    elif "A_p" in data_dict:
        # Independent SISO filters for P and Q
        A_p = np.array(data_dict["A_p"])
        B_p = np.array(data_dict["B_p"])
        C_p = np.array(data_dict["C_p"])
        D_p = np.array(data_dict["D_p"])

        sys_p = ss_num2sym(f"p_{name}", A_p, B_p, C_p, D_p)
        sys_p["dx"] = sys_p["dx"].replace(sys_p["u"][0, 0], p_s_ppc_sym)
        sys_p["z_evaluated"] = sys_p["z_evaluated"].replace(
            sys_p["u"][0, 0], p_s_ppc_sym
        )

        p_s_ppc_d = sys_p["z_evaluated"][0, 0]

        grid.dae["f"] += list(sys_p["dx"])
        grid.dae["x"] += list(sys_p["x"])
        grid.dae["params_dict"].update(sys_p["params_dict"])

        A_q = np.array(data_dict.get("A_q", A_p))
        B_q = np.array(data_dict.get("B_q", B_p))
        C_q = np.array(data_dict.get("C_q", C_p))
        D_q = np.array(data_dict.get("D_q", D_p))

        sys_q = ss_num2sym(f"q_{name}", A_q, B_q, C_q, D_q)
        sys_q["dx"] = sys_q["dx"].replace(sys_q["u"][0, 0], q_s_ppc_sym)
        sys_q["z_evaluated"] = sys_q["z_evaluated"].replace(
            sys_q["u"][0, 0], q_s_ppc_sym
        )

        q_s_ppc_d = sys_q["z_evaluated"][0, 0]

        grid.dae["f"] += list(sys_q["dx"])
        grid.dae["x"] += list(sys_q["x"])
        grid.dae["params_dict"].update(sys_q["params_dict"])

    # 14. Power reference limiting (clip to MPP)
    p_s_ref = backend.Piecewise(
        (p_s_ppc_d, p_s_ppc_d < p_mp),
        (p_mp, p_s_ppc_d >= p_mp),
    )
    q_s_ref = q_s_ppc_d

    # 15. Current reference computation (PQ mode)
    denom = v_sd**2 + v_sq**2
    i_sd_pq_ref = (p_s_ref * v_sd + q_s_ref * v_sq) / denom
    i_sq_pq_ref = (p_s_ref * v_sq - q_s_ref * v_sd) / denom

    # 16. Current reference computation (arbitrary / current mode)
    v_m_abc = backend.sqrt(denom)
    i_sd_ar_ref = i_sa_ref * v_sd / v_m_abc + i_sr_ref * v_sq / v_m_abc
    i_sq_ar_ref = i_sa_ref * v_sq / v_m_abc - i_sr_ref * v_sd / v_m_abc

    # 17. Blend PQ and arbitrary mode via LVRT flag
    i_sd_ref_nosat = (1.0 - lvrt) * i_sd_pq_ref + lvrt * i_sd_ar_ref
    i_sq_ref_nosat = (1.0 - lvrt) * i_sq_pq_ref + lvrt * i_sq_ar_ref

    # 18. Current saturation (+/-1.2 pu)
    I_MAX = 1.2
    i_sd_ref_sat = backend.Piecewise(
        (-I_MAX, i_sd_ref_nosat < -I_MAX),
        (I_MAX, i_sd_ref_nosat > I_MAX),
        (i_sd_ref_nosat, True),
    )
    i_sq_ref_sat = backend.Piecewise(
        (-I_MAX, i_sq_ref_nosat < -I_MAX),
        (I_MAX, i_sq_ref_nosat > I_MAX),
        (i_sq_ref_nosat, True),
    )

    g_i_sd_ref = -i_sd_ref + i_sd_ref_sat
    g_i_sq_ref = -i_sq_ref + i_sq_ref_sat

    # 19. Voltage references in dq frame
    v_td_ref = R_s * i_sd_ref - X_s * i_sq_ref + v_sd
    v_tq_ref = R_s * i_sq_ref + X_s * i_sd_ref + v_sq

    # 20. Transform back to stationary frame
    v_tD_ref = v_td_ref * cos(delta) + v_tq_ref * sin(delta)
    v_tQ_ref = -v_td_ref * sin(delta) + v_tq_ref * cos(delta)
    v_ti_ref = v_tD_ref
    v_tr_ref = v_tQ_ref

    # 21. Modulation index and angle
    m_ref = backend.sqrt(v_tr_ref**2 + v_ti_ref**2) / v_dc
    theta_t_ref = backend.atan2(v_ti_ref, v_tr_ref)

    # 22. VSC filter algebraic equations
    v_si = V_s * sin(theta_s)
    v_sr = V_s * cos(theta_s)
    v_ti = v_ti_ref
    v_tr = v_tr_ref

    g_i_si = v_ti - R_s * i_si + X_s * i_sr - v_si
    g_i_sr = v_tr - R_s * i_sr - X_s * i_si - v_sr
    g_p_s = i_si * v_si + i_sr * v_sr - p_s
    g_q_s = i_si * v_sr - i_sr * v_si - q_s

    # 23. Assembly
    f_vsg = []
    x_vsg = []
    g_vsg = [g_v_dc, g_i_sd_ref, g_i_sq_ref, g_i_si, g_i_sr, g_p_s, g_q_s]
    y_vsg = [v_dc, i_sd_ref, i_sq_ref, i_si, i_sr, p_s, q_s]

    grid.dae["f"] += f_vsg
    grid.dae["x"] += x_vsg
    grid.dae["g"] += g_vsg
    grid.dae["y_ini"] += y_vsg
    grid.dae["y_run"] += y_vsg

    # 24. Dynamic input handling
    grid.dae["u_ini_dict"].update({f"{lvrt_ext}": 0.0})
    grid.dae["u_run_dict"].update({f"{lvrt_ext}": 0.0})

    p_s_ppc_val = data_dict.get("p_s_ppc", default_map.get("p_s_ppc", 0.5))
    q_s_ppc_val = data_dict.get("q_s_ppc", default_map.get("q_s_ppc", 0.0))

    grid.dae["u_ini_dict"].update({f"{p_s_ppc}": p_s_ppc_val})
    grid.dae["u_run_dict"].update({f"{p_s_ppc}": p_s_ppc_val})

    grid.dae["u_ini_dict"].update({f"{q_s_ppc}": q_s_ppc_val})
    grid.dae["u_run_dict"].update({f"{q_s_ppc}": q_s_ppc_val})

    grid.dae["u_ini_dict"].update({f"{i_sa_ref}": 0.0})
    grid.dae["u_run_dict"].update({f"{i_sa_ref}": 0.0})

    grid.dae["u_ini_dict"].update({f"{i_sr_ref}": 0.0})
    grid.dae["u_run_dict"].update({f"{i_sr_ref}": 0.0})

    grid.dae["u_ini_dict"].update({f"{irrad}": 1000.0})
    grid.dae["u_run_dict"].update({f"{irrad}": 1000.0})

    grid.dae["u_ini_dict"].update({f"{temp_deg}": 25.0})
    grid.dae["u_run_dict"].update({f"{temp_deg}": 25.0})

    # 25. Initialization hints
    v_dc_N = data_dict.get("v_dc", 1.5)
    grid.dae["xy_0_dict"].update({str(v_dc): v_dc_N})
    grid.dae["xy_0_dict"].update({str(p_s): 0.5})
    grid.dae["xy_0_dict"].update({str(q_s): 0.0})
    grid.dae["xy_0_dict"].update({str(i_sd_ref): 0.5})
    grid.dae["xy_0_dict"].update({str(i_sq_ref): 0.0})
    grid.dae["xy_0_dict"].update({str(i_si): 0.5})
    grid.dae["xy_0_dict"].update({str(i_sr): 0.0})

    # 26. Parameters - LVRT threshold
    v_lvrt_val = data_dict.get("v_lvrt", default_map.get("v_lvrt", 0.8))
    grid.dae["params_dict"].update({str(v_lvrt): v_lvrt_val})

    # 27. Parameters - Inverter
    for item in ["S_n", "F_n", "U_n", "X_s", "R_s"]:
        val = data_dict.get(item, default_map.get(item, 0.0))
        grid.dae["params_dict"].update({f"{item}_{name}": val})

    # 28. Parameters - PV module
    for item in ["I_sc", "V_oc", "I_mp", "V_mp", "K_vt", "K_it",
                 "N_pv_s", "N_pv_p"]:
        val = data_dict.get(item, default_map.get(item, 0.0))
        grid.dae["params_dict"].update({f"{item}_{name}": val})

    # 29. Outputs
    grid.dae["h_dict"].update({f"m_ref_{name}": m_ref})
    grid.dae["h_dict"].update({f"theta_t_ref_{name}": theta_t_ref})
    grid.dae["h_dict"].update({f"v_sd_{name}": v_sd})
    grid.dae["h_dict"].update({f"v_sq_{name}": v_sq})
    grid.dae["h_dict"].update({f"lvrt_{name}": lvrt})
    grid.dae["h_dict"].update({f"p_s_ppc_{name}": p_s_ppc})
    grid.dae["h_dict"].update({f"q_s_ppc_{name}": q_s_ppc})

    if data_dict.get("monitor", False):
        grid.dae["h_dict"].update({f"v_dc_v_{name}": v_dc * V_dc_b})
        grid.dae["h_dict"].update({f"v_ac_v_{name}": m_ref * v_dc * U_n})
        grid.dae["h_dict"].update({f"v_dc_pv_{name}": v_dc_v})
        grid.dae["h_dict"].update({f"p_mp_{name}": p_mp})
        grid.dae["h_dict"].update({f"i_pv_{name}": i_pv})
        grid.dae["h_dict"].update({f"v_dc_{name}": v_dc})
        grid.dae["h_dict"].update({f"p_s_{name}": p_s})
        grid.dae["h_dict"].update({f"q_s_{name}": q_s})
        grid.dae["h_dict"].update({f"v_ti_{name}": v_ti})
        grid.dae["h_dict"].update({f"v_tr_{name}": v_tr})
        grid.dae["h_dict"].update({f"i_si_{name}": i_si})
        grid.dae["h_dict"].update({f"i_sr_{name}": i_sr})
        i_s = backend.sqrt(i_sr**2 + i_si**2)
        grid.dae["h_dict"].update({f"i_s_{name}": i_s})

    p_W = p_s * S_n
    q_var = q_s * S_n

    return p_W, q_var


# =============================================================================
# Sphinx Documentation Auto-Generator
# =============================================================================
def dict_list_to_aligned_markdown_table(data: list[dict]) -> str:
    if not data or not isinstance(data, list):
        return ""
    dict_data = [item for item in data if isinstance(item, dict)]
    if not dict_data:
        return ""

    all_headers = []
    header_set = set()
    for row_dict in dict_data:
        for key in row_dict.keys():
            if key not in header_set:
                header_set.add(key)
                all_headers.append(key)

    max_widths = {header: len(header) for header in all_headers}
    for row_dict in dict_data:
        for header in all_headers:
            max_widths[header] = max(
                max_widths[header], len(str(row_dict.get(header, "")))
            )

    output = io.StringIO()
    padded_headers = [
        header.ljust(max_widths[header]) for header in all_headers
    ]
    output.write("| " + " | ".join(padded_headers) + " |\n")
    separators = ["-" * max_widths[header] for header in all_headers]
    output.write("| " + " | ".join(separators) + " |\n")

    for row_dict in dict_data:
        padded_values = [
            str(row_dict.get(header, "")).ljust(max_widths[header])
            for header in all_headers
        ]
        output.write("| " + " | ".join(padded_values) + " |\n")

    return output.getvalue().strip()


def generate_sphinx_tables():
    docs = ""
    categories = [
        "Parameter", "Input", "Dynamic State", "Algebraic State", "Output"
    ]
    full_list = descriptions()

    for cat in categories:
        cat_list = [
            {k: v for k, v in item.items() if k != "type"}
            for item in full_list if item.get("type") == cat
        ]

        if cat_list:
            docs += f"\n### {cat}s\n\n"
            docs += dict_list_to_aligned_markdown_table(cat_list)
            docs += "\n"

    return docs


__doc__ += generate_sphinx_tables()


def symbolic_dev():
    """
    Symbolic steady-state analysis for PV inverter model.
    Solves algebraic equations for equilibrium currents and voltages.
    """
    # Inputs
    V = sym.Symbol("V", real=True, positive=True)

    # Parameters
    R_s = sym.Symbol("R_s", real=True, positive=True)
    X_s = sym.Symbol("X_s", real=True, positive=True)

    # Algebraic states
    i_si = sym.Symbol("i_si", real=True)
    i_sr = sym.Symbol("i_sr", real=True)
    p_s = sym.Symbol("p_s", real=True)
    q_s = sym.Symbol("q_s", real=True)

    # Simplified: ideal PLL, no LVRT, no saturation
    v_sd_val = 0
    v_sq_val = V
    v_ti = R_s * i_si - X_s * i_sr + v_sd_val
    v_tr = R_s * i_sr + X_s * i_si + v_sq_val

    # Algebraic equations
    g_i_si = v_ti - R_s * i_si + X_s * i_sr - v_sd_val
    g_i_sr = v_tr - R_s * i_sr - X_s * i_si - v_sq_val
    g_p_s = i_si * v_sd_val + i_sr * v_sq_val - p_s
    g_q_s = i_si * v_sq_val - i_sr * v_sd_val - q_s

    unknown = [i_si, i_sr, p_s, q_s]
    solution = sym.solve([g_i_si, g_i_sr, g_p_s, g_q_s], unknown)
    print(solution)

    print("Symbolic solution for steady state:")
    for item in solution:
        print(sym.simplify(item))
        print("\n")


# =============================================================================
# Testing Block
# =============================================================================
def test_build():
    from pydae.bps import BpsBuilder
    from pydae.core import Builder

    grid = BpsBuilder("pv_dq_ss.hjson")
    grid.checker()
    grid.uz_jacs = False
    grid.construct("temp_pv_dq_ss")
    bld = Builder(grid.sys_dict, target="ctypes", sparse=False)
    bld.build()


def test_ini():
    from pydae.core import Model

    model = Model("temp_pv_dq_ss")

    model.ini({}, "xy_0.json")
    model.report_x()
    model.report_y()


def test_run():
    import matplotlib.pyplot as plt
    from pydae.core import Model

    model = Model("temp_pv_dq_ss")

    model.ini({"p_s_ppc_1": 0.5, "q_s_ppc_1": 0.0}, "xy_0.json")
    model.report_x()
    model.report_y()

    model.run(1.0, {})

    model.run(10.0, {"p_s_ppc_1": 0.8, "q_s_ppc_1": 0.2})

    model.post()

    fig, axes = plt.subplots(3, 1, figsize=(8, 8))

    axes[0].plot(model.Time, model.get_values("p_s_1"),
                 label="$p_s$ (pu)", color="b")
    axes[0].set_ylabel("Active Power")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(model.Time, model.get_values("q_s_1"),
                 label="$q_s$ (pu)", color="r")
    axes[1].set_ylabel("Reactive Power")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(model.Time, model.get_values("m_ref_1"),
                 label="$m_{ref}$", color="g")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Modulation Index")
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig("pv_dq_ss_response.svg")
    print("Test completed. Plot saved as 'pv_dq_ss_response.svg'.")


if __name__ == "__main__":
    # symbolic_dev()
    test_build()
    test_ini()
    # test_run()
