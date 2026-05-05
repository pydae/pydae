# -*- coding: utf-8 -*-
r"""
Battery energy storage system (BESS) with VSC dq control, SOC dynamics,
and voltage ride-through (VRT) capability.

**Auxiliar equations**

$$H = \frac{E_{kWh} \cdot 1000 \cdot 3600}{S_n}$$
$$\epsilon = soc_{ref} - soc$$
$$p_{soc} = -(K_p \epsilon + K_i \xi_{soc})$$
$$e = f_{spline}(soc) \quad \text{(OCV-SOC interpolation)}$$
$$v_{sr} = V \cos(\theta), \quad v_{si} = V \sin(\theta)$$
$$v_{sq\_mag} = v_{sr}^2 + v_{si}^2 + \epsilon_{reg}$$

**Dynamic equations**

$$\frac{d\,soc}{dt} = \frac{1}{H} (-i_{dc} \cdot e)$$
$$\frac{d\,\xi_{soc}}{dt} = soc_{ref} - soc$$
$$\frac{d\,\text{lvrt\_ext\_ramp}}{dt} = \frac{\text{lvrt\_ext} - \text{lvrt\_ext\_ramp}}{T_{lvrt}}$$

**Algebraic equations (DC side)**

$$0 = p_s + p_{loss} - p_{dc}$$
$$0 = v_{dc} i_{dc} - p_{dc}$$
$$0 = e - i_{dc} R_{bat} - v_{dc}$$

**Algebraic equations (AC side)**

$$0 = v_{sr} i_{sr} + v_{si} i_{si} - p_s$$
$$0 = v_{si} i_{sr} - v_{sr} i_{si} - q_s$$

**Current blending with soft saturation**

$$i_{sr,nosat} = (1 - \text{lvrt\_ramp}) i_{sr,pq} + \text{lvrt\_ramp} \cdot i_{sr,ar}$$
$$i_{si,nosat} = (1 - \text{lvrt\_ramp}) i_{si,pq} + \text{lvrt\_ramp} \cdot i_{si,ar}$$
$$i_{mod} = \sqrt{i_{sr,nosat}^2 + i_{si,nosat}^2 + \epsilon}$$
$$i_{mod,sat} = \frac{1}{2} \left(i_{mod} + I_{max} - \sqrt{(i_{mod} - I_{max})^2 + \epsilon}\right)$$
"""
import io

import numpy as np
import sympy as sym  # noqa: F401 - used by symbolic_dev()
from sympy import interpolating_spline


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.
    """
    descriptions_list = []

    # Parameters - BESS
    descriptions_list += [
        {"type": "Parameter", "tex": "S_n", "data": "S_n", "model": "S_n",
         "default": 1e6, "description": "Nominal power", "units": "VA"}]
    descriptions_list += [
        {"type": "Parameter", "tex": "E_{kWh}", "data": "E_kWh",
         "model": "E_kWh", "default": 100.0,
         "description": "Battery energy capacity", "units": "kWh"}]
    descriptions_list += [
        {"type": "Parameter", "tex": "soc_{min}", "data": "soc_min",
         "model": "soc_min", "default": 0.0,
         "description": "Minimum state of charge", "units": "pu"}]
    descriptions_list += [
        {"type": "Parameter", "tex": "soc_{max}", "data": "soc_max",
         "model": "soc_max", "default": 1.0,
         "description": "Maximum state of charge", "units": "pu"}]
    descriptions_list += [
        {"type": "Parameter", "tex": "R_{bat}", "data": "R_bat",
         "model": "R_bat", "default": 0.0,
         "description": "Battery internal resistance", "units": "ohm"}]

    # Parameters - Loss model
    descriptions_list += [
        {"type": "Parameter", "tex": "A_{loss}", "data": "A_loss",
         "model": "A_loss", "default": 0.0001,
         "description": "Quadratic loss coefficient", "units": "-"}]
    descriptions_list += [
        {"type": "Parameter", "tex": "B_{loss}", "data": "B_loss",
         "model": "B_loss", "default": 0.0,
         "description": "Linear loss coefficient", "units": "-"}]
    descriptions_list += [
        {"type": "Parameter", "tex": "C_{loss}", "data": "C_loss",
         "model": "C_loss", "default": 0.0001,
         "description": "Constant loss coefficient", "units": "-"}]

    # Parameters - SOC PI controller
    descriptions_list += [
        {"type": "Parameter", "tex": "K_p", "data": "K_p",
         "model": "K_p", "default": 1e-6,
         "description": "SOC proportional gain", "units": "-"}]
    descriptions_list += [
        {"type": "Parameter", "tex": "K_i", "data": "K_i",
         "model": "K_i", "default": 1e-6,
         "description": "SOC integral gain", "units": "-"}]

    # Parameters - OCV-SOC interpolation
    descriptions_list += [
        {"type": "Parameter", "tex": "e_{soc\\_order}", "data": "e_soc_order",
         "model": "e_soc_order", "default": 1,
         "description": "Spline interpolation order for OCV-SOC", "units": "-"}]

    # Parameters - VRT
    descriptions_list += [
        {"type": "Parameter", "tex": "I_{max}", "data": "I_max",
         "model": "I_max", "default": 1.2,
         "description": "Maximum current magnitude", "units": "pu"}]
    descriptions_list += [
        {"type": "Parameter", "tex": "T_{lvrt}", "data": "T_lvrt",
         "model": "T_lvrt", "default": 0.02,
         "description": "LVRT ramp time constant", "units": "s"}]
    descriptions_list += [
        {"type": "Parameter", "tex": "\\epsilon", "data": "Epsilon",
         "model": "Epsilon", "default": 1e-8,
         "description": "Regularization for soft saturation", "units": "-"}]

    # Inputs
    descriptions_list += [
        {"type": "Input", "tex": "p_{s,ppc}", "data": "p_s_ppc",
         "model": "p_s_ppc", "default": 0.0,
         "description": "Active power reference from PPC", "units": "pu"}]
    descriptions_list += [
        {"type": "Input", "tex": "q_{s,ppc}", "data": "q_s_ppc",
         "model": "q_s_ppc", "default": 0.0,
         "description": "Reactive power reference from PPC", "units": "pu"}]
    descriptions_list += [
        {"type": "Input", "tex": "soc_{ref}", "data": "soc_ref",
         "model": "soc_ref", "default": 0.5,
         "description": "SOC reference setpoint", "units": "pu"}]
    descriptions_list += [
        {"type": "Input", "tex": "\\text{lvrt}_{\\text{ext}}",
         "data": "lvrt_ext", "model": "lvrt_ext", "default": 0.0,
         "description": "External LVRT trigger", "units": "-"}]
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

    # Dynamic States
    descriptions_list += [
        {"type": "Dynamic State", "tex": "soc", "data": "",
         "model": "soc", "default": "",
         "description": "State of charge", "units": "pu"}]
    descriptions_list += [
        {"type": "Dynamic State", "tex": "\\xi_{soc}", "data": "",
         "model": "xi_soc", "default": "",
         "description": "SOC integral error state", "units": "pu"}]
    descriptions_list += [
        {"type": "Dynamic State", "tex": "\\text{lvrt}_{\\text{ext,ramp}}",
         "data": "", "model": "lvrt_ext_ramp", "default": "",
         "description": "LVRT trigger ramped signal", "units": "-"}]

    # Algebraic States
    descriptions_list += [
        {"type": "Algebraic State", "tex": "p_{dc}", "data": "",
         "model": "p_dc", "default": "",
         "description": "DC power", "units": "pu"}]
    descriptions_list += [
        {"type": "Algebraic State", "tex": "i_{dc}", "data": "",
         "model": "i_dc", "default": "",
         "description": "DC current", "units": "pu"}]
    descriptions_list += [
        {"type": "Algebraic State", "tex": "v_{dc}", "data": "",
         "model": "v_dc", "default": "",
         "description": "DC voltage", "units": "pu"}]
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
        {"type": "Output", "tex": "p_{loss}", "data": "",
         "model": "p_loss", "default": "",
         "description": "VSC power losses", "units": "pu"}]
    descriptions_list += [
        {"type": "Output", "tex": "i_s", "data": "",
         "model": "i_s", "default": "",
         "description": "AC current magnitude", "units": "pu"}]
    descriptions_list += [
        {"type": "Output", "tex": "e", "data": "",
         "model": "e", "default": "",
         "description": "Battery OCV (from SOC curve)", "units": "pu"}]
    descriptions_list += [
        {"type": "Output", "tex": "i_{mod}", "data": "",
         "model": "i_mod", "default": "",
         "description": "Current magnitude before saturation", "units": "pu"}]
    descriptions_list += [
        {"type": "Output", "tex": "i_{mod,sat}", "data": "",
         "model": "i_mod_sat", "default": "",
         "description": "Saturated current magnitude", "units": "pu"}]

    return descriptions_list


def bess_dq_vrt(grid, name, bus_name, data_dict):
    """
    BESS with VSC dq control, SOC dynamics, and VRT capability.

    Combines battery energy storage (SOC dynamics + OCV-SOC curve) with a
    grid-following VSC that supports voltage ride-through.  During normal
    operation the VSC tracks PQ references from a plant controller.  When
    an LVRT trigger is asserted (externally or by undervoltage detection)
    the controller blends to an arbitrary current reference mode.  Current
    magnitude is limited via a smooth (differentiable) saturation function.

    Parameters
    ----------
    grid : BpsBuilder
        Builder instance that accumulates DAE components.
    name : str
        Unique instance identifier (e.g. ``"1"``).
    bus_name : str
        Name of the bus the BESS is connected to.
    data_dict : dict
        User-supplied parameter values.  Must contain ``socs`` and ``es``
        arrays for OCV-SOC interpolation, or ``B_0``/``B_1`` for a
        linear OCV model.

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
    soc_ref = backend.symbols(f"soc_ref_{name}")
    p_s_ppc = backend.symbols(f"p_s_ppc_{name}")
    q_s_ppc = backend.symbols(f"q_s_ppc_{name}")
    lvrt_ext = backend.symbols(f"lvrt_ext_{name}")
    i_sa_ref = backend.symbols(f"i_sa_ref_{name}")
    i_sr_ref = backend.symbols(f"i_sr_ref_{name}")

    # 4. Dynamic states - battery + LVRT ramp
    soc = backend.symbols(f"soc_{name}")
    xi_soc = backend.symbols(f"xi_soc_{name}")
    lvrt_ext_ramp = backend.symbols(f"lvrt_ext_ramp_{name}")

    # 5. Algebraic states - DC side + AC power
    p_dc = backend.symbols(f"p_dc_{name}")
    i_dc = backend.symbols(f"i_dc_{name}")
    v_dc = backend.symbols(f"v_dc_{name}")
    p_s = backend.symbols(f"p_s_{name}")
    q_s = backend.symbols(f"q_s_{name}")

    # 6. Parameters - BESS
    S_n = backend.symbols(f"S_n_{name}")
    E_kWh = backend.symbols(f"E_kWh_{name}")
    soc_min = backend.symbols(f"soc_min_{name}")
    soc_max = backend.symbols(f"soc_max_{name}")
    A_loss = backend.symbols(f"A_loss_{name}")
    B_loss = backend.symbols(f"B_loss_{name}")
    C_loss = backend.symbols(f"C_loss_{name}")
    K_p = backend.symbols(f"K_p_{name}")
    K_i = backend.symbols(f"K_i_{name}")
    R_bat = backend.symbols(f"R_bat_{name}")

    # 7. Parameters - VRT
    I_max = backend.symbols(f"I_max_{name}")
    T_lvrt = backend.symbols(f"T_lvrt_{name}")
    Epsilon = backend.symbols(f"Epsilon_{name}")

    # 8. OCV-SOC curve (piecewise interpolation or linear model)
    soc_sym = sym.Symbol(f"soc_{name}", real=True)
    if "socs" in data_dict:
        socs = np.array(data_dict["socs"])
        es = np.array(data_dict["es"])
        e_max = float(np.max(es))
        e_soc_order = data_dict.get("e_soc_order", 1)
        interpolation = interpolating_spline(
            e_soc_order, soc_sym, socs, es
        )
        interpolation._args = tuple(
            list(interpolation._args)
            + [sym.functions.elementary.piecewise.ExprCondPair(e_max, True)]
        )
        e = interpolation.subs(soc_sym, soc)
        soc_ref_N = data_dict["soc_ref"]
        e_ini = float(np.interp(soc_ref_N, socs, es))
    else:
        B_0 = backend.symbols(f"B_0_{name}")
        B_1 = backend.symbols(f"B_1_{name}")
        e = B_0 + B_1 * soc
        e_ini = data_dict.get("B_0", 1.0)

    # 9. Auxiliary equations
    H = E_kWh * 1000.0 * 3600 / S_n
    epsilon_soc = soc_ref - soc
    p_soc = -(K_p * epsilon_soc + K_i * xi_soc)

    # 10. Grid voltages in stationary (r-i) frame
    v_sr = V_s * cos(theta_s)
    v_si = V_s * sin(theta_s)
    v_sq_mag = v_sr**2 + v_si**2 + Epsilon

    # 11. PQ-mode current references (stationary frame)
    i_sr_pq = (p_s_ppc * v_sr + q_s_ppc * v_si) / v_sq_mag
    i_si_pq = (p_s_ppc * v_si - q_s_ppc * v_sr) / v_sq_mag

    # 12. Arbitrary-mode current references
    v_m = backend.sqrt(v_sq_mag)
    i_sr_ar = (i_sa_ref * v_sr + i_sr_ref * v_si) / v_m
    i_si_ar = (i_sa_ref * v_si - i_sr_ref * v_sr) / v_m

    # 13. Blend modes via LVRT ramp
    i_sr_nosat = (1.0 - lvrt_ext_ramp) * i_sr_pq + lvrt_ext_ramp * i_sr_ar
    i_si_nosat = (1.0 - lvrt_ext_ramp) * i_si_pq + lvrt_ext_ramp * i_si_ar

    # 14. Soft saturation (smooth, differentiable)
    i_mod = backend.sqrt(i_sr_nosat**2 + i_si_nosat**2 + Epsilon)
    i_mod_sat = 0.5 * (i_mod + I_max - backend.sqrt(
        (i_mod - I_max)**2 + Epsilon
    ))
    ratio = i_mod_sat / i_mod
    i_sr = i_sr_nosat * ratio
    i_si = i_si_nosat * ratio

    # 15. Injected power from saturated currents
    p_s_expr = v_sr * i_sr + v_si * i_si
    q_s_expr = v_si * i_sr - v_sr * i_si

    # 16. SOC-based power modulation
    p_s_limited = backend.Piecewise(
        (p_s_expr, (p_s_expr <= 0.0) & (soc < soc_max)),
        (p_s_expr, (p_s_expr > 0.0) & (soc > soc_min)),
        (0.0, True),
    ) + p_soc

    g_p_s = p_s_limited - p_s
    g_q_s = q_s_expr - q_s

    # 17. Loss model
    s_s = backend.sqrt(p_s**2 + q_s**2)
    i_s = s_s / V_s
    p_loss = A_loss * i_s**2 + B_loss * i_s + C_loss

    # 18. Dynamic equations
    dsoc = 1 / H * (-i_dc * e)
    dxi_soc = epsilon_soc
    dlvrt_ext_ramp = (lvrt_ext - lvrt_ext_ramp) / T_lvrt

    # 19. Algebraic equations - DC side
    g_p_dc = p_s + p_loss - p_dc
    g_i_dc = v_dc * i_dc - p_dc
    g_v_dc = e - i_dc * R_bat - v_dc

    # 20. Assembly
    grid.dae["f"] += [dsoc, dxi_soc, dlvrt_ext_ramp]
    grid.dae["x"] += [soc, xi_soc, lvrt_ext_ramp]
    grid.dae["g"] += [g_p_dc, g_i_dc, g_v_dc, g_p_s, g_q_s]
    grid.dae["y_ini"] += [p_dc, i_dc, v_dc, p_s, q_s]
    grid.dae["y_run"] += [p_dc, i_dc, v_dc, p_s, q_s]

    # 21. Dynamic input handling
    p_s_ppc_val = data_dict.get("p_s_ppc", default_map.get("p_s_ppc", 0.0))
    q_s_ppc_val = data_dict.get("q_s_ppc", default_map.get("q_s_ppc", 0.0))
    soc_ref_val = data_dict.get("soc_ref", default_map.get("soc_ref", 0.5))

    grid.dae["u_ini_dict"].update({str(p_s_ppc): p_s_ppc_val})
    grid.dae["u_run_dict"].update({str(p_s_ppc): p_s_ppc_val})

    grid.dae["u_ini_dict"].update({str(q_s_ppc): q_s_ppc_val})
    grid.dae["u_run_dict"].update({str(q_s_ppc): q_s_ppc_val})

    grid.dae["u_ini_dict"].update({str(soc_ref): soc_ref_val})
    grid.dae["u_run_dict"].update({str(soc_ref): soc_ref_val})

    grid.dae["u_ini_dict"].update({str(lvrt_ext): 0.0})
    grid.dae["u_run_dict"].update({str(lvrt_ext): 0.0})

    grid.dae["u_ini_dict"].update({str(i_sa_ref): 0.0})
    grid.dae["u_run_dict"].update({str(i_sa_ref): 0.0})

    grid.dae["u_ini_dict"].update({str(i_sr_ref): 0.0})
    grid.dae["u_run_dict"].update({str(i_sr_ref): 0.0})

    # 22. Parameters - VRT
    I_max_val = data_dict.get("I_max", default_map.get("I_max", 1.2))
    T_lvrt_val = data_dict.get("T_lvrt", default_map.get("T_lvrt", 0.02))
    Epsilon_val = data_dict.get("Epsilon", default_map.get("Epsilon", 1e-8))
    grid.dae["params_dict"].update({str(I_max): I_max_val})
    grid.dae["params_dict"].update({str(T_lvrt): T_lvrt_val})
    grid.dae["params_dict"].update({str(Epsilon): Epsilon_val})

    # 23. Parameters - SOC controller
    K_p_val = data_dict.get("K_p", default_map.get("K_p", 1e-6))
    K_i_val = data_dict.get("K_i", default_map.get("K_i", 1e-6))
    soc_min_val = data_dict.get("soc_min", default_map.get("soc_min", 0.0))
    soc_max_val = data_dict.get("soc_max", default_map.get("soc_max", 1.0))
    grid.dae["params_dict"].update({str(K_p): K_p_val})
    grid.dae["params_dict"].update({str(K_i): K_i_val})
    grid.dae["params_dict"].update({str(soc_min): soc_min_val})
    grid.dae["params_dict"].update({str(soc_max): soc_max_val})

    # 24. Parameters - BESS sizing
    grid.dae["params_dict"].update({str(S_n): data_dict["S_n"]})
    grid.dae["params_dict"].update({str(E_kWh): data_dict["E_kWh"]})

    # 25. Parameters - Loss model
    A_loss_N = data_dict.get("A_loss", default_map.get("A_loss", 0.0001))
    B_loss_N = data_dict.get("B_loss", default_map.get("B_loss", 0.0))
    C_loss_N = data_dict.get("C_loss", default_map.get("C_loss", 0.0001))
    grid.dae["params_dict"].update({str(A_loss): A_loss_N})
    grid.dae["params_dict"].update({str(B_loss): B_loss_N})
    grid.dae["params_dict"].update({str(C_loss): C_loss_N})

    # 26. Parameters - Battery resistance
    R_bat_N = data_dict.get("R_bat", default_map.get("R_bat", 0.0))
    grid.dae["params_dict"].update({str(R_bat): R_bat_N})

    # 27. Initialization hints
    grid.dae["xy_0_dict"].update({str(v_dc): e_ini})
    grid.dae["xy_0_dict"].update({str(soc): data_dict.get("soc_ref", 0.5)})
    grid.dae["xy_0_dict"].update({str(xi_soc): 0.0})
    grid.dae["xy_0_dict"].update({str(lvrt_ext_ramp): 0.0})
    grid.dae["xy_0_dict"].update({str(p_s): p_s_ppc_val})
    grid.dae["xy_0_dict"].update({str(q_s): q_s_ppc_val})

    # 28. Outputs
    grid.dae["h_dict"].update({f"p_loss_{name}": p_loss})
    grid.dae["h_dict"].update({f"i_s_{name}": i_s})
    grid.dae["h_dict"].update({f"e_{name}": e})
    grid.dae["h_dict"].update({f"i_dc_{name}": i_dc})
    grid.dae["h_dict"].update({f"p_s_{name}": p_s})
    grid.dae["h_dict"].update({f"q_s_{name}": q_s})
    grid.dae["h_dict"].update({f"p_s_ppc_{name}": p_s_ppc})
    grid.dae["h_dict"].update({f"q_s_ppc_{name}": q_s_ppc})
    grid.dae["h_dict"].update({f"i_mod_{name}": i_mod})
    grid.dae["h_dict"].update({f"i_mod_sat_{name}": i_mod_sat})
    grid.dae["h_dict"].update({f"lvrt_ext_ramp_{name}": lvrt_ext_ramp})

    # 29. Optional monitor outputs
    if data_dict.get("monitor", False):
        grid.dae["h_dict"].update({f"i_sr_{name}": i_sr})
        grid.dae["h_dict"].update({f"i_si_{name}": i_si})
        grid.dae["h_dict"].update({f"v_sr_{name}": v_sr})
        grid.dae["h_dict"].update({f"v_si_{name}": v_si})
        grid.dae["h_dict"].update({f"v_m_{name}": v_m})
        grid.dae["h_dict"].update({f"soc_{name}": soc})
        grid.dae["h_dict"].update({f"v_dc_{name}": v_dc})

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
    Symbolic steady-state analysis for BESS VRT model.
    Solves algebraic equations for equilibrium DC quantities.
    """
    V_s = sym.Symbol("V_s", real=True, positive=True)
    R_bat = sym.Symbol("R_bat", real=True, positive=True)
    A_loss = sym.Symbol("A_loss", real=True)
    B_loss = sym.Symbol("B_loss", real=True)
    C_loss = sym.Symbol("C_loss", real=True)
    e = sym.Symbol("e", real=True, positive=True)

    p_dc = sym.Symbol("p_dc", real=True)
    i_dc = sym.Symbol("i_dc", real=True)
    v_dc = sym.Symbol("v_dc", real=True)
    p_s = sym.Symbol("p_s", real=True)
    q_s = sym.Symbol("q_s", real=True)

    s_s = sym.sqrt(p_s**2 + q_s**2)
    i_s = s_s / V_s
    p_loss = A_loss * i_s**2 + B_loss * i_s + C_loss

    g_p_dc = p_s + p_loss - p_dc
    g_i_dc = v_dc * i_dc - p_dc
    g_v_dc = e - i_dc * R_bat - v_dc

    unknown = [p_dc, i_dc, v_dc]
    solution = sym.solve([g_p_dc, g_i_dc, g_v_dc], unknown)
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

    grid = BpsBuilder("bess_dq_vrt.hjson")
    grid.checker()
    grid.uz_jacs = False
    grid.construct("temp_bess_dq_vrt")
    bld = Builder(grid.sys_dict, target="ctypes", sparse=False)
    bld.build()


def test_ini():
    from pydae.core import Model

    model = Model("temp_bess_dq_vrt")

    model.ini({"soc_ref_1": 0.5}, "xy_0.json")
    model.report_x()
    model.report_y()


def test_run():
    import matplotlib.pyplot as plt
    from pydae.core import Model

    model = Model("temp_bess_dq_vrt")

    model.ini({"soc_ref_1": 0.5, "p_s_ppc_1": 0.0, "q_s_ppc_1": 0.0},
              "xy_0.json")
    model.report_x()
    model.report_y()

    model.run(1.0, {})

    # Discharge for 10 seconds
    model.run(10.0, {"p_s_ppc_1": 1.0})

    # Trigger LVRT at t=11s
    model.run(2.0, {"lvrt_ext_1": 1.0, "i_sa_ref_1": 0.5, "i_sr_ref_1": 0.3})

    model.post()

    fig, axes = plt.subplots(3, 1, figsize=(8, 8))

    axes[0].plot(model.Time, model.get_values("soc_1"),
                 label="$soc$ (pu)", color="b")
    axes[0].set_ylabel("SOC")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(model.Time, model.get_values("p_s_1"),
                 label="$p_s$ (pu)", color="r")
    axes[1].set_ylabel("Active Power")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(model.Time, model.get_values("i_mod_1"),
                 label="$i_{mod}$ (pu)", color="g")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Current Magnitude")
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig("bess_dq_vrt_response.svg")
    print("Test completed. Plot saved as 'bess_dq_vrt_response.svg'.")


if __name__ == "__main__":
    # symbolic_dev()
    test_build()
    test_ini()
    # test_run()
