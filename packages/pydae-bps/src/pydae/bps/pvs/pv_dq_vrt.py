# -*- coding: utf-8 -*-
r"""
PV inverter model with VRT (voltage ride-through) capability.

**Auxiliar equations**

$$v_{sr} = V \cos(\theta), \quad v_{si} = V \sin(\theta)$$
$$v_{sq\_mag} = v_{sr}^2 + v_{si}^2 + \epsilon$$
$$v_m = \sqrt{v_{sr}^2 + v_{si}^2}$$

**Dynamic equations**

$$\frac{d\,\text{lvrt\_ext\_ramp}}{dt} = \frac{\text{lvrt\_ext} - \text{lvrt\_ext\_ramp}}{T_{lvrt}}$$

**Current references (PQ mode)**

$$i_{sr,pq} = \frac{p_{s,ppc}\, v_{sr} + q_{s,ppc}\, v_{si}}{v_{sq\_mag}}$$
$$i_{si,pq} = \frac{p_{s,ppc}\, v_{si} - q_{s,ppc}\, v_{sr}}{v_{sq\_mag}}$$

**Current references (arbitrary mode)**

$$i_{sr,ar} = \frac{i_{sa,ref}\, v_{sr} + i_{sr,ref}\, v_{si}}{v_m}$$
$$i_{si,ar} = \frac{i_{sa,ref}\, v_{si} - i_{sr,ref}\, v_{sr}}{v_m}$$

**Blending and soft saturation**

$$i_{sr,nosat} = (1 - \text{lvrt\_ramp}) i_{sr,pq} + \text{lvrt\_ramp} \cdot i_{sr,ar}$$
$$i_{si,nosat} = (1 - \text{lvrt\_ramp}) i_{si,pq} + \text{lvrt\_ramp} \cdot i_{si,ar}$$
$$i_{mod} = \sqrt{i_{sr,nosat}^2 + i_{si,nosat}^2 + \epsilon}$$
$$i_{mod,sat} = \frac{1}{2} \left(i_{mod} + I_{max} - \sqrt{(i_{mod} - I_{max})^2 + \epsilon}\right)$$

**Algebraic equations**

$$0 = v_{sr} i_{sr} + v_{si} i_{si} - p_s$$
$$0 = v_{si} i_{sr} - v_{sr} i_{si} - q_s$$
"""
import io

import numpy as np
import sympy as sym  # noqa: F401 - used by symbolic_dev()


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.
    """
    descriptions_list = []

    # Parameters
    descriptions_list += [
        {"type": "Parameter", "tex": "S_n", "data": "S_n", "model": "S_n",
         "default": 1e6, "description": "Nominal power", "units": "VA"}]
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
         "model": "p_s_ppc", "default": 1.0,
         "description": "Active power reference from PPC", "units": "pu"}]
    descriptions_list += [
        {"type": "Input", "tex": "q_{s,ppc}", "data": "q_s_ppc",
         "model": "q_s_ppc", "default": 0.0,
         "description": "Reactive power reference from PPC", "units": "pu"}]
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
        {"type": "Dynamic State", "tex": "\\text{lvrt}_{\\text{ext,ramp}}",
         "data": "", "model": "lvrt_ext_ramp", "default": "",
         "description": "LVRT trigger ramped signal", "units": "-"}]

    # Algebraic States
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
        {"type": "Output", "tex": "i_{mod}", "data": "",
         "model": "i_mod", "default": "",
         "description": "Current magnitude before saturation", "units": "pu"}]
    descriptions_list += [
        {"type": "Output", "tex": "i_{mod,sat}", "data": "",
         "model": "i_mod_sat", "default": "",
         "description": "Saturated current magnitude", "units": "pu"}]

    return descriptions_list


def pv_dq_vrt(grid, name, bus_name, data_dict):
    """
    PV inverter with VRT capability in stationary (r-i) coordinates.

    The model represents a grid-following inverter whose inner current
    loops are assumed ideal (algebraic).  Power references come from a
    plant-level controller (PPC) and can be overridden by an arbitrary
    current reference mode when an LVRT trigger is asserted.  Current
    magnitude is limited via a smooth (differentiable) saturation
    function to maintain numerical stability during Newton solves.

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
    lvrt_ext = backend.symbols(f"lvrt_ext_{name}")
    i_sa_ref = backend.symbols(f"i_sa_ref_{name}")
    i_sr_ref = backend.symbols(f"i_sr_ref_{name}")

    # 4. Dynamic states
    lvrt_ext_ramp = backend.symbols(f"lvrt_ext_ramp_{name}")

    # 5. Algebraic states
    p_s = backend.symbols(f"p_s_{name}")
    q_s = backend.symbols(f"q_s_{name}")

    # 6. Parameters
    S_n = backend.symbols(f"S_n_{name}")
    I_max = backend.symbols(f"I_max_{name}")
    T_lvrt = backend.symbols(f"T_lvrt_{name}")
    Epsilon = backend.symbols(f"Epsilon_{name}")

    # 7. Dynamic equation for the LVRT ramp
    dlvrt_ext_ramp = (lvrt_ext - lvrt_ext_ramp) / T_lvrt

    # 8. Grid voltages in stationary (r-i) frame
    v_sr = V_s * cos(theta_s)
    v_si = V_s * sin(theta_s)
    v_sq_mag = v_sr**2 + v_si**2 + Epsilon
    v_m = backend.sqrt(v_sq_mag)

    # 9. PQ-mode current references
    i_sr_pq = (p_s_ppc * v_sr + q_s_ppc * v_si) / v_sq_mag
    i_si_pq = (p_s_ppc * v_si - q_s_ppc * v_sr) / v_sq_mag

    # 10. Arbitrary-mode current references
    i_sr_ar = (i_sa_ref * v_sr + i_sr_ref * v_si) / v_m
    i_si_ar = (i_sa_ref * v_si - i_sr_ref * v_sr) / v_m

    # 11. Blend modes via LVRT ramp
    i_sr_nosat = (1.0 - lvrt_ext_ramp) * i_sr_pq + lvrt_ext_ramp * i_sr_ar
    i_si_nosat = (1.0 - lvrt_ext_ramp) * i_si_pq + lvrt_ext_ramp * i_si_ar

    # 12. Soft saturation (smooth, differentiable)
    i_mod = backend.sqrt(i_sr_nosat**2 + i_si_nosat**2 + Epsilon)
    i_mod_sat = 0.5 * (i_mod + I_max - backend.sqrt(
        (i_mod - I_max)**2 + Epsilon
    ))
    ratio = i_mod_sat / i_mod
    i_sr = i_sr_nosat * ratio
    i_si = i_si_nosat * ratio

    # 13. Injected power from saturated currents
    p_s_expr = v_sr * i_sr + v_si * i_si
    q_s_expr = v_si * i_sr - v_sr * i_si

    g_p_s = p_s_expr - p_s
    g_q_s = q_s_expr - q_s

    # 14. Assembly
    grid.dae["f"] += [dlvrt_ext_ramp]
    grid.dae["x"] += [lvrt_ext_ramp]
    grid.dae["g"] += [g_p_s, g_q_s]
    grid.dae["y_ini"] += [p_s, q_s]
    grid.dae["y_run"] += [p_s, q_s]

    # 15. Dynamic input handling
    p_s_ppc_val = data_dict.get("p_s_ppc", default_map.get("p_s_ppc", 1.0))
    q_s_ppc_val = data_dict.get("q_s_ppc", default_map.get("q_s_ppc", 0.0))

    grid.dae["u_ini_dict"].update({str(lvrt_ext): 0.0})
    grid.dae["u_run_dict"].update({str(lvrt_ext): 0.0})

    grid.dae["u_ini_dict"].update({str(p_s_ppc): p_s_ppc_val})
    grid.dae["u_run_dict"].update({str(p_s_ppc): p_s_ppc_val})

    grid.dae["u_ini_dict"].update({str(q_s_ppc): q_s_ppc_val})
    grid.dae["u_run_dict"].update({str(q_s_ppc): q_s_ppc_val})

    grid.dae["u_ini_dict"].update({str(i_sa_ref): 0.0})
    grid.dae["u_run_dict"].update({str(i_sa_ref): 0.0})

    grid.dae["u_ini_dict"].update({str(i_sr_ref): 0.0})
    grid.dae["u_run_dict"].update({str(i_sr_ref): 0.0})

    # 16. Parameters
    S_n_val = data_dict.get("S_n", default_map.get("S_n", 1e6))
    I_max_val = data_dict.get("I_max", default_map.get("I_max", 1.2))
    T_lvrt_val = data_dict.get("T_lvrt", default_map.get("T_lvrt", 0.02))
    Epsilon_val = data_dict.get("Epsilon", default_map.get("Epsilon", 1e-8))

    grid.dae["params_dict"].update({str(S_n): S_n_val})
    grid.dae["params_dict"].update({str(I_max): I_max_val})
    grid.dae["params_dict"].update({str(T_lvrt): T_lvrt_val})
    grid.dae["params_dict"].update({str(Epsilon): Epsilon_val})

    # 17. Initialization hints
    grid.dae["xy_0_dict"].update({str(lvrt_ext_ramp): 0.0})
    grid.dae["xy_0_dict"].update({str(p_s): p_s_ppc_val})
    grid.dae["xy_0_dict"].update({str(q_s): q_s_ppc_val})

    # 18. Outputs
    grid.dae["h_dict"].update({f"p_s_{name}": p_s})
    grid.dae["h_dict"].update({f"q_s_{name}": q_s})
    grid.dae["h_dict"].update({f"i_mod_{name}": i_mod})
    grid.dae["h_dict"].update({f"i_mod_sat_{name}": i_mod_sat})
    grid.dae["h_dict"].update({f"lvrt_ext_ramp_{name}": lvrt_ext_ramp})

    # 19. Optional monitor outputs
    if data_dict.get("monitor", False):
        R_s = backend.symbols(f"R_s_{name}")
        X_s = backend.symbols(f"X_s_{name}")
        U_n = backend.symbols(f"U_n_{name}")
        V_dc_b = U_n * np.sqrt(2)

        v_tr = v_sr + R_s * i_sr - X_s * i_si
        v_ti = v_si + R_s * i_si + X_s * i_sr
        v_t_m = backend.sqrt(v_tr**2 + v_ti**2)

        K_vt = backend.symbols(f"K_vt_{name}")
        K_it = backend.symbols(f"K_it_{name}")
        V_oc = backend.symbols(f"V_oc_{name}")
        V_mp = backend.symbols(f"V_mp_{name}")
        I_sc = backend.symbols(f"I_sc_{name}")
        I_mp = backend.symbols(f"I_mp_{name}")
        temp_deg = backend.symbols(f"temp_deg_{name}")
        irrad = backend.symbols(f"irrad_{name}")
        N_pv_s = backend.symbols(f"N_pv_s_{name}")
        N_pv_p = backend.symbols(f"N_pv_p_{name}")

        T_stc_deg = 25.0
        V_oc_t = N_pv_s * V_oc * (1 + K_vt / 100.0 * (temp_deg - T_stc_deg))
        V_mp_t = N_pv_s * V_mp * (1 + K_vt / 100.0 * (temp_deg - T_stc_deg))
        I_mp_t = N_pv_p * I_mp * (1 + K_it / 100.0 * (temp_deg - T_stc_deg))
        I_mp_i = I_mp_t * irrad / 1000.0

        v_1, i_1 = V_mp_t, I_mp_i
        v_2, i_2 = V_oc_t, 0.0

        m_pv = (v_1 - v_2) / (i_1 - i_2)
        B_coef = -(v_1 - i_1 * m_pv)
        C_coef = -(p_s * S_n * m_pv)

        v_dc_v_expr = (-B_coef + backend.sqrt(B_coef**2 - 4 * C_coef)) / 2.0
        i_pv_expr = (p_s * S_n) / v_dc_v_expr
        v_dc_expr = v_dc_v_expr / V_dc_b
        m_ref_expr = v_t_m / v_dc_expr

        grid.dae["u_ini_dict"].update({str(irrad): 1000.0})
        grid.dae["u_run_dict"].update({str(irrad): 1000.0})
        grid.dae["u_ini_dict"].update({str(temp_deg): 25.0})
        grid.dae["u_run_dict"].update({str(temp_deg): 25.0})

        grid.dae["params_dict"].update({str(U_n): data_dict["U_n"]})
        grid.dae["params_dict"].update({str(R_s): data_dict["R_s"]})
        grid.dae["params_dict"].update({str(X_s): data_dict["X_s"]})
        grid.dae["params_dict"].update({str(I_sc): data_dict["I_sc"]})
        grid.dae["params_dict"].update({str(I_mp): data_dict["I_mp"]})
        grid.dae["params_dict"].update({str(V_mp): data_dict["V_mp"]})
        grid.dae["params_dict"].update({str(V_oc): data_dict["V_oc"]})
        grid.dae["params_dict"].update({str(N_pv_s): data_dict["N_pv_s"]})
        grid.dae["params_dict"].update({str(N_pv_p): data_dict["N_pv_p"]})
        grid.dae["params_dict"].update({str(K_vt): data_dict["K_vt"]})
        grid.dae["params_dict"].update({str(K_it): data_dict["K_it"]})

        grid.dae["h_dict"].update({f"v_m_{name}": v_m})
        grid.dae["h_dict"].update({f"i_sr_{name}": i_sr})
        grid.dae["h_dict"].update({f"i_si_{name}": i_si})
        grid.dae["h_dict"].update({f"v_tr_{name}": v_tr})
        grid.dae["h_dict"].update({f"v_ti_{name}": v_ti})
        grid.dae["h_dict"].update({f"v_dc_v_{name}": v_dc_v_expr})
        grid.dae["h_dict"].update({f"v_dc_{name}": v_dc_expr})
        grid.dae["h_dict"].update({f"i_pv_{name}": i_pv_expr})
        grid.dae["h_dict"].update({f"m_ref_{name}": m_ref_expr})

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
    Symbolic steady-state analysis for PV VRT model.
    Derives current reference expressions from power balance.
    """
    v_sd = sym.Symbol("v_sd", real=True)
    v_sq = sym.Symbol("v_sq", real=True)
    p_s_ref = sym.Symbol("p_s_ref", real=True)
    q_s_ref = sym.Symbol("q_s_ref", real=True)
    i_sa_ref = sym.Symbol("i_sa_ref", real=True)
    i_sr_ref = sym.Symbol("i_sr_ref", real=True)
    i_sd_pq_ref, i_sq_pq_ref = sym.symbols("i_sd_pq_ref, i_sq_pq_ref",
                                            real=True)
    i_sd_ar_ref, i_sq_ar_ref = sym.symbols("i_sd_ar_ref, i_sq_ar_ref",
                                            real=True)

    g_i_sd_pq_ref = i_sd_pq_ref * v_sd + i_sq_pq_ref * v_sq - p_s_ref
    g_i_sq_pq_ref = -i_sq_pq_ref * v_sd + i_sd_pq_ref * v_sq - q_s_ref

    sol = sym.solve([g_i_sd_pq_ref, g_i_sq_pq_ref],
                    [i_sd_pq_ref, i_sq_pq_ref])
    for item in sol:
        print(f"{item} = {sol[item]}")

    v_m = sym.sqrt(v_sd**2 + v_sq**2)
    g_i_sd_ar_ref = i_sd_ar_ref * v_sd / v_m + i_sq_ar_ref * v_sq / v_m - i_sa_ref
    g_i_sq_ar_ref = -i_sq_ar_ref * v_sd / v_m + i_sd_ar_ref * v_sq / v_m - i_sr_ref

    sol = sym.solve([g_i_sd_ar_ref, g_i_sq_ar_ref],
                    [i_sd_ar_ref, i_sq_ar_ref])
    for item in sol:
        print(f"{item} = {sol[item]}")

    v_tr, v_ti = sym.symbols("v_tr, v_ti", real=True)
    i_sr, i_si = sym.symbols("i_sr, i_si", real=True)
    v_sr, v_si = sym.symbols("v_sr, v_si", real=True)
    R_s, X_s = sym.symbols("R_s, X_s", real=True)

    g_i_si = v_ti - R_s * i_si + X_s * i_sr - v_si
    g_i_sr = v_tr - R_s * i_sr - X_s * i_si - v_sr
    sol = sym.solve([g_i_sr, g_i_si], [i_sr, i_si])
    for item in sol:
        print(f"{item} = {sol[item]}")


# =============================================================================
# Testing Block
# =============================================================================
def test_build():
    from pydae.bps import BpsBuilder
    from pydae.core import Builder

    grid = BpsBuilder("pv_dq_vrt.hjson")
    grid.checker()
    grid.uz_jacs = False
    grid.construct("temp_pv_dq_vrt")
    bld = Builder(grid.sys_dict, target="ctypes", sparse=False)
    bld.build()


def test_ini():
    from pydae.core import Model

    model = Model("temp_pv_dq_vrt")

    model.ini({}, "xy_0.json")
    model.report_x()
    model.report_y()


def test_run():
    import matplotlib.pyplot as plt
    from pydae.core import Model

    model = Model("temp_pv_dq_vrt")

    model.ini({"p_s_ppc_1": 0.5, "q_s_ppc_1": 0.0}, "xy_0.json")
    model.report_x()
    model.report_y()

    model.run(1.0, {})

    # LVRT trigger at t=1s
    model.run(2.0, {"lvrt_ext_1": 1.0, "i_sa_ref_1": 0.5, "i_sr_ref_1": 0.3})

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

    axes[2].plot(model.Time, model.get_values("i_mod_1"),
                 label="$i_{mod}$ (pu)", color="g")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Current Magnitude")
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig("pv_dq_vrt_response.svg")
    print("Test completed. Plot saved as 'pv_dq_vrt_response.svg'.")


if __name__ == "__main__":
    # symbolic_dev()
    test_build()
    test_ini()
    # test_run()
