# -*- coding: utf-8 -*-
r"""
Battery energy storage system (BESS) with VSC PQ control and SOC dynamics
(steady-state DC side, filtered AC power references).

**Auxiliar equations**

$$H = \frac{E_{kWh} \cdot 1000 \cdot 3600}{S_n}$$
$$\epsilon = soc_{ref} - soc$$
$$p_{soc} = -(K_p \epsilon + K_i \xi_{soc})$$
$$e = f_{spline}(soc) \quad \text{(OCV-SOC interpolation)}$$
$$s_s = \sqrt{p_s^2 + q_s^2}$$
$$i_s = s_s / V$$
$$p_{loss} = A_{loss} i_s^2 + B_{loss} i_s + C_{loss}$$

**Dynamic equations**

$$\frac{d\,soc}{dt} = \frac{1}{H} (-i_{dc} \cdot e)$$
$$\frac{d\,\xi_{soc}}{dt} = soc_{ref} - soc$$
$$\dot{x} = A_{pq} x + B_{pq} \begin{bmatrix} p_{s,ppc} \\ q_{s,ppc} \end{bmatrix}$$

**Algebraic equations**

$$0 = p_s + p_{loss} - p_{dc}$$
$$0 = v_{dc} i_{dc} - p_{dc}$$
$$0 = e - i_{dc} R_{bat} - v_{dc}$$

**Power reference with SOC limits**

$$p_s = \begin{cases} p_{s,ref} & p_{s,ref} \leq 0 \land soc < soc_{max} \\
p_{s,ref} & p_{s,ref} > 0 \land soc > soc_{min} \\
0 & \text{otherwise} \end{cases} + p_{soc}$$
$$q_s = q_{s,ref}$$
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
        {"type": "Dynamic State", "tex": "x_{pq}", "data": "",
         "model": "x_pq", "default": "",
         "description": "State-space filter states", "units": "-"}]

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
        {"type": "Output", "tex": "p_s", "data": "",
         "model": "p_s", "default": "",
         "description": "Injected active power", "units": "pu"}]
    descriptions_list += [
        {"type": "Output", "tex": "q_s", "data": "",
         "model": "q_s", "default": "",
         "description": "Injected reactive power", "units": "pu"}]
    descriptions_list += [
        {"type": "Output", "tex": "p_{s,ref}", "data": "",
         "model": "p_s_ref", "default": "",
         "description": "Filtered active power reference", "units": "pu"}]
    descriptions_list += [
        {"type": "Output", "tex": "q_{s,ref}", "data": "",
         "model": "q_s_ref", "default": "",
         "description": "Filtered reactive power reference", "units": "pu"}]

    return descriptions_list


def bess_pq_ss(grid, name, bus_name, data_dict):
    """
    BESS with VSC PQ control and SOC dynamics (steady-state DC side).

    The model combines a battery (SOC dynamics + OCV-SOC curve) with a
    grid-following VSC whose inner current loops are assumed ideal.
    Power references from a plant controller pass through an optional
    state-space filter (A_pq/B_pq/C_pq/D_pq).  SOC is regulated via a
    PI controller that modulates the active power setpoint.

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
    # NOTE: bess_pq_ss uses sympy symbols directly for the OCV-SOC spline
    # (interpolating_spline requires sympy).  All other symbols and math
    # operations use the backend abstraction for SymPy/CasADi compatibility.

    # 1. Fetch metadata and defaults
    meta = descriptions()
    default_map = {
        item["data"]: item["default"]
        for item in meta if "data" in item and item["data"]
    }

    # 2. Bus-side inputs
    V_s = backend.symbols(f"V_{bus_name}")

    # 3. PPC / external inputs
    soc_ref = backend.symbols(f"soc_ref_{name}")
    p_s_ppc = backend.symbols(f"p_s_ppc_{name}")
    q_s_ppc = backend.symbols(f"q_s_ppc_{name}")

    # 4. Dynamic states - battery
    soc = backend.symbols(f"soc_{name}")
    xi_soc = backend.symbols(f"xi_soc_{name}")

    # 5. Algebraic states - DC side
    p_dc = backend.symbols(f"p_dc_{name}")
    i_dc = backend.symbols(f"i_dc_{name}")
    v_dc = backend.symbols(f"v_dc_{name}")

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

    # 7. OCV-SOC curve (piecewise interpolation or linear model)
    soc_sym = sym.Symbol(f"soc_{name}", real=True)
    if "socs" in data_dict:
        socs = np.array(data_dict["socs"])
        es = np.array(data_dict["es"])
        e_max = float(np.max(es))
        soc_ref_N = data_dict["soc_ref"]
        e_ini = float(np.interp(soc_ref_N, socs, es))
        if backend.use_casadi:
            import casadi as ca
            # Build CasADi piecewise linear interpolation with if_else
            e = ca.SX(e_max)
            for _i in range(len(socs) - 2, -1, -1):
                _x0, _x1 = float(socs[_i]), float(socs[_i + 1])
                _y0, _y1 = float(es[_i]), float(es[_i + 1])
                _seg = _y0 + (_y1 - _y0) / (_x1 - _x0) * (soc - _x0)
                e = ca.if_else(soc < _x1, _seg, e)
            e = ca.if_else(soc < float(socs[0]), float(es[0]), e)
        else:
            e_soc_order = data_dict.get("e_soc_order", 1)
            interpolation = interpolating_spline(e_soc_order, soc_sym, socs, es)
            interpolation._args = tuple(
                list(interpolation._args)
                + [sym.functions.elementary.piecewise.ExprCondPair(e_max, True)]
            )
            e = interpolation.subs(soc_sym, soc)
    else:
        B_0 = backend.symbols(f"B_0_{name}")
        B_1 = backend.symbols(f"B_1_{name}")
        e = B_0 + B_1 * soc
        e_ini = data_dict.get("B_0", 1.0)

    # 8. Auxiliary equations
    H = E_kWh * 1000.0 * 3600 / S_n
    epsilon = soc_ref - soc
    p_soc = -(K_p * epsilon + K_i * xi_soc)

    # 9. State-space filter for power references
    from pydae.bps.utils.ss_num2sym import ss_num2sym

    def _replace(expr, old, new):
        if backend.use_casadi:
            import casadi as ca
            return ca.substitute(expr, old, new)
        return expr.replace(old, new)

    def _to_list(mat):
        if backend.use_casadi:
            return [mat[i, 0] for i in range(mat.shape[0])]
        return list(mat)

    A_mat = np.array([[-10.0, 0.0],
                      [0.0, -10.0]])
    B_mat = np.array([[10.0, 0.0],
                      [0.0, 10.0]])
    C_mat = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
    D_mat = np.array([[0.0, 0.0],
                      [0.0, 0.0]])

    if "A_pq" in data_dict:
        A_mat = np.array(data_dict["A_pq"])
        B_mat = np.array(data_dict["B_pq"])
        C_mat = np.array(data_dict["C_pq"])
        D_mat = np.array(data_dict["D_pq"])

    sys = ss_num2sym(f"{name}", A_mat, B_mat, C_mat, D_mat, backend=backend)
    p_s_ppc_sym = backend.symbols(f"p_s_ppc_{name}")
    q_s_ppc_sym = backend.symbols(f"q_s_ppc_{name}")
    sys["dx"] = _replace(sys["dx"], sys["u"][0, 0], p_s_ppc_sym)
    sys["dx"] = _replace(sys["dx"], sys["u"][1, 0], q_s_ppc_sym)
    sys["z_evaluated"] = _replace(sys["z_evaluated"], sys["u"][0, 0], p_s_ppc_sym)
    sys["z_evaluated"] = _replace(sys["z_evaluated"], sys["u"][1, 0], q_s_ppc_sym)
    p_s_ref = sys["z_evaluated"][0, 0]
    q_s_ref = sys["z_evaluated"][1, 0]

    # 10. Power limiting based on SOC bounds
    if backend.use_casadi:
        import casadi as ca
        _cond_dch = ca.logic_and(p_s_ref <= 0.0, soc < soc_max)
        _cond_ch  = ca.logic_and(p_s_ref > 0.0,  soc > soc_min)
    else:
        _cond_dch = (p_s_ref <= 0.0) & (soc < soc_max)
        _cond_ch  = (p_s_ref > 0.0)  & (soc > soc_min)
    p_s = backend.Piecewise(
        (p_s_ref, _cond_dch),
        (p_s_ref, _cond_ch),
        (0.0, True),
    ) + p_soc
    q_s = q_s_ref

    # 11. Loss model  (epsilon avoids d(sqrt)/d(p_s)=0/0 when p_s=q_s=0)
    s_s = backend.sqrt(p_s**2 + q_s**2 + 1e-8)
    i_s = s_s / V_s
    p_loss = A_loss * i_s**2 + B_loss * i_s + C_loss

    # 12. Dynamic equations
    dsoc = 1 / H * (-i_dc * e)
    dxi_soc = epsilon

    # 13. Algebraic equations - DC side
    g_p_dc = p_s + p_loss - p_dc
    g_i_dc = v_dc * i_dc - p_dc
    g_v_dc = e - i_dc * R_bat - v_dc

    # 14. Assembly
    grid.dae["f"] += [dsoc, dxi_soc] + _to_list(sys["dx"])
    grid.dae["x"] += [soc, xi_soc] + _to_list(sys["x"])
    grid.dae["g"] += [g_p_dc, g_i_dc, g_v_dc]
    grid.dae["y_ini"] += [p_dc, i_dc, v_dc]
    grid.dae["y_run"] += [p_dc, i_dc, v_dc]

    # 15. Dynamic input handling
    p_s_ppc_val = data_dict.get("p_s_ppc", default_map.get("p_s_ppc", 0.0))
    q_s_ppc_val = data_dict.get("q_s_ppc", default_map.get("q_s_ppc", 0.0))
    soc_ref_val = data_dict.get("soc_ref", default_map.get("soc_ref", 0.5))

    grid.dae["u_ini_dict"].update({str(p_s_ppc): p_s_ppc_val})
    grid.dae["u_run_dict"].update({str(p_s_ppc): p_s_ppc_val})

    grid.dae["u_ini_dict"].update({str(q_s_ppc): q_s_ppc_val})
    grid.dae["u_run_dict"].update({str(q_s_ppc): q_s_ppc_val})

    grid.dae["u_ini_dict"].update({str(soc_ref): soc_ref_val})
    grid.dae["u_run_dict"].update({str(soc_ref): soc_ref_val})

    # 16. Outputs
    grid.dae["h_dict"].update({f"p_loss_{name}": p_loss})
    grid.dae["h_dict"].update({f"i_s_{name}": i_s})
    grid.dae["h_dict"].update({f"e_{name}": e})
    grid.dae["h_dict"].update({f"i_dc_{name}": i_dc})
    grid.dae["h_dict"].update({f"p_s_{name}": p_s})
    grid.dae["h_dict"].update({f"q_s_{name}": q_s})
    grid.dae["h_dict"].update({f"p_s_ref_{name}": p_s_ref})
    grid.dae["h_dict"].update({f"q_s_ref_{name}": q_s_ref})
    grid.dae["h_dict"].update({f"p_s_ppc_{name}": p_s_ppc})
    grid.dae["h_dict"].update({f"q_s_ppc_{name}": q_s_ppc})

    # 17. Parameters - SOC controller
    K_p_val = data_dict.get("K_p", default_map.get("K_p", 1e-6))
    K_i_val = data_dict.get("K_i", default_map.get("K_i", 1e-6))
    soc_min_val = data_dict.get("soc_min", default_map.get("soc_min", 0.0))
    soc_max_val = data_dict.get("soc_max", default_map.get("soc_max", 1.0))
    grid.dae["params_dict"].update({str(K_p): K_p_val})
    grid.dae["params_dict"].update({str(K_i): K_i_val})
    grid.dae["params_dict"].update({str(soc_min): soc_min_val})
    grid.dae["params_dict"].update({str(soc_max): soc_max_val})

    # 18. Parameters - BESS sizing
    grid.dae["params_dict"].update({str(S_n): data_dict["S_n"]})
    grid.dae["params_dict"].update({str(E_kWh): data_dict["E_kWh"]})

    # 19. Parameters - Loss model
    A_loss_N = data_dict.get("A_loss", default_map.get("A_loss", 0.0001))
    B_loss_N = data_dict.get("B_loss", default_map.get("B_loss", 0.0))
    C_loss_N = data_dict.get("C_loss", default_map.get("C_loss", 0.0001))
    grid.dae["params_dict"].update({str(A_loss): A_loss_N})
    grid.dae["params_dict"].update({str(B_loss): B_loss_N})
    grid.dae["params_dict"].update({str(C_loss): C_loss_N})

    # 20. Parameters - Battery resistance
    R_bat_N = data_dict.get("R_bat", default_map.get("R_bat", 0.0))
    grid.dae["params_dict"].update({str(R_bat): R_bat_N})

    # 21. Parameters - State-space filter matrices
    grid.dae["params_dict"].update(sys["params_dict"])

    # 22. Initialization hints
    grid.dae["xy_0_dict"].update({str(v_dc): e_ini})
    grid.dae["xy_0_dict"].update({str(soc): data_dict.get("soc_ref", 0.5)})
    grid.dae["xy_0_dict"].update({str(xi_soc): 0.0})

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
    Symbolic steady-state analysis for BESS model.
    Solves algebraic equations for equilibrium DC quantities.
    """
    # Parameters
    V_s = sym.Symbol("V_s", real=True, positive=True)
    R_bat = sym.Symbol("R_bat", real=True, positive=True)
    A_loss = sym.Symbol("A_loss", real=True)
    B_loss = sym.Symbol("B_loss", real=True)
    C_loss = sym.Symbol("C_loss", real=True)
    e = sym.Symbol("e", real=True, positive=True)

    # Algebraic states
    p_dc = sym.Symbol("p_dc", real=True)
    i_dc = sym.Symbol("i_dc", real=True)
    v_dc = sym.Symbol("v_dc", real=True)
    p_s = sym.Symbol("p_s", real=True)
    q_s = sym.Symbol("q_s", real=True)

    # Auxiliary
    s_s = sym.sqrt(p_s**2 + q_s**2)
    i_s = s_s / V_s
    p_loss = A_loss * i_s**2 + B_loss * i_s + C_loss

    # Algebraic equations
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

    grid = BpsBuilder("bess_pq_ss.hjson")
    grid.checker()
    grid.uz_jacs = False
    grid.construct("temp_bess_pq_ss")
    bld = Builder(grid.sys_dict, target="ctypes", sparse=False)
    bld.build()


def test_ini():
    from pydae.core import Model

    model = Model("temp_bess_pq_ss")

    model.ini({"soc_ref_1": 0.5}, "xy_0.json")
    model.report_x()
    model.report_y()


def test_run():
    import matplotlib.pyplot as plt
    from pydae.core import Model

    model = Model("temp_bess_pq_ss")

    model.ini({"soc_ref_1": 0.5, "p_s_ppc_1": 0.0, "q_s_ppc_1": 0.0},
              "xy_0.json")
    model.report_x()
    model.report_y()

    # Run to steady state
    model.run(1.0, {})

    # Discharge for 10 seconds
    model.run(10.0, {"p_s_ppc_1": 1.0})

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

    axes[2].plot(model.Time, model.get_values("v_dc_1"),
                 label="$v_{dc}$ (pu)", color="g")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("DC Voltage")
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig("bess_pq_ss_response.svg")
    print("Test completed. Plot saved as 'bess_pq_ss_response.svg'.")


if __name__ == "__main__":
    # symbolic_dev()
    test_build()
    test_ini()
    # test_run()
