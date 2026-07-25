# -*- coding: utf-8 -*-
r"""
PMSM full-converter wind turbine — simplified p/q model.

A reduced version of :func:`pmsm_1` that keeps the **wind-turbine dynamics**
(aerodynamics, 2-mass drivetrain, pitch control, MPPT) but replaces the whole
**electrical stack** (PMSM machine equations, machine-side VSC, DC-link, PLL,
grid-side current control, L-filter) with a purely **algebraic grid-following
p/q injection** in stationary (r-i) coordinates — exactly the simplification
``pv_pq_ss`` applies to ``pv_dq_ss`` on the PV side.

The available active power comes from the wind through the MPPT reference; the
grid injection is current-limited (smooth saturation) and supports a VRT blend
into an arbitrary current-injection mode.  An optional state-space low-pass
filter can shape the reactive-power reference.

**Wind / mechanical side (retained)**

$$\nu_w = \nu_{w,0} + \nu_{w,\text{ramp}}, \qquad
  \lambda = \Lambda_b\,\frac{\omega_t/\Omega_{t,b}}{\nu_w/N_{u_w,b}}$$
$$c_p = C_1(C_2/\lambda_i - C_3\beta - C_4)e^{-C_5/\lambda_i} + C_6\lambda,
  \qquad p_w = K_{pow}\,\frac{c_p}{c_{p,b}}\left(\frac{\nu_w}{N_{u_w,b}}\right)^3$$
$$2H_t\,\dot\omega_t = p_w - K_{tr}\theta_{tr} - D_{tr}(\omega_t-\omega_r), \quad
  2H_r\,\dot\omega_r = K_{tr}\theta_{tr} + D_{tr}(\omega_t-\omega_r) - p_r$$

MPPT sets the electrical (rotor-braking) power reference:

$$p_{mppt} = K_{mppt3}\,\omega_t^3, \qquad
  T_{mppt}\,\dot p_{w,mppt} = p_{mppt} + p_{ref,ext} - p_{w,mppt}, \qquad
  p_r = p_{w,mppt}$$

**Grid injection (algebraic p/q, from pv_pq_ss)**

$$p_{s,ref} = p_{w,mppt}, \qquad q_{s,ref} = q_{s,ppc}$$
$$i_{sr,pq} = \frac{p_{s,ref} v_{sr} + q_{s,ref} v_{si}}{v_{sr}^2+v_{si}^2+\epsilon},
  \quad
  i_{si,pq} = \frac{p_{s,ref} v_{si} - q_{s,ref} v_{sr}}{v_{sr}^2+v_{si}^2+\epsilon}$$
$$i_{mod} = \sqrt{i_{sr}^2+i_{si}^2+\epsilon}, \quad
  i_{mod,sat} = \tfrac12\!\left(i_{mod}+I_{max}-\sqrt{(i_{mod}-I_{max})^2+\epsilon}\right)$$
$$0 = v_{sr} i_{sr} + v_{si} i_{si} - p_s, \qquad
  0 = v_{si} i_{sr} - v_{sr} i_{si} - q_s$$
"""
import io

import numpy as np
import sympy as sym


def descriptions():
    """Single source of truth for model parameters, inputs, states, outputs."""
    d = []

    # Parameters
    d += [{"type": "Parameter", "tex": "S_n", "data": "S_n", "model": "S_n",
           "default": 3e6, "description": "Nominal power", "units": "VA"}]
    d += [{"type": "Parameter", "tex": "F_n", "data": "F_n", "model": "F_n",
           "default": 50.0, "description": "Nominal frequency", "units": "Hz"}]
    d += [{"type": "Parameter", "tex": "H_t", "data": "H_t", "model": "H_t",
           "default": 4.0, "description": "Turbine inertia constant", "units": "s"}]
    d += [{"type": "Parameter", "tex": "H_r", "data": "H_r", "model": "H_r",
           "default": 0.8, "description": "Rotor (generator) inertia constant", "units": "s"}]
    d += [{"type": "Parameter", "tex": "w_{tr}", "data": "w_tr", "model": "w_tr",
           "default": 2.0, "description": "Drivetrain torsional frequency", "units": "Hz"}]
    d += [{"type": "Parameter", "tex": "d_{tr}", "data": "d_tr", "model": "d_tr",
           "default": 0.1, "description": "Drivetrain torsional damping", "units": "Hz"}]
    d += [{"type": "Parameter", "tex": "K_{mppt3}", "data": "K_mppt3", "model": "K_mppt3",
           "default": 0.4, "description": "MPPT cubic gain (p = K_mppt3·ω_t³)", "units": "pu"}]
    d += [{"type": "Parameter", "tex": "T_{mppt}", "data": "T_mppt", "model": "T_mppt",
           "default": 5.0, "description": "MPPT low-pass time constant", "units": "s"}]
    d += [{"type": "Parameter", "tex": "I_{max}", "data": "I_max", "model": "I_max",
           "default": 1.2, "description": "Maximum current magnitude", "units": "pu"}]
    d += [{"type": "Parameter", "tex": "T_{lvrt}", "data": "T_lvrt", "model": "T_lvrt",
           "default": 0.02, "description": "LVRT ramp time constant", "units": "s"}]
    d += [{"type": "Parameter", "tex": "\\epsilon", "data": "Epsilon", "model": "Epsilon",
           "default": 1e-8, "description": "Regularization for soft saturation", "units": "-"}]

    # Inputs
    d += [{"type": "Input", "tex": "\\nu_{w,0}", "data": "nu_w_0", "model": "nu_w_0",
           "default": 10.0, "description": "Base wind speed", "units": "m/s"}]
    d += [{"type": "Input", "tex": "q_{s,ppc}", "data": "q_s_ppc", "model": "q_s_ppc",
           "default": 0.0, "description": "Reactive power reference from PPC", "units": "pu"}]
    d += [{"type": "Input", "tex": "p_{ref,ext}", "data": "p_ref_ext", "model": "p_ref_ext",
           "default": 0.0, "description": "External active-power adjustment (≤0 curtails)", "units": "pu"}]
    d += [{"type": "Input", "tex": "\\beta_{ext}", "data": "beta_ext", "model": "beta_ext",
           "default": 0.0, "description": "External pitch-angle adder", "units": "deg"}]
    d += [{"type": "Input", "tex": "\\text{lvrt}_{ext}", "data": "lvrt_ext", "model": "lvrt_ext",
           "default": 0.0, "description": "External LVRT trigger (0=PQ, 1=arbitrary)", "units": "-"}]
    d += [{"type": "Input", "tex": "i_{sa,ref}", "data": "i_sa_ref", "model": "i_sa_ref",
           "default": 0.0, "description": "Active current reference (arbitrary mode)", "units": "pu"}]
    d += [{"type": "Input", "tex": "i_{sr,ref}", "data": "i_sr_ref", "model": "i_sr_ref",
           "default": 0.0, "description": "Reactive current reference (arbitrary mode)", "units": "pu"}]

    # Dynamic States
    d += [{"type": "Dynamic State", "tex": "\\nu_{w,ramp}", "model": "nu_w_ramp",
           "data": "", "default": "", "description": "Wind-speed ramp state", "units": "m/s"}]
    d += [{"type": "Dynamic State", "tex": "\\theta_{tr}", "model": "theta_tr",
           "data": "", "default": "", "description": "Drivetrain torsion angle", "units": "rad"}]
    d += [{"type": "Dynamic State", "tex": "\\omega_t", "model": "omega_t",
           "data": "", "default": "", "description": "Turbine speed", "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "\\omega_r", "model": "omega_r",
           "data": "", "default": "", "description": "Rotor (generator) speed", "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "\\xi_\\beta", "model": "xi_beta",
           "data": "", "default": "", "description": "Pitch PI integrator", "units": "-"}]
    d += [{"type": "Dynamic State", "tex": "\\beta", "model": "beta",
           "data": "", "default": "", "description": "Pitch angle", "units": "deg"}]
    d += [{"type": "Dynamic State", "tex": "p_{w,mppt}", "model": "p_w_mppt_lpf",
           "data": "", "default": "", "description": "Filtered MPPT power reference", "units": "pu"}]

    # Algebraic States
    d += [{"type": "Algebraic State", "tex": "p_s", "model": "p_s", "data": "", "default": "",
           "description": "Injected active power", "units": "pu"}]
    d += [{"type": "Algebraic State", "tex": "q_s", "model": "q_s", "data": "", "default": "",
           "description": "Injected reactive power", "units": "pu"}]

    # Outputs
    d += [{"type": "Output", "tex": "\\nu_w", "model": "nu_w", "data": "", "default": "",
           "description": "Wind speed", "units": "m/s"}]
    d += [{"type": "Output", "tex": "p_w", "model": "p_w", "data": "", "default": "",
           "description": "Aerodynamic power", "units": "pu"}]
    d += [{"type": "Output", "tex": "i_{mod,sat}", "model": "i_mod_sat", "data": "", "default": "",
           "description": "Saturated current magnitude", "units": "pu"}]

    return d


def pmsm_pq_ss(grid, name, bus_name, data_dict):
    r"""
    Simplified PMSM full-converter wind turbine (wind dynamics + algebraic p/q).

    Retains the aerodynamic, 2-mass mechanical, pitch and MPPT dynamics of
    :func:`pmsm_1`; the converter/machine electrical stack is reduced to an
    ideal, current-limited grid-following p/q injection in the r-i frame.  The
    MPPT reference drives both the rotor-braking power and the grid active-power
    injection (lossless assumption).

    Parameters
    ----------
    grid : BpsBuilder
        Builder instance (SymPy backend — this model is not CasADi-ported).
    name : str
        Unique instance identifier.
    bus_name : str
        Bus the turbine is connected to.
    data_dict : dict
        Parameters.  Optionally ``A_q``/``B_q``/``C_q``/``D_q`` to enable a
        state-space low-pass filter on the reactive-power reference.

    Returns
    -------
    p_W, q_var : expressions
        Active (W) and reactive (var) injection into the bus.
    """
    sin = sym.sin
    cos = sym.cos

    meta = descriptions()
    default_map = {it["data"]: it["default"] for it in meta if it.get("data")}

    def par(key):
        return data_dict.get(key, default_map.get(key))

    # 1. Bus voltages
    V_s = sym.Symbol(f"V_{bus_name}", real=True)
    theta_s = sym.Symbol(f"theta_{bus_name}", real=True)

    # 2. Inputs
    nu_w_0 = sym.Symbol(f"nu_w_0_{name}", real=True)
    dnu_w_ramp_ref = sym.Symbol(f"dnu_w_ramp_ref_{name}", real=True)
    q_s_ppc = sym.Symbol(f"q_s_ppc_{name}", real=True)
    p_ref_ext = sym.Symbol(f"p_ref_ext_{name}", real=True)
    beta_ext = sym.Symbol(f"beta_ext_{name}", real=True)
    lvrt_ext = sym.Symbol(f"lvrt_ext_{name}", real=True)
    i_sa_ref = sym.Symbol(f"i_sa_ref_{name}", real=True)
    i_sr_ref = sym.Symbol(f"i_sr_ref_{name}", real=True)

    # 3. Dynamic states
    nu_w_ramp = sym.Symbol(f"nu_w_ramp_{name}", real=True)
    theta_tr = sym.Symbol(f"theta_tr_{name}", real=True)
    omega_t = sym.Symbol(f"omega_t_{name}", real=True)
    omega_r = sym.Symbol(f"omega_r_{name}", real=True)
    xi_beta = sym.Symbol(f"xi_beta_{name}", real=True)
    beta = sym.Symbol(f"beta_{name}", real=True)
    p_w_mppt_lpf = sym.Symbol(f"p_w_mppt_lpf_{name}", real=True)

    # 4. Algebraic states
    p_s = sym.Symbol(f"p_s_{name}", real=True)
    q_s = sym.Symbol(f"q_s_{name}", real=True)

    # 5. Parameters
    S_n = sym.Symbol(f"S_n_{name}", real=True)
    K_mppt3 = sym.Symbol(f"K_mppt3_{name}", real=True)
    T_mppt = sym.Symbol(f"T_mppt_{name}", real=True)
    T_beta = sym.Symbol(f"T_beta_{name}", real=True)
    K_p_beta = sym.Symbol(f"K_p_beta_{name}", real=True)
    K_i_beta = sym.Symbol(f"K_i_beta_{name}", real=True)
    Omega_r_max = sym.Symbol(f"Omega_r_max_{name}", real=True)
    H_t = sym.Symbol(f"H_t_{name}", real=True)
    H_r = sym.Symbol(f"H_r_{name}", real=True)
    K_tr = sym.Symbol(f"K_tr_{name}", real=True)
    D_tr = sym.Symbol(f"D_tr_{name}", real=True)
    C_1 = sym.Symbol(f"C_1_{name}", real=True)
    C_2 = sym.Symbol(f"C_2_{name}", real=True)
    C_3 = sym.Symbol(f"C_3_{name}", real=True)
    C_4 = sym.Symbol(f"C_4_{name}", real=True)
    C_5 = sym.Symbol(f"C_5_{name}", real=True)
    C_6 = sym.Symbol(f"C_6_{name}", real=True)
    Nu_w_b = sym.Symbol(f"Nu_w_b_{name}", real=True)
    Lam_b = sym.Symbol(f"Lam_b_{name}", real=True)
    Omega_t_b = sym.Symbol(f"Omega_t_b_{name}", real=True)
    K_pow = sym.Symbol(f"K_pow_{name}", real=True)
    C_p_b = sym.Symbol(f"C_p_b_{name}", real=True)
    I_max = sym.Symbol(f"I_max_{name}", real=True)
    Epsilon = sym.Symbol(f"Epsilon_{name}", real=True)

    # 6. Aerodynamics
    nu_w = nu_w_0 + nu_w_ramp
    lam = Lam_b * (omega_t / Omega_t_b) / (nu_w / Nu_w_b)
    inv_lam_i = 1 / (lam + 0.08 * beta) - 0.035 / (beta**3 + 1.0)
    c_p = C_1 * (C_2 * inv_lam_i - C_3 * beta - C_4) * sym.exp(-C_5 * inv_lam_i) + C_6 * lam
    c_p_pu = c_p / C_p_b
    p_w_nosat = K_pow * c_p_pu * (nu_w / Nu_w_b)**3
    p_w = sym.Piecewise((0.0, p_w_nosat < 0.0), (p_w_nosat, True))
    dnu_w_ramp = dnu_w_ramp_ref - 1e-4 * nu_w_ramp

    # 7. MPPT: cubic power–speed law, low-pass filtered → rotor-braking power
    p_mppt_ref = K_mppt3 * omega_t**3
    p_m_ref = sym.Piecewise((p_mppt_ref + p_ref_ext, p_ref_ext < 0.0), (p_mppt_ref, True))
    dp_w_mppt_lpf = 1.0 / T_mppt * (p_m_ref - p_w_mppt_lpf)
    p_r = p_w_mppt_lpf                                # electrical power extracted from the rotor

    # 8. Pitch control (limits rotor overspeed above Omega_r_max)
    epsilon_beta_nosat = omega_r - Omega_r_max
    epsilon_beta = sym.Piecewise((0.0, epsilon_beta_nosat < 0.0), (epsilon_beta_nosat, True))
    beta_ref = K_p_beta * epsilon_beta + K_i_beta * xi_beta
    dxi_beta = epsilon_beta - xi_beta * 1e-8
    dbeta = 1.0 / T_beta * (beta_ref + beta_ext - beta)

    # 9. Two-mass mechanical drivetrain
    omega_t_abs = sym.Abs(omega_t) + 0.01
    omega_r_abs = sym.Abs(omega_r) + 0.01
    dtheta_tr = omega_t - omega_r
    domega_t = 1.0 / (2 * H_t * omega_t_abs) * (p_w - K_tr * theta_tr - D_tr * (omega_t - omega_r))
    domega_r = 1.0 / (2 * H_r * omega_r_abs) * (K_tr * theta_tr + D_tr * (omega_t - omega_r) - p_r)

    # 10. Optional state-space filter on the reactive-power reference
    q_s_ppc_d = q_s_ppc
    if "A_q" in data_dict:
        from pydae.bps.utils.ss_num2sym import ss_num2sym
        A_q = np.array(data_dict["A_q"])
        B_q = np.array(data_dict["B_q"])
        C_q = np.array(data_dict["C_q"])
        D_q = np.array(data_dict["D_q"])
        sys_q = ss_num2sym(f"q_{name}", A_q, B_q, C_q, D_q)
        sys_q["dx"] = sys_q["dx"].replace(sys_q["u"][0, 0], q_s_ppc)
        sys_q["z_evaluated"] = sys_q["z_evaluated"].replace(sys_q["u"][0, 0], q_s_ppc)
        q_s_ppc_d = sys_q["z_evaluated"][0, 0]
        grid.dae["f"] += list(sys_q["dx"])
        grid.dae["x"] += list(sys_q["x"])
        grid.dae["params_dict"].update(sys_q["params_dict"])

    # 11. Grid injection references (active from MPPT, reactive from PPC)
    p_s_ref = p_w_mppt_lpf
    q_s_ref = q_s_ppc_d

    # 12. Grid voltages in stationary (r-i) frame
    v_sr = V_s * cos(theta_s)
    v_si = V_s * sin(theta_s)
    v_sq_mag = v_sr**2 + v_si**2 + Epsilon
    v_m = sym.sqrt(v_sq_mag)

    # 13. PQ-mode current references
    i_sr_pq = (p_s_ref * v_sr + q_s_ref * v_si) / v_sq_mag
    i_si_pq = (p_s_ref * v_si - q_s_ref * v_sr) / v_sq_mag

    # 14. Arbitrary-mode current references (VRT)
    i_sr_ar = (i_sa_ref * v_sr + i_sr_ref * v_si) / v_m
    i_si_ar = (i_sa_ref * v_si - i_sr_ref * v_sr) / v_m

    # 15. Blend via LVRT trigger + smooth current saturation
    i_sr_nosat = (1.0 - lvrt_ext) * i_sr_pq + lvrt_ext * i_sr_ar
    i_si_nosat = (1.0 - lvrt_ext) * i_si_pq + lvrt_ext * i_si_ar
    i_mod = sym.sqrt(i_sr_nosat**2 + i_si_nosat**2 + Epsilon)
    i_mod_sat = 0.5 * (i_mod + I_max - sym.sqrt((i_mod - I_max)**2 + Epsilon))
    ratio = i_mod_sat / i_mod
    i_sr = i_sr_nosat * ratio
    i_si = i_si_nosat * ratio

    # 16. Injected power (algebraic)
    g_p_s = v_sr * i_sr + v_si * i_si - p_s
    g_q_s = v_si * i_sr - v_sr * i_si - q_s

    # 17. Assemble dynamics
    grid.dae["f"] += [dnu_w_ramp, dtheta_tr, domega_t, domega_r,
                      dxi_beta, dbeta, dp_w_mppt_lpf]
    grid.dae["x"] += [nu_w_ramp, theta_tr, omega_t, omega_r,
                      xi_beta, beta, p_w_mppt_lpf]
    grid.dae["g"] += [g_p_s, g_q_s]
    grid.dae["y_ini"] += [p_s, q_s]
    grid.dae["y_run"] += [p_s, q_s]

    # 18. Inputs
    inputs = {
        nu_w_0: par("nu_w_0"), dnu_w_ramp_ref: 0.0, q_s_ppc: par("q_s_ppc"),
        p_ref_ext: par("p_ref_ext"), beta_ext: par("beta_ext"),
        lvrt_ext: 0.0, i_sa_ref: 0.0, i_sr_ref: 0.0,
    }
    for sym_in, val in inputs.items():
        grid.dae["u_ini_dict"].update({str(sym_in): val})
        grid.dae["u_run_dict"].update({str(sym_in): val})

    # 19. Parameters
    S_n_val = par("S_n")
    H_t_val = par("H_t")
    H_r_val = par("H_r")
    w_tr = 2 * np.pi * par("w_tr")
    d_tr = 2 * np.pi * par("d_tr")
    K_tr_val = 2 * H_r_val * H_t_val * w_tr**2 / (H_r_val + H_t_val)
    D_tr_val = 4 * np.pi * H_r_val * H_t_val * d_tr * w_tr / (H_r_val + H_t_val)

    # aerodynamic curve constants (standard c_p fit, as in pmsm_1)
    C_vals = {"C_1_": 0.5176, "C_2_": 116.0, "C_3_": 0.4, "C_4_": 5.0,
              "C_5_": 21.0, "C_6_": 0.0068}
    Nu_w_b_v, Lam_b_v, Omega_t_b_v, K_pow_v = 12.0, 8.1, 1.2, 1.0
    inv_lam_i_b = 1 / Lam_b_v - 0.035
    C_p_b_v = (C_vals["C_1_"] * (C_vals["C_2_"] * inv_lam_i_b - C_vals["C_4_"])
               * np.exp(-C_vals["C_5_"] * inv_lam_i_b) + C_vals["C_6_"] * Lam_b_v)

    params = {
        S_n: S_n_val, K_mppt3: par("K_mppt3"), T_mppt: par("T_mppt"),
        T_beta: 2.0, K_p_beta: 100.0, K_i_beta: 1.0, Omega_r_max: 1.2,
        H_t: H_t_val, H_r: H_r_val, K_tr: K_tr_val, D_tr: D_tr_val,
        Nu_w_b: Nu_w_b_v, Lam_b: Lam_b_v, Omega_t_b: Omega_t_b_v,
        K_pow: K_pow_v, C_p_b: C_p_b_v,
        I_max: par("I_max"), Epsilon: par("Epsilon"),
    }
    params.update({sym.Symbol(f"{k}{name}", real=True): v for k, v in C_vals.items()})
    for sym_p, val in params.items():
        grid.dae["params_dict"].update({str(sym_p): val})

    # 20. Initialization hints
    p_0 = par("K_mppt3") * 1.2**3
    grid.dae["xy_0_dict"].update({
        str(nu_w_ramp): 0.0, str(omega_t): 1.2, str(omega_r): 1.2,
        str(theta_tr): p_0 / max(K_tr_val, 1e-6), str(xi_beta): 0.0, str(beta): 0.0,
        str(p_w_mppt_lpf): p_0, str(p_s): p_0, str(q_s): par("q_s_ppc"),
    })

    # 21. Outputs
    grid.dae["h_dict"].update({
        f"nu_w_{name}": nu_w, f"p_w_{name}": p_w,
        f"omega_t_{name}": omega_t, f"omega_r_{name}": omega_r,
        f"beta_{name}": beta, f"p_w_mppt_lpf_{name}": p_w_mppt_lpf,
        f"p_s_{name}": p_s, f"q_s_{name}": q_s,
        f"i_mod_{name}": i_mod, f"i_mod_sat_{name}": i_mod_sat,
    })

    p_W = p_s * S_n
    q_var = q_s * S_n
    return p_W, q_var


# =============================================================================
# Sphinx Documentation Auto-Generator
# =============================================================================
def dict_list_to_aligned_markdown_table(data: list) -> str:
    dict_data = [item for item in data if isinstance(item, dict)]
    if not dict_data:
        return ""
    headers = []
    seen = set()
    for row in dict_data:
        for key in row:
            if key not in seen:
                seen.add(key)
                headers.append(key)
    widths = {h: len(h) for h in headers}
    for row in dict_data:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))
    out = io.StringIO()
    out.write("| " + " | ".join(h.ljust(widths[h]) for h in headers) + " |\n")
    out.write("| " + " | ".join("-" * widths[h] for h in headers) + " |\n")
    for row in dict_data:
        out.write("| " + " | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers) + " |\n")
    return out.getvalue().strip()


def generate_sphinx_tables():
    docs = ""
    full = descriptions()
    for cat in ["Parameter", "Input", "Dynamic State", "Algebraic State", "Output"]:
        rows = [{k: v for k, v in it.items() if k != "type"}
                for it in full if it.get("type") == cat]
        if rows:
            docs += f"\n### {cat}s\n\n" + dict_list_to_aligned_markdown_table(rows) + "\n"
    return docs


__doc__ += generate_sphinx_tables()


# =============================================================================
# Testing Block
# =============================================================================
def test_build():
    from pydae.bps import BpsBuilder
    from pydae.core import Builder

    grid = BpsBuilder("pmsm_pq_ss.hjson")
    grid.checker()
    grid.uz_jacs = False
    grid.construct("pmsm_pq_ss")
    Builder(grid.sys_dict, target="ctypes", sparse=False).build()
    return grid


def test_ini():
    from pydae.bps import BpsBuilder
    from pydae.core import Model

    grid = BpsBuilder("pmsm_pq_ss.hjson")
    grid.checker()
    grid.uz_jacs = False
    grid.construct("pmsm_pq_ss")

    model = Model("pmsm_pq_ss")
    model.ini({}, grid.dae["xy_0_dict"])
    # MPPT steady state: p_s = K_mppt3 * omega_t^3, and p_w = p_s (mechanical balance)
    p_s = float(model.get_value("p_s_WT1"))
    omega_t = float(model.get_value("omega_t_WT1"))
    assert abs(p_s - 0.4 * omega_t**3) < 1e-4
    assert abs(p_s - float(model.get_value("p_w_WT1"))) < 1e-4
    assert abs(float(model.get_value("q_s_WT1"))) < 1e-6
    return model


if __name__ == "__main__":
    test_build()
    test_ini()
    print("pmsm_pq_ss: build + ini OK")
