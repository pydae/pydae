# -*- coding: utf-8 -*-
r"""
DFIG (doubly-fed induction generator) wind turbine — simplified p/q model.

Shares the **exact same mechanical part** as :func:`pmsm_pq_ss` (aerodynamics,
2-mass drivetrain, pitch control, MPPT) but models the particular **doubly-fed
induction machine and the converters feeding the rotor** instead of a full
converter.

In a DFIG the **stator is connected directly to the grid** and only the **slip
power flows through the rotor back-to-back converter** (rotor-side converter
RSC + grid-side converter GSC), so the converter is rated to the slip fraction
(~±30 %) and the machine must be magnetized (magnetizing reactance ``X_m``).

**Power split.** The command ``p_s_ppc`` sets the **total** grid active power
``p_g`` (subject to the available wind power); with slip ``s = 1 − ω_r``
(ω_s = 1 pu) the total splits into the stator power ``p_s`` and the converter
power ``P_rotor``:

$$p_g = \min(p_{s,ppc},\,\text{available}) = p_{w,mppt}, \qquad s = 1 - \omega_r$$
$$p_s = \frac{p_g}{\omega_r}\ \text{(stator)}, \qquad
  P_{rotor} = p_g - p_s = -s\,p_s\ \text{(converter, GSC)}$$

Sub-synchronous (``ω_r < 1``): the converter **absorbs** slip power
(``P_rotor < 0``); super-synchronous (``ω_r > 1``): it **delivers**
(``P_rotor > 0``). The stator (via the RSC) controls the **total** reactive to
``q_s_ppc``; the GSC runs at unity power factor.

**Rotor (RSC) current.** In stator-flux-oriented coordinates the rotor current
provides the machine torque and magnetization:

$$i_{ra} = \frac{X_s}{X_m}\frac{p_s}{V_s}, \qquad
  i_{rr} = \frac{V_s}{X_m} + \frac{X_s}{X_m}\frac{q_s}{V_s}, \qquad
  |i_r| = \sqrt{i_{ra}^2 + i_{rr}^2 + \epsilon}$$

``|i_r|`` is soft-limited to the converter rating ``I_max_r`` (only the slip
power flows through it); ``lvrt_ext`` blends to a reactive-priority current
injection. When unsaturated the terminals reduce to ``p_g = p_s_ppc``,
``q_g = q_s_ppc`` — the total equals the command, split into stator + rotor by
slip.

**Grid injection (total):** ``p_W = p_g·S_n``, ``q_var = q_g·S_n``.
"""
import io

import numpy as np


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
    d += [{"type": "Parameter", "tex": "X_m", "data": "X_m", "model": "X_m",
           "default": 3.0, "description": "Magnetizing reactance", "units": "pu"}]
    d += [{"type": "Parameter", "tex": "X_s", "data": "X_s", "model": "X_s",
           "default": 3.05, "description": "Stator reactance (X_ls + X_m)", "units": "pu"}]
    d += [{"type": "Parameter", "tex": "I_{max,r}", "data": "I_max_r", "model": "I_max_r",
           "default": 1.2, "description": "Rotor converter current limit", "units": "pu"}]
    d += [{"type": "Parameter", "tex": "\\epsilon", "data": "Epsilon", "model": "Epsilon",
           "default": 1e-8, "description": "Regularization for soft saturation", "units": "-"}]

    # Inputs
    d += [{"type": "Input", "tex": "\\nu_{w,0}", "data": "nu_w_0", "model": "nu_w_0",
           "default": 10.0, "description": "Base wind speed", "units": "m/s"}]
    d += [{"type": "Input", "tex": "p_{s,ppc}", "data": "p_s_ppc", "model": "p_s_ppc",
           "default": 1.2, "description": "Total active-power command from PPC; p_g = min(p_s_ppc, available wind power)", "units": "pu"}]
    d += [{"type": "Input", "tex": "q_{s,ppc}", "data": "q_s_ppc", "model": "q_s_ppc",
           "default": 0.0, "description": "Total reactive power reference from PPC (RSC control)", "units": "pu"}]
    d += [{"type": "Input", "tex": "\\beta_{ext}", "data": "beta_ext", "model": "beta_ext",
           "default": 0.0, "description": "External pitch-angle adder", "units": "deg"}]
    d += [{"type": "Input", "tex": "\\text{lvrt}_{ext}", "data": "lvrt_ext", "model": "lvrt_ext",
           "default": 0.0, "description": "External LVRT trigger (0=PQ, 1=arbitrary)", "units": "-"}]
    d += [{"type": "Input", "tex": "i_{ra,ref}", "data": "i_sa_ref", "model": "i_sa_ref",
           "default": 0.0, "description": "Active rotor-current reference (arbitrary mode)", "units": "pu"}]
    d += [{"type": "Input", "tex": "i_{rr,ref}", "data": "i_sr_ref", "model": "i_sr_ref",
           "default": 0.0, "description": "Reactive rotor-current reference (arbitrary mode)", "units": "pu"}]

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
           "data": "", "default": "", "description": "Filtered MPPT power reference (= total p_g)", "units": "pu"}]

    # Algebraic States
    d += [{"type": "Algebraic State", "tex": "p_s", "model": "p_s", "data": "", "default": "",
           "description": "Stator active power", "units": "pu"}]
    d += [{"type": "Algebraic State", "tex": "q_s", "model": "q_s", "data": "", "default": "",
           "description": "Stator reactive power (= total q_g, GSC unity PF)", "units": "pu"}]

    # Outputs
    d += [{"type": "Output", "tex": "s", "model": "slip", "data": "", "default": "",
           "description": "Slip (1 − ω_r)", "units": "pu"}]
    d += [{"type": "Output", "tex": "P_{rotor}", "model": "p_r", "data": "", "default": "",
           "description": "Rotor/converter (GSC) active power", "units": "pu"}]
    d += [{"type": "Output", "tex": "p_g", "model": "p_g", "data": "", "default": "",
           "description": "Total grid active power (stator + rotor)", "units": "pu"}]
    d += [{"type": "Output", "tex": "q_g", "model": "q_g", "data": "", "default": "",
           "description": "Total grid reactive power", "units": "pu"}]
    d += [{"type": "Output", "tex": "i_r", "model": "i_r", "data": "", "default": "",
           "description": "Rotor converter current magnitude", "units": "pu"}]
    d += [{"type": "Output", "tex": "\\nu_w", "model": "nu_w", "data": "", "default": "",
           "description": "Wind speed", "units": "m/s"}]
    d += [{"type": "Output", "tex": "p_w", "model": "p_w", "data": "", "default": "",
           "description": "Aerodynamic power", "units": "pu"}]

    return d


def dfim_pq_ss(grid, name, bus_name, data_dict):
    r"""
    Simplified DFIG wind turbine (shared wind mechanics + slip power-split).

    Reuses the aerodynamic, 2-mass mechanical, pitch and MPPT dynamics of
    :func:`pmsm_pq_ss`. The electrical part is a doubly-fed induction machine:
    the total grid power ``p_g = min(p_s_ppc, available)`` splits by slip into the
    stator power ``p_s = p_g/ω_r`` (direct to grid) and the rotor/converter power
    ``P_rotor = −s·p_s``; the rotor (RSC) current, which magnetizes the machine
    and sets ``p_s``/``q_s``, is limited to the converter rating ``I_max_r``.

    Parameters
    ----------
    grid : BpsBuilder
        Builder instance. Dual-backend (SymPy and CasADi via ``grid.backend``).
    name : str
        Unique instance identifier.
    bus_name : str
        Bus the turbine is connected to.
    data_dict : dict
        Parameters. Optionally ``A_q``/``B_q``/``C_q``/``D_q`` to enable a
        state-space low-pass filter on the reactive-power reference.

    Returns
    -------
    p_W, q_var : expressions
        Total active (W) and reactive (var) injection into the bus.
    """
    backend = grid.backend
    sin = backend.sin
    cos = backend.cos

    meta = descriptions()
    default_map = {it["data"]: it["default"] for it in meta if it.get("data")}

    def par(key):
        return data_dict.get(key, default_map.get(key))

    # 1. Bus voltages
    V_s = backend.symbols(f"V_{bus_name}", real=True)
    theta_s = backend.symbols(f"theta_{bus_name}", real=True)

    # 2. Inputs
    nu_w_0 = backend.symbols(f"nu_w_0_{name}", real=True)
    dnu_w_ramp_ref = backend.symbols(f"dnu_w_ramp_ref_{name}", real=True)
    p_s_ppc = backend.symbols(f"p_s_ppc_{name}", real=True)
    q_s_ppc = backend.symbols(f"q_s_ppc_{name}", real=True)
    beta_ext = backend.symbols(f"beta_ext_{name}", real=True)
    lvrt_ext = backend.symbols(f"lvrt_ext_{name}", real=True)
    i_sa_ref = backend.symbols(f"i_sa_ref_{name}", real=True)
    i_sr_ref = backend.symbols(f"i_sr_ref_{name}", real=True)

    # 3. Dynamic states
    nu_w_ramp = backend.symbols(f"nu_w_ramp_{name}", real=True)
    theta_tr = backend.symbols(f"theta_tr_{name}", real=True)
    omega_t = backend.symbols(f"omega_t_{name}", real=True)
    omega_r = backend.symbols(f"omega_r_{name}", real=True)
    xi_beta = backend.symbols(f"xi_beta_{name}", real=True)
    beta = backend.symbols(f"beta_{name}", real=True)
    p_w_mppt_lpf = backend.symbols(f"p_w_mppt_lpf_{name}", real=True)

    # 4. Algebraic states (stator power)
    p_s = backend.symbols(f"p_s_{name}", real=True)
    q_s = backend.symbols(f"q_s_{name}", real=True)

    # 5. Parameters
    S_n = backend.symbols(f"S_n_{name}", real=True)
    K_mppt3 = backend.symbols(f"K_mppt3_{name}", real=True)
    T_mppt = backend.symbols(f"T_mppt_{name}", real=True)
    T_beta = backend.symbols(f"T_beta_{name}", real=True)
    K_p_beta = backend.symbols(f"K_p_beta_{name}", real=True)
    K_i_beta = backend.symbols(f"K_i_beta_{name}", real=True)
    Omega_r_max = backend.symbols(f"Omega_r_max_{name}", real=True)
    H_t = backend.symbols(f"H_t_{name}", real=True)
    H_r = backend.symbols(f"H_r_{name}", real=True)
    K_tr = backend.symbols(f"K_tr_{name}", real=True)
    D_tr = backend.symbols(f"D_tr_{name}", real=True)
    C_1 = backend.symbols(f"C_1_{name}", real=True)
    C_2 = backend.symbols(f"C_2_{name}", real=True)
    C_3 = backend.symbols(f"C_3_{name}", real=True)
    C_4 = backend.symbols(f"C_4_{name}", real=True)
    C_5 = backend.symbols(f"C_5_{name}", real=True)
    C_6 = backend.symbols(f"C_6_{name}", real=True)
    Nu_w_b = backend.symbols(f"Nu_w_b_{name}", real=True)
    Lam_b = backend.symbols(f"Lam_b_{name}", real=True)
    Omega_t_b = backend.symbols(f"Omega_t_b_{name}", real=True)
    K_pow = backend.symbols(f"K_pow_{name}", real=True)
    C_p_b = backend.symbols(f"C_p_b_{name}", real=True)
    X_m = backend.symbols(f"X_m_{name}", real=True)
    X_s = backend.symbols(f"X_s_{name}", real=True)
    I_max_r = backend.symbols(f"I_max_r_{name}", real=True)
    Epsilon = backend.symbols(f"Epsilon_{name}", real=True)

    # ── Mechanical / aero / pitch / MPPT — identical to pmsm_pq_ss ───────────

    # 6. Aerodynamics
    nu_w = nu_w_0 + nu_w_ramp
    lam = Lam_b * (omega_t / Omega_t_b) / (nu_w / Nu_w_b)
    inv_lam_i = 1 / (lam + 0.08 * beta) - 0.035 / (beta**3 + 1.0)
    c_p = C_1 * (C_2 * inv_lam_i - C_3 * beta - C_4) * backend.exp(-C_5 * inv_lam_i) + C_6 * lam
    c_p_pu = c_p / C_p_b
    p_w_nosat = K_pow * c_p_pu * (nu_w / Nu_w_b)**3
    p_w = backend.Piecewise((0.0, p_w_nosat < 0.0), (p_w_nosat, True))
    dnu_w_ramp = dnu_w_ramp_ref - 1e-4 * nu_w_ramp

    # 7. Power reference: total p_g = min(PPC command, available wind power)
    p_mppt_ref = K_mppt3 * omega_t**3                 # available power at the operating point
    p_cmd = 0.5 * (p_s_ppc + p_mppt_ref
                   - backend.sqrt((p_s_ppc - p_mppt_ref)**2 + Epsilon))
    dp_w_mppt_lpf = 1.0 / T_mppt * (p_cmd - p_w_mppt_lpf)
    p_r_brake = p_w_mppt_lpf                          # rotor-braking power (total, drives ω_r)

    # 8. Pitch control (limits rotor overspeed above Omega_r_max)
    epsilon_beta_nosat = omega_r - Omega_r_max
    epsilon_beta = backend.Piecewise((0.0, epsilon_beta_nosat < 0.0), (epsilon_beta_nosat, True))
    beta_ref = K_p_beta * epsilon_beta + K_i_beta * xi_beta
    dxi_beta = epsilon_beta - xi_beta * 1e-8
    dbeta = 1.0 / T_beta * (beta_ref + beta_ext - beta)

    # 9. Two-mass mechanical drivetrain
    omega_t_abs = backend.Abs(omega_t) + 0.01
    omega_r_abs = backend.Abs(omega_r) + 0.01
    dtheta_tr = omega_t - omega_r
    domega_t = 1.0 / (2 * H_t * omega_t_abs) * (p_w - K_tr * theta_tr - D_tr * (omega_t - omega_r))
    domega_r = 1.0 / (2 * H_r * omega_r_abs) * (K_tr * theta_tr + D_tr * (omega_t - omega_r) - p_r_brake)

    # ── DFIG electrical: slip power-split + rotor (RSC) current ──────────────

    # 10. Optional state-space filter on the reactive-power reference
    q_s_ppc_d = q_s_ppc
    if "A_q" in data_dict:
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

        A_q = np.array(data_dict["A_q"])
        B_q = np.array(data_dict["B_q"])
        C_q = np.array(data_dict["C_q"])
        D_q = np.array(data_dict["D_q"])
        sys_q = ss_num2sym(f"q_{name}", A_q, B_q, C_q, D_q, backend=backend)
        sys_q["dx"] = _replace(sys_q["dx"], sys_q["u"][0, 0], q_s_ppc)
        sys_q["z_evaluated"] = _replace(sys_q["z_evaluated"], sys_q["u"][0, 0], q_s_ppc)
        q_s_ppc_d = sys_q["z_evaluated"][0, 0]
        grid.dae["f"] += _to_list(sys_q["dx"])
        grid.dae["x"] += _to_list(sys_q["x"])
        grid.dae["params_dict"].update(sys_q["params_dict"])

    # 11. Slip and power-split references (total is the command; stator derived)
    slip = 1.0 - omega_r
    p_tot_ref = p_w_mppt_lpf                          # total grid power = command (via MPPT min)
    p_s_ref = p_tot_ref / omega_r                     # stator power (P_grid = p_s·ω_r)
    q_s_ref = q_s_ppc_d                               # RSC controls total Q; GSC unity PF

    # 12. Rotor (RSC) current, stator-flux-oriented (regularized V in denominators)
    V_reg = backend.sqrt(V_s**2 + Epsilon)
    i_mag = V_s / X_m                                 # magnetizing current (RSC supplies it)
    i_ra = (X_s / X_m) * p_s_ref / V_reg              # active (torque) rotor current
    i_rr = i_mag + (X_s / X_m) * q_s_ref / V_reg      # reactive (magnetizing + Q) rotor current

    # 13. LVRT blend to an arbitrary (reactive-priority) rotor-current mode
    i_ra_ar = i_sa_ref
    i_rr_ar = i_mag + i_sr_ref
    i_ra_pre = (1.0 - lvrt_ext) * i_ra + lvrt_ext * i_ra_ar
    i_rr_pre = (1.0 - lvrt_ext) * i_rr + lvrt_ext * i_rr_ar

    # 14. Converter current limit (only slip power flows through it): soft-sat |i_r|
    i_r = backend.sqrt(i_ra_pre**2 + i_rr_pre**2 + Epsilon)
    i_r_sat = 0.5 * (i_r + I_max_r - backend.sqrt((i_r - I_max_r)**2 + Epsilon))
    ratio = i_r_sat / i_r
    i_ra_lim = i_ra_pre * ratio
    i_rr_lim = i_rr_pre * ratio

    # 15. Delivered stator power from the (limited) rotor current, stator-flux
    #     oriented: P_s = (X_m/X_s)·V_s·i_ra, Q_s = (X_m/X_s)·V_s·(i_rr − i_mag).
    #     The (X_m/X_s) factor is the exact inverse of the (X_s/X_m) in the
    #     current references above, so p_s → p_s_ref, q_s → q_s_ref when
    #     unsaturated (round-trip closes).
    k_c = X_m / X_s
    g_p_s = k_c * i_ra_lim * V_s - p_s
    g_q_s = k_c * (i_rr_lim - i_mag) * V_s - q_s

    # 16. Rotor / converter and total grid power
    p_r = -slip * p_s                                 # converter (GSC) power = P_rotor
    p_g = p_s + p_r                                   # = p_s·ω_r (total to grid)
    q_g = q_s                                         # GSC unity PF

    # 17. Assemble dynamics
    grid.dae["f"] += [dnu_w_ramp, dtheta_tr, domega_t, domega_r,
                      dxi_beta, dbeta, dp_w_mppt_lpf]
    grid.dae["x"] += [nu_w_ramp, theta_tr, omega_t, omega_r,
                      xi_beta, beta, p_w_mppt_lpf]
    grid.dae["g"] += [g_p_s, g_q_s]
    grid.dae["y_ini"] += [p_s, q_s]
    grid.dae["y_run"] += [p_s, q_s]

    # 18. Inputs (string-keyed: CasADi SX symbols are not usable as dict keys)
    input_defaults = {
        str(nu_w_0): par("nu_w_0"), str(dnu_w_ramp_ref): 0.0,
        str(p_s_ppc): par("p_s_ppc"), str(q_s_ppc): par("q_s_ppc"),
        str(beta_ext): par("beta_ext"),
        str(lvrt_ext): 0.0, str(i_sa_ref): 0.0, str(i_sr_ref): 0.0,
    }
    grid.dae["u_ini_dict"].update(input_defaults)
    grid.dae["u_run_dict"].update(input_defaults)

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
        str(S_n): S_n_val, str(K_mppt3): par("K_mppt3"), str(T_mppt): par("T_mppt"),
        str(T_beta): 2.0, str(K_p_beta): 100.0, str(K_i_beta): 1.0, str(Omega_r_max): 1.2,
        str(H_t): H_t_val, str(H_r): H_r_val, str(K_tr): K_tr_val, str(D_tr): D_tr_val,
        str(Nu_w_b): Nu_w_b_v, str(Lam_b): Lam_b_v, str(Omega_t_b): Omega_t_b_v,
        str(K_pow): K_pow_v, str(C_p_b): C_p_b_v,
        str(X_m): par("X_m"), str(X_s): par("X_s"),
        str(I_max_r): par("I_max_r"), str(Epsilon): par("Epsilon"),
    }
    params.update({f"{k}{name}": v for k, v in C_vals.items()})
    grid.dae["params_dict"].update(params)

    # 20. Initialization hints (adaptive pitched seed, as in pmsm_pq_ss).
    p_rated = par("K_mppt3") * 1.2**3
    p_seed = min(par("p_s_ppc"), p_rated)             # total power seed
    X_m_v, X_s_v = par("X_m"), par("X_s")

    def _p_w_aero(b):
        lam_ = Lam_b_v * (1.2 / Omega_t_b_v) / (par("nu_w_0") / Nu_w_b_v)
        inv_li = 1.0 / (lam_ + 0.08 * b) - 0.035 / (b**3 + 1.0)
        cp = (C_vals["C_1_"] * (C_vals["C_2_"] * inv_li - C_vals["C_3_"] * b - C_vals["C_4_"])
              * np.exp(-C_vals["C_5_"] * inv_li) + C_vals["C_6_"] * lam_)
        return K_pow_v * max(cp, 0.0) / C_p_b_v * (par("nu_w_0") / Nu_w_b_v)**3

    if _p_w_aero(0.0) <= p_seed:
        beta_seed = 0.0
    else:
        lo, hi = 0.0, 60.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if _p_w_aero(mid) > p_seed:
                lo = mid
            else:
                hi = mid
        beta_seed = 0.5 * (lo + hi)

    omega_r_seed = 1.2
    p_s_seed = p_seed / omega_r_seed                  # stator power seed (total/ω_r)
    q_s_seed = par("q_s_ppc")
    grid.dae["xy_0_dict"].update({
        str(nu_w_ramp): 0.0, str(omega_t): omega_r_seed, str(omega_r): omega_r_seed,
        str(theta_tr): p_seed / max(K_tr_val, 1e-6),
        str(xi_beta): beta_seed, str(beta): beta_seed,
        str(p_w_mppt_lpf): p_seed, str(p_s): p_s_seed, str(q_s): q_s_seed,
    })

    # 21. Outputs
    grid.dae["h_dict"].update({
        f"nu_w_{name}": nu_w, f"p_w_{name}": p_w,
        f"omega_t_{name}": omega_t, f"omega_r_{name}": omega_r,
        f"beta_{name}": beta, f"p_w_mppt_lpf_{name}": p_w_mppt_lpf,
        f"slip_{name}": slip,
        f"p_s_{name}": p_s, f"q_s_{name}": q_s,
        f"p_r_{name}": p_r, f"p_g_{name}": p_g, f"q_g_{name}": q_g,
        f"i_r_{name}": i_r, f"i_r_sat_{name}": i_r_sat,
    })

    p_W = p_g * S_n
    q_var = q_g * S_n
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

    grid = BpsBuilder("dfim_pq_ss.hjson")
    grid.checker()
    grid.uz_jacs = False
    grid.construct("dfim_pq_ss")
    Builder(grid.sys_dict, target="ctypes", sparse=False).build()
    return grid


def test_ini():
    from pydae.bps import BpsBuilder
    from pydae.core import Model

    grid = BpsBuilder("dfim_pq_ss.hjson")
    grid.checker()
    grid.uz_jacs = False
    grid.construct("dfim_pq_ss")

    model = Model("dfim_pq_ss")
    model.ini({}, grid.dae["xy_0_dict"])
    # MPPT steady state: total p_g = K_mppt3*omega_t^3 = p_w; stator p_s = p_g/omega_r
    p_g = float(model.get_value("p_g_WT1"))
    omega_t = float(model.get_value("omega_t_WT1"))
    omega_r = float(model.get_value("omega_r_WT1"))
    p_s = float(model.get_value("p_s_WT1"))
    assert abs(p_g - 0.4 * omega_t**3) < 1e-4
    assert abs(p_g - float(model.get_value("p_w_WT1"))) < 1e-4
    assert abs(p_s - p_g / omega_r) < 1e-4
    return model


if __name__ == "__main__":
    test_build()
    test_ini()
    print("dfim_pq_ss: build + ini OK")
