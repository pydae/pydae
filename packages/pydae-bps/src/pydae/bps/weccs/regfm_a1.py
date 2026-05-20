# -*- coding: utf-8 -*-
r"""
WECC Renewable Energy Grid-Forming Inverter model A1 (REGFM_A1).

Reference: PNNL-32278, *Model Specification of Droop-Controlled Grid-Forming
Inverters (REGFM_A1)*, Pacific Northwest National Laboratory, September 2023.

REGFM_A1 represents a **droop-controlled grid-forming (GFM) inverter** as a
controllable voltage source of magnitude :math:`E` and virtual angle
:math:`\delta` behind a coupling reactance :math:`X_L`.  Unlike grid-following
converters (REGC_A/B), the GFM does not rely on a PLL — it sets its own
internal frequency from active power measurements.

The network interface is structurally identical to a synchronous machine with
internal EMF on the q-axis, coupling reactance :math:`X_L`, and zero armature
resistance, so the same dq-frame algebraic equations apply.

Signal chain::

    REPC_A  (plant level, ppcs/)     [optional]
        │  Pref
    REGFM_A1  (standalone weccs entry, no reec layer)
        │  p_g, q_g
    Network  (voltage source behind XL)

**Differential equations**

.. math::

    \dot{\delta}   &= \Omega_b\,(\omega_{droop} - \omega_{coi})
                      - K_{\delta}(\delta - \delta_{ref})  \\
    \dot{x}_{Pe}   &= (P_e - x_{Pe}) / T_{PI}  \\
    \dot{x}_{Qe}   &= (Q_e - x_{Qe}) / T_{QI}  \\
    \dot{E}        &= (E_{ref} - E) / T_v

where the droop quantities are:

.. math::

    \omega_{droop} &= \mathrm{clip}\!\left(
                        1 - m_P(x_{Pe} - P_{ref}),\;
                        \omega_{min},\; \omega_{max}\right)  \\
    E_{droop}      &= \mathrm{clip}\!\left(
                        V_{ref} + k_{pv}(V_{ref} - V_t)
                        - n_Q(x_{Qe} - Q_{ref}),\;
                        E_{min},\; E_{max}\right)  \\
    E_{ref}        &= E_{droop} \cdot f_{cl}, \qquad
    f_{cl} = \min\!\left(\frac{I_{maxF}}{|I|},\, 1\right)

and the current magnitude is :math:`|I| = \sqrt{i_d^2 + i_q^2 + \varepsilon}`.

**Algebraic equations** (coupling reactance, :math:`R_a = 0`)

.. math::

    0 &= E - X_L\,i_d - v_q  \\
    0 &= X_L\,i_q - v_d  \\
    0 &= i_d v_d + i_q v_q - p_g  \\
    0 &= i_d v_q - i_q v_d - q_g

with :math:`v_d = V\sin(\delta-\theta)`,\; :math:`v_q = V\cos(\delta-\theta)`.

**Power injection**

.. math::

    P = p_g\,S_n, \qquad Q = q_g\,S_n
"""
import numpy as np


def descriptions():
    r"""Single source of truth for REGFM_A1 parameters, states, and I/O."""
    d = []

    # ── parameters ────────────────────────────────────────────────────────
    d += [{"type": "Parameter", "tex": "S_n",       "data": "S_n",     "model": "S_n",     "default": 100e6, "description": "Converter MVA base",                              "units": "VA"}]
    d += [{"type": "Parameter", "tex": "F_n",       "data": "F_n",     "model": "F_n",     "default": 50.0,  "description": "Nominal frequency",                               "units": "Hz"}]
    d += [{"type": "Parameter", "tex": "X_L",       "data": "XL",      "model": "XL",      "default": 0.1,   "description": "Coupling reactance",                              "units": "pu"}]
    d += [{"type": "Parameter", "tex": "m_P",       "data": "mP",      "model": "mP",      "default": 0.04,  "description": "P-f droop gain",                                  "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "n_Q",       "data": "nQ",      "model": "nQ",      "default": 0.05,  "description": "Q-V droop gain",                                  "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "k_{pv}",    "data": "kpv",     "model": "kpv",     "default": 0.0,   "description": "Voltage proportional feedback gain",               "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "T_{PI}",    "data": "TPI",     "model": "TPI",     "default": 0.05,  "description": "Active power filter time constant",               "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_{QI}",    "data": "TQI",     "model": "TQI",     "default": 0.05,  "description": "Reactive power filter time constant",             "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_v",       "data": "Tv",      "model": "Tv",      "default": 0.02,  "description": "Voltage control time constant",                   "units": "s"}]
    d += [{"type": "Parameter", "tex": "V_{ref}",   "data": "Vref",    "model": "Vref",    "default": 1.0,   "description": "Voltage reference",                               "units": "pu"}]
    d += [{"type": "Parameter", "tex": "P_{ref}",   "data": "Pref",    "model": "Pref",    "default": 0.8,   "description": "Active power setpoint",                           "units": "pu"}]
    d += [{"type": "Parameter", "tex": "Q_{ref}",   "data": "Qref",    "model": "Qref",    "default": 0.0,   "description": "Reactive power reference",                        "units": "pu"}]
    d += [{"type": "Parameter", "tex": "P_{max}",   "data": "Pmax",    "model": "Pmax",    "default": 1.0,   "description": "Maximum active power",                            "units": "pu"}]
    d += [{"type": "Parameter", "tex": "P_{min}",   "data": "Pmin",    "model": "Pmin",    "default": 0.0,   "description": "Minimum active power",                            "units": "pu"}]
    d += [{"type": "Parameter", "tex": "Q_{max}",   "data": "Qmax",    "model": "Qmax",    "default": 0.4,   "description": "Maximum reactive power",                          "units": "pu"}]
    d += [{"type": "Parameter", "tex": "Q_{min}",   "data": "Qmin",    "model": "Qmin",    "default": -0.4,  "description": "Minimum reactive power",                          "units": "pu"}]
    d += [{"type": "Parameter", "tex": "E_{max}",   "data": "Emax",    "model": "Emax",    "default": 1.2,   "description": "Internal voltage upper limit",                    "units": "pu"}]
    d += [{"type": "Parameter", "tex": "E_{min}",   "data": "Emin",    "model": "Emin",    "default": 0.8,   "description": "Internal voltage lower limit",                    "units": "pu"}]
    d += [{"type": "Parameter", "tex": "I_{maxF}",  "data": "ImaxF",   "model": "ImaxF",   "default": 1.5,   "description": "Fault current limit (fault current limiting)",      "units": "pu"}]
    d += [{"type": "Parameter", "tex": "\\omega_{max}", "data": "omega_max", "model": "omega_max", "default": 1.05, "description": "Maximum droop frequency",                  "units": "pu"}]
    d += [{"type": "Parameter", "tex": "\\omega_{min}", "data": "omega_min", "model": "omega_min", "default": 0.95, "description": "Minimum droop frequency",                  "units": "pu"}]
    d += [{"type": "Parameter", "tex": "M_{vir}",   "data": "Mvir",    "model": "Mvir",    "default": 0.0,   "description": "Virtual inertia contribution to COI (0 = pure droop)", "units": "s"}]
    d += [{"type": "Parameter", "tex": "K_{\\delta}", "data": "K_delta", "model": "K_delta", "default": 0.0, "description": "Angle reference stabiliser",                      "units": "pu"}]

    # ── inputs ────────────────────────────────────────────────────────────
    d += [{"type": "Input", "tex": "P_{ref}", "data": "Pref_ext", "model": "Pref_ext", "default": 0.8, "description": "Active power setpoint (overrides param when present)", "units": "pu"}]

    # ── dynamic states ────────────────────────────────────────────────────
    d += [{"type": "Dynamic State", "tex": "\\delta", "data": "", "model": "delta",  "default": "", "description": "Virtual rotor angle",              "units": "rad"}]
    d += [{"type": "Dynamic State", "tex": "x_{Pe}",  "data": "", "model": "x_Pe",   "default": "", "description": "Filtered active power",            "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "x_{Qe}",  "data": "", "model": "x_Qe",   "default": "", "description": "Filtered reactive power",          "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "E",       "data": "", "model": "E",      "default": "", "description": "Internal voltage magnitude",       "units": "pu"}]

    # ── algebraic states ──────────────────────────────────────────────────
    d += [{"type": "Algebraic State", "tex": "i_d", "data": "", "model": "i_d", "default": "", "description": "d-axis current",    "units": "pu"}]
    d += [{"type": "Algebraic State", "tex": "i_q", "data": "", "model": "i_q", "default": "", "description": "q-axis current",    "units": "pu"}]
    d += [{"type": "Algebraic State", "tex": "p_g", "data": "", "model": "p_g", "default": "", "description": "Active power output (pu on S_n)",  "units": "pu"}]
    d += [{"type": "Algebraic State", "tex": "q_g", "data": "", "model": "q_g", "default": "", "description": "Reactive power output (pu on S_n)", "units": "pu"}]

    return d


def regfm_a1(grid, name, bus_name, data_dict):
    """Build REGFM_A1 and inject it into *grid*.

    Parameters
    ----------
    grid : BpsBuilder
    name : str
        Unique identifier for this converter instance.
    bus_name : str
        Terminal bus name.
    data_dict : dict
        Entry from the ``weccs`` list in the HJSON file.

    Returns
    -------
    p_W : symbolic expression
        Active power injection in Watts.
    q_var : symbolic expression
        Reactive power injection in VAR.
    """
    backend = grid.backend
    sin = backend.sin
    cos = backend.cos

    meta = descriptions()
    default_map = {item['data']: item['default']
                   for item in meta if item.get('data')}

    def param(key):
        return data_dict.get(key, default_map.get(key, 0.0))

    # ── network symbols ───────────────────────────────────────────────────
    V     = backend.symbols(f"V_{bus_name}")
    theta = backend.symbols(f"theta_{bus_name}")
    omega_coi = backend.symbols("omega_coi")

    # ── dynamic states ────────────────────────────────────────────────────
    delta = backend.symbols(f"delta_{name}")
    x_Pe  = backend.symbols(f"x_Pe_{name}")
    x_Qe  = backend.symbols(f"x_Qe_{name}")
    E     = backend.symbols(f"E_{name}")

    # ── algebraic states ──────────────────────────────────────────────────
    i_d = backend.symbols(f"i_d_{name}")
    i_q = backend.symbols(f"i_q_{name}")
    p_g = backend.symbols(f"p_g_{name}")
    q_g = backend.symbols(f"q_g_{name}")

    # ── parameters ────────────────────────────────────────────────────────
    S_n       = backend.symbols(f"S_n_{name}")
    Omega_b   = backend.symbols(f"Omega_b_{name}")
    XL        = backend.symbols(f"XL_{name}")
    mP        = backend.symbols(f"mP_{name}")
    nQ        = backend.symbols(f"nQ_{name}")
    kpv       = backend.symbols(f"kpv_{name}")
    TPI       = backend.symbols(f"TPI_{name}")
    TQI       = backend.symbols(f"TQI_{name}")
    Tv        = backend.symbols(f"Tv_{name}")
    Vref      = backend.symbols(f"Vref_{name}")
    Pref      = backend.symbols(f"Pref_{name}")
    Qref      = backend.symbols(f"Qref_{name}")
    Pmax      = backend.symbols(f"Pmax_{name}")
    Pmin      = backend.symbols(f"Pmin_{name}")
    Qmax      = backend.symbols(f"Qmax_{name}")
    Qmin      = backend.symbols(f"Qmin_{name}")
    Emax      = backend.symbols(f"Emax_{name}")
    Emin      = backend.symbols(f"Emin_{name}")
    ImaxF     = backend.symbols(f"ImaxF_{name}")
    omega_max = backend.symbols(f"omega_max_{name}")
    omega_min = backend.symbols(f"omega_min_{name}")
    K_delta   = backend.symbols(f"K_delta_{name}")
    Delta_ref = backend.symbols(f"Delta_ref_{name}")

    # ── dq-frame terminal voltage ─────────────────────────────────────────
    v_d = V * sin(delta - theta)
    v_q = V * cos(delta - theta)

    # ── measured power (from algebraic states) ────────────────────────────
    Pe = i_d * v_d + i_q * v_q   # active power
    Qe = i_d * v_q - i_q * v_d   # reactive power

    # ── P-f droop: virtual frequency ─────────────────────────────────────
    omega_droop = backend.hard_limits(
        1.0 - mP * (x_Pe - Pref), omega_min, omega_max
    )

    # ── Q-V droop: internal voltage reference ─────────────────────────────
    # VFlag=1 includes local voltage feedback (kpv > 0 activates it).
    E_droop = backend.hard_limits(
        Vref + kpv * (Vref - V) - nQ * (x_Qe - Qref), Emin, Emax
    )

    # ── Fault Current Limiting (FCL) ──────────────────────────────────────
    EPS  = 1e-6
    Imag = backend.sqrt(i_d**2 + i_q**2 + EPS)
    # When |I| > ImaxF, scale down E_ref to limit the current.
    # backend.min maps to fmin (CasADi) / Min (SymPy) — no Piecewise issues.
    f_cl  = backend.min(ImaxF / Imag, 1.0)
    E_ref = E_droop * f_cl

    # ── Differential equations ────────────────────────────────────────────
    # Angle: P-f droop angle dynamics, with optional angle stabiliser.
    ddelta = Omega_b * (omega_droop - omega_coi) - K_delta * (delta - Delta_ref)

    # Power filters (expose filtered signals for droop computation)
    dx_Pe = (Pe - x_Pe) / TPI
    dx_Qe = (Qe - x_Qe) / TQI

    # Voltage: first-order lag toward E_droop (limited by FCL)
    dE = (E_ref - E) / Tv

    # ── Algebraic equations (voltage source behind XL, Ra=0) ─────────────
    g_id = E   - XL * i_d - v_q   # q-axis: E = v_q + XL*id
    g_iq = XL * i_q - v_d          # d-axis: 0 = v_d - XL*iq  (Ed=0)
    g_pg = i_d * v_d + i_q * v_q - p_g
    g_qg = i_d * v_q - i_q * v_d - q_g

    # ── Power injection ───────────────────────────────────────────────────
    p_W   = p_g * S_n
    q_var = q_g * S_n

    # ── Assembly ──────────────────────────────────────────────────────────
    grid.dae['f'] += [ddelta, dx_Pe, dx_Qe, dE]
    grid.dae['x'] += [delta,  x_Pe,  x_Qe,  E]

    grid.dae['g']     += [g_id, g_iq, g_pg, g_qg]
    grid.dae['y_ini'] += [i_d,  i_q,  p_g,  q_g]
    grid.dae['y_run'] += [i_d,  i_q,  p_g,  q_g]

    # ── Virtual inertia contribution to COI ──────────────────────────────
    Mvir_val = float(param('Mvir'))
    if Mvir_val > 0.0:
        Mvir = backend.symbols(f"Mvir_{name}")
        grid.H_total                += Mvir_val
        grid.omega_coi_numerator    += omega_droop * Mvir * S_n
        grid.omega_coi_denominator  += Mvir * S_n

    # ── Parameters ────────────────────────────────────────────────────────
    F_n_val = param('F_n')
    grid.dae['params_dict'].update({
        f"Omega_b_{name}":  2 * np.pi * F_n_val,
        f"S_n_{name}":      param('S_n'),
        f"XL_{name}":       param('XL'),
        f"mP_{name}":       param('mP'),
        f"nQ_{name}":       param('nQ'),
        f"kpv_{name}":      param('kpv'),
        f"TPI_{name}":      param('TPI'),
        f"TQI_{name}":      param('TQI'),
        f"Tv_{name}":       param('Tv'),
        f"Vref_{name}":     param('Vref'),
        f"Pref_{name}":     param('Pref'),
        f"Qref_{name}":     param('Qref'),
        f"Pmax_{name}":     param('Pmax'),
        f"Pmin_{name}":     param('Pmin'),
        f"Qmax_{name}":     param('Qmax'),
        f"Qmin_{name}":     param('Qmin'),
        f"Emax_{name}":     param('Emax'),
        f"Emin_{name}":     param('Emin'),
        f"ImaxF_{name}":    param('ImaxF'),
        f"omega_max_{name}": param('omega_max'),
        f"omega_min_{name}": param('omega_min'),
        f"K_delta_{name}":  param('K_delta'),
        f"Delta_ref_{name}": 0.0,
    })
    if Mvir_val > 0.0:
        grid.dae['params_dict'][f"Mvir_{name}"] = Mvir_val

    # ── Initialization hints ──────────────────────────────────────────────
    Pref_0  = float(param('Pref'))
    Qref_0  = float(param('Qref'))
    Vref_0  = float(param('Vref'))
    XL_val  = float(param('XL'))
    V_0     = data_dict.get('V_ini', 1.0)

    # At steady state with V=V_0, theta=0, p_g=Pref_0, q_g=Qref_0:
    #   p_g = E*V*sin(delta)/XL  → sin(delta) = Pref_0*XL/(E_0*V_0)
    #   q_g = (E*V*cos(delta) - V²)/XL → E*cos(delta) = (q_g*XL + V²)/V
    # Approximation (small angle, unity voltage):
    E_0     = V_0 + XL_val * Qref_0 / max(V_0, 0.01)
    delta_0 = float(np.arctan(XL_val * Pref_0 / (E_0 * V_0)))
    i_q_0   = Pref_0 * np.sin(delta_0)    # ≈ Pref*delta for small delta
    i_d_0   = Pref_0 * np.cos(delta_0)
    # More precise: from algebraic equations at delta_0
    i_q_0   = V_0 * np.sin(delta_0) / XL_val
    i_d_0   = (E_0 - V_0 * np.cos(delta_0)) / XL_val

    grid.dae['xy_0_dict'].update({
        str(delta): delta_0,
        str(x_Pe):  Pref_0,
        str(x_Qe):  Qref_0,
        str(E):     E_0,
        str(i_d):   i_d_0,
        str(i_q):   i_q_0,
        str(p_g):   Pref_0,
        str(q_g):   Qref_0,
    })

    # ── Outputs ───────────────────────────────────────────────────────────
    grid.dae['h_dict'].update({
        f"p_g_{name}":       p_g,
        f"q_g_{name}":       q_g,
        f"E_{name}":         E,
        f"delta_{name}":     delta,
        f"omega_d_{name}":   omega_droop,
        f"E_droop_{name}":   E_droop,
        f"Imag_{name}":      Imag,
        f"f_cl_{name}":      f_cl,
    })

    return p_W, q_var


# ======================================================================
# In-module test
# ======================================================================
def test():
    import os, time
    from pydae.bps import BpsBuilder
    from pydae.core import Builder, Model

    hjson_path = os.path.join(os.path.dirname(__file__), 'regfm_a1.hjson')
    grid = BpsBuilder(hjson_path)
    grid.construct('temp_regfm_a1')
    bld = Builder(grid.sys_dict, target='cffi', sparse=False)
    bld.build()

    model = Model('temp_regfm_a1')
    t0 = time.perf_counter_ns()
    model.ini({}, 'temp_regfm_a1_xy_0.json')
    t1 = time.perf_counter_ns()
    print(f"ini time: {(t1-t0)/1e6:.2f} ms")

    p_g   = model.get_value('p_g_1')
    q_g   = model.get_value('q_g_1')
    E     = model.get_value('E_1')
    delta = model.get_value('delta_1')
    omega = model.get_value('omega_d_1')
    f_cl  = model.get_value('f_cl_1')
    Imag  = model.get_value('Imag_1')
    print(f"p_g={p_g:.4f}  q_g={q_g:.4f}  E={E:.4f}  "
          f"delta={np.rad2deg(delta):.2f}°  omega={omega:.6f}  f_cl={f_cl:.4f}")
    print(f"|I|={Imag:.4f}  (ImaxF=1.5, so f_cl should be 1.0)")

    assert abs(p_g - 0.8) < 0.05,  f"p_g={p_g:.4f} not close to 0.8 pu"
    assert abs(q_g - 0.0) < 0.05,  f"q_g={q_g:.4f} not close to 0.0 pu"
    assert abs(omega - 1.0) < 1e-4, f"omega_droop={omega:.6f} should be 1.0 at steady state"
    assert f_cl > 0.99,             f"f_cl={f_cl:.4f} should be 1.0 (no overcurrent)"

    # Step-response: Pref from 0.8 → 1.0, check that droop lowers omega
    model.run(5.0, {'Pref_1': 1.0})
    model.post()
    omega_vals = model.get_values('omega_d_1')
    assert min(omega_vals) < 1.0, "omega_droop should dip below 1.0 during Pref step"
    print(f"min(omega_d) during Pref step = {min(omega_vals):.6f}  ✓")

    print("test() PASSED")


if __name__ == '__main__':
    test()
