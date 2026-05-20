# -*- coding: utf-8 -*-
r"""
WECC Virtual Synchronous Machine Grid-Forming Inverter model B1 (REGFM_B1).

Reference: NREL/TP-5D00-90260, *Virtual Synchronous Machine Grid-Forming
Inverter Model Specification (REGFM_B1)*, June 2024 (UNIFI-2024-6-1).

REGFM_B1 is a **Virtual Synchronous Machine (VSM)** grid-forming inverter.
Unlike the droop-based REGFM_A1, it includes an explicit swing equation with
inertia constant *H*, algebraic damping *D1*, and washout-based transient
damping *D2*.  A dedicated PLL tracks the bus angle and contributes its output
to the total VSM angle :math:`\delta_{VSM} = \delta_{IT} + \delta_{PLL}`.

The four major control subsystems (per Section 2 of the spec):

1. **VSM active power / frequency control** — P-f droop filter → swing equation
   :math:`1/(2Hs)` → angle integrator :math:`\omega_0/s`.
2. **Voltage control** — Q-V droop → PI voltage regulator with dynamic
   :math:`E_{min}/E_{max}` limits computed from the PQ priority algorithm.
3. **PLL** — PI synchronisation loop; frozen when :math:`V < V_{PLL,frz}`.
4. **Fault current limiting** — multiplicative FCL applied to :math:`E_{VSM}`.

Active-current angle limiting (:math:`\delta_{IT,max}`, :math:`\delta_{IT,min}`)
is implemented here as static limits computed from :math:`\delta_{max}` and the
ESFlag flag, rather than the two dynamic integrators in Fig. 6 of the spec.

**Network interface**

REGFM_B1 is a **voltage source behind impedance** (Thevenin equivalent):

.. math::

    P + jQ = V e^{j\theta} \cdot
             \left(\frac{E_{VSM} e^{j\delta_{VSM}} - V e^{j\theta}}{R_e + jX_L}\right)^*

The pydae dq-frame form (with :math:`R_e = 0` default):

.. math::

    0 &= E_{VSM} - X_L i_d - v_q  \\
    0 &= X_L i_q - v_d

where :math:`v_d = V\sin(\delta_{VSM}-\theta)`,\;
:math:`v_q = V\cos(\delta_{VSM}-\theta)`.

**Differential equations** (11 states)

.. math::

    \dot\delta_{IT}   &= \omega_0\,(\Delta\omega_m + \Delta\omega_{PLL})  \\
    \dot\Delta\omega_m &= \frac{P_{cond} - P_{inv} - D_1\Delta\omega_m
                                - D_2(\Delta\omega_m - x_{D2})}{2H}  \\
    \dot x_{D2}        &= \omega_D\,(\Delta\omega_m - x_{D2})  \\
    \dot P_{inv}       &= (P_e - P_{inv}) / T_{pf}  \\
    \dot I_{d,inv}     &= (i_d - I_{d,inv}) / T_{if}  \\
    \dot Q_{inv}       &= (Q_e - Q_{inv}) / T_{Qf}  \\
    \dot V_{inv}       &= (V - V_{inv}) / T_{Vf}  \\
    \dot I_{q,inv}     &= (i_q - I_{q,inv}) / T_{if}  \\
    \dot x_{iv}        &= k_{iv}\,(V_{cmd} - V_{inv})  \\
    \dot\delta_{PLL}   &= \omega_0\,\Delta\omega_{PLL}  \\
    \dot x_{PLL}       &= k_{i,PLL}\,V_q^{PLL} \cdot f_{frz}

where :math:`P_{cond} = P_{ref} + \Delta P`,\;
:math:`V_q^{PLL} = V\sin(\theta - \delta_{PLL})`,\;
:math:`f_{frz} = 0` when :math:`V < V_{PLL,frz}`.

**Algebraic equations** (4 states)

.. math::

    0 &= E_{VSM,lim} - X_L i_d - v_q  \\
    0 &= X_L i_q - v_d  \\
    0 &= i_d v_d + i_q v_q - p_g  \\
    0 &= i_d v_q - i_q v_d - q_g
"""
import numpy as np


def descriptions():
    r"""Single source of truth for REGFM_B1 parameters, states, and I/O."""
    d = []
    # ── electrical ─────────────────────────────────────────────────────────
    d += [{"type": "Parameter", "tex": "S_n",         "data": "S_n",       "model": "S_n",       "default": 100e6, "description": "Converter MVA base",                            "units": "VA"}]
    d += [{"type": "Parameter", "tex": "F_n",         "data": "F_n",       "model": "F_n",       "default": 50.0,  "description": "Nominal frequency",                             "units": "Hz"}]
    d += [{"type": "Parameter", "tex": "R_e",         "data": "Re",        "model": "Re",        "default": 0.0,   "description": "Coupling resistance (0 ≤ Re ≤ XL/4)",           "units": "pu"}]
    d += [{"type": "Parameter", "tex": "X_L",         "data": "XL",        "model": "XL",        "default": 0.1,   "description": "Coupling reactance (0.04 ≤ XL ≤ 0.4)",          "units": "pu"}]
    # ── VSM active power control ───────────────────────────────────────────
    d += [{"type": "Parameter", "tex": "m_p",         "data": "mp",        "model": "mp",        "default": 0.02,  "description": "P-f droop gain (= 1/mr)",                        "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "T_p",         "data": "Tp",        "model": "Tp",        "default": 0.0,   "description": "P-f droop filter time constant (0 = bypass)",   "units": "s"}]
    d += [{"type": "Parameter", "tex": "H",           "data": "H",         "model": "H",         "default": 0.5,   "description": "Virtual inertia constant",                      "units": "s"}]
    d += [{"type": "Parameter", "tex": "D_1",         "data": "D1",        "model": "D1",        "default": 0.0,   "description": "Algebraic damping coefficient",                  "units": "pu"}]
    d += [{"type": "Parameter", "tex": "D_2",         "data": "D2",        "model": "D2",        "default": 100.0, "description": "Transient damping coefficient (washout)",        "units": "pu"}]
    d += [{"type": "Parameter", "tex": "\\omega_D",   "data": "omegaD",    "model": "omegaD",    "default": 50.0,  "description": "Washout corner frequency",                       "units": "rad/s"}]
    d += [{"type": "Parameter", "tex": "\\Delta\\omega_{max}", "data": "Dωmax", "model": "Domega_max", "default": 0.05,  "description": "VSM speed deviation upper limit",         "units": "pu"}]
    d += [{"type": "Parameter", "tex": "\\Delta\\omega_{min}", "data": "Dωmin", "model": "Domega_min", "default": -0.05, "description": "VSM speed deviation lower limit",         "units": "pu"}]
    # ── voltage control ────────────────────────────────────────────────────
    d += [{"type": "Parameter", "tex": "m_q",         "data": "mq",        "model": "mq",        "default": 0.05,  "description": "Q-V droop gain (or virtual impedance when VdrpFlag=1)", "units": "pu"}]
    d += [{"type": "Parameter", "tex": "k_{pv}",      "data": "kpv",       "model": "kpv",       "default": 0.0,   "description": "Voltage PI proportional gain",                  "units": "pu"}]
    d += [{"type": "Parameter", "tex": "k_{iv}",      "data": "kiv",       "model": "kiv",       "default": 5.0,   "description": "Voltage PI integral gain",                      "units": "pu/s"}]
    d += [{"type": "Parameter", "tex": "V_{ref}",     "data": "Vref",      "model": "Vref",      "default": 1.0,   "description": "Voltage reference",                             "units": "pu"}]
    d += [{"type": "Parameter", "tex": "Q_{ref}",     "data": "Qref",      "model": "Qref",      "default": 0.0,   "description": "Reactive power reference",                      "units": "pu"}]
    d += [{"type": "Parameter", "tex": "P_{ref}",     "data": "Pref",      "model": "Pref",      "default": 0.8,   "description": "Active power setpoint",                         "units": "pu"}]
    # ── current limits ─────────────────────────────────────────────────────
    d += [{"type": "Parameter", "tex": "I_{maxSS}",   "data": "ImaxSS",    "model": "ImaxSS",    "default": 1.0,   "description": "Steady-state current limit (≤0 → 1/XL)",        "units": "pu"}]
    d += [{"type": "Parameter", "tex": "I_{maxF}",    "data": "ImaxF",     "model": "ImaxF",     "default": 1.5,   "description": "Transient (fault) current limit",                "units": "pu"}]
    d += [{"type": "Parameter", "tex": "k_f",         "data": "kf",        "model": "kf",        "default": 0.9,   "description": "Priority factor for Q/P max (0 → reset to 1)", "units": "pu"}]
    d += [{"type": "Parameter", "tex": "K_e",         "data": "Ke",        "model": "Ke",        "default": 1.0,   "description": "Scalar for negative active current limit",       "units": "-"}]
    # ── measurement filters ────────────────────────────────────────────────
    d += [{"type": "Parameter", "tex": "T_{pf}",      "data": "Tpf",       "model": "Tpf",       "default": 0.02,  "description": "Active power filter time constant",             "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_{Qf}",      "data": "TQf",       "model": "TQf",       "default": 0.02,  "description": "Reactive power filter time constant",           "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_{Vf}",      "data": "TVf",       "model": "TVf",       "default": 0.02,  "description": "Voltage filter time constant",                  "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_{if}",      "data": "Tif",       "model": "Tif",       "default": 0.02,  "description": "Current filter time constant",                  "units": "s"}]
    # ── PLL ───────────────────────────────────────────────────────────────
    d += [{"type": "Parameter", "tex": "k_{pPLL}",    "data": "kpPLL",     "model": "kpPLL",     "default": 0.265, "description": "PLL proportional gain",                         "units": "pu"}]
    d += [{"type": "Parameter", "tex": "k_{iPLL}",    "data": "kiPLL",     "model": "kiPLL",     "default": 2.65,  "description": "PLL integral gain",                             "units": "pu/s"}]
    d += [{"type": "Parameter", "tex": "\\Delta\\omega_{PLL,max}", "data": "DwPLLmax", "model": "DwPLLmax", "default": 0.2,  "description": "PLL output upper limit",             "units": "pu"}]
    d += [{"type": "Parameter", "tex": "\\Delta\\omega_{PLL,min}", "data": "DwPLLmin", "model": "DwPLLmin", "default": -0.2, "description": "PLL output lower limit",             "units": "pu"}]
    d += [{"type": "Parameter", "tex": "V_{PLL,frz}", "data": "VPLLfrz",   "model": "VPLLfrz",   "default": 0.05,  "description": "PLL freeze voltage threshold (≤0 = disabled)", "units": "pu"}]
    # ── flags ──────────────────────────────────────────────────────────────
    d += [{"type": "Parameter", "tex": "\\omega Flag", "data": "omegaFlag", "model": "omegaFlag", "default": 0, "description": "0=use omega_coi  1=use omega_PLL as droop input", "units": "-"}]
    d += [{"type": "Parameter", "tex": "VdrpFlag",    "data": "VdrpFlag",  "model": "VdrpFlag",  "default": 0, "description": "0=Q droop  1=Iq droop",                          "units": "-"}]
    d += [{"type": "Parameter", "tex": "QVFlag",      "data": "QVFlag",    "model": "QVFlag",    "default": 1, "description": "0=Qref input  1=Vref input for plant ctrl",      "units": "-"}]
    d += [{"type": "Parameter", "tex": "PQFlag",      "data": "PQFlag",    "model": "PQFlag",    "default": 1, "description": "0=Q priority  1=P priority",                     "units": "-"}]
    d += [{"type": "Parameter", "tex": "FFlag",       "data": "FFlag",     "model": "FFlag",     "default": 1, "description": "0=droop disabled  1=droop enabled",              "units": "-"}]
    d += [{"type": "Parameter", "tex": "ESFlag",      "data": "ESFlag",    "model": "ESFlag",    "default": 1, "description": "0=non-battery (δITmin=0)  1=battery",            "units": "-"}]
    # ── inputs ─────────────────────────────────────────────────────────────
    d += [{"type": "Input", "tex": "P_{ref}", "data": "Pref_ext", "model": "Pref_ext", "default": 0.8, "description": "Active power setpoint", "units": "pu"}]
    # ── states ─────────────────────────────────────────────────────────────
    d += [{"type": "Dynamic State", "tex": "\\delta_{IT}", "data": "", "model": "deltaIT",  "default": "", "description": "VSM angle integrator output",    "units": "rad"}]
    d += [{"type": "Dynamic State", "tex": "\\Delta\\omega_m", "data": "", "model": "Domegam","default": "", "description": "VSM speed deviation",          "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "x_{D2}",      "data": "", "model": "x_D2",    "default": "", "description": "Washout damping filter state",     "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "P_{inv}",     "data": "", "model": "Pinv",    "default": "", "description": "Filtered active power",            "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "I_{d,inv}",   "data": "", "model": "Idinv",   "default": "", "description": "Filtered d-axis current",          "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "Q_{inv}",     "data": "", "model": "Qinv",    "default": "", "description": "Filtered reactive power",          "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "V_{inv}",     "data": "", "model": "Vinv",    "default": "", "description": "Filtered terminal voltage",        "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "I_{q,inv}",   "data": "", "model": "Iqinv",   "default": "", "description": "Filtered q-axis current",          "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "x_{iv}",      "data": "", "model": "xiv",     "default": "", "description": "Voltage PI integrator state",      "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "\\delta_{PLL}", "data": "", "model": "deltaPLL", "default": "", "description": "PLL angle",                     "units": "rad"}]
    d += [{"type": "Dynamic State", "tex": "x_{PLL}",     "data": "", "model": "xPLL",    "default": "", "description": "PLL PI integrator state",          "units": "pu"}]
    # ── algebraic states ───────────────────────────────────────────────────
    d += [{"type": "Algebraic State", "tex": "i_d", "data": "", "model": "i_d", "default": "", "description": "d-axis current (δVSM frame)", "units": "pu"}]
    d += [{"type": "Algebraic State", "tex": "i_q", "data": "", "model": "i_q", "default": "", "description": "q-axis current (δVSM frame)", "units": "pu"}]
    d += [{"type": "Algebraic State", "tex": "p_g", "data": "", "model": "p_g", "default": "", "description": "Active power output (pu on S_n)",  "units": "pu"}]
    d += [{"type": "Algebraic State", "tex": "q_g", "data": "", "model": "q_g", "default": "", "description": "Reactive power output (pu on S_n)", "units": "pu"}]
    return d


def regfm_b1(grid, name, bus_name, data_dict):
    """Build REGFM_B1 and inject it into *grid*.

    Returns
    -------
    p_W : symbolic expression  — active power in Watts
    q_var : symbolic expression — reactive power in VAR
    """
    backend = grid.backend
    sin = backend.sin
    cos = backend.cos

    meta = descriptions()
    default_map = {item['data']: item['default']
                   for item in meta if item.get('data')}

    def param(key, fallback=0.0):
        return data_dict.get(key, default_map.get(key, fallback))

    # ── Python-level parameter validation (spec constraints) ───────────────
    XL_val    = float(param('XL'))
    ImaxSS_v  = float(param('ImaxSS'))
    if ImaxSS_v <= 0:
        ImaxSS_v = 1.0 / XL_val       # spec: treat ≤0 as 1/XL
    kf_v      = float(param('kf'))
    if kf_v == 0.0:
        kf_v = 1.0                     # spec: kf=0 → reset to 1

    # flags (Python values → clean expression tree)
    omegaFlag = int(param('omegaFlag', 0))
    VdrpFlag  = int(param('VdrpFlag',  0))
    PQFlag    = int(param('PQFlag',    1))
    FFlag     = int(param('FFlag',     1))
    ESFlag    = int(param('ESFlag',    1))

    # δIT limits (from steady-state current limit and ESFlag)
    delta_max_val = float(np.arcsin(min(XL_val * ImaxSS_v, 0.9999)))
    delta_IT_min  = -delta_max_val if ESFlag else 0.0

    # ── network symbols ───────────────────────────────────────────────────
    V      = backend.symbols(f"V_{bus_name}")
    theta  = backend.symbols(f"theta_{bus_name}")
    omega_coi = backend.symbols("omega_coi")

    # ── states ────────────────────────────────────────────────────────────
    deltaIT  = backend.symbols(f"deltaIT_{name}")
    Domegam  = backend.symbols(f"Domegam_{name}")
    x_D2     = backend.symbols(f"x_D2_{name}")
    Pinv     = backend.symbols(f"Pinv_{name}")
    Idinv    = backend.symbols(f"Idinv_{name}")
    Qinv     = backend.symbols(f"Qinv_{name}")
    Vinv     = backend.symbols(f"Vinv_{name}")
    Iqinv    = backend.symbols(f"Iqinv_{name}")
    xiv      = backend.symbols(f"xiv_{name}")
    deltaPLL = backend.symbols(f"deltaPLL_{name}")
    xPLL     = backend.symbols(f"xPLL_{name}")

    # ── algebraic states ──────────────────────────────────────────────────
    i_d = backend.symbols(f"i_d_{name}")
    i_q = backend.symbols(f"i_q_{name}")
    p_g = backend.symbols(f"p_g_{name}")
    q_g = backend.symbols(f"q_g_{name}")

    # ── symbolic parameters ────────────────────────────────────────────────
    S_n      = backend.symbols(f"S_n_{name}")
    Omega_b  = backend.symbols(f"Omega_b_{name}")
    XL       = backend.symbols(f"XL_{name}")
    Re       = backend.symbols(f"Re_{name}")
    mp       = backend.symbols(f"mp_{name}")
    Tp       = backend.symbols(f"Tp_{name}")
    H        = backend.symbols(f"H_{name}")
    D1       = backend.symbols(f"D1_{name}")
    D2       = backend.symbols(f"D2_{name}")
    omegaD   = backend.symbols(f"omegaD_{name}")
    Domega_max = backend.symbols(f"Domega_max_{name}")
    Domega_min = backend.symbols(f"Domega_min_{name}")
    mq       = backend.symbols(f"mq_{name}")
    kpv      = backend.symbols(f"kpv_{name}")
    kiv      = backend.symbols(f"kiv_{name}")
    Vref     = backend.symbols(f"Vref_{name}")
    Qref     = backend.symbols(f"Qref_{name}")
    Pref     = backend.symbols(f"Pref_{name}")
    ImaxSS   = backend.symbols(f"ImaxSS_{name}")
    ImaxF    = backend.symbols(f"ImaxF_{name}")
    kf       = backend.symbols(f"kf_{name}")
    Tpf      = backend.symbols(f"Tpf_{name}")
    TQf      = backend.symbols(f"TQf_{name}")
    TVf      = backend.symbols(f"TVf_{name}")
    Tif      = backend.symbols(f"Tif_{name}")
    kpPLL    = backend.symbols(f"kpPLL_{name}")
    kiPLL    = backend.symbols(f"kiPLL_{name}")
    DwPLLmax = backend.symbols(f"DwPLLmax_{name}")
    DwPLLmin = backend.symbols(f"DwPLLmin_{name}")
    VPLLfrz  = backend.symbols(f"VPLLfrz_{name}")
    delta_IT_min_sym = backend.symbols(f"delta_IT_min_{name}")
    delta_max_sym    = backend.symbols(f"delta_max_{name}")
    # P-f droop filter time constant: use small value when Tp=0
    Tp_eff   = backend.symbols(f"Tp_eff_{name}")

    # ── VSM angle ─────────────────────────────────────────────────────────
    deltaVSM = deltaIT + deltaPLL

    # ── dq-frame terminal voltage (δVSM reference frame) ──────────────────
    v_d = V * sin(deltaVSM - theta)
    v_q = V * cos(deltaVSM - theta)

    # ── actual electrical power (from network algebraic states) ───────────
    Pe = i_d * v_d + i_q * v_q
    Qe = i_d * v_q - i_q * v_d

    # ── PLL ───────────────────────────────────────────────────────────────
    # q-axis voltage in PLL reference frame: drives to zero when δPLL → θ
    Vq_pll = V * sin(theta - deltaPLL)

    # PLL freeze: smooth gate (1 = active, 0 = frozen)
    if float(param('VPLLfrz')) > 0:
        f_pll_frz = backend.hard_limits(
            (V - VPLLfrz) / (VPLLfrz * 0.1 + 1e-6), 0.0, 1.0
        )
    else:
        f_pll_frz = 1.0   # no freeze

    DeltaomegaPLL = backend.hard_limits(
        kpPLL * Vq_pll + xPLL, DwPLLmin, DwPLLmax
    )
    d_xPLL     = kiPLL * Vq_pll * f_pll_frz
    d_deltaPLL = Omega_b * DeltaomegaPLL * f_pll_frz

    # ── P-f droop filter ──────────────────────────────────────────────────
    # Droop input: frequency deviation
    if omegaFlag == 0:
        omega_in = omega_coi
    else:
        omega_in = 1.0 + DeltaomegaPLL   # use PLL frequency

    if FFlag:
        # ΔP = droop gain × frequency error, passed through low-pass Tp
        droop_err = (1.0 / mp) * (1.0 - omega_in)   # ωref = 1
        # State x_DP carries the filtered ΔP (Tp_eff avoids Tp=0 singularity)
        x_DP = backend.symbols(f"x_DP_{name}")
        d_x_DP = (droop_err - x_DP) / Tp_eff
        DeltaP = x_DP
    else:
        # Droop disabled: ΔP = 0
        x_DP   = backend.symbols(f"x_DP_{name}")   # declared but unused
        d_x_DP = x_DP * 0.0
        DeltaP = 0.0

    Pcond = Pref + DeltaP   # conditioned power command

    # ── VSM swing equation ────────────────────────────────────────────────
    # D2 washout: G(s) = s·D2 / (s + ωD)
    # State x_D2; output = D2*(Δωm − x_D2)
    d_x_D2   = omegaD * (Domegam - x_D2)
    D2_damp  = D2 * (Domegam - x_D2)

    dDomegam_raw = (Pcond - Pinv - D1 * Domegam - D2_damp) / (2.0 * H)

    # Anti-windup: freeze integrator at limits
    Domegam_eff = backend.hard_limits(Domegam, Domega_min, Domega_max)
    dDomegam = backend.hard_limits(dDomegam_raw, -999.0, 999.0)

    # ── VSM angle integrator ──────────────────────────────────────────────
    d_deltaIT_raw = Omega_b * (Domegam_eff + DeltaomegaPLL)
    # Angle limits from active current limiting (static, from δmax)
    d_deltaIT = backend.hard_limits(d_deltaIT_raw, -999.0, 999.0)

    # ── measurement filters ────────────────────────────────────────────────
    d_Pinv  = (Pe  - Pinv)  / Tpf
    d_Idinv = (i_d - Idinv) / Tif
    d_Qinv  = (Qe  - Qinv)  / TQf
    d_Vinv  = (V   - Vinv)  / TVf
    d_Iqinv = (i_q - Iqinv) / Tif

    # ── PQ priority → steady-state current limits ─────────────────────────
    EPS = 1e-4
    if PQFlag == 0:   # Q priority
        IqmaxSS = kf * ImaxSS
        IdmaxSS = backend.sqrt(backend.max(ImaxSS**2 - Iqinv**2, EPS))
    else:             # P priority
        IdmaxSS = kf * ImaxSS
        IqmaxSS = backend.sqrt(backend.max(ImaxSS**2 - Idinv**2, EPS))

    # Dynamic E limits (eqs. 10–11 of spec)
    Emin_dyn = backend.sqrt((Vinv - IqmaxSS * XL)**2 + (Idinv * XL)**2 + EPS)
    Emax_dyn = backend.sqrt((Vinv + IqmaxSS * XL)**2 + (Idinv * XL)**2 + EPS)

    # ── Q-V droop → voltage command ────────────────────────────────────────
    if VdrpFlag == 0:
        Vcmd = Vref + mq * (Qref - Qinv)
    else:
        Vcmd = Vref - mq * Iqinv

    # ── Voltage PI controller ─────────────────────────────────────────────
    v_err   = Vcmd - Vinv
    d_xiv   = kiv * v_err   # integrator with dynamic limits
    EVSM    = backend.hard_limits(kpv * v_err + xiv, Emin_dyn, Emax_dyn)

    # ── Fault current limiting (transient, multiplicative) ────────────────
    Imag     = backend.sqrt(i_d**2 + i_q**2 + EPS)
    f_cl     = backend.min(ImaxF / Imag, 1.0)
    EVSM_lim = EVSM * f_cl

    # ── Algebraic equations: voltage source behind (Re + jXL) ────────────
    g_id = EVSM_lim - Re * i_q - XL * i_d - v_q
    g_iq = Re * i_d + XL * i_q - v_d
    # Sign check (with Re=0): E = v_q + XL*id  and  v_d = XL*iq  ✓
    g_pg = i_d * v_d + i_q * v_q - p_g
    g_qg = i_d * v_q - i_q * v_d - q_g

    # ── Power injection ───────────────────────────────────────────────────
    p_W   = p_g * S_n
    q_var = q_g * S_n

    # ── Assembly ──────────────────────────────────────────────────────────
    if FFlag:
        grid.dae['f'] += [d_deltaIT, dDomegam, d_x_D2,
                          d_Pinv, d_Idinv, d_Qinv, d_Vinv, d_Iqinv,
                          d_xiv, d_deltaPLL, d_xPLL, d_x_DP]
        grid.dae['x'] += [deltaIT,  Domegam,  x_D2,
                          Pinv,  Idinv,  Qinv,  Vinv,  Iqinv,
                          xiv,  deltaPLL,  xPLL,  x_DP]
    else:
        grid.dae['f'] += [d_deltaIT, dDomegam, d_x_D2,
                          d_Pinv, d_Idinv, d_Qinv, d_Vinv, d_Iqinv,
                          d_xiv, d_deltaPLL, d_xPLL]
        grid.dae['x'] += [deltaIT,  Domegam,  x_D2,
                          Pinv,  Idinv,  Qinv,  Vinv,  Iqinv,
                          xiv,  deltaPLL,  xPLL]

    grid.dae['g']     += [g_id, g_iq, g_pg, g_qg]
    grid.dae['y_ini'] += [i_d,  i_q,  p_g,  q_g]
    grid.dae['y_run'] += [i_d,  i_q,  p_g,  q_g]

    # ── Virtual inertia contribution to COI ───────────────────────────────
    H_val = float(param('H'))
    if H_val > 0.0:
        H_sym = backend.symbols(f"H_{name}")
        grid.H_total               += H_val
        grid.omega_coi_numerator   += (1.0 + Domegam_eff) * H_sym * S_n
        grid.omega_coi_denominator += H_sym * S_n

    # ── Parameters ────────────────────────────────────────────────────────
    Tp_val = float(param('Tp'))
    Tp_eff_val = max(Tp_val, 5e-4)   # avoid Tp=0 singularity

    grid.dae['params_dict'].update({
        f"Omega_b_{name}":    2 * np.pi * float(param('F_n')),
        f"S_n_{name}":        param('S_n'),
        f"XL_{name}":         XL_val,
        f"Re_{name}":         param('Re'),
        f"mp_{name}":         param('mp'),
        f"Tp_eff_{name}":     Tp_eff_val,
        f"H_{name}":          H_val,
        f"D1_{name}":         param('D1'),
        f"D2_{name}":         param('D2'),
        f"omegaD_{name}":     param('omegaD'),
        f"Domega_max_{name}": param('Dωmax', 0.05),
        f"Domega_min_{name}": param('Dωmin', -0.05),
        f"mq_{name}":         param('mq'),
        f"kpv_{name}":        param('kpv'),
        f"kiv_{name}":        param('kiv'),
        f"Vref_{name}":       param('Vref'),
        f"Qref_{name}":       param('Qref'),
        f"Pref_{name}":       param('Pref'),
        f"ImaxSS_{name}":     ImaxSS_v,
        f"ImaxF_{name}":      param('ImaxF'),
        f"kf_{name}":         kf_v,
        f"Tpf_{name}":        param('Tpf'),
        f"TQf_{name}":        param('TQf'),
        f"TVf_{name}":        param('TVf'),
        f"Tif_{name}":        param('Tif'),
        f"kpPLL_{name}":      param('kpPLL'),
        f"kiPLL_{name}":      param('kiPLL'),
        f"DwPLLmax_{name}":   param('DwPLLmax', 0.2),
        f"DwPLLmin_{name}":   param('DwPLLmin', -0.2),
        f"VPLLfrz_{name}":    param('VPLLfrz'),
        f"delta_IT_min_{name}": delta_IT_min,
        f"delta_max_{name}":    delta_max_val,
    })

    # ── Initialization hints ──────────────────────────────────────────────
    Pref_0  = float(param('Pref'))
    Qref_0  = float(param('Qref'))
    Vref_0  = float(param('Vref'))
    V_0     = data_dict.get('V_ini', 1.0)

    # Steady state: Δωm=0, δPLL→θ (bus angle ≈0), ΔωPLL=0
    # E ≈ V + XL*Qref/V (small-angle approximation)
    E_0     = V_0 + XL_val * Qref_0 / max(V_0, 0.01)
    delta_0 = float(np.arctan(XL_val * Pref_0 / (E_0 * V_0)))
    i_q_0   = V_0 * np.sin(delta_0) / XL_val
    i_d_0   = (E_0 - V_0 * np.cos(delta_0)) / XL_val

    xy0 = {
        str(deltaIT):  delta_0,   # δVSM = δIT + δPLL ≈ δIT (PLL→θ≈0)
        str(Domegam):  0.0,
        str(x_D2):     0.0,
        str(Pinv):     Pref_0,
        str(Idinv):    i_d_0,
        str(Qinv):     Qref_0,
        str(Vinv):     V_0,
        str(Iqinv):    i_q_0,
        str(xiv):      E_0,       # voltage PI at steady state ≈ E_0
        str(deltaPLL): 0.0,
        str(xPLL):     0.0,
        str(i_d):      i_d_0,
        str(i_q):      i_q_0,
        str(p_g):      Pref_0,
        str(q_g):      Qref_0,
    }
    if FFlag:
        xy0[str(x_DP)] = 0.0   # ΔP = 0 at steady state
    grid.dae['xy_0_dict'].update(xy0)

    # ── Outputs ───────────────────────────────────────────────────────────
    grid.dae['h_dict'].update({
        f"p_g_{name}":       p_g,
        f"q_g_{name}":       q_g,
        f"EVSM_{name}":      EVSM,
        f"deltaIT_{name}":   deltaIT,
        f"deltaPLL_{name}":  deltaPLL,
        f"deltaVSM_{name}":  deltaVSM,
        f"Domegam_{name}":   Domegam,
        f"DomegaPLL_{name}": DeltaomegaPLL,
        f"Imag_{name}":      Imag,
        f"f_cl_{name}":      f_cl,
        f"Vcmd_{name}":      Vcmd,
    })

    return p_W, q_var


# ======================================================================
# In-module test
# ======================================================================
def test():
    import os, time
    from pydae.bps import BpsBuilder
    from pydae.core import Builder, Model

    hjson_path = os.path.join(os.path.dirname(__file__), 'regfm_b1.hjson')
    grid = BpsBuilder(hjson_path)
    grid.construct('temp_regfm_b1')
    bld = Builder(grid.sys_dict, target='cffi', sparse=False)
    bld.build()

    model = Model('temp_regfm_b1')
    t0 = time.perf_counter_ns()
    model.ini({}, 'temp_regfm_b1_xy_0.json')
    t1 = time.perf_counter_ns()
    print(f"ini time: {(t1-t0)/1e6:.2f} ms")

    p_g      = model.get_value('p_g_1')
    q_g      = model.get_value('q_g_1')
    EVSM     = model.get_value('EVSM_1')
    deltaVSM = model.get_value('deltaVSM_1')
    Domegam  = model.get_value('Domegam_1')
    f_cl     = model.get_value('f_cl_1')
    Imag     = model.get_value('Imag_1')
    deltaPLL = model.get_value('deltaPLL_1')

    print(f"p_g={p_g:.4f}  q_g={q_g:.4f}  EVSM={EVSM:.4f}  "
          f"δVSM={np.rad2deg(deltaVSM):.2f}°  Δωm={Domegam:.6f}")
    print(f"|I|={Imag:.4f}  f_cl={f_cl:.4f}  δPLL={np.rad2deg(deltaPLL):.4f}°")

    assert abs(p_g  - 0.8) < 0.05, f"p_g={p_g:.4f} not close to 0.8 pu"
    assert abs(q_g  - 0.0) < 0.05, f"q_g={q_g:.4f} not close to 0.0 pu"
    assert abs(Domegam) < 1e-4,     f"Δωm={Domegam:.6f} should be ~0 at steady state"
    assert f_cl > 0.99,             f"f_cl={f_cl:.4f} should be 1.0 (no overcurrent)"

    # Step-response: Pref step → Δωm should go negative (inertia slows)
    model.run(5.0, {'Pref_1': 1.0})
    model.post()
    Domegam_min = min(model.get_values('Domegam_1'))
    assert Domegam_min < -1e-5, (
        f"Δωm should dip negative during Pref step; got min={Domegam_min:.6f}"
    )
    print(f"min(Δωm) during Pref step = {Domegam_min:.6f}  ✓ (inertia response)")

    print("test() PASSED")


if __name__ == '__main__':
    test()
