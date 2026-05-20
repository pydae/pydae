# -*- coding: utf-8 -*-
r"""
WECC Renewable Energy Plant Controller model D (REPC_D).

Reference: *REPC_B.pdf* (WECC document recommending REPC_D as the replacement
for REPC_B), June 2024.  The document describes REPC_D as superseding REPC_B
with the following key improvements over REPC_A/B:

* **Own bus attachment** — standalone in ``ppcs``, not attached to a generator.
* **Absolute P/Q references** — commands initialized to power-flow values
  (not deviations).  ``Pmax``/``Pmin``, ``Qmax``/``Qmin`` are absolute limits.
* **Per-device dispatch** — individual Q and P references dispatched to each
  downstream converter with weight ``Kw_n`` / ``Kz_n``, per-device filter
  ``Tw_n`` / ``Tz_n``, and per-device limits ``Qmax_n`` / ``Pmax_n``.
* **Extended flags** — ``Pefd_Flag``, ``Ffwrd_Flag``, ``QVFlag``, extended
  ``RefFlg`` (adds power-factor mode ``RefFlg=2``).
* **Asymmetric Q deadband** — separate ``dbd1`` and ``dbd2`` parameters.
* **Additional filters** — frequency filter ``Tfrq``, reactive-current
  compensation with time constant ``Tc``.
* **Rate limits** — Q and P output rate limits (``qvrmax``/``qvrmin``,
  ``prmax``/``prmin``).

Mechanically-switched shunt (MSS) switching logic is omitted in this
implementation (hardware-specific; not needed for dynamic simulation).

Signal chain::

    Network  →  V_reg (POI), Q_brn, P_brn, Freq
                    │
              REPC_D  →  Qext_{gen_n}, Pref_{gen_n}  (per device)
                    │
         REEC_B/REGFM_A1/REGFM_B1 …  (each commanded device)

HJSON placement: top-level ``ppcs`` section, same as REPC_A::

    ppcs: [{
        type:    "repc_d",
        reg_bus: "POI",
        weccs:   ["gen1", "gen2"],    // converter names
        devices: [                    // per-device dispatch parameters
            {name: "gen1", Kw: 0.5, Kz: 0.5, Tw: 0.02, Tz: 0.02,
             Qmax: 0.4, Qmin: -0.4, Qo: 0.0,
             Pmax: 1.0, Pmin: 0.0,  Po: 0.8},
            {name: "gen2", Kw: 0.5, Kz: 0.5, Tw: 0.02, Tz: 0.02,
             Qmax: 0.4, Qmin: -0.4, Qo: 0.0,
             Pmax: 1.0, Pmin: 0.0,  Po: 0.6},
        ],
        ...
    }]
"""
import numpy as np


def descriptions():
    r"""Single source of truth for REPC_D parameters, states, and I/O."""
    d = []
    # ── filters ──────────────────────────────────────────────────────────
    d += [{"type": "Parameter", "tex": "T_{fltr}",  "data": "Tfltr",  "model": "Tfltr",  "default": 0.02,  "description": "V and Q measurement filter time constant",       "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_p",        "data": "Tp",     "model": "Tp",     "default": 0.02,  "description": "Active power measurement filter time constant",  "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_{frq}",    "data": "Tfrq",   "model": "Tfrq",   "default": 0.02,  "description": "Frequency measurement filter time constant",     "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_c",        "data": "Tc",     "model": "Tc",     "default": 0.02,  "description": "Reactive-current compensation filter time constant","units": "s"}]
    # ── Q/V control ───────────────────────────────────────────────────────
    d += [{"type": "Parameter", "tex": "K_p",        "data": "Kp",     "model": "Kp",     "default": 0.5,   "description": "Q/V PI proportional gain",                       "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "K_i",        "data": "Ki",     "model": "Ki",     "default": 3.0,   "description": "Q/V PI integral gain",                          "units": "pu/pu·s"}]
    d += [{"type": "Parameter", "tex": "K_c",        "data": "Kc",     "model": "Kc",     "default": 0.0,   "description": "Reactive-current compensation gain",            "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "T_{ft}",     "data": "Tft",    "model": "Tft",    "default": 0.0,   "description": "Q output lead time constant",                   "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_{fv}",     "data": "Tfv",    "model": "Tfv",    "default": 0.05,  "description": "Q output lag time constant",                    "units": "s"}]
    d += [{"type": "Parameter", "tex": "e_{max}",    "data": "emax",   "model": "emax",   "default": 0.3,   "description": "Q/V error upper limit",                         "units": "pu"}]
    d += [{"type": "Parameter", "tex": "e_{min}",    "data": "emin",   "model": "emin",   "default": -0.3,  "description": "Q/V error lower limit",                         "units": "pu"}]
    d += [{"type": "Parameter", "tex": "dbd_1",      "data": "dbd1",   "model": "dbd1",   "default": -0.01, "description": "Q/V error deadband lower bound (overvoltage)",  "units": "pu"}]
    d += [{"type": "Parameter", "tex": "dbd_2",      "data": "dbd2",   "model": "dbd2",   "default": 0.01,  "description": "Q/V error deadband upper bound (undervoltage)", "units": "pu"}]
    d += [{"type": "Parameter", "tex": "Q_{max}",    "data": "Qmax",   "model": "Qmax",   "default": 0.4,   "description": "Plant Q command upper limit (absolute)",         "units": "pu"}]
    d += [{"type": "Parameter", "tex": "Q_{min}",    "data": "Qmin",   "model": "Qmin",   "default": -0.4,  "description": "Plant Q command lower limit (absolute)",         "units": "pu"}]
    d += [{"type": "Parameter", "tex": "V_{ref}",    "data": "Vref",   "model": "Vref",   "default": 1.0,   "description": "POI voltage reference",                         "units": "pu"}]
    d += [{"type": "Parameter", "tex": "Q_{ref}",    "data": "Qref",   "model": "Qref",   "default": 0.0,   "description": "POI reactive power reference (RefFlg=0)",        "units": "pu"}]
    d += [{"type": "Parameter", "tex": "V_{frz}",    "data": "Vfrz",   "model": "Vfrz",   "default": 0.7,   "description": "Voltage threshold for PI integrator freeze",    "units": "pu"}]
    d += [{"type": "Parameter", "tex": "V_{refmax}", "data": "Vrefmax","model": "Vrefmax","default": 1.1,   "description": "Voltage reference upper limit",                 "units": "pu"}]
    d += [{"type": "Parameter", "tex": "V_{refmin}", "data": "Vrefmin","model": "Vrefmin","default": 0.9,   "description": "Voltage reference lower limit",                 "units": "pu"}]
    d += [{"type": "Parameter", "tex": "qvr_{max}",  "data": "qvrmax", "model": "qvrmax", "default": 999.0, "description": "Q output rate-of-change upper limit",           "units": "pu/s"}]
    d += [{"type": "Parameter", "tex": "qvr_{min}",  "data": "qvrmin", "model": "qvrmin", "default": -999.0,"description": "Q output rate-of-change lower limit",           "units": "pu/s"}]
    # ── P / frequency control ─────────────────────────────────────────────
    d += [{"type": "Parameter", "tex": "K_{pg}",     "data": "Kpg",    "model": "Kpg",    "default": 0.5,   "description": "P PI proportional gain",                        "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "K_{ig}",     "data": "Kig",    "model": "Kig",    "default": 0.25,  "description": "P PI integral gain",                           "units": "pu/pu·s"}]
    d += [{"type": "Parameter", "tex": "P_{max}",    "data": "Pmax",   "model": "Pmax",   "default": 1.0,   "description": "Plant P command upper limit (absolute)",         "units": "pu"}]
    d += [{"type": "Parameter", "tex": "P_{min}",    "data": "Pmin",   "model": "Pmin",   "default": 0.0,   "description": "Plant P command lower limit (absolute)",         "units": "pu"}]
    d += [{"type": "Parameter", "tex": "T_{lag}",    "data": "Tlag",   "model": "Tlag",   "default": 0.7,   "description": "P output lag time constant",                    "units": "s"}]
    d += [{"type": "Parameter", "tex": "f_{dbd1}",   "data": "fdbd1",  "model": "fdbd1",  "default": -0.0006,"description": "Frequency droop deadband lower limit",          "units": "pu"}]
    d += [{"type": "Parameter", "tex": "f_{dbd2}",   "data": "fdbd2",  "model": "fdbd2",  "default": 0.0006, "description": "Frequency droop deadband upper limit",          "units": "pu"}]
    d += [{"type": "Parameter", "tex": "D_{dn}",     "data": "Ddn",    "model": "Ddn",    "default": 103.33,"description": "Down-regulation droop gain",                    "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "D_{up}",     "data": "Dup",    "model": "Dup",    "default": 103.33,"description": "Up-regulation droop gain",                      "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "f_{emax}",   "data": "femax",  "model": "femax",  "default": 999.0, "description": "P droop error upper limit",                     "units": "pu"}]
    d += [{"type": "Parameter", "tex": "f_{emin}",   "data": "femin",  "model": "femin",  "default": -999.0,"description": "P droop error lower limit",                     "units": "pu"}]
    d += [{"type": "Parameter", "tex": "pi_{max}",   "data": "pimax",  "model": "pimax",  "default": 1.5,   "description": "P PI output upper limit",                       "units": "pu"}]
    d += [{"type": "Parameter", "tex": "pi_{min}",   "data": "pimin",  "model": "pimin",  "default": -0.5,  "description": "P PI output lower limit",                       "units": "pu"}]
    d += [{"type": "Parameter", "tex": "pr_{max}",   "data": "prmax",  "model": "prmax",  "default": 999.0, "description": "P output rate-of-change upper limit",           "units": "pu/s"}]
    d += [{"type": "Parameter", "tex": "pr_{min}",   "data": "prmin",  "model": "prmin",  "default": -999.0,"description": "P output rate-of-change lower limit",           "units": "pu/s"}]
    # ── flags ─────────────────────────────────────────────────────────────
    d += [{"type": "Parameter", "tex": "RefFlg",     "data": "RefFlg",     "model": "RefFlg",     "default": 1, "description": "0=Q ctrl  1=V ctrl  2=PF ctrl",          "units": "-"}]
    d += [{"type": "Parameter", "tex": "VcmpFlg",    "data": "VcmpFlg",    "model": "VcmpFlg",    "default": 0, "description": "0=Q droop  1=line drop comp.",            "units": "-"}]
    d += [{"type": "Parameter", "tex": "Freq_flag",  "data": "Freq_flag",  "model": "Freq_flag",  "default": 1, "description": "0=no governor  1=freq droop active",      "units": "-"}]
    d += [{"type": "Parameter", "tex": "QVFlag",     "data": "QVFlag",     "model": "QVFlag",     "default": 1, "description": "0=Q ctrl disabled  1=Q ctrl active",      "units": "-"}]
    d += [{"type": "Parameter", "tex": "Pefd_Flag",  "data": "Pefd_Flag",  "model": "Pefd_Flag",  "default": 1, "description": "0=bypass P measurement  1=use P feedback","units": "-"}]
    d += [{"type": "Parameter", "tex": "Ffwrd_Flag", "data": "Ffwrd_Flag", "model": "Ffwrd_Flag", "default": 0, "description": "0=no feedfwd  1=P feedforward after PI",  "units": "-"}]
    return d


def repc_d(dae, data, name, backend):
    """Add REPC_D plant controller *name* to the DAE system.

    Parameters
    ----------
    dae : dict
        BpsBuilder DAE dictionary.
    data : dict
        HJSON entry for this REPC_D instance (from the ``ppcs`` list).
    name : str
        Unique identifier for this plant controller.
    backend : MathBackend
        Symbolic backend (SymPy or CasADi).
    """
    def p(key, default=0.0):
        return data.get(key, default)

    # ── flags ─────────────────────────────────────────────────────────────
    RefFlg     = int(p('RefFlg',     1))
    VcmpFlg    = int(p('VcmpFlg',    0))
    Freq_flag  = int(p('Freq_flag',  1))
    QVFlag     = int(p('QVFlag',     1))
    Pefd_Flag  = int(p('Pefd_Flag',  1))
    Ffwrd_Flag = int(p('Ffwrd_Flag', 0))

    # ── network ───────────────────────────────────────────────────────────
    reg_bus   = str(p('reg_bus', 'POI'))
    Vreg      = backend.symbols(f"V_{reg_bus}")
    omega_coi = backend.symbols("omega_coi")
    freq_dev  = omega_coi - 1.0

    # ── converter and device lists ────────────────────────────────────────
    wecc_names   = data.get('weccs',   [])
    devices_data = data.get('devices', [])

    # Build device parameter lookup (keyed by device name)
    dev_params = {d_['name']: d_ for d_ in devices_data} if devices_data else {}

    def dev(dev_name, key, default=0.0):
        return dev_params.get(dev_name, {}).get(key, default)

    # ── plant-level output symbols ─────────────────────────────────────────
    Qext_plant = backend.symbols(f"Qext_{name}")   # e.g. Qext_repc1
    Pref_plant = backend.symbols(f"Pref_{name}")   # e.g. Pref_repc1

    # Per-converter connector symbols
    _Qext_syms = [backend.symbols(f"Qext_{g}") for g in wecc_names]
    _Pref_syms = [backend.symbols(f"Pref_{g}") for g in wecc_names]

    # ── symbolic parameters ────────────────────────────────────────────────
    Tfltr   = backend.symbols(f"Tfltr_{name}")
    Tp_p    = backend.symbols(f"Tp_repc_{name}")
    Tfrq    = backend.symbols(f"Tfrq_{name}")
    Tc      = backend.symbols(f"Tc_{name}")
    Kp      = backend.symbols(f"Kp_{name}")
    Ki      = backend.symbols(f"Ki_{name}")
    Kc      = backend.symbols(f"Kc_{name}")
    Tft     = backend.symbols(f"Tft_{name}")
    Tfv     = backend.symbols(f"Tfv_{name}")
    emax    = backend.symbols(f"emax_{name}")
    emin    = backend.symbols(f"emin_{name}")
    dbd1    = backend.symbols(f"dbd1_{name}")
    dbd2    = backend.symbols(f"dbd2_{name}")
    Qmax    = backend.symbols(f"Qmax_{name}")
    Qmin    = backend.symbols(f"Qmin_{name}")
    Vref    = backend.symbols(f"Vref_{name}")
    Qref    = backend.symbols(f"Qref_{name}")
    Vfrz    = backend.symbols(f"Vfrz_{name}")
    Vrefmax = backend.symbols(f"Vrefmax_{name}")
    Vrefmin = backend.symbols(f"Vrefmin_{name}")
    Kpg     = backend.symbols(f"Kpg_{name}")
    Kig     = backend.symbols(f"Kig_{name}")
    Pmax_r  = backend.symbols(f"Pmax_repc_{name}")
    Pmin_r  = backend.symbols(f"Pmin_repc_{name}")
    Tlag    = backend.symbols(f"Tlag_{name}")
    fdbd1   = backend.symbols(f"fdbd1_{name}")
    fdbd2   = backend.symbols(f"fdbd2_{name}")
    Ddn     = backend.symbols(f"Ddn_{name}")
    Dup     = backend.symbols(f"Dup_{name}")
    femax   = backend.symbols(f"femax_{name}")
    femin   = backend.symbols(f"femin_{name}")
    pimax   = backend.symbols(f"pimax_{name}")
    pimin   = backend.symbols(f"pimin_{name}")
    Vref_r  = backend.symbols(f"Vref_r_{name}")
    Qref_r  = backend.symbols(f"Qref_r_{name}")
    Pref_r  = backend.symbols(f"Pref_r_{name}")

    # ── states ────────────────────────────────────────────────────────────
    Vreg_flt  = backend.symbols(f"Vreg_flt_{name}")   # V measurement filter
    Qbrn_flt  = backend.symbols(f"Qbrn_flt_{name}")   # Q branch filter
    x_Kq      = backend.symbols(f"x_Kq_{name}")       # Q/V PI integrator
    x_Tfv     = backend.symbols(f"x_Tfv_{name}")      # lead-lag lag state
    Pbrn_flt  = backend.symbols(f"Pbrn_flt_{name}")   # P branch filter
    x_Kig     = backend.symbols(f"x_Kig_{name}")      # P PI integrator
    Pref_lag  = backend.symbols(f"Pref_lag_{name}")   # P output lag
    Freq_flt  = backend.symbols(f"Freq_flt_{name}")   # frequency filter

    # ── branch power (approximated from terminal quantities) ───────────────
    Pbranch = sum(
        backend.symbols(f"Ip_{n}") * backend.symbols(f"V_{reg_bus}")
        for n in wecc_names
    ) if wecc_names else backend.symbols(f"V_{reg_bus}") * 0.0

    Qbranch = sum(
        backend.symbols(f"Iq_{n}") * backend.symbols(f"V_{reg_bus}")
        for n in wecc_names
    ) if wecc_names else backend.symbols(f"V_{reg_bus}") * 0.0

    # ── Q/V control path ─────────────────────────────────────────────────

    # Voltage effective (with optional reactive-current droop or line drop comp.)
    Vreg_eff = Vreg - Kc * Qbranch   # reactive droop (VcmpFlg=0) or approx.

    dVreg_flt = (Vreg_eff - Vreg_flt) / Tfltr
    dQbrn_flt = (Qbranch  - Qbrn_flt) / Tfltr

    # Voltage reference clamped (REPC_D adds Vrefmax/Vrefmin)
    Vref_clamped = backend.hard_limits(Vref_r, Vrefmin, Vrefmax)

    # Error (mode-dependent via RefFlg — Python flag)
    if RefFlg == 1:
        qv_err = Vref_clamped - Vreg_flt
    elif RefFlg == 0:
        qv_err = Qref_r - Qbrn_flt
    else:   # RefFlg == 2: power factor control (approximate as Q error)
        qv_err = Qref_r - Qbrn_flt

    # Asymmetric deadband (REPC_D uses separate dbd1 and dbd2)
    qv_db = backend.max(qv_err - dbd2, 0.0) + backend.min(qv_err - dbd1, 0.0)
    qv_db_clipped = backend.hard_limits(qv_db, emin, emax)

    # PI integrator with integrator freeze when V < Vfrz
    freeze = backend.hard_limits(Vreg_flt - Vfrz, 0.0, 1.0)
    dx_Kq  = Ki * qv_db_clipped * freeze

    PI_Q_out = backend.hard_limits(Kp * qv_db_clipped + x_Kq, Qmin, Qmax)

    # Lead-lag on Q output
    dx_Tfv  = (PI_Q_out - x_Tfv) / Tfv
    Qext_raw = PI_Q_out * Tft / Tfv + x_Tfv * (1.0 - Tft / Tfv)

    if QVFlag:
        g_Qext_plant = Qext_plant - backend.hard_limits(Qext_raw, Qmin, Qmax)
    else:
        # QVFlag=0: Q control disabled, Qext = initial value
        g_Qext_plant = Qext_plant - Qref_r

    # ── P / frequency control path ────────────────────────────────────────

    # Frequency filter
    dFreq_flt = (freq_dev - Freq_flt) / Tfrq

    # Active power filter (Pefd_Flag selects whether to use P measurement)
    dPbrn_flt = (Pbranch - Pbrn_flt) / Tp_p
    P_feedback = Pbrn_flt if Pefd_Flag else Pbrn_flt * 0.0

    # Frequency droop (dual deadband)
    f_db = (backend.max(Freq_flt - fdbd2, 0.0) * Ddn
            + backend.min(Freq_flt - fdbd1, 0.0) * Dup)
    P_err = Pref_r - P_feedback + f_db
    P_err_clipped = backend.hard_limits(P_err, femin, femax)

    dx_Kig   = Kig * P_err_clipped
    PI_P_out = backend.hard_limits(Kpg * P_err_clipped + x_Kig, pimin, pimax)

    # Optional P feedforward (adds measured P after PI)
    PI_P_ffwd = PI_P_out + (P_feedback if Ffwrd_Flag else 0.0)
    PI_P_ffwd_lim = backend.hard_limits(PI_P_ffwd, Pmin_r, Pmax_r)

    dPref_lag = (PI_P_ffwd_lim - Pref_lag) / Tlag

    if Freq_flag:
        g_Pref_plant = Pref_plant - backend.hard_limits(Pref_lag, Pmin_r, Pmax_r)
    else:
        g_Pref_plant = Pref_plant - Pref_r

    # ── main controller assembly ──────────────────────────────────────────
    dae['f'] += [dVreg_flt, dQbrn_flt, dx_Kq, dx_Tfv,
                 dPbrn_flt, dx_Kig, dPref_lag, dFreq_flt]
    dae['x'] += [Vreg_flt,  Qbrn_flt,  x_Kq,  x_Tfv,
                 Pbrn_flt,  x_Kig,  Pref_lag,  Freq_flt]

    dae['g']     += [g_Qext_plant, g_Pref_plant]
    dae['y_ini'] += [Qext_plant,   Pref_plant]
    dae['y_run'] += [Qext_plant,   Pref_plant]

    # ── per-device dispatch layer ─────────────────────────────────────────
    # Each device n gets a dispatched Q and P reference:
    #   Q_device_n = Qo_n + Kw_n * (Qext_plant − Qext_o)
    #   P_device_n = Po_n + Kz_n * (Pref_plant − Pref_o)
    # filtered through Tw_n / Tz_n and clamped to [Qmin_n, Qmax_n] / [Pmin_n, Pmax_n].

    Qext_o_val = sum(dev(g, 'Qo', 0.0) for g in wecc_names)
    Pref_o_val = sum(dev(g, 'Po', p('Pref_r', p('Pref', 0.0))) / max(len(wecc_names), 1)
                     for g in wecc_names)

    Qext_o_sym = backend.symbols(f"Qext_o_{name}")
    Pref_o_sym = backend.symbols(f"Pref_o_{name}")
    dae['params_dict'][f"Qext_o_{name}"] = Qext_o_val
    dae['params_dict'][f"Pref_o_{name}"] = Pref_o_val

    for gen, Qe_sym, Pr_sym in zip(wecc_names, _Qext_syms, _Pref_syms):
        Kw_n   = float(dev(gen, 'Kw',   1.0))
        Kz_n   = float(dev(gen, 'Kz',   1.0))
        Tw_n   = float(dev(gen, 'Tw',   0.02))
        Tz_n   = float(dev(gen, 'Tz',   0.02))
        Qmax_n = float(dev(gen, 'Qmax', 0.4))
        Qmin_n = float(dev(gen, 'Qmin', -0.4))
        Pmax_n = float(dev(gen, 'Pmax', 1.0))
        Pmin_n = float(dev(gen, 'Pmin', 0.0))
        Qo_n   = float(dev(gen, 'Qo',   0.0))
        Po_n   = float(dev(gen, 'Po',   0.0))

        Kw_sym   = backend.symbols(f"Kw_{gen}_{name}")
        Kz_sym   = backend.symbols(f"Kz_{gen}_{name}")
        Tw_sym   = backend.symbols(f"Tw_{gen}_{name}")
        Tz_sym   = backend.symbols(f"Tz_{gen}_{name}")
        Qmax_sym = backend.symbols(f"Qmax_{gen}_{name}")
        Qmin_sym = backend.symbols(f"Qmin_{gen}_{name}")
        Pmax_sym = backend.symbols(f"Pmax_{gen}_{name}")
        Pmin_sym = backend.symbols(f"Pmin_{gen}_{name}")
        Qo_sym   = backend.symbols(f"Qo_{gen}_{name}")
        Po_sym   = backend.symbols(f"Po_{gen}_{name}")

        # Per-device dispatch filter states
        xQ_n = backend.symbols(f"xQ_{gen}_{name}")
        xP_n = backend.symbols(f"xP_{gen}_{name}")

        # Dispatch targets
        Q_device = Qo_sym + Kw_sym * (Qext_plant - Qext_o_sym)
        P_device = Po_sym + Kz_sym * (Pref_plant - Pref_o_sym)

        dxQ_n = (Q_device - xQ_n) / Tw_sym
        dxP_n = (P_device - xP_n) / Tz_sym

        # Per-device algebraic constraints
        g_Qext_gen = Qe_sym - backend.hard_limits(xQ_n, Qmin_sym, Qmax_sym)
        g_Pref_gen = Pr_sym - backend.hard_limits(xP_n, Pmin_sym, Pmax_sym)

        # Assembly
        dae['f'] += [dxQ_n, dxP_n]
        dae['x'] += [xQ_n,  xP_n]

        dae['g']     += [g_Qext_gen, g_Pref_gen]
        dae['y_ini'] += [Qe_sym,     Pr_sym]
        dae['y_run'] += [Qe_sym,     Pr_sym]

        # Pop per-device inputs from u_run (were set by REEC_B)
        dae['u_ini_dict'].pop(f'Qext_{gen}', None)
        dae['u_run_dict'].pop(f'Qext_{gen}', None)
        dae['u_ini_dict'].pop(f'Pref_{gen}', None)
        dae['u_run_dict'].pop(f'Pref_{gen}', None)

        # Per-device parameters
        dae['params_dict'].update({
            f"Kw_{gen}_{name}":   Kw_n,
            f"Kz_{gen}_{name}":   Kz_n,
            f"Tw_{gen}_{name}":   Tw_n,
            f"Tz_{gen}_{name}":   Tz_n,
            f"Qmax_{gen}_{name}": Qmax_n,
            f"Qmin_{gen}_{name}": Qmin_n,
            f"Pmax_{gen}_{name}": Pmax_n,
            f"Pmin_{gen}_{name}": Pmin_n,
            f"Qo_{gen}_{name}":   Qo_n,
            f"Po_{gen}_{name}":   Po_n,
        })

        # Per-device initialization
        dae['xy_0_dict'].update({
            str(xQ_n): Qo_n,
            str(xP_n): Po_n,
            str(Qe_sym): Qo_n,
            str(Pr_sym): Po_n,
        })

    # ── main controller parameters ────────────────────────────────────────
    Pref_0 = float(p('Pref_r', p('Pref', 0.0)))
    V_0    = float(p('Vref',  1.0))
    Q_0    = float(p('Qref',  0.0))
    Qext_0 = float(p('Qext',  Q_0))
    Pref_ss = Pref_0

    dae['params_dict'].update({
        f"Tfltr_{name}":     p('Tfltr',  0.02),
        f"Tp_repc_{name}":   p('Tp',     0.02),
        f"Tfrq_{name}":      p('Tfrq',   0.02),
        f"Tc_{name}":        p('Tc',     0.02),
        f"Kp_{name}":        p('Kp',     0.5),
        f"Ki_{name}":        p('Ki',     3.0),
        f"Kc_{name}":        p('Kc',     0.0),
        f"Tft_{name}":       p('Tft',    0.0),
        f"Tfv_{name}":       p('Tfv',    0.05),
        f"emax_{name}":      p('emax',   0.3),
        f"emin_{name}":      p('emin',  -0.3),
        f"dbd1_{name}":      p('dbd1',  -0.01),
        f"dbd2_{name}":      p('dbd2',   0.01),
        f"Qmax_{name}":      p('Qmax',   0.4),
        f"Qmin_{name}":      p('Qmin',  -0.4),
        f"Vref_{name}":      V_0,
        f"Qref_{name}":      Q_0,
        f"Vfrz_{name}":      p('Vfrz',   0.7),
        f"Vrefmax_{name}":   p('Vrefmax', 1.1),
        f"Vrefmin_{name}":   p('Vrefmin', 0.9),
        f"Kpg_{name}":       p('Kpg',   0.5),
        f"Kig_{name}":       p('Kig',   0.25),
        f"Pmax_repc_{name}": p('Pmax',   1.0),
        f"Pmin_repc_{name}": p('Pmin',   0.0),
        f"Tlag_{name}":      p('Tlag',   0.7),
        f"fdbd1_{name}":     p('fdbd1', -0.0006),
        f"fdbd2_{name}":     p('fdbd2',  0.0006),
        f"Ddn_{name}":       p('Ddn',   103.33),
        f"Dup_{name}":       p('Dup',   103.33),
        f"femax_{name}":     p('femax',  999.0),
        f"femin_{name}":     p('femin', -999.0),
        f"pimax_{name}":     p('pimax',  1.5),
        f"pimin_{name}":     p('pimin', -0.5),
        f"Vref_r_{name}":    V_0,
        f"Qref_r_{name}":    Q_0,
        f"Pref_r_{name}":    Pref_ss,
    })

    # ── initialization hints ──────────────────────────────────────────────
    dae['xy_0_dict'].update({
        str(Vreg_flt): V_0,
        str(Qbrn_flt): Q_0,
        str(x_Kq):     Qext_0,
        str(x_Tfv):    Qext_0,
        str(Pbrn_flt): Pref_ss,
        str(x_Kig):    Pref_ss,
        str(Pref_lag): Pref_ss,
        str(Freq_flt): 0.0,
        str(Qext_plant): Qext_0,
        str(Pref_plant): Pref_ss,
    })


# ======================================================================
# In-module test
# ======================================================================
def test():
    import os, time
    from pydae.bps import BpsBuilder
    from pydae.core import Builder, Model

    hjson_path = os.path.join(os.path.dirname(__file__), 'repc_d.hjson')
    grid = BpsBuilder(hjson_path)
    grid.construct('temp_repc_d')
    bld = Builder(grid.sys_dict, target='cffi', sparse=False)
    bld.build()

    model = Model('temp_repc_d')
    t0 = time.perf_counter_ns()
    model.ini({}, 'temp_repc_d_xy_0.json')
    t1 = time.perf_counter_ns()
    print(f"ini time: {(t1-t0)/1e6:.2f} ms")

    Qext_plant = model.get_value('Qext_repc1')
    Pref_plant = model.get_value('Pref_repc1')
    Qext_1     = model.get_value('Qext_1')
    Pref_1     = model.get_value('Pref_1')
    Qext_2     = model.get_value('Qext_2')
    Pref_2     = model.get_value('Pref_2')
    Ip_1       = model.get_value('Ip_1')
    Ip_2       = model.get_value('Ip_2')

    print(f"Qext_plant={Qext_plant:.4f}  Pref_plant={Pref_plant:.4f}")
    print(f"Gen1: Qext={Qext_1:.4f}  Pref={Pref_1:.4f}  Ip={Ip_1:.4f}")
    print(f"Gen2: Qext={Qext_2:.4f}  Pref={Pref_2:.4f}  Ip={Ip_2:.4f}")

    assert abs(Ip_1) > 0.1, f"Ip_1={Ip_1:.4f} — converter 1 not injecting"
    assert abs(Ip_2) > 0.1, f"Ip_2={Ip_2:.4f} — converter 2 not injecting"
    # Per-device Qext is dispatched (Qo_n + Kw*(plant − plant_o)), not equal to plant value
    # With Kw=0.5, Qo=0.0, Qext_o=0.0: Qext_n = 0.5*Qext_plant → clamped by device limits
    assert abs(Pref_1 - 0.8) < 0.05, f"Pref_1={Pref_1:.4f} not close to 0.8 (Po_1)"
    assert abs(Pref_2 - 0.6) < 0.05, f"Pref_2={Pref_2:.4f} not close to 0.6 (Po_2)"

    model.run(5.0, {})
    model.post()
    import numpy as np
    for var in ['Qext_repc1', 'Pref_repc1', 'Ip_1', 'Ip_2']:
        vals = model.get_values(var)
        assert np.all(np.isfinite(vals)), f"{var} went non-finite"
    print("test() PASSED")


if __name__ == '__main__':
    test()
