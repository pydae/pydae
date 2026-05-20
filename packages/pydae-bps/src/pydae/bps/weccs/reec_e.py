# -*- coding: utf-8 -*-
r"""
WECC Renewable Energy Electrical Control model E (REEC_E) — BESS.

Reference: WECC Battery Energy Storage System (BESS) Model Specification,
second-generation models (post-2019).

REEC_E is the **local electrical control** layer for Battery Energy Storage
Systems (BESS). It extends :mod:`reec_b` with:

* A **State of Charge (SOC)** integrator state.
* **Bidirectional active power**: ``P_flt`` and ``Ipcmd`` can be negative
  (charging from the network) or positive (discharging to the network).
* **SOC-based power limit modulation**: discharge capability ramps to zero as
  SOC → ``SOC_min``; charge capability ramps to zero as SOC → ``SOC_max``.

The reactive current control path (V-PI, Q-PI, VRT injection, current limit
logic) is identical to REEC_B.

Signal chain::

    REPC_A  (plant level, ppcs/)     [optional]
        │  Pref, Qext
    REEC_E  (local BESS control)     ← this file
        │  Ipcmd  (signed: + discharge, − charge)
        │  Iqcmd
    REGC_A/B  (grid interface)
        │  Ip, Iq
    Network

.. note::

    For charging (``Ipcmd < 0``) to actually reach the network the connected
    REGC model must support negative ``Ip``.  When used with the standard
    REGC_A or REGC_B (which clip ``Ip`` to ``[0, Ipmax]``) only the
    discharge direction is active.  A bidirectional REGC variant or a direct
    current injection scheme is required for full charge/discharge operation.

**Differential equations**

.. math::

    \dot{V}_{t,flt} &= (V_t - V_{t,flt}) / T_{rv}  \\
    \dot{x}_{Pe}    &= (P_e - x_{Pe}) / T_p  \\
    \dot{x}_{Kqi}   &= K_{qi}(Q_{ref} - Q_e)  \quad (V_{flag}=0)  \\
    \dot{x}_{Kvi}   &= K_{vi}(V_{t,flt} - V_{ref,ctrl})  \\
    \dot{V}_{l,flt} &= \mathrm{clip}((K_{vi,out} - V_{l,flt})/T_{iq},
                                      \,dP_{min},\,dP_{max})  \\
    \dot{P}_{flt}   &= \mathrm{clip}((P_{ref} - P_{flt})/T_{pord},
                                      \,dP_{min},\,dP_{max})  \\
    \dot{SOC}       &= -I_p V_t / E_{cap}

**Algebraic outputs**

.. math::

    I_{pcmd} &= \mathrm{clip}\!\left(
                    P_{flt,lim}/\max(V_{t,flt},\,0.01),\;
                    I_{pcmd,min},\; I_{pmax}\right)  \\
    I_{qcmd} &= \mathrm{clip}(V_{l,flt} + I_{q,inj},\;
                               I_{q\min},\; I_{q\max})

where ``P_flt_lim`` is ``P_flt`` after SOC-based limit modulation.
"""
import numpy as np


def descriptions():
    r"""Single source of truth for REEC_E parameters, states, and I/O."""
    d = []
    # ── time constants ────────────────────────────────────────────────────
    d += [{"type": "Parameter", "tex": "T_{rv}",    "data": "Trv",    "model": "Trv",    "default": 0.02,   "description": "Terminal voltage filter time constant",             "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_p",       "data": "Tp",     "model": "Tp",     "default": 0.02,   "description": "Active power filter time constant",                "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_{iq}",    "data": "Tiq",    "model": "Tiq",    "default": 0.02,   "description": "Reactive current lag time constant",               "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_{pord}",  "data": "Tpord",  "model": "Tpord",  "default": 0.05,   "description": "Active power order lag time constant",             "units": "s"}]
    # ── V/Q control ───────────────────────────────────────────────────────
    d += [{"type": "Parameter", "tex": "K_{vp}",    "data": "Kvp",    "model": "Kvp",    "default": 1.0,    "description": "Inner voltage PI proportional gain",               "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "K_{vi}",    "data": "Kvi",    "model": "Kvi",    "default": 40.0,   "description": "Inner voltage PI integral gain",                   "units": "pu/pu·s"}]
    d += [{"type": "Parameter", "tex": "K_{qp}",    "data": "Kqp",    "model": "Kqp",    "default": 0.0,    "description": "Q-PI proportional gain (Vflag=0)",                 "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "K_{qi}",    "data": "Kqi",    "model": "Kqi",    "default": 0.0,    "description": "Q-PI integral gain (Vflag=0)",                     "units": "pu/pu·s"}]
    d += [{"type": "Parameter", "tex": "V_{ref0}",  "data": "Vref0",  "model": "Vref0",  "default": 1.0,    "description": "Initial voltage reference",                        "units": "pu"}]
    d += [{"type": "Parameter", "tex": "dbd_1",     "data": "dbd1",   "model": "dbd1",   "default": -0.05,  "description": "VRT deadband lower bound (overvoltage, <0)",        "units": "pu"}]
    d += [{"type": "Parameter", "tex": "dbd_2",     "data": "dbd2",   "model": "dbd2",   "default": 0.05,   "description": "VRT deadband upper bound (undervoltage, >0)",       "units": "pu"}]
    d += [{"type": "Parameter", "tex": "K_{qv}",    "data": "Kqv",    "model": "Kqv",    "default": 2.0,    "description": "VRT reactive current injection gain",               "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "I_{qhl}",   "data": "Iqhl",   "model": "Iqhl",   "default": 1.05,   "description": "VRT Iqinj upper limit",                            "units": "pu"}]
    d += [{"type": "Parameter", "tex": "I_{qll}",   "data": "Iqll",   "model": "Iqll",   "default": -1.05,  "description": "VRT Iqinj lower limit (<0)",                       "units": "pu"}]
    d += [{"type": "Parameter", "tex": "V_{max}",   "data": "Vmax",   "model": "Vmax",   "default": 1.1,    "description": "Q-PI output upper voltage limit",                  "units": "pu"}]
    d += [{"type": "Parameter", "tex": "V_{min}",   "data": "Vmin",   "model": "Vmin",   "default": 0.9,    "description": "Q-PI output lower voltage limit",                  "units": "pu"}]
    d += [{"type": "Parameter", "tex": "I_{max}",   "data": "Imax",   "model": "Imax",   "default": 1.1,    "description": "Maximum apparent current magnitude",               "units": "pu"}]
    # ── BESS power limits ─────────────────────────────────────────────────
    d += [{"type": "Parameter", "tex": "P_{max,dis}","data": "Pmax_dis","model": "Pmax_dis","default": 1.0,  "description": "Maximum discharge power (>0)",                      "units": "pu"}]
    d += [{"type": "Parameter", "tex": "P_{max,chg}","data": "Pmax_chg","model": "Pmax_chg","default": -1.0, "description": "Maximum charge power (≤0)",                        "units": "pu"}]
    d += [{"type": "Parameter", "tex": "dP_{max}",  "data": "dPmax",  "model": "dPmax",  "default": 999.0,  "description": "Active power up-ramp rate limit",                  "units": "pu/s"}]
    d += [{"type": "Parameter", "tex": "dP_{min}",  "data": "dPmin",  "model": "dPmin",  "default": -999.0, "description": "Active power down-ramp rate limit",                "units": "pu/s"}]
    # ── BESS energy / SOC ─────────────────────────────────────────────────
    d += [{"type": "Parameter", "tex": "E_{cap}",   "data": "E_cap",  "model": "E_cap",  "default": 3600.0, "description": "Battery energy capacity (S_n × E_cap = energy in J)","units": "s"}]
    d += [{"type": "Parameter", "tex": "SOC_{ini}", "data": "SOC_ini","model": "SOC_ini","default": 0.5,    "description": "Initial state of charge",                          "units": "pu"}]
    d += [{"type": "Parameter", "tex": "SOC_{max}", "data": "SOC_max","model": "SOC_max","default": 0.9,    "description": "SOC upper limit (charge capability ramps to 0 above)", "units": "pu"}]
    d += [{"type": "Parameter", "tex": "SOC_{min}", "data": "SOC_min","model": "SOC_min","default": 0.1,    "description": "SOC lower limit (discharge capability ramps to 0 below)","units": "pu"}]
    d += [{"type": "Parameter", "tex": "\\Delta SOC_{db}","data": "dSOC_db","model": "dSOC_db","default": 0.05, "description": "SOC limit transition band half-width",       "units": "pu"}]
    # ── flags ─────────────────────────────────────────────────────────────
    d += [{"type": "Parameter", "tex": "Vflag",  "data": "Vflag",  "model": "Vflag",  "default": 1, "description": "0=Q control  1=voltage control",   "units": "-"}]
    d += [{"type": "Parameter", "tex": "Qflag",  "data": "Qflag",  "model": "Qflag",  "default": 1, "description": "0=bypass inner V-PI  1=engage",    "units": "-"}]
    d += [{"type": "Parameter", "tex": "PFflag", "data": "PFflag", "model": "PFflag", "default": 0, "description": "0=constant Q/Vref  1=constant PF", "units": "-"}]
    d += [{"type": "Parameter", "tex": "Pqflag", "data": "Pqflag", "model": "Pqflag", "default": 0, "description": "0=Q priority  1=P priority",       "units": "-"}]
    # ── inputs ────────────────────────────────────────────────────────────
    d += [{"type": "Input", "tex": "P_{ref}",  "data": "Pref",  "model": "Pref",  "default": 0.5,  "description": "Active power reference (+discharge, −charge)", "units": "pu"}]
    d += [{"type": "Input", "tex": "Q_{ext}",  "data": "Qext",  "model": "Qext",  "default": 1.0,  "description": "Reactive / voltage reference",                 "units": "pu"}]
    # ── dynamic states ────────────────────────────────────────────────────
    d += [{"type": "Dynamic State", "tex": "Vt_{flt}", "data": "", "model": "Vt_flt", "default": "", "description": "Filtered terminal voltage",           "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "x_{Pe}",   "data": "", "model": "x_Pe",   "default": "", "description": "Filtered active power",                "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "x_{Kqi}",  "data": "", "model": "x_Kqi",  "default": "", "description": "Q-PI integrator state",                "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "x_{Kvi}",  "data": "", "model": "x_Kvi",  "default": "", "description": "Inner V-PI integrator state",          "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "Vl_{flt}", "data": "", "model": "Vl_flt", "default": "", "description": "Reactive current regulator output lag", "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "P_{flt}",  "data": "", "model": "P_flt",  "default": "", "description": "Active power order lag (signed)",       "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "SOC",      "data": "", "model": "SOC",    "default": "", "description": "State of charge (0=empty, 1=full)",     "units": "pu"}]
    # ── algebraic outputs ─────────────────────────────────────────────────
    d += [{"type": "Algebraic State", "tex": "I_{pcmd}", "data": "", "model": "Ipcmd", "default": "", "description": "Active current command (+discharge, −charge)", "units": "pu"}]
    d += [{"type": "Algebraic State", "tex": "I_{qcmd}", "data": "", "model": "Iqcmd", "default": "", "description": "Reactive current command to REGC_A/B",         "units": "pu"}]
    return d


def reec_e(dae, data, name, bus_name, backend):
    """Add REEC_E BESS electrical controls for converter *name*.

    Must be called **after** :func:`~pydae.bps.weccs.regc_a.regc_a` (or
    :func:`~pydae.bps.weccs.regc_b.regc_b`) so that ``Ipcmd_{name}`` and
    ``Iqcmd_{name}`` already exist in *u_run_dict*.

    New inputs added to *u_run_dict*: ``Pref_{name}`` and ``Qext_{name}``.
    """
    reec = data['reec']

    def p(key, default=0.0):
        return reec.get(key, default)

    Vflag  = int(p('Vflag',  1))
    Qflag  = int(p('Qflag',  1))
    PFflag = int(p('PFflag', 0))
    Pqflag = int(p('Pqflag', 0))

    # ── network & REGC symbols ────────────────────────────────────────────
    Vt = backend.symbols(f"V_{bus_name}")
    Ip = backend.symbols(f"Ip_{name}")   # REGC actual active current (for SOC)
    Iq = backend.symbols(f"Iq_{name}")

    # ── inputs ────────────────────────────────────────────────────────────
    Pref  = backend.symbols(f"Pref_{name}")
    Qext  = backend.symbols(f"Qext_{name}")
    Ipcmd = backend.symbols(f"Ipcmd_{name}")
    Iqcmd = backend.symbols(f"Iqcmd_{name}")

    # ── parameters ────────────────────────────────────────────────────────
    Trv      = backend.symbols(f"Trv_{name}")
    Tp       = backend.symbols(f"Tp_reec_{name}")
    Tiq      = backend.symbols(f"Tiq_{name}")
    Tpord    = backend.symbols(f"Tpord_{name}")
    Kvp      = backend.symbols(f"Kvp_{name}")
    Kvi      = backend.symbols(f"Kvi_{name}")
    Kqp      = backend.symbols(f"Kqp_{name}")
    Kqi      = backend.symbols(f"Kqi_{name}")
    Vref0    = backend.symbols(f"Vref0_{name}")
    dbd1     = backend.symbols(f"dbd1_{name}")
    dbd2     = backend.symbols(f"dbd2_{name}")
    Kqv      = backend.symbols(f"Kqv_{name}")
    Iqhl     = backend.symbols(f"Iqhl_{name}")
    Iqll     = backend.symbols(f"Iqll_{name}")
    Vmax     = backend.symbols(f"Vmax_{name}")
    Vmin     = backend.symbols(f"Vmin_{name}")
    Imax     = backend.symbols(f"Imax_{name}")
    Pmax_dis = backend.symbols(f"Pmax_dis_{name}")
    Pmax_chg = backend.symbols(f"Pmax_chg_{name}")
    dPmax    = backend.symbols(f"dPmax_{name}")
    dPmin    = backend.symbols(f"dPmin_{name}")
    E_cap    = backend.symbols(f"E_cap_{name}")
    SOC_max  = backend.symbols(f"SOC_max_{name}")
    SOC_min  = backend.symbols(f"SOC_min_{name}")
    dSOC_db  = backend.symbols(f"dSOC_db_{name}")

    # ── states ────────────────────────────────────────────────────────────
    Vt_flt = backend.symbols(f"Vt_flt_{name}")
    x_Pe   = backend.symbols(f"x_Pe_{name}")
    x_Kqi  = backend.symbols(f"x_Kqi_{name}")
    x_Kvi  = backend.symbols(f"x_Kvi_{name}")
    Vl_flt = backend.symbols(f"Vl_flt_{name}")
    P_flt  = backend.symbols(f"P_flt_{name}")
    SOC    = backend.symbols(f"SOC_{name}")

    # ── measured power ────────────────────────────────────────────────────
    Pe = Ip * Vt
    Qe = Iq * Vt

    # ── SOC-based power limit modulation ──────────────────────────────────
    # Smooth multiplicative scaling avoids nested Piecewise with compound
    # symbolic bounds (which cause 'false' in SymPy ccode for Jacobians).
    #
    # dis_factor: 0 when SOC ≤ SOC_min, 1 when SOC ≥ SOC_min + dSOC_db
    dis_factor = backend.hard_limits(
        (SOC - SOC_min) / dSOC_db, 0.0, 1.0
    )
    # chg_factor: 0 when SOC ≥ SOC_max, 1 when SOC ≤ SOC_max − dSOC_db
    chg_factor = backend.hard_limits(
        (SOC_max - SOC) / dSOC_db, 0.0, 1.0
    )

    # Scale positive (discharge) and negative (charge) parts of P_flt
    # separately so both factors use fixed scalar bounds in hard_limits:
    P_flt_dis = backend.max(P_flt, 0.0) * dis_factor   # ≥ 0, reduced near SOC_min
    P_flt_chg = backend.min(P_flt, 0.0) * chg_factor   # ≤ 0, reduced near SOC_max
    P_flt_lim = P_flt_dis + P_flt_chg                  # signed active power

    # ── VRT reactive injection ────────────────────────────────────────────
    vrt_err = Vt_flt - Vref0
    db_out  = backend.max(vrt_err - dbd2, 0.0) + backend.min(vrt_err - dbd1, 0.0)
    Iqinj   = backend.hard_limits(Kqv * db_out, Iqll, Iqhl)

    # ── current limit logic ───────────────────────────────────────────────
    EPS = 1e-4
    if Pqflag == 0:   # Q priority
        Iqmax  =  Imax
        Iqmin  = -Imax
        Ipmax  = backend.sqrt(backend.max(Imax**2 - Vl_flt**2, EPS))
    else:             # P priority
        P_flt_pu = P_flt_lim / backend.max(Vt_flt, 0.01)
        Ipmax    = Imax
        Iqmax    =  backend.sqrt(backend.max(Imax**2 - P_flt_pu**2, EPS))
        Iqmin    = -Iqmax

    # Lower current bound allows negative Ipcmd (charging).
    # Use -Imax as the fixed lower bound (avoids nested Piecewise with
    # symbolic bounds that break SymPy ccode Jacobian generation).
    Ipmin = -Imax

    # ── Q / voltage control path ──────────────────────────────────────────
    if PFflag == 0:
        Qref = Qext
    else:
        Pfaref = float(np.arctan(p('Q0', 0.0) / max(float(p('P0', p('Pref', 0.5))), 1e-6)))
        Qref   = x_Pe * float(np.tan(Pfaref))

    if Vflag == 0:
        Vcorr     = backend.hard_limits(Kqp * (Qref - Qe) + x_Kqi, Vmin, Vmax)
        Vref_ctrl = Vcorr
        dx_Kqi    = Kqi * (Qref - Qe)
    else:
        Vref_ctrl = Qref
        dx_Kqi    = x_Kqi * 0.0

    if Qflag == 1:
        verr    = Vt_flt - Vref_ctrl
        Kvi_out = backend.hard_limits(Kvp * verr + x_Kvi, Iqmin, Iqmax)
        dx_Kvi  = Kvi * verr
        dVl_flt = backend.hard_limits((Kvi_out - Vl_flt) / Tiq, dPmin, dPmax)
    else:
        Kvi_out = backend.hard_limits(-Qref / backend.max(Vt_flt, 0.01), Iqmin, Iqmax)
        dx_Kvi  = x_Kvi * 0.0
        dVl_flt = backend.hard_limits((Kvi_out - Vl_flt) / Tiq, dPmin, dPmax)

    # ── active power path ─────────────────────────────────────────────────
    dP_flt = backend.hard_limits((Pref - P_flt) / Tpord, dPmin, dPmax)

    # ── SOC integrator ────────────────────────────────────────────────────
    # dSOC/dt = −Pe / E_cap   (Pe > 0 discharging → SOC decreases)
    dSOC = -Pe / E_cap

    # ── differential equations ────────────────────────────────────────────
    dVt_flt = (Vt - Vt_flt) / Trv
    dx_Pe   = (Pe - x_Pe)   / Tp

    # ── algebraic output constraints ──────────────────────────────────────
    g_Ipcmd = Ipcmd - backend.hard_limits(
        P_flt_lim / backend.max(Vt_flt, 0.01), Ipmin, Ipmax
    )
    g_Iqcmd = Iqcmd - backend.hard_limits(Vl_flt + Iqinj, Iqmin, Iqmax)

    # ── assembly ──────────────────────────────────────────────────────────
    dae['f'] += [dVt_flt, dx_Pe, dx_Kqi, dx_Kvi, dVl_flt, dP_flt, dSOC]
    dae['x'] += [Vt_flt,  x_Pe,  x_Kqi,  x_Kvi,  Vl_flt,  P_flt,  SOC]

    dae['g']     += [g_Ipcmd, g_Iqcmd]
    dae['y_ini'] += [Ipcmd,   Iqcmd]
    dae['y_run'] += [Ipcmd,   Iqcmd]

    # Promote Ipcmd/Iqcmd from REGC inputs to computed algebraic states
    dae['u_ini_dict'].pop(f'Ipcmd_{name}', None)
    dae['u_run_dict'].pop(f'Ipcmd_{name}', None)
    dae['u_ini_dict'].pop(f'Iqcmd_{name}', None)
    dae['u_run_dict'].pop(f'Iqcmd_{name}', None)

    # New inputs (overridden by REPC_A when present)
    Pref_0 = p('Pref', 0.5)
    Qext_0 = p('Qext', p('Vref0', 1.0))
    dae['u_ini_dict'].update({f'Pref_{name}': Pref_0})
    dae['u_run_dict'].update({f'Pref_{name}': Pref_0})
    dae['u_ini_dict'].update({f'Qext_{name}': Qext_0})
    dae['u_run_dict'].update({f'Qext_{name}': Qext_0})

    # ── parameters ────────────────────────────────────────────────────────
    SOC_ini = p('SOC_ini', 0.5)
    dae['params_dict'].update({
        f'Trv_{name}':       p('Trv',     0.02),
        f'Tp_reec_{name}':   p('Tp',      0.02),
        f'Tiq_{name}':       p('Tiq',     0.02),
        f'Tpord_{name}':     p('Tpord',   0.05),
        f'Kvp_{name}':       p('Kvp',     1.0),
        f'Kvi_{name}':       p('Kvi',     40.0),
        f'Kqp_{name}':       p('Kqp',     0.0),
        f'Kqi_{name}':       p('Kqi',     0.0),
        f'Vref0_{name}':     p('Vref0',   1.0),
        f'dbd1_{name}':      p('dbd1',   -0.05),
        f'dbd2_{name}':      p('dbd2',    0.05),
        f'Kqv_{name}':       p('Kqv',     2.0),
        f'Iqhl_{name}':      p('Iqhl',    1.05),
        f'Iqll_{name}':      p('Iqll',   -1.05),
        f'Vmax_{name}':      p('Vmax',    1.1),
        f'Vmin_{name}':      p('Vmin',    0.9),
        f'Imax_{name}':      p('Imax',    1.1),
        f'Pmax_dis_{name}':  p('Pmax_dis', 1.0),
        f'Pmax_chg_{name}':  p('Pmax_chg', -1.0),
        f'dPmax_{name}':     p('dPmax',   999.0),
        f'dPmin_{name}':     p('dPmin',  -999.0),
        f'E_cap_{name}':     p('E_cap',   3600.0),
        f'SOC_max_{name}':   p('SOC_max', 0.9),
        f'SOC_min_{name}':   p('SOC_min', 0.1),
        f'dSOC_db_{name}':   p('dSOC_db', 0.05),
    })

    # ── initialization hints ──────────────────────────────────────────────
    V_0      = data.get('V_ini', 1.0)
    Ip_0     = Pref_0 / max(V_0, 0.01)    # initial discharge current
    Iqcmd_0  = -data.get('Iqcmd', 0.0)    # REGC sign

    dae['xy_0_dict'].update({
        str(Vt_flt): V_0,
        str(x_Pe):   max(Ip_0 * V_0, 0.0),
        str(x_Kqi):  0.0,
        str(x_Kvi):  Iqcmd_0,
        str(Vl_flt): Iqcmd_0,
        str(P_flt):  Pref_0,
        str(SOC):    SOC_ini,
        str(Ipcmd):  Ip_0,
        str(Iqcmd):  Iqcmd_0,
    })

    # Expose SOC, factors and limited power order as outputs
    dae['h_dict'][f"SOC_{name}"]        = SOC
    dae['h_dict'][f"dis_factor_{name}"] = dis_factor
    dae['h_dict'][f"chg_factor_{name}"] = chg_factor
    dae['h_dict'][f"P_flt_lim_{name}"]  = P_flt_lim


# ======================================================================
# In-module test
# ======================================================================
def test():
    import os, time
    from pydae.bps import BpsBuilder
    from pydae.core import Builder, Model

    hjson_path = os.path.join(os.path.dirname(__file__), 'reec_e.hjson')
    grid = BpsBuilder(hjson_path)
    grid.construct('temp_reec_e')
    bld = Builder(grid.sys_dict, target='cffi', sparse=False)
    bld.build()

    model = Model('temp_reec_e')
    t0 = time.perf_counter_ns()
    model.ini({}, 'temp_reec_e_xy_0.json')
    t1 = time.perf_counter_ns()
    print(f"ini time: {(t1-t0)/1e6:.2f} ms")

    Ip        = model.get_value('Ip_1')
    Iq        = model.get_value('Iq_1')
    Ipc       = model.get_value('Ipcmd_1')
    Iqc       = model.get_value('Iqcmd_1')
    SOC       = model.get_value('SOC_1')
    dis_fac   = model.get_value('dis_factor_1')
    print(f"Ip={Ip:.4f}  Iq={Iq:.4f}  Ipcmd={Ipc:.4f}  Iqcmd={Iqc:.4f}")
    print(f"SOC={SOC:.4f}  dis_factor={dis_fac:.4f}")

    assert abs(Ip  - 0.5) < 0.05, f"Ip={Ip:.4f} not close to 0.5 pu"
    assert abs(Iq)         < 0.02, f"Iq={Iq:.4f} should be ~0"
    assert 0.0 < SOC < 1.0,        f"SOC={SOC:.4f} out of range [0,1]"

    # SOC should decrease during discharge
    model.run(10.0, {})
    model.post()
    SOC_fin = model.get_values('SOC_1')[-1]
    assert SOC_fin < SOC, f"SOC did not decrease during discharge: {SOC:.4f} → {SOC_fin:.4f}"
    print(f"SOC after 10 s discharge: {SOC:.4f} → {SOC_fin:.4f}")

    print("test() PASSED")


if __name__ == '__main__':
    test()
