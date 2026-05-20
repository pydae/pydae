# -*- coding: utf-8 -*-
r"""
WECC Renewable Energy Plant Control and Electrical Control models.

Reference: WECC Solar Plant Dynamic Modeling Guidelines, April 2014.

Contains two functions that are layered on top of REGC_A (analogous to avrs/govs on syns):

* :func:`reec_b` — Renewable Energy Electrical Control model B.
  Sits between REPC_A and REGC_A.  Receives *Pref* and *Qext* (or *Vref*)
  commands, computes the active and reactive current commands *Ipcmd*, *Iqcmd*
  that feed REGC_A.

* :func:`repc_a` — Renewable Energy Plant Control model A.
  Optional supervisory layer above REEC_B.  Monitors the regulated bus and
  branch, runs a Volt/VAR PI and a frequency-droop P regulator, and outputs
  *Qext* and *Pref* to REEC_B.

Signal chain (with both layers active)::

    Network  →  Vreg, Qbrn, Pbrn, omega_coi
                    │
                REPC_A  →  Qext (voltage/Q ref), Pref (P ref)
                    │
                REEC_B  →  Ipcmd, Iqcmd
                    │
                REGC_A  →  Ip, Iq  →  network injection

Calling convention (in weccs.py dispatcher)::

    p_W, q_var = regc_a(grid, name, bus_name, item)          # always first
    if 'reec' in item:
        reec_b(grid.dae, item, name, bus_name, grid.backend)  # second
    if 'repc' in item:
        repc_a(grid.dae, item, name, bus_name, grid.backend)  # third

HJSON example::

    weccs: [{
        type: "regc_a",  bus: "1",  S_n: 100e6,  Tg: 0.02,
        // ... all REGC_A params ...
        reec: {
            Trv: 0.02, Tp: 0.02, Tiq: 0.02, Tpord: 0.1,
            Vflag: 1, Qflag: 1, PFflag: 0, Pqflag: 0,
            Kvp: 1.0, Kvi: 40.0, Kqp: 0.0, Kqi: 0.0,
            Vref0: 1.0, dbd1: -0.05, dbd2: 0.05, Kqv: 2.0,
            Iqhl: 1.05, Iqll: -1.05, Vmax: 1.1, Vmin: 0.9,
            Imax: 1.1, Pmax: 1.0, Pmin: 0.0,
            dPmax: 999.0, dPmin: -999.0,
            Pref: 0.8, Qext: 1.0
        },
        repc: {
            RefFlag: 1, VcompFlag: 0, Freq_flag: 0,
            Tfltr: 0.02, Tp: 0.02,
            Kp: 0.0, Kq: 0.05, Kc: 0.0, dbd: 0.0,
            emax: 0.3, emin: -0.3, Qmax: 0.4, Qmin: -0.4,
            Vfrz: 0.7, Tft: 0.0, Tfv: 0.15,
            Kpg: 0.0, Kig: 0.05, Pmax: 1.0, Pmin: 0.0,
            fdbd1: 0.01, fdbd2: -0.01, Ddn: 20.0, Dup: 0.0,
            femax: 0.3, femin: -0.3, Tlag: 0.15,
            reg_bus: "1"
        }
    }]
"""
import numpy as np


# ======================================================================
# REEC_B — Renewable Energy Electrical Control model B
# ======================================================================

def descriptions_reec_b():
    r"""Parameter/state/IO descriptions for REEC_B."""
    d = []
    d += [{"type": "Parameter", "tex": "T_{rv}",    "data": "Trv",    "model": "Trv",    "default": 0.02,   "description": "Terminal voltage filter time constant",             "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_p",        "data": "Tp",     "model": "Tp",     "default": 0.02,   "description": "Active power filter time constant",                "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_{iq}",     "data": "Tiq",    "model": "Tiq",    "default": 0.02,   "description": "Reactive current lag time constant",               "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_{pord}",   "data": "Tpord",  "model": "Tpord",  "default": 0.1,    "description": "Inverter active power order lag time constant",     "units": "s"}]
    d += [{"type": "Parameter", "tex": "K_{vp}",     "data": "Kvp",    "model": "Kvp",    "default": 1.0,    "description": "Voltage PI proportional gain",                     "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "K_{vi}",     "data": "Kvi",    "model": "Kvi",    "default": 40.0,   "description": "Voltage PI integral gain",                         "units": "pu/pu·s"}]
    d += [{"type": "Parameter", "tex": "K_{qp}",     "data": "Kqp",    "model": "Kqp",    "default": 0.0,    "description": "Q PI proportional gain (Vflag=0)",                 "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "K_{qi}",     "data": "Kqi",    "model": "Kqi",    "default": 0.0,    "description": "Q PI integral gain (Vflag=0)",                     "units": "pu/pu·s"}]
    d += [{"type": "Parameter", "tex": "V_{ref0}",   "data": "Vref0",  "model": "Vref0",  "default": 1.0,    "description": "Initial voltage reference",                        "units": "pu"}]
    d += [{"type": "Parameter", "tex": "dbd_1",      "data": "dbd1",   "model": "dbd1",   "default": -0.05,  "description": "VRT deadband lower limit (overvoltage, <0)",        "units": "pu"}]
    d += [{"type": "Parameter", "tex": "dbd_2",      "data": "dbd2",   "model": "dbd2",   "default": 0.05,   "description": "VRT deadband upper limit (undervoltage, >0)",       "units": "pu"}]
    d += [{"type": "Parameter", "tex": "K_{qv}",     "data": "Kqv",    "model": "Kqv",    "default": 2.0,    "description": "VRT reactive current injection gain",               "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "I_{qhl}",    "data": "Iqhl",   "model": "Iqhl",   "default": 1.05,   "description": "VRT reactive current injection maximum",            "units": "pu"}]
    d += [{"type": "Parameter", "tex": "I_{qll}",    "data": "Iqll",   "model": "Iqll",   "default": -1.05,  "description": "VRT reactive current injection minimum (<0)",       "units": "pu"}]
    d += [{"type": "Parameter", "tex": "V_{max}",    "data": "Vmax",   "model": "Vmax",   "default": 1.1,    "description": "Q-PI output upper voltage limit",                  "units": "pu"}]
    d += [{"type": "Parameter", "tex": "V_{min}",    "data": "Vmin",   "model": "Vmin",   "default": 0.9,    "description": "Q-PI output lower voltage limit",                  "units": "pu"}]
    d += [{"type": "Parameter", "tex": "I_{max}",    "data": "Imax",   "model": "Imax",   "default": 1.1,    "description": "Maximum apparent current magnitude",               "units": "pu"}]
    d += [{"type": "Parameter", "tex": "P_{max}",    "data": "Pmax",   "model": "Pmax",   "default": 1.0,    "description": "Maximum active power",                             "units": "pu"}]
    d += [{"type": "Parameter", "tex": "P_{min}",    "data": "Pmin",   "model": "Pmin",   "default": 0.0,    "description": "Minimum active power",                             "units": "pu"}]
    d += [{"type": "Parameter", "tex": "dP_{max}",   "data": "dPmax",  "model": "dPmax",  "default": 999.0,  "description": "Active power up-ramp rate limit",                  "units": "pu/s"}]
    d += [{"type": "Parameter", "tex": "dP_{min}",   "data": "dPmin",  "model": "dPmin",  "default": -999.0, "description": "Active power down-ramp rate limit",                "units": "pu/s"}]
    # flags
    d += [{"type": "Parameter", "tex": "Vflag",      "data": "Vflag",  "model": "Vflag",  "default": 1,      "description": "0=Q control  1=voltage control",                  "units": "-"}]
    d += [{"type": "Parameter", "tex": "Qflag",      "data": "Qflag",  "model": "Qflag",  "default": 1,      "description": "0=bypass inner V-PI  1=engage",                   "units": "-"}]
    d += [{"type": "Parameter", "tex": "PFflag",     "data": "PFflag", "model": "PFflag", "default": 0,      "description": "0=constant Q/Vref  1=constant PF",                "units": "-"}]
    d += [{"type": "Parameter", "tex": "Pqflag",     "data": "Pqflag", "model": "Pqflag", "default": 0,      "description": "0=Q priority  1=P priority current limits",        "units": "-"}]
    # inputs
    d += [{"type": "Input", "tex": "P_{ref}",   "data": "Pref",  "model": "Pref",  "default": 0.8, "description": "Active power reference (from REPC_A or external)", "units": "pu"}]
    d += [{"type": "Input", "tex": "Q_{ext}",   "data": "Qext",  "model": "Qext",  "default": 1.0, "description": "Reactive/voltage reference (from REPC_A or external)", "units": "pu"}]
    # states
    d += [{"type": "Dynamic State", "tex": "Vt_{flt}", "data": "", "model": "Vt_flt", "default": "", "description": "Filtered terminal voltage",          "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "x_{Pe}",   "data": "", "model": "x_Pe",   "default": "", "description": "Filtered active power",               "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "x_{Kqi}",  "data": "", "model": "x_Kqi",  "default": "", "description": "Q-PI integrator state",               "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "x_{Kvi}",  "data": "", "model": "x_Kvi",  "default": "", "description": "Inner voltage PI integrator state",   "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "Vl_{flt}", "data": "", "model": "Vl_flt", "default": "", "description": "Reactive current regulator output lag","units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "P_{flt}",  "data": "", "model": "P_flt",  "default": "", "description": "Active power order (Tpord lag)",       "units": "pu"}]
    # algebraic outputs
    d += [{"type": "Algebraic State", "tex": "I_{pcmd}", "data": "", "model": "Ipcmd", "default": "", "description": "Active current command to REGC_A",   "units": "pu"}]
    d += [{"type": "Algebraic State", "tex": "I_{qcmd}", "data": "", "model": "Iqcmd", "default": "", "description": "Reactive current command to REGC_A", "units": "pu"}]
    return d


def reec_b(dae, data, name, bus_name, backend):
    """Add REEC_B electrical controls for the converter *name*.

    Must be called **after** :func:`~pydae.bps.weccs.regc_a.regc_a` so that
    the ``Ipcmd_{name}`` and ``Iqcmd_{name}`` symbols already exist in
    *u_run_dict*.  This function pops them, promotes them to algebraic states,
    and drives them from the REEC_B control equations.

    New inputs added to *u_run_dict* (can be overridden by :func:`repc_a`):
    ``Pref_{name}`` and ``Qext_{name}``.
    """
    reec = data['reec']

    def p(key, default=0.0):
        return reec.get(key, default)

    # ── flags (handled as Python booleans → clean expression tree) ──────
    Vflag  = int(p('Vflag',  1))
    Qflag  = int(p('Qflag',  1))
    PFflag = int(p('PFflag', 0))
    Pqflag = int(p('Pqflag', 0))

    # ── network & REGC_A symbols ─────────────────────────────────────────
    Vt = backend.symbols(f"V_{bus_name}")
    Ip = backend.symbols(f"Ip_{name}")
    Iq = backend.symbols(f"Iq_{name}")

    # ── inputs from higher layer ─────────────────────────────────────────
    Pref = backend.symbols(f"Pref_{name}")
    Qext = backend.symbols(f"Qext_{name}")

    # ── REEC_B algebraic outputs (were u_run inputs from REGC_A) ─────────
    Ipcmd = backend.symbols(f"Ipcmd_{name}")
    Iqcmd = backend.symbols(f"Iqcmd_{name}")

    # ── parameters ───────────────────────────────────────────────────────
    Trv   = backend.symbols(f"Trv_{name}")
    Tp    = backend.symbols(f"Tp_reec_{name}")
    Tiq   = backend.symbols(f"Tiq_{name}")
    Tpord = backend.symbols(f"Tpord_{name}")
    Kvp   = backend.symbols(f"Kvp_{name}")
    Kvi   = backend.symbols(f"Kvi_{name}")
    Kqp   = backend.symbols(f"Kqp_{name}")
    Kqi   = backend.symbols(f"Kqi_{name}")
    Vref0 = backend.symbols(f"Vref0_{name}")
    dbd1  = backend.symbols(f"dbd1_{name}")
    dbd2  = backend.symbols(f"dbd2_{name}")
    Kqv   = backend.symbols(f"Kqv_{name}")
    Iqhl  = backend.symbols(f"Iqhl_{name}")
    Iqll  = backend.symbols(f"Iqll_{name}")
    Vmax  = backend.symbols(f"Vmax_{name}")
    Vmin  = backend.symbols(f"Vmin_{name}")
    Imax  = backend.symbols(f"Imax_{name}")
    Pmax  = backend.symbols(f"Pmax_reec_{name}")
    Pmin  = backend.symbols(f"Pmin_reec_{name}")
    dPmax = backend.symbols(f"dPmax_{name}")
    dPmin = backend.symbols(f"dPmin_{name}")

    # ── states ───────────────────────────────────────────────────────────
    Vt_flt = backend.symbols(f"Vt_flt_{name}")
    x_Pe   = backend.symbols(f"x_Pe_{name}")
    x_Kqi  = backend.symbols(f"x_Kqi_{name}")
    x_Kvi  = backend.symbols(f"x_Kvi_{name}")
    Vl_flt = backend.symbols(f"Vl_flt_{name}")
    P_flt  = backend.symbols(f"P_flt_{name}")

    # ── measured power ────────────────────────────────────────────────────
    Pe = Ip * Vt   # active power (pu on S_n)
    Qe = Iq * Vt   # reactive power (negative sign in REGC_A convention)

    # ── VRT reactive injection (supplementary during voltage dips) ────────
    # error = Vt_flt - Vref0:  >0 overvoltage, <0 undervoltage
    vrt_err = Vt_flt - Vref0
    db_out  = backend.max(vrt_err - dbd2, 0.0) + backend.min(vrt_err - dbd1, 0.0)
    Iqinj   = backend.hard_limits(Kqv * db_out, Iqll, Iqhl)

    # ── current limit logic ───────────────────────────────────────────────
    EPS = 1e-4
    if Pqflag == 0:   # Q priority
        Iqmax =  Imax
        Iqmin = -Imax
        Ipmax = backend.sqrt(backend.max(Imax**2 - Vl_flt**2, EPS))
    else:             # P priority
        P_flt_pu = P_flt / backend.max(Vt_flt, 0.01)
        Ipmax = Imax
        Iqmax =  backend.sqrt(backend.max(Imax**2 - P_flt_pu**2, EPS))
        Iqmin = -Iqmax

    # ── Q / voltage control path ──────────────────────────────────────────
    if PFflag == 0:
        Qref = Qext          # external Q reference (or Vref when Vflag=1)
    else:
        Pfaref = float(np.arctan(p('Q0', 0.0) / max(p('P0', p('Pref', 0.8)), 1e-6)))
        Qref = x_Pe * float(np.tan(Pfaref))

    if Vflag == 0:
        # Q-PI computes a voltage reference correction
        Vcorr     = backend.hard_limits(Kqp * (Qref - Qe) + x_Kqi, Vmin, Vmax)
        Vref_ctrl = Vcorr
        dx_Kqi    = Kqi * (Qref - Qe)
    else:
        # Qext (from REPC_A) is used directly as voltage reference
        Vref_ctrl = Qref
        dx_Kqi    = backend.symbols(f"x_Kqi_{name}") * 0.0  # frozen (not used)

    # Inner voltage PI
    # error = Vt_flt - Vref_ctrl (positive error → over-voltage → less injection)
    verr = Vt_flt - Vref_ctrl
    if Qflag == 1:
        Kvi_out    = backend.hard_limits(Kvp * verr + x_Kvi, Iqmin, Iqmax)
        dx_Kvi     = Kvi * verr
        dVl_flt    = backend.hard_limits((Kvi_out - Vl_flt) / Tiq, dPmin, dPmax)
    else:
        # bypass inner PI — Iqcmd computed directly from Q reference
        Kvi_out    = backend.hard_limits(-Qref / backend.max(Vt_flt, 0.01), Iqmin, Iqmax)
        dx_Kvi     = Kvi * verr * 0.0   # frozen
        dVl_flt    = backend.hard_limits((Kvi_out - Vl_flt) / Tiq, dPmin, dPmax)

    # ── active power path ─────────────────────────────────────────────────
    dP_flt = backend.hard_limits((Pref - P_flt) / Tpord, dPmin, dPmax)

    Ipcmd_raw = backend.hard_limits(P_flt / backend.max(Vt_flt, 0.01), Pmin, Ipmax)

    # ── algebraic constraint equations ────────────────────────────────────
    # Iqcmd: steady state of reactive lag + VRT injection
    g_Iqcmd = Iqcmd - backend.hard_limits(Vl_flt + Iqinj, Iqmin, Iqmax)
    # Ipcmd: active current from P order, voltage-divided
    g_Ipcmd = Ipcmd - Ipcmd_raw

    # ── differential equations ─────────────────────────────────────────────
    dVt_flt = (Vt - Vt_flt) / Trv
    dx_Pe   = (Pe - x_Pe)   / Tp

    # ── assembly ──────────────────────────────────────────────────────────
    dae['f'] += [dVt_flt, dx_Pe, dx_Kqi, dx_Kvi, dVl_flt, dP_flt]
    dae['x'] += [Vt_flt,  x_Pe,  x_Kqi,  x_Kvi,  Vl_flt,  P_flt]

    dae['g'] += [g_Ipcmd, g_Iqcmd]
    dae['y_ini'] += [Ipcmd, Iqcmd]
    dae['y_run'] += [Ipcmd, Iqcmd]

    # ── promote Ipcmd/Iqcmd: were inputs, now computed ────────────────────
    dae['u_ini_dict'].pop(f'Ipcmd_{name}', None)
    dae['u_run_dict'].pop(f'Ipcmd_{name}', None)
    dae['u_ini_dict'].pop(f'Iqcmd_{name}', None)
    dae['u_run_dict'].pop(f'Iqcmd_{name}', None)

    # ── new inputs (to be overridden by REPC_A if present) ───────────────
    Pref_0 = p('Pref', 0.8)
    Qext_0 = p('Qext', p('Vref0', 1.0))
    dae['u_ini_dict'].update({f'Pref_{name}': Pref_0})
    dae['u_run_dict'].update({f'Pref_{name}': Pref_0})
    dae['u_ini_dict'].update({f'Qext_{name}': Qext_0})
    dae['u_run_dict'].update({f'Qext_{name}': Qext_0})

    # ── parameters ────────────────────────────────────────────────────────
    _params = {
        f'Trv_{name}':        p('Trv',    0.02),
        f'Tp_reec_{name}':    p('Tp',     0.02),
        f'Tiq_{name}':        p('Tiq',    0.02),
        f'Tpord_{name}':      p('Tpord',  0.1),
        f'Kvp_{name}':        p('Kvp',    1.0),
        f'Kvi_{name}':        p('Kvi',    40.0),
        f'Kqp_{name}':        p('Kqp',    0.0),
        f'Kqi_{name}':        p('Kqi',    0.0),
        f'Vref0_{name}':      p('Vref0',  1.0),
        f'dbd1_{name}':       p('dbd1',  -0.05),
        f'dbd2_{name}':       p('dbd2',   0.05),
        f'Kqv_{name}':        p('Kqv',    2.0),
        f'Iqhl_{name}':       p('Iqhl',   1.05),
        f'Iqll_{name}':       p('Iqll',  -1.05),
        f'Vmax_{name}':       p('Vmax',   1.1),
        f'Vmin_{name}':       p('Vmin',   0.9),
        f'Imax_{name}':       p('Imax',   1.1),
        f'Pmax_reec_{name}':  p('Pmax',   1.0),
        f'Pmin_reec_{name}':  p('Pmin',   0.0),
        f'dPmax_{name}':      p('dPmax',  999.0),
        f'dPmin_{name}':      p('dPmin', -999.0),
    }
    dae['params_dict'].update(_params)

    # ── initialization hints ──────────────────────────────────────────────
    V_0    = data.get('V_ini', 1.0)
    Ip_0   = data.get('Ipcmd', Pref_0)
    Iq_0   = -data.get('Iqcmd', 0.0)   # REGC_A sign: Iq_0 = -Iqcmd_0
    Pe_0   = Ip_0 * V_0
    # At steady state: Vl_flt = Iqcmd, P_flt = Pref
    Iqcmd_0 = -Iq_0   # REGC_A convention
    dae['xy_0_dict'].update({
        str(Vt_flt): V_0,
        str(x_Pe):   Pe_0,
        str(x_Kqi):  0.0,
        str(x_Kvi):  Iqcmd_0,
        str(Vl_flt): Iqcmd_0,
        str(P_flt):  Pref_0,
        str(Ipcmd):  Ip_0,
        str(Iqcmd):  Iqcmd_0,
    })


# ======================================================================
# REPC_A — Renewable Energy Plant Control model A
# ======================================================================

def descriptions_repc_a():
    r"""Parameter/state/IO descriptions for REPC_A."""
    d = []
    d += [{"type": "Parameter", "tex": "T_{fltr}", "data": "Tfltr", "model": "Tfltr", "default": 0.02,  "description": "Voltage and Q filter time constant",            "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_p",       "data": "Tp",    "model": "Tp",    "default": 0.02,  "description": "Active power filter time constant",             "units": "s"}]
    d += [{"type": "Parameter", "tex": "K_p",       "data": "Kp",    "model": "Kp",    "default": 0.0,   "description": "Volt/VAR PI proportional gain",                 "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "K_q",       "data": "Kq",    "model": "Kq",    "default": 0.05,  "description": "Volt/VAR PI integral gain",                     "units": "pu/pu·s"}]
    d += [{"type": "Parameter", "tex": "K_c",       "data": "Kc",    "model": "Kc",    "default": 0.0,   "description": "Reactive droop gain (VcompFlag=0)",             "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "dbd",       "data": "dbd",   "model": "dbd",   "default": 0.0,   "description": "Volt/VAR error deadband half-width",            "units": "pu"}]
    d += [{"type": "Parameter", "tex": "e_{max}",   "data": "emax",  "model": "emax",  "default": 0.3,   "description": "Volt/VAR error upper limit",                    "units": "pu"}]
    d += [{"type": "Parameter", "tex": "e_{min}",   "data": "emin",  "model": "emin",  "default": -0.3,  "description": "Volt/VAR error lower limit",                    "units": "pu"}]
    d += [{"type": "Parameter", "tex": "Q_{max}",   "data": "Qmax",  "model": "Qmax",  "default": 0.4,   "description": "Plant Q command upper limit",                   "units": "pu"}]
    d += [{"type": "Parameter", "tex": "Q_{min}",   "data": "Qmin",  "model": "Qmin",  "default": -0.4,  "description": "Plant Q command lower limit",                   "units": "pu"}]
    d += [{"type": "Parameter", "tex": "V_{frz}",   "data": "Vfrz",  "model": "Vfrz",  "default": 0.7,   "description": "Voltage threshold for Volt/VAR integrator freeze","units": "pu"}]
    d += [{"type": "Parameter", "tex": "T_{ft}",    "data": "Tft",   "model": "Tft",   "default": 0.0,   "description": "Q output lead time constant",                   "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_{fv}",    "data": "Tfv",   "model": "Tfv",   "default": 0.15,  "description": "Q output lag time constant",                    "units": "s"}]
    d += [{"type": "Parameter", "tex": "K_{pg}",    "data": "Kpg",   "model": "Kpg",   "default": 0.0,   "description": "P droop PI proportional gain",                  "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "K_{ig}",    "data": "Kig",   "model": "Kig",   "default": 0.05,  "description": "P droop PI integral gain",                      "units": "pu/pu·s"}]
    d += [{"type": "Parameter", "tex": "P_{max}",   "data": "Pmax",  "model": "Pmax",  "default": 1.0,   "description": "Plant P command upper limit",                   "units": "pu"}]
    d += [{"type": "Parameter", "tex": "P_{min}",   "data": "Pmin",  "model": "Pmin",  "default": 0.0,   "description": "Plant P command lower limit",                   "units": "pu"}]
    d += [{"type": "Parameter", "tex": "T_{lag}",   "data": "Tlag",  "model": "Tlag",  "default": 0.15,  "description": "P output lag time constant",                    "units": "s"}]
    d += [{"type": "Parameter", "tex": "f_{dbd1}",  "data": "fdbd1", "model": "fdbd1", "default": 0.01,  "description": "Over-frequency deadband for governor (pu)",     "units": "pu"}]
    d += [{"type": "Parameter", "tex": "f_{dbd2}",  "data": "fdbd2", "model": "fdbd2", "default": -0.01, "description": "Under-frequency deadband for governor (pu)",    "units": "pu"}]
    d += [{"type": "Parameter", "tex": "D_{dn}",    "data": "Ddn",   "model": "Ddn",   "default": 20.0,  "description": "Down-regulation droop gain",                    "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "D_{up}",    "data": "Dup",   "model": "Dup",   "default": 0.0,   "description": "Up-regulation droop gain",                      "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "f_{emax}",  "data": "femax", "model": "femax", "default": 0.3,   "description": "P droop error upper limit",                     "units": "pu"}]
    d += [{"type": "Parameter", "tex": "f_{emin}",  "data": "femin", "model": "femin", "default": -0.3,  "description": "P droop error lower limit",                     "units": "pu"}]
    d += [{"type": "Flag",      "tex": "RefFlag",   "data": "RefFlag",   "model": "RefFlag",   "default": 1, "description": "0=Q control  1=V control",       "units": "-"}]
    d += [{"type": "Flag",      "tex": "VcompFlag", "data": "VcompFlag", "model": "VcompFlag", "default": 0, "description": "0=Q droop  1=line drop comp.",   "units": "-"}]
    d += [{"type": "Flag",      "tex": "Freq_flag", "data": "Freq_flag", "model": "Freq_flag", "default": 0, "description": "0=no governor  1=governor active","units": "-"}]
    # states
    d += [{"type": "Dynamic State", "tex": "V_{rflt}",   "data": "", "model": "Vreg_flt",  "default": "", "description": "Filtered regulated bus voltage",    "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "Q_{bflt}",   "data": "", "model": "Qbrn_flt",  "default": "", "description": "Filtered branch reactive power",     "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "x_{Kq}",     "data": "", "model": "x_Kq",      "default": "", "description": "Volt/VAR PI integrator state",       "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "x_{Tfv}",    "data": "", "model": "x_Tfv",     "default": "", "description": "Q output lead-lag lag state",        "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "P_{bflt}",   "data": "", "model": "Pbrn_flt",  "default": "", "description": "Filtered branch active power",       "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "x_{Kig}",    "data": "", "model": "x_Kig",     "default": "", "description": "P droop PI integrator state",        "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "P_{ref,lag}", "data": "", "model": "Pref_lag",  "default": "", "description": "P command output lag state",         "units": "pu"}]
    # algebraic outputs
    d += [{"type": "Algebraic State", "tex": "Q_{ext}",  "data": "", "model": "Qext",  "default": "", "description": "Reactive/voltage command to REEC_B", "units": "pu"}]
    d += [{"type": "Algebraic State", "tex": "P_{ref}",  "data": "", "model": "Pref",  "default": "", "description": "Active power command to REEC_B",     "units": "pu"}]
    return d


def repc_a(dae, data, name, bus_name, backend):
    """Add REPC_A plant controller for the converter *name*.

    Must be called **after** :func:`reec_b`.  Pops ``Pref_{name}`` and
    ``Qext_{name}`` from *u_run_dict* and replaces them with plant-level
    controlled algebraic variables.
    """
    repc = data['repc']

    def p(key, default=0.0):
        return repc.get(key, default)

    # ── flags (Python values → clean expression tree) ────────────────────
    RefFlag   = int(p('RefFlag',   1))
    VcompFlag = int(p('VcompFlag', 0))
    Freq_flag = int(p('Freq_flag', 0))

    # ── regulated bus (default = terminal bus) ────────────────────────────
    reg_bus = str(p('reg_bus', bus_name))
    Vreg = backend.symbols(f"V_{reg_bus}")

    # ── Ip/Iq for branch flow approximation ───────────────────────────────
    Ip = backend.symbols(f"Ip_{name}")
    Iq = backend.symbols(f"Iq_{name}")
    Vt = backend.symbols(f"V_{bus_name}")
    Pbranch = Ip * Vt    # approximated from terminal power
    Qbranch = Iq * Vt

    # ── frequency reference ───────────────────────────────────────────────
    omega_coi = backend.symbols("omega_coi")
    freq_dev  = omega_coi - 1.0   # per-unit frequency deviation

    # ── REPC_A algebraic outputs (were u_run inputs from REEC_B) ─────────
    Qext = backend.symbols(f"Qext_{name}")
    Pref = backend.symbols(f"Pref_{name}")

    # ── parameters ────────────────────────────────────────────────────────
    Tfltr  = backend.symbols(f"Tfltr_{name}")
    Tp_p   = backend.symbols(f"Tp_repc_{name}")
    Kp_qv  = backend.symbols(f"Kp_qv_{name}")
    Kq_qv  = backend.symbols(f"Kq_qv_{name}")
    Kc     = backend.symbols(f"Kc_{name}")
    dbd    = backend.symbols(f"dbd_{name}")
    emax   = backend.symbols(f"emax_{name}")
    emin   = backend.symbols(f"emin_{name}")
    Qmax   = backend.symbols(f"Qmax_{name}")
    Qmin   = backend.symbols(f"Qmin_{name}")
    Vfrz   = backend.symbols(f"Vfrz_{name}")
    Tft    = backend.symbols(f"Tft_{name}")
    Tfv    = backend.symbols(f"Tfv_{name}")
    Kpg    = backend.symbols(f"Kpg_{name}")
    Kig    = backend.symbols(f"Kig_{name}")
    Pmax_r = backend.symbols(f"Pmax_repc_{name}")
    Pmin_r = backend.symbols(f"Pmin_repc_{name}")
    Tlag   = backend.symbols(f"Tlag_{name}")
    fdbd1  = backend.symbols(f"fdbd1_{name}")
    fdbd2  = backend.symbols(f"fdbd2_{name}")
    Ddn    = backend.symbols(f"Ddn_{name}")
    Dup    = backend.symbols(f"Dup_{name}")
    femax  = backend.symbols(f"femax_{name}")
    femin  = backend.symbols(f"femin_{name}")
    Vref_r = backend.symbols(f"Vref_r_{name}")   # initial regulated bus V (from p/f)
    Qref_r = backend.symbols(f"Qref_r_{name}")   # initial branch Q (from p/f)
    Pref_r = backend.symbols(f"Pref_r_{name}")   # initial branch P (from p/f)

    # ── states ────────────────────────────────────────────────────────────
    Vreg_flt = backend.symbols(f"Vreg_flt_{name}")
    Qbrn_flt = backend.symbols(f"Qbrn_flt_{name}")
    x_Kq     = backend.symbols(f"x_Kq_{name}")
    x_Tfv    = backend.symbols(f"x_Tfv_{name}")
    Pbrn_flt = backend.symbols(f"Pbrn_flt_{name}")
    x_Kig    = backend.symbols(f"x_Kig_{name}")
    Pref_lag = backend.symbols(f"Pref_lag_{name}")

    # ── Volt/VAR path ─────────────────────────────────────────────────────
    if VcompFlag == 0:
        # Reactive droop: effective V = Vreg - Kc*Qbranch
        Vreg_eff = Vreg - Kc * Qbranch
    else:
        # Line drop compensation (simplified: ignores angle, uses |Vreg - Kc*Qbranch|)
        Vreg_eff = Vreg - Kc * Qbranch   # simplified (full form needs Ibranch angle)

    dVreg_flt = (Vreg_eff - Vreg_flt) / Tfltr
    dQbrn_flt = (Qbranch  - Qbrn_flt) / Tfltr

    if RefFlag == 1:
        qv_err = Vref_r - Vreg_flt
    else:
        qv_err = Qref_r - Qbrn_flt

    # deadband: signals in [-dbd, +dbd] → 0
    qv_db = backend.max(qv_err - dbd, 0.0) + backend.min(qv_err + dbd, 0.0)
    qv_db_clipped = backend.hard_limits(qv_db, emin, emax)

    # PI — integrator frozen (soft) when Vreg < Vfrz using hard_limits
    freeze = backend.hard_limits(Vreg_flt - Vfrz, 0.0, 1.0)  # 0 when frozen, 1 when active
    dx_Kq = Kq_qv * qv_db_clipped * freeze

    PI_Q_out = backend.hard_limits(Kp_qv * qv_db_clipped + x_Kq, Qmin, Qmax)

    # Lead-lag on Q output: Qext = PI_Q_out*(Tft/Tfv) + x_Tfv*(1 - Tft/Tfv)
    dx_Tfv = (PI_Q_out - x_Tfv) / Tfv
    Qext_raw = PI_Q_out * Tft / Tfv + x_Tfv * (1.0 - Tft / Tfv)
    g_Qext = Qext - backend.hard_limits(Qext_raw, Qmin, Qmax)

    # ── Active power / frequency droop path ───────────────────────────────
    dPbrn_flt = (Pbranch - Pbrn_flt) / Tp_p

    # Frequency deadband droop
    f_err = freq_dev
    f_db  = backend.max(f_err - fdbd1, 0.0) * Ddn + backend.min(f_err - fdbd2, 0.0) * Dup
    P_err = Pref_r - Pbrn_flt + f_db
    P_err_clipped = backend.hard_limits(P_err, femin, femax)

    dx_Kig = Kig * P_err_clipped
    PI_P_out = backend.hard_limits(Kpg * P_err_clipped + x_Kig, Pmin_r, Pmax_r)

    dPref_lag = (PI_P_out - Pref_lag) / Tlag

    if Freq_flag == 1:
        g_Pref = Pref - backend.hard_limits(Pref_lag, Pmin_r, Pmax_r)
    else:
        g_Pref = Pref - Pref_r   # constant, tracks initial P

    # ── assembly ──────────────────────────────────────────────────────────
    dae['f'] += [dVreg_flt, dQbrn_flt, dx_Kq, dx_Tfv, dPbrn_flt, dx_Kig, dPref_lag]
    dae['x'] += [Vreg_flt,  Qbrn_flt,  x_Kq,  x_Tfv,  Pbrn_flt,  x_Kig,  Pref_lag]

    dae['g'] += [g_Qext, g_Pref]
    dae['y_ini'] += [Qext, Pref]
    dae['y_run'] += [Qext, Pref]

    # ── promote Pref/Qext: were u_run inputs from REEC_B, now computed ────
    dae['u_ini_dict'].pop(f'Pref_{name}', None)
    dae['u_run_dict'].pop(f'Pref_{name}', None)
    dae['u_ini_dict'].pop(f'Qext_{name}', None)
    dae['u_run_dict'].pop(f'Qext_{name}', None)

    # ── parameters ────────────────────────────────────────────────────────
    Pref_0 = data.get('reec', {}).get('Pref', 0.8)
    V_0    = data.get('V_ini', 1.0)
    Q_0    = -data.get('Iqcmd', 0.0) * V_0  # approx reactive power

    _params = {
        f'Tfltr_{name}':     p('Tfltr',  0.02),
        f'Tp_repc_{name}':   p('Tp',     0.02),
        f'Kp_qv_{name}':     p('Kp',     0.0),
        f'Kq_qv_{name}':     p('Kq',     0.05),
        f'Kc_{name}':        p('Kc',     0.0),
        f'dbd_{name}':       p('dbd',    0.0),
        f'emax_{name}':      p('emax',   0.3),
        f'emin_{name}':      p('emin',  -0.3),
        f'Qmax_{name}':      p('Qmax',   0.4),
        f'Qmin_{name}':      p('Qmin',  -0.4),
        f'Vfrz_{name}':      p('Vfrz',   0.7),
        f'Tft_{name}':       p('Tft',    0.0),
        f'Tfv_{name}':       p('Tfv',    0.15),
        f'Kpg_{name}':       p('Kpg',    0.0),
        f'Kig_{name}':       p('Kig',    0.05),
        f'Pmax_repc_{name}': p('Pmax',   1.0),
        f'Pmin_repc_{name}': p('Pmin',   0.0),
        f'Tlag_{name}':      p('Tlag',   0.15),
        f'fdbd1_{name}':     p('fdbd1',  0.01),
        f'fdbd2_{name}':     p('fdbd2', -0.01),
        f'Ddn_{name}':       p('Ddn',   20.0),
        f'Dup_{name}':       p('Dup',    0.0),
        f'femax_{name}':     p('femax',  0.3),
        f'femin_{name}':     p('femin', -0.3),
        # reference setpoints initialized from power flow
        f'Vref_r_{name}':    p('Vref', V_0),
        f'Qref_r_{name}':    p('Qref', Q_0),
        f'Pref_r_{name}':    p('Pref_r', Pref_0),
    }
    dae['params_dict'].update(_params)

    # ── initialization hints ──────────────────────────────────────────────
    Qext_0 = data.get('reec', {}).get('Qext', data.get('reec', {}).get('Vref0', 1.0))
    dae['xy_0_dict'].update({
        str(Vreg_flt): V_0,
        str(Qbrn_flt): Q_0,
        str(x_Kq):     Qext_0,
        str(x_Tfv):    Qext_0,
        str(Pbrn_flt): Pref_0,
        str(x_Kig):    Pref_0,
        str(Pref_lag): Pref_0,
        str(Qext):     Qext_0,
        str(Pref):     Pref_0,
    })


# ======================================================================
# In-module test
# ======================================================================
def test():
    import os
    import time
    from pydae.bps import BpsBuilder
    from pydae.core import Builder, Model

    hjson_path = os.path.join(os.path.dirname(__file__), 'repc_a.hjson')
    grid = BpsBuilder(hjson_path)
    grid.construct('temp_repc_a')
    bld = Builder(grid.sys_dict, target='cffi', sparse=False)
    bld.build()

    model = Model('temp_repc_a')
    t0 = time.perf_counter_ns()
    model.ini({}, 'temp_repc_a_xy_0.json')
    t1 = time.perf_counter_ns()
    print(f"ini time: {(t1-t0)/1e6:.2f} ms")

    Vt  = model.get_value('V_1')
    Ip  = model.get_value('Ip_1')
    Iq  = model.get_value('Iq_1')
    Ipc = model.get_value('Ipcmd_1')
    Iqc = model.get_value('Iqcmd_1')
    p_g = model.get_value('p_g_1')
    print(f"V={Vt:.4f}  Ip={Ip:.4f}  Iq={Iq:.4f}  Ipcmd={Ipc:.4f}  Iqcmd={Iqc:.4f}  p_g={p_g:.4f}")

    assert abs(Ip - 0.8) < 0.05, f"Ip mismatch: {Ip}"
    print("test() PASSED")


if __name__ == '__main__':
    test()
