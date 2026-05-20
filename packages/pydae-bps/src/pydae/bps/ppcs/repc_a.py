# -*- coding: utf-8 -*-
r"""
WECC Renewable Energy Plant Control model A (REPC_A).

Reference: WECC Solar Plant Dynamic Modeling Guidelines, April 2014, Section 5.3.

REPC_A is a **plant-level** supervisory controller.  Unlike REEC_B (which is
local to each converter), REPC_A monitors the point-of-interconnection (POI)
bus and can command one or several REGC_A/REEC_B units via *Pref* and *Qext*.

It lives in ``bps/ppcs/`` and is declared as a separate top-level section in
the HJSON (``ppcs``), not nested inside a ``weccs`` entry::

    ppcs: [{
        type:    "repc_a",
        reg_bus: "POI",       // regulated (POI) bus
        weccs:   ["1", "2"],  // list of weccs names (or bus names) it drives
        ...
    }]

The dispatcher ``add_ppcs`` (called by BpsBuilder after ``add_weccs``) pops
``Pref_{name}`` and ``Qext_{name}`` from the *u_run_dict* of every weccs unit
listed in ``weccs`` and replaces them with REPC_A-driven algebraic variables.

**Differential equations**

.. math::

    \dot{V}_{reg,flt}  &= (V_{reg,eff} - V_{reg,flt}) / T_{fltr}  \\
    \dot{Q}_{brn,flt}  &= (Q_{brn}     - Q_{brn,flt}) / T_{fltr}  \\
    \dot{x}_{Kq}       &= K_q \cdot e_{QV,clip} \cdot f_{frz}  \\
    \dot{x}_{Tfv}      &= (Q_{PI,out} - x_{Tfv}) / T_{fv}  \\
    \dot{P}_{brn,flt}  &= (P_{brn}    - P_{brn,flt}) / T_p  \\
    \dot{x}_{Kig}      &= K_{ig} \cdot e_{P,clip}  \\
    \dot{P}_{ref,lag}  &= (P_{PI,out} - P_{ref,lag}) / T_{lag}

**Algebraic outputs** (replace *Qext* and *Pref* inputs of each driven REEC_B)

.. math::

    Q_{ext} &= \mathrm{clip}(x_{Tfv} + T_{ft}/T_{fv}\,(Q_{PI,out}-x_{Tfv}),\;Q_{min},\;Q_{max})  \\
    P_{ref} &= \mathrm{clip}(P_{ref,lag},\;P_{min},\;P_{max})  \quad (F_{req\_flag}=1)
"""
import numpy as np


def descriptions():
    r"""Single source of truth for REPC_A parameters, states, and I/O."""
    d = []
    d += [{"type": "Parameter", "tex": "T_{fltr}",  "data": "Tfltr",  "model": "Tfltr",  "default": 0.02,   "description": "Voltage and Q filter time constant",             "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_p",        "data": "Tp",     "model": "Tp",     "default": 0.02,   "description": "Active power filter time constant",             "units": "s"}]
    d += [{"type": "Parameter", "tex": "K_p",        "data": "Kp",     "model": "Kp",     "default": 0.0,    "description": "Volt/VAR PI proportional gain",                 "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "K_q",        "data": "Kq",     "model": "Kq",     "default": 0.05,   "description": "Volt/VAR PI integral gain",                     "units": "pu/pu·s"}]
    d += [{"type": "Parameter", "tex": "K_c",        "data": "Kc",     "model": "Kc",     "default": 0.0,    "description": "Reactive droop gain (VcompFlag=0)",             "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "dbd",        "data": "dbd",    "model": "dbd",    "default": 0.0,    "description": "Volt/VAR error deadband half-width",            "units": "pu"}]
    d += [{"type": "Parameter", "tex": "e_{max}",    "data": "emax",   "model": "emax",   "default": 0.3,    "description": "Volt/VAR error upper limit",                    "units": "pu"}]
    d += [{"type": "Parameter", "tex": "e_{min}",    "data": "emin",   "model": "emin",   "default": -0.3,   "description": "Volt/VAR error lower limit",                    "units": "pu"}]
    d += [{"type": "Parameter", "tex": "Q_{max}",    "data": "Qmax",   "model": "Qmax",   "default": 0.4,    "description": "Plant Q command upper limit",                   "units": "pu"}]
    d += [{"type": "Parameter", "tex": "Q_{min}",    "data": "Qmin",   "model": "Qmin",   "default": -0.4,   "description": "Plant Q command lower limit",                   "units": "pu"}]
    d += [{"type": "Parameter", "tex": "V_{frz}",    "data": "Vfrz",   "model": "Vfrz",   "default": 0.7,    "description": "Voltage threshold for Volt/VAR integrator freeze","units": "pu"}]
    d += [{"type": "Parameter", "tex": "T_{ft}",     "data": "Tft",    "model": "Tft",    "default": 0.0,    "description": "Q output lead time constant",                   "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_{fv}",     "data": "Tfv",    "model": "Tfv",    "default": 0.15,   "description": "Q output lag time constant",                    "units": "s"}]
    d += [{"type": "Parameter", "tex": "K_{pg}",     "data": "Kpg",    "model": "Kpg",    "default": 0.0,    "description": "P droop PI proportional gain",                  "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "K_{ig}",     "data": "Kig",    "model": "Kig",    "default": 0.05,   "description": "P droop PI integral gain",                      "units": "pu/pu·s"}]
    d += [{"type": "Parameter", "tex": "P_{max}",    "data": "Pmax",   "model": "Pmax",   "default": 1.0,    "description": "Plant P command upper limit",                   "units": "pu"}]
    d += [{"type": "Parameter", "tex": "P_{min}",    "data": "Pmin",   "model": "Pmin",   "default": 0.0,    "description": "Plant P command lower limit",                   "units": "pu"}]
    d += [{"type": "Parameter", "tex": "T_{lag}",    "data": "Tlag",   "model": "Tlag",   "default": 0.15,   "description": "P output lag time constant",                    "units": "s"}]
    d += [{"type": "Parameter", "tex": "f_{dbd1}",   "data": "fdbd1",  "model": "fdbd1",  "default": 0.01,   "description": "Over-frequency governor deadband (pu)",         "units": "pu"}]
    d += [{"type": "Parameter", "tex": "f_{dbd2}",   "data": "fdbd2",  "model": "fdbd2",  "default": -0.01,  "description": "Under-frequency governor deadband (pu)",        "units": "pu"}]
    d += [{"type": "Parameter", "tex": "D_{dn}",     "data": "Ddn",    "model": "Ddn",    "default": 20.0,   "description": "Down-regulation droop gain",                    "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "D_{up}",     "data": "Dup",    "model": "Dup",    "default": 0.0,    "description": "Up-regulation droop gain",                      "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "f_{emax}",   "data": "femax",  "model": "femax",  "default": 0.3,    "description": "P droop error upper limit",                     "units": "pu"}]
    d += [{"type": "Parameter", "tex": "f_{emin}",   "data": "femin",  "model": "femin",  "default": -0.3,   "description": "P droop error lower limit",                     "units": "pu"}]
    d += [{"type": "Flag", "tex": "RefFlag",   "data": "RefFlag",   "model": "RefFlag",   "default": 1, "description": "0=Q control  1=V control",       "units": "-"}]
    d += [{"type": "Flag", "tex": "VcompFlag", "data": "VcompFlag", "model": "VcompFlag", "default": 0, "description": "0=Q droop  1=line drop comp.",   "units": "-"}]
    d += [{"type": "Flag", "tex": "Freq_flag", "data": "Freq_flag", "model": "Freq_flag", "default": 0, "description": "0=no governor  1=governor active","units": "-"}]
    d += [{"type": "Dynamic State", "tex": "V_{rflt}",   "data": "", "model": "Vreg_flt",  "default": "", "description": "Filtered regulated bus voltage",   "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "Q_{bflt}",   "data": "", "model": "Qbrn_flt",  "default": "", "description": "Filtered branch reactive power",    "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "x_{Kq}",     "data": "", "model": "x_Kq",      "default": "", "description": "Volt/VAR PI integrator state",      "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "x_{Tfv}",    "data": "", "model": "x_Tfv",     "default": "", "description": "Q output lead-lag lag state",       "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "P_{bflt}",   "data": "", "model": "Pbrn_flt",  "default": "", "description": "Filtered branch active power",      "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "x_{Kig}",    "data": "", "model": "x_Kig",     "default": "", "description": "P droop PI integrator state",       "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "P_{rlag}",   "data": "", "model": "Pref_lag",  "default": "", "description": "P command output lag state",        "units": "pu"}]
    d += [{"type": "Algebraic State","tex": "Q_{ext}",   "data": "", "model": "Qext",      "default": "", "description": "Reactive/voltage command to REEC_B","units": "pu"}]
    d += [{"type": "Algebraic State","tex": "P_{ref}",   "data": "", "model": "Pref",      "default": "", "description": "Active power command to REEC_B",    "units": "pu"}]
    return d


def repc_a(dae, data, name, backend):
    """Add REPC_A plant controller *name* to the DAE system.

    Parameters
    ----------
    dae : dict
        The BpsBuilder DAE dictionary.
    data : dict
        The HJSON entry for this REPC_A instance (from the ``ppcs`` list).
    name : str
        Unique identifier for this plant controller.
    backend : MathBackend
        Symbolic backend (SymPy or CasADi).

    The function:

    1. Reads ``reg_bus`` from *data* (regulated/POI bus) and ``weccs`` (list
       of converter names or bus names it commands).
    2. Pops ``Pref_{gen}`` and ``Qext_{gen}`` from *u_run_dict* for every
       listed converter and adds shared plant-level algebraic outputs.
    3. Adds REPC_A differential states and algebraic equations.
    """
    def p(key, default=0.0):
        return data.get(key, default)

    RefFlag   = int(p('RefFlag',   1))
    VcompFlag = int(p('VcompFlag', 0))
    Freq_flag = int(p('Freq_flag', 0))

    reg_bus   = str(p('reg_bus', 'POI'))
    Vreg      = backend.symbols(f"V_{reg_bus}")
    omega_coi = backend.symbols("omega_coi")
    freq_dev  = omega_coi - 1.0

    # ── converter names commanded by this plant controller ───────────────
    wecc_names = data.get('weccs', [])

    # ── plant-level output symbols (internal to this controller) ─────────
    # Named after the plant controller instance, e.g. Qext_repc1 / Pref_repc1
    Qext = backend.symbols(f"Qext_{name}")
    Pref = backend.symbols(f"Pref_{name}")
    # Per-converter connector symbols (e.g. Qext_1, Pref_1) collected here
    _conv_Qext_syms = [backend.symbols(f"Qext_{g}") for g in wecc_names]
    _conv_Pref_syms = [backend.symbols(f"Pref_{g}") for g in wecc_names]

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
    Vref_r = backend.symbols(f"Vref_r_{name}")
    Qref_r = backend.symbols(f"Qref_r_{name}")
    Pref_r = backend.symbols(f"Pref_r_{name}")

    # ── states ────────────────────────────────────────────────────────────
    Vreg_flt = backend.symbols(f"Vreg_flt_{name}")
    Qbrn_flt = backend.symbols(f"Qbrn_flt_{name}")
    x_Kq     = backend.symbols(f"x_Kq_{name}")
    x_Tfv    = backend.symbols(f"x_Tfv_{name}")
    Pbrn_flt = backend.symbols(f"Pbrn_flt_{name}")
    x_Kig    = backend.symbols(f"x_Kig_{name}")
    Pref_lag = backend.symbols(f"Pref_lag_{name}")

    # ── branch flows: sum over all commanded converters ───────────────────
    Pbranch = sum(
        backend.symbols(f"Ip_{n}") * backend.symbols(f"V_{reg_bus}") for n in wecc_names
    ) if wecc_names else backend.symbols(f"V_{reg_bus}") * 0.0
    Qbranch = sum(
        backend.symbols(f"Iq_{n}") * backend.symbols(f"V_{reg_bus}") for n in wecc_names
    ) if wecc_names else backend.symbols(f"V_{reg_bus}") * 0.0

    # ── Volt/VAR path ─────────────────────────────────────────────────────
    Vreg_eff = Vreg - Kc * Qbranch   # reactive droop (VcompFlag=0) or approx line drop

    dVreg_flt = (Vreg_eff - Vreg_flt) / Tfltr
    dQbrn_flt = (Qbranch  - Qbrn_flt) / Tfltr

    qv_err = (Vref_r - Vreg_flt) if RefFlag == 1 else (Qref_r - Qbrn_flt)

    qv_db = backend.max(qv_err - dbd, 0.0) + backend.min(qv_err + dbd, 0.0)
    qv_db_clipped = backend.hard_limits(qv_db, emin, emax)

    # Integrator freeze when Vreg < Vfrz (soft gate: 0 frozen, 1 active)
    freeze  = backend.hard_limits(Vreg_flt - Vfrz, 0.0, 1.0)
    dx_Kq   = Kq_qv * qv_db_clipped * freeze

    PI_Q_out = backend.hard_limits(Kp_qv * qv_db_clipped + x_Kq, Qmin, Qmax)

    # Lead-lag on Q output
    dx_Tfv  = (PI_Q_out - x_Tfv) / Tfv
    Qext_raw = PI_Q_out * Tft / Tfv + x_Tfv * (1.0 - Tft / Tfv)
    g_Qext   = Qext - backend.hard_limits(Qext_raw, Qmin, Qmax)

    # ── Active power / frequency droop path ───────────────────────────────
    dPbrn_flt = (Pbranch - Pbrn_flt) / Tp_p

    f_db      = backend.max(freq_dev - fdbd1, 0.0) * Ddn + backend.min(freq_dev - fdbd2, 0.0) * Dup
    P_err     = Pref_r - Pbrn_flt + f_db
    P_err_clipped = backend.hard_limits(P_err, femin, femax)

    dx_Kig   = Kig * P_err_clipped
    PI_P_out = backend.hard_limits(Kpg * P_err_clipped + x_Kig, Pmin_r, Pmax_r)
    dPref_lag = (PI_P_out - Pref_lag) / Tlag

    if Freq_flag == 1:
        g_Pref = Pref - backend.hard_limits(Pref_lag, Pmin_r, Pmax_r)
    else:
        g_Pref = Pref - Pref_r   # constant at initial P setpoint

    # ── assembly ──────────────────────────────────────────────────────────
    dae['f'] += [dVreg_flt, dQbrn_flt, dx_Kq, dx_Tfv, dPbrn_flt, dx_Kig, dPref_lag]
    dae['x'] += [Vreg_flt,  Qbrn_flt,  x_Kq,  x_Tfv,  Pbrn_flt,  x_Kig,  Pref_lag]

    # Plant-level outputs (solved algebraically)
    dae['g']     += [g_Qext, g_Pref]
    dae['y_ini'] += [Qext,   Pref]
    dae['y_run'] += [Qext,   Pref]

    # Per-converter connector equations: Qext_gen_i = Qext_plant (& same for Pref)
    # This promotes the symbols used inside REEC_B's f-equations into y_run
    # so the codegen can resolve them instead of leaving them as free identifiers.
    for gen, Qe_sym, Pr_sym in zip(wecc_names, _conv_Qext_syms, _conv_Pref_syms):
        dae['g']     += [Qe_sym - Qext, Pr_sym - Pref]
        dae['y_ini'] += [Qe_sym,        Pr_sym]
        dae['y_run'] += [Qe_sym,        Pr_sym]
        dae['u_ini_dict'].pop(f'Qext_{gen}', None)
        dae['u_run_dict'].pop(f'Qext_{gen}', None)
        dae['u_ini_dict'].pop(f'Pref_{gen}', None)
        dae['u_run_dict'].pop(f'Pref_{gen}', None)

    # ── parameters ────────────────────────────────────────────────────────
    Pref_0 = float(p('Pref_r', p('Pref', 0.8)))
    V_0    = float(p('Vref',   1.0))
    Q_0    = float(p('Qref',   0.0))
    Qext_0 = float(p('Qext',  V_0))

    dae['params_dict'].update({
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
        f'Vref_r_{name}':    V_0,
        f'Qref_r_{name}':    Q_0,
        f'Pref_r_{name}':    Pref_0,
    })

    # ── initialization hints ──────────────────────────────────────────────
    xy0 = {
        str(Vreg_flt): V_0,
        str(Qbrn_flt): Q_0,
        str(x_Kq):     Qext_0,
        str(x_Tfv):    Qext_0,
        str(Pbrn_flt): Pref_0,
        str(x_Kig):    Pref_0,
        str(Pref_lag): Pref_0,
        str(Qext):     Qext_0,
        str(Pref):     Pref_0,
    }
    for Qe_sym, Pr_sym in zip(_conv_Qext_syms, _conv_Pref_syms):
        xy0[str(Qe_sym)] = Qext_0
        xy0[str(Pr_sym)] = Pref_0
    dae['xy_0_dict'].update(xy0)


# ======================================================================
# In-module test
# ======================================================================
def test():
    import os, time
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

    Ip  = model.get_value('Ip_1')
    Iq  = model.get_value('Iq_1')
    Qe  = model.get_value('Qext_repc1')
    Pe  = model.get_value('Pref_repc1')
    print(f"Ip={Ip:.4f}  Iq={Iq:.4f}  Qext={Qe:.4f}  Pref={Pe:.4f}")

    assert abs(Ip - 0.8) < 0.05, f"Ip mismatch: {Ip}"
    print("test() PASSED")


if __name__ == '__main__':
    test()
