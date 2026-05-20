# -*- coding: utf-8 -*-
r"""
WECC Renewable Energy Electrical Control model B (REEC_B).

Reference: WECC Solar Plant Dynamic Modeling Guidelines, April 2014, Section 5.2.

REEC_B is the **local electrical control** layer of the WECC renewable model
stack.  It sits between the plant controller (REPC_A, optional) and the
generator/converter interface (REGC_A), and is wired into the HJSON entry of
the parent ``weccs`` generator via the ``reec`` sub-dict — exactly as an AVR
is nested inside a ``syns`` entry.

Signal chain::

    REPC_A  (plant level, ppcs/)
        │ Pref, Qext
    REEC_B  (local control, weccs/)   ← this file
        │ Ipcmd, Iqcmd
    REGC_A  (generator/converter, weccs/)
        │ Ip, Iq
    Network

**Differential equations**

.. math::

    \dot{V}_{t,flt} &= (V_t - V_{t,flt}) / T_{rv}  \\
    \dot{x}_{Pe}    &= (P_e - x_{Pe}) / T_p  \\
    \dot{x}_{Kqi}   &= K_{qi}\,(Q_{ref} - Q_e)  \quad (V_{flag}=0)  \\
    \dot{x}_{Kvi}   &= K_{vi}\,(V_{t,flt} - V_{ref,ctrl})  \\
    \dot{V}_{l,flt} &= \mathrm{clip}((K_{vi,out} - V_{l,flt})/T_{iq},\;dP_{min},\;dP_{max})  \\
    \dot{P}_{flt}   &= \mathrm{clip}((P_{ref} - P_{flt})/T_{pord},\;dP_{min},\;dP_{max})

**Algebraic outputs** (replace the ``Ipcmd`` / ``Iqcmd`` inputs of REGC_A)

.. math::

    I_{pcmd} &= \mathrm{clip}(P_{flt}/\max(V_{t,flt},\,0.01),\;0,\;I_{pmax})  \\
    I_{qcmd} &= \mathrm{clip}(V_{l,flt} + I_{q,inj},\;I_{qmin},\;I_{qmax})

where :math:`I_{q,inj}` is the supplementary VRT reactive injection.
"""
import numpy as np


def descriptions():
    r"""Single source of truth for REEC_B parameters, states, and I/O."""
    d = []
    d += [{"type": "Parameter", "tex": "T_{rv}",   "data": "Trv",   "model": "Trv",   "default": 0.02,   "description": "Terminal voltage filter time constant",             "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_p",       "data": "Tp",    "model": "Tp",    "default": 0.02,   "description": "Active power filter time constant",                "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_{iq}",    "data": "Tiq",   "model": "Tiq",   "default": 0.02,   "description": "Reactive current lag time constant",               "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_{pord}",  "data": "Tpord", "model": "Tpord", "default": 0.1,    "description": "Active power order lag time constant",             "units": "s"}]
    d += [{"type": "Parameter", "tex": "K_{vp}",    "data": "Kvp",   "model": "Kvp",   "default": 1.0,    "description": "Inner voltage PI proportional gain",               "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "K_{vi}",    "data": "Kvi",   "model": "Kvi",   "default": 40.0,   "description": "Inner voltage PI integral gain",                   "units": "pu/pu·s"}]
    d += [{"type": "Parameter", "tex": "K_{qp}",    "data": "Kqp",   "model": "Kqp",   "default": 0.0,    "description": "Q-PI proportional gain (Vflag=0)",                 "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "K_{qi}",    "data": "Kqi",   "model": "Kqi",   "default": 0.0,    "description": "Q-PI integral gain (Vflag=0)",                     "units": "pu/pu·s"}]
    d += [{"type": "Parameter", "tex": "V_{ref0}",  "data": "Vref0", "model": "Vref0", "default": 1.0,    "description": "Initial voltage reference",                        "units": "pu"}]
    d += [{"type": "Parameter", "tex": "dbd_1",     "data": "dbd1",  "model": "dbd1",  "default": -0.05,  "description": "VRT deadband lower bound (overvoltage, <0)",        "units": "pu"}]
    d += [{"type": "Parameter", "tex": "dbd_2",     "data": "dbd2",  "model": "dbd2",  "default": 0.05,   "description": "VRT deadband upper bound (undervoltage, >0)",       "units": "pu"}]
    d += [{"type": "Parameter", "tex": "K_{qv}",    "data": "Kqv",   "model": "Kqv",   "default": 2.0,    "description": "VRT reactive current injection gain",               "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "I_{qhl}",   "data": "Iqhl",  "model": "Iqhl",  "default": 1.05,   "description": "VRT Iqinj upper limit",                            "units": "pu"}]
    d += [{"type": "Parameter", "tex": "I_{qll}",   "data": "Iqll",  "model": "Iqll",  "default": -1.05,  "description": "VRT Iqinj lower limit (<0)",                       "units": "pu"}]
    d += [{"type": "Parameter", "tex": "V_{max}",   "data": "Vmax",  "model": "Vmax",  "default": 1.1,    "description": "Q-PI output upper voltage limit",                  "units": "pu"}]
    d += [{"type": "Parameter", "tex": "V_{min}",   "data": "Vmin",  "model": "Vmin",  "default": 0.9,    "description": "Q-PI output lower voltage limit",                  "units": "pu"}]
    d += [{"type": "Parameter", "tex": "I_{max}",   "data": "Imax",  "model": "Imax",  "default": 1.1,    "description": "Maximum apparent current magnitude",               "units": "pu"}]
    d += [{"type": "Parameter", "tex": "P_{max}",   "data": "Pmax",  "model": "Pmax",  "default": 1.0,    "description": "Maximum active power",                             "units": "pu"}]
    d += [{"type": "Parameter", "tex": "P_{min}",   "data": "Pmin",  "model": "Pmin",  "default": 0.0,    "description": "Minimum active power",                             "units": "pu"}]
    d += [{"type": "Parameter", "tex": "dP_{max}",  "data": "dPmax", "model": "dPmax", "default": 999.0,  "description": "Active power up-ramp rate limit",                  "units": "pu/s"}]
    d += [{"type": "Parameter", "tex": "dP_{min}",  "data": "dPmin", "model": "dPmin", "default": -999.0, "description": "Active power down-ramp rate limit",                "units": "pu/s"}]
    d += [{"type": "Parameter", "tex": "Vflag",     "data": "Vflag",  "model": "Vflag",  "default": 1, "description": "0=Q control  1=voltage control",   "units": "-"}]
    d += [{"type": "Parameter", "tex": "Qflag",     "data": "Qflag",  "model": "Qflag",  "default": 1, "description": "0=bypass inner V-PI  1=engage",    "units": "-"}]
    d += [{"type": "Parameter", "tex": "PFflag",    "data": "PFflag", "model": "PFflag", "default": 0, "description": "0=constant Q/Vref  1=constant PF", "units": "-"}]
    d += [{"type": "Parameter", "tex": "Pqflag",    "data": "Pqflag", "model": "Pqflag", "default": 0, "description": "0=Q priority  1=P priority",       "units": "-"}]
    d += [{"type": "Input",         "tex": "P_{ref}",   "data": "Pref",  "model": "Pref",  "default": 0.8, "description": "Active power reference (from REPC_A or external)",  "units": "pu"}]
    d += [{"type": "Input",         "tex": "Q_{ext}",   "data": "Qext",  "model": "Qext",  "default": 1.0, "description": "Reactive/voltage reference (from REPC_A or external)","units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "Vt_{flt}",  "data": "", "model": "Vt_flt", "default": "", "description": "Filtered terminal voltage",           "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "x_{Pe}",    "data": "", "model": "x_Pe",   "default": "", "description": "Filtered active power",                "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "x_{Kqi}",   "data": "", "model": "x_Kqi",  "default": "", "description": "Q-PI integrator state",                "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "x_{Kvi}",   "data": "", "model": "x_Kvi",  "default": "", "description": "Inner voltage PI integrator state",    "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "Vl_{flt}",  "data": "", "model": "Vl_flt", "default": "", "description": "Reactive current regulator output lag", "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "P_{flt}",   "data": "", "model": "P_flt",  "default": "", "description": "Active power order (Tpord lag)",        "units": "pu"}]
    d += [{"type": "Algebraic State","tex": "I_{pcmd}",  "data": "", "model": "Ipcmd",  "default": "", "description": "Active current command to REGC_A",    "units": "pu"}]
    d += [{"type": "Algebraic State","tex": "I_{qcmd}",  "data": "", "model": "Iqcmd",  "default": "", "description": "Reactive current command to REGC_A",  "units": "pu"}]
    return d


def reec_b(dae, data, name, bus_name, backend):
    """Add REEC_B electrical controls for converter *name*.

    Must be called **after** :func:`~pydae.bps.weccs.regc_a.regc_a` so that
    ``Ipcmd_{name}`` and ``Iqcmd_{name}`` already exist in *u_run_dict*.
    This function promotes them to algebraic states driven by REEC_B equations.

    New inputs added to *u_run_dict* (overridden by REPC_A when present):
    ``Pref_{name}`` and ``Qext_{name}``.
    """
    reec = data['reec']

    def p(key, default=0.0):
        return reec.get(key, default)

    Vflag  = int(p('Vflag',  1))
    Qflag  = int(p('Qflag',  1))
    PFflag = int(p('PFflag', 0))
    Pqflag = int(p('Pqflag', 0))

    # ── network & REGC_A symbols ─────────────────────────────────────────
    Vt    = backend.symbols(f"V_{bus_name}")
    Ip    = backend.symbols(f"Ip_{name}")
    Iq    = backend.symbols(f"Iq_{name}")
    Pref  = backend.symbols(f"Pref_{name}")
    Qext  = backend.symbols(f"Qext_{name}")
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

    Pe = Ip * Vt
    Qe = Iq * Vt

    # ── VRT supplementary reactive injection ─────────────────────────────
    # error = Vt_flt - Vref0: >0 overvoltage, <0 undervoltage
    vrt_err = Vt_flt - Vref0
    db_out  = backend.max(vrt_err - dbd2, 0.0) + backend.min(vrt_err - dbd1, 0.0)
    Iqinj   = backend.hard_limits(Kqv * db_out, Iqll, Iqhl)

    # ── current limit logic ───────────────────────────────────────────────
    EPS = 1e-4
    if Pqflag == 0:      # Q priority
        Iqmax =  Imax
        Iqmin = -Imax
        Ipmax = backend.sqrt(backend.max(Imax**2 - Vl_flt**2, EPS))
    else:                # P priority
        P_flt_pu = P_flt / backend.max(Vt_flt, 0.01)
        Ipmax    = Imax
        Iqmax    =  backend.sqrt(backend.max(Imax**2 - P_flt_pu**2, EPS))
        Iqmin    = -Iqmax

    # ── Q / voltage control path ──────────────────────────────────────────
    if PFflag == 0:
        Qref = Qext
    else:
        Pfaref = float(np.arctan(p('Q0', 0.0) / max(float(p('P0', p('Pref', 0.8))), 1e-6)))
        Qref   = x_Pe * float(np.tan(Pfaref))

    if Vflag == 0:
        Vcorr     = backend.hard_limits(Kqp * (Qref - Qe) + x_Kqi, Vmin, Vmax)
        Vref_ctrl = Vcorr
        dx_Kqi    = Kqi * (Qref - Qe)
    else:
        Vref_ctrl = Qref      # Qext acts as voltage reference
        dx_Kqi    = x_Kqi * 0.0

    if Qflag == 1:
        # error: positive when over-voltage → less reactive injection
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

    # ── differential equations ────────────────────────────────────────────
    dVt_flt = (Vt - Vt_flt) / Trv
    dx_Pe   = (Pe - x_Pe) / Tp

    # ── algebraic output constraints ──────────────────────────────────────
    g_Ipcmd = Ipcmd - backend.hard_limits(P_flt / backend.max(Vt_flt, 0.01), Pmin, Ipmax)
    g_Iqcmd = Iqcmd - backend.hard_limits(Vl_flt + Iqinj, Iqmin, Iqmax)

    # ── assembly ──────────────────────────────────────────────────────────
    dae['f'] += [dVt_flt, dx_Pe, dx_Kqi, dx_Kvi, dVl_flt, dP_flt]
    dae['x'] += [Vt_flt,  x_Pe,  x_Kqi,  x_Kvi,  Vl_flt,  P_flt]

    dae['g']     += [g_Ipcmd, g_Iqcmd]
    dae['y_ini'] += [Ipcmd,   Iqcmd]
    dae['y_run'] += [Ipcmd,   Iqcmd]

    # Ipcmd/Iqcmd were REGC_A inputs; now they are computed algebraic states
    dae['u_ini_dict'].pop(f'Ipcmd_{name}', None)
    dae['u_run_dict'].pop(f'Ipcmd_{name}', None)
    dae['u_ini_dict'].pop(f'Iqcmd_{name}', None)
    dae['u_run_dict'].pop(f'Iqcmd_{name}', None)

    # New inputs — overridden by REPC_A when present
    Pref_0 = p('Pref', 0.8)
    Qext_0 = p('Qext', p('Vref0', 1.0))
    dae['u_ini_dict'].update({f'Pref_{name}': Pref_0})
    dae['u_run_dict'].update({f'Pref_{name}': Pref_0})
    dae['u_ini_dict'].update({f'Qext_{name}': Qext_0})
    dae['u_run_dict'].update({f'Qext_{name}': Qext_0})

    # ── parameters ────────────────────────────────────────────────────────
    dae['params_dict'].update({
        f'Trv_{name}':       p('Trv',    0.02),
        f'Tp_reec_{name}':   p('Tp',     0.02),
        f'Tiq_{name}':       p('Tiq',    0.02),
        f'Tpord_{name}':     p('Tpord',  0.1),
        f'Kvp_{name}':       p('Kvp',    1.0),
        f'Kvi_{name}':       p('Kvi',    40.0),
        f'Kqp_{name}':       p('Kqp',    0.0),
        f'Kqi_{name}':       p('Kqi',    0.0),
        f'Vref0_{name}':     p('Vref0',  1.0),
        f'dbd1_{name}':      p('dbd1',  -0.05),
        f'dbd2_{name}':      p('dbd2',   0.05),
        f'Kqv_{name}':       p('Kqv',    2.0),
        f'Iqhl_{name}':      p('Iqhl',   1.05),
        f'Iqll_{name}':      p('Iqll',  -1.05),
        f'Vmax_{name}':      p('Vmax',   1.1),
        f'Vmin_{name}':      p('Vmin',   0.9),
        f'Imax_{name}':      p('Imax',   1.1),
        f'Pmax_reec_{name}': p('Pmax',   1.0),
        f'Pmin_reec_{name}': p('Pmin',   0.0),
        f'dPmax_{name}':     p('dPmax',  999.0),
        f'dPmin_{name}':     p('dPmin', -999.0),
    })

    # ── initialization hints ──────────────────────────────────────────────
    V_0      = data.get('V_ini', 1.0)
    Ip_0     = data.get('Ipcmd', Pref_0)
    Iqcmd_0  = -data.get('Iqcmd', 0.0)   # sign: Iq_0 = -Iqcmd_0 in REGC_A
    dae['xy_0_dict'].update({
        str(Vt_flt): V_0,
        str(x_Pe):   Ip_0 * V_0,
        str(x_Kqi):  0.0,
        str(x_Kvi):  Iqcmd_0,
        str(Vl_flt): Iqcmd_0,
        str(P_flt):  Pref_0,
        str(Ipcmd):  Ip_0,
        str(Iqcmd):  Iqcmd_0,
    })


# ======================================================================
# In-module test (REGC_A + REEC_B only, no plant controller)
# ======================================================================
def test():
    import os, time
    from pydae.bps import BpsBuilder
    from pydae.core import Builder, Model

    hjson_path = os.path.join(os.path.dirname(__file__), 'reec_b.hjson')
    grid = BpsBuilder(hjson_path)
    grid.construct('temp_reec_b')
    bld = Builder(grid.sys_dict, target='cffi', sparse=False)
    bld.build()

    model = Model('temp_reec_b')
    t0 = time.perf_counter_ns()
    model.ini({}, 'temp_reec_b_xy_0.json')
    t1 = time.perf_counter_ns()
    print(f"ini time: {(t1-t0)/1e6:.2f} ms")

    Ip  = model.get_value('Ip_1')
    Iq  = model.get_value('Iq_1')
    Ipc = model.get_value('Ipcmd_1')
    Iqc = model.get_value('Iqcmd_1')
    print(f"Ip={Ip:.4f}  Iq={Iq:.4f}  Ipcmd={Ipc:.4f}  Iqcmd={Iqc:.4f}")

    assert abs(Ip - 0.8) < 0.05, f"Ip mismatch: {Ip}"
    print("test() PASSED")


if __name__ == '__main__':
    test()
