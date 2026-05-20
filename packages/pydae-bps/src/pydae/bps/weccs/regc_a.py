# -*- coding: utf-8 -*-
r"""
WECC Renewable Energy Generator/Converter Model — Type A (REGC_A).

Reference: WECC Solar Plant Dynamic Modeling Guidelines, April 2014, Section 5.1.

The model is the grid-interface layer in the WECC three-module renewable plant stack::

    REPC_A  ←  Plant controller  (optional)
       |
    REEC_B  ←  Electrical controls
       |
    REGC_A  ←  Generator / converter  ← this file
       |
    Network

REGC_A receives real (``Ipcmd``) and reactive (``Iqcmd``) current commands and
injects them into the network as voltage-scaled active and reactive power.

**Differential equations**

.. math::

    \dot{V}_{flt} = \frac{V_t - V_{flt}}{T_{fltr}}

    \dot{I}_p = \mathrm{clip}\!\left(\frac{I_{p,\mathrm{eff}} - I_p}{T_g},\;
                                     -r_{rpwr},\; r_{rpwr}\right)

    \dot{I}_q = \mathrm{clip}\!\left(\frac{-I_{q,\mathrm{cmd}} - I_q}{T_g},\;
                                     I_{qr\min},\; I_{qr\max}\right)

**Algebraic signals**

.. math::

    \mathrm{gain} &= \mathrm{clip}\!\left(
        \frac{V_{flt} - l_{vpnt0}}{l_{vpnt1} - l_{vpnt0}},\; 0,\; 1\right)

    \mathrm{LVPL} &= \mathrm{clip}\!\left(
        \frac{V_{flt} - Z_{erox}}{B_{rkpt} - Z_{erox}} L_{vpl1},\;
        0,\; L_{vpl1}\right)  \quad (L_{vplsw}=1)

    I_{p,\mathrm{eff}} &= \min(I_{p,\mathrm{cmd}},\,\mathrm{LVPL})\cdot\mathrm{gain}

    I_{q,\mathrm{out}} &= \mathrm{clip}\!\left(
        I_q + K_{hv}\max(V_t - V_{olim},\,0),\;
        I_{olim},\; +\infty\right)

**Power injection into the network**

.. math::

    P = I_p \, V_t \, S_n, \qquad Q = I_{q,\mathrm{out}} \, V_t \, S_n
"""
import numpy as np


def descriptions():
    r"""Single source of truth for REGC_A parameters, inputs, states, and outputs."""
    d = []

    # Parameters
    d += [{"type": "Parameter", "tex": "S_n",       "data": "S_n",      "model": "S_n",      "default": 100e6,  "description": "Converter MVA base",                           "units": "VA"}]
    d += [{"type": "Parameter", "tex": "T_{fltr}",  "data": "Tfltr",    "model": "Tfltr",    "default": 0.02,   "description": "Terminal voltage filter time constant (LVPL)",  "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_g",        "data": "Tg",       "model": "Tg",       "default": 0.02,   "description": "Inverter current regulator lag time constant",  "units": "s"}]
    d += [{"type": "Parameter", "tex": "L_{vpl1}",  "data": "Lvpl1",    "model": "Lvpl1",    "default": 1.22,   "description": "LVPL gain (current at breakpoint voltage)",      "units": "pu"}]
    d += [{"type": "Parameter", "tex": "Z_{erox}",  "data": "Zerox",    "model": "Zerox",    "default": 0.4,    "description": "LVPL zero-crossing voltage",                    "units": "pu"}]
    d += [{"type": "Parameter", "tex": "B_{rkpt}",  "data": "Brkpt",    "model": "Brkpt",    "default": 0.9,    "description": "LVPL breakpoint voltage",                       "units": "pu"}]
    d += [{"type": "Parameter", "tex": "L_{vplsw}", "data": "Lvplsw",   "model": "Lvplsw",   "default": 1,      "description": "Enable (1) / disable (0) LVPL",                 "units": "-"}]
    d += [{"type": "Parameter", "tex": "r_{rpwr}",  "data": "rrpwr",    "model": "rrpwr",    "default": 10.0,   "description": "Active current up-ramp rate limit on recovery",  "units": "pu/s"}]
    d += [{"type": "Parameter", "tex": "V_{olim}",  "data": "Volim",    "model": "Volim",    "default": 1.2,    "description": "High-voltage clamp threshold",                  "units": "pu"}]
    d += [{"type": "Parameter", "tex": "I_{olim}",  "data": "Iolim",    "model": "Iolim",    "default": -1.3,   "description": "High-voltage clamp current lower bound (<0)",    "units": "pu"}]
    d += [{"type": "Parameter", "tex": "K_{hv}",    "data": "Khv",      "model": "Khv",      "default": 0.7,    "description": "High-voltage clamp gain",                       "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "l_{vpnt0}", "data": "lvpnt0",   "model": "lvpnt0",   "default": 0.4,    "description": "Low-voltage active current management lower breakpoint (gain=0 below)", "units": "pu"}]
    d += [{"type": "Parameter", "tex": "l_{vpnt1}", "data": "lvpnt1",   "model": "lvpnt1",   "default": 0.8,    "description": "Low-voltage active current management upper breakpoint (gain=1 above)", "units": "pu"}]
    d += [{"type": "Parameter", "tex": "I_{qrmax}", "data": "Iqrmax",   "model": "Iqrmax",   "default": 999.0,  "description": "Maximum reactive current rate of change",        "units": "pu/s"}]
    d += [{"type": "Parameter", "tex": "I_{qrmin}", "data": "Iqrmin",   "model": "Iqrmin",   "default": -999.0, "description": "Minimum reactive current rate of change",        "units": "pu/s"}]

    # Inputs
    d += [{"type": "Input", "tex": "I_{pcmd}", "data": "Ipcmd", "model": "Ipcmd", "default": 0.8, "description": "Active current command from REEC_B",   "units": "pu"}]
    d += [{"type": "Input", "tex": "I_{qcmd}", "data": "Iqcmd", "model": "Iqcmd", "default": 0.0, "description": "Reactive current command from REEC_B",  "units": "pu"}]

    # Dynamic states
    d += [{"type": "Dynamic State", "tex": "V_{flt}", "data": "", "model": "V_flt", "default": "", "description": "Filtered terminal voltage (LVPL filter)",  "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "I_p",     "data": "", "model": "Ip",    "default": "", "description": "Actual active current injection",           "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "I_q",     "data": "", "model": "Iq",    "default": "", "description": "Actual reactive current injection",          "units": "pu"}]

    # Outputs
    d += [{"type": "Output", "tex": "p_g", "data": "", "model": "p_g", "default": "", "description": "Active power injection (pu on S_n)",   "units": "pu"}]
    d += [{"type": "Output", "tex": "q_g", "data": "", "model": "q_g", "default": "", "description": "Reactive power injection (pu on S_n)", "units": "pu"}]

    return d


def regc_a(grid, name, bus_name, data_dict):
    """Build the REGC_A model and inject it into *grid*.

    Parameters
    ----------
    grid : BpsBuilder
    name : str
        Unique identifier for this converter instance.
    bus_name : str
        Terminal bus name in the network.
    data_dict : dict
        Entry from the ``weccs`` list in the HJSON file.

    Returns
    -------
    p_W : symbolic expression
        Active power injection in Watts (for the bus balance equation).
    q_var : symbolic expression
        Reactive power injection in VAR.
    """
    backend = grid.backend

    meta = descriptions()
    default_map = {item['data']: item['default']
                   for item in meta if item.get('data')}

    def param(key):
        return data_dict.get(key, default_map.get(key, 0.0))

    # ------------------------------------------------------------------ #
    # Symbols — network                                                    #
    # ------------------------------------------------------------------ #
    Vt = backend.symbols(f"V_{bus_name}")

    # ------------------------------------------------------------------ #
    # Symbols — inputs                                                     #
    # ------------------------------------------------------------------ #
    Ipcmd = backend.symbols(f"Ipcmd_{name}")
    Iqcmd = backend.symbols(f"Iqcmd_{name}")

    # ------------------------------------------------------------------ #
    # Symbols — dynamic states                                             #
    # ------------------------------------------------------------------ #
    V_flt = backend.symbols(f"V_flt_{name}")
    Ip    = backend.symbols(f"Ip_{name}")
    Iq    = backend.symbols(f"Iq_{name}")

    # ------------------------------------------------------------------ #
    # Symbols — parameters                                                 #
    # ------------------------------------------------------------------ #
    S_n    = backend.symbols(f"S_n_{name}")
    Tfltr  = backend.symbols(f"Tfltr_{name}")
    Tg     = backend.symbols(f"Tg_{name}")
    Lvpl1  = backend.symbols(f"Lvpl1_{name}")
    Zerox  = backend.symbols(f"Zerox_{name}")
    Brkpt  = backend.symbols(f"Brkpt_{name}")
    rrpwr  = backend.symbols(f"rrpwr_{name}")
    Volim  = backend.symbols(f"Volim_{name}")
    Iolim  = backend.symbols(f"Iolim_{name}")
    Khv    = backend.symbols(f"Khv_{name}")
    lvpnt0 = backend.symbols(f"lvpnt0_{name}")
    lvpnt1 = backend.symbols(f"lvpnt1_{name}")
    Iqrmax = backend.symbols(f"Iqrmax_{name}")
    Iqrmin = backend.symbols(f"Iqrmin_{name}")

    # ------------------------------------------------------------------ #
    # Low Voltage Active Current Management gain  [0, 1]                  #
    # ------------------------------------------------------------------ #
    gain = backend.hard_limits(
        (V_flt - lvpnt0) / (lvpnt1 - lvpnt0), 0.0, 1.0
    )

    # ------------------------------------------------------------------ #
    # Low Voltage Power Logic (LVPL)                                       #
    # Lvplsw is treated as a Python flag so the expression tree is clean. #
    # ------------------------------------------------------------------ #
    Lvplsw_val = int(param('Lvplsw'))
    if Lvplsw_val:
        LVPL = backend.hard_limits(
            (V_flt - Zerox) / (Brkpt - Zerox) * Lvpl1, 0.0, Lvpl1
        )
        Ipcmd_eff = backend.min(Ipcmd, LVPL) * gain
    else:
        Ipcmd_eff = Ipcmd * gain

    # ------------------------------------------------------------------ #
    # High Voltage Reactive Current Clamp                                  #
    # Positive Iq → reactive injection; Iolim < 0 → absorption limit.    #
    # ------------------------------------------------------------------ #
    hv_correction = Khv * backend.max(Vt - Volim, 0.0)
    Iq_out = backend.hard_limits(Iq + hv_correction, Iolim, 999.0)

    # ------------------------------------------------------------------ #
    # Differential equations                                               #
    # ------------------------------------------------------------------ #
    dV_flt = (Vt - V_flt) / Tfltr

    # Active current lag with symmetric rate limiter (rrpwr).
    # The standard specifies only an upward rate limit on recovery;
    # applying it symmetrically is the common simulation simplification.
    dIp = backend.hard_limits(
        (Ipcmd_eff - Ip) / Tg, -rrpwr, rrpwr
    )

    # Reactive current lag: note the sign inversion (−Iqcmd) at the block input.
    dIq = backend.hard_limits(
        (-Iqcmd - Iq) / Tg, Iqrmin, Iqrmax
    )

    # ------------------------------------------------------------------ #
    # Power injection                                                      #
    # ------------------------------------------------------------------ #
    p_W   = Ip     * Vt * S_n
    q_var = Iq_out * Vt * S_n

    # ------------------------------------------------------------------ #
    # Assembly                                                             #
    # ------------------------------------------------------------------ #
    grid.dae['f'] += [dV_flt, dIp, dIq]
    grid.dae['x'] += [V_flt,  Ip,  Iq]

    # ------------------------------------------------------------------ #
    # Inputs                                                               #
    # ------------------------------------------------------------------ #
    Ipcmd_0 = param('Ipcmd')
    Iqcmd_0 = param('Iqcmd')
    grid.dae['u_ini_dict'].update({str(Ipcmd): Ipcmd_0})
    grid.dae['u_run_dict'].update({str(Ipcmd): Ipcmd_0})
    grid.dae['u_ini_dict'].update({str(Iqcmd): Iqcmd_0})
    grid.dae['u_run_dict'].update({str(Iqcmd): Iqcmd_0})

    # ------------------------------------------------------------------ #
    # Parameters                                                           #
    # ------------------------------------------------------------------ #
    _param_keys = [
        'S_n', 'Tfltr', 'Tg', 'Lvpl1', 'Zerox', 'Brkpt',
        'rrpwr', 'Volim', 'Iolim', 'Khv', 'lvpnt0', 'lvpnt1',
        'Iqrmax', 'Iqrmin',
    ]
    for key in _param_keys:
        grid.dae['params_dict'][f"{key}_{name}"] = param(key)

    # ------------------------------------------------------------------ #
    # Initialization hints                                                 #
    # ------------------------------------------------------------------ #
    S_n_val  = param('S_n')
    V_0      = data_dict.get('V_ini', 1.0)
    Ip_0     = Ipcmd_0          # gain = 1, LVPL inactive at nominal voltage
    Iq_0     = -Iqcmd_0         # steady state of lag with sign inversion

    grid.dae['xy_0_dict'][str(V_flt)] = V_0
    grid.dae['xy_0_dict'][str(Ip)]    = Ip_0
    grid.dae['xy_0_dict'][str(Iq)]    = Iq_0

    # ------------------------------------------------------------------ #
    # Outputs                                                              #
    # ------------------------------------------------------------------ #
    grid.dae['h_dict'][f"p_g_{name}"]   = Ip * Vt
    grid.dae['h_dict'][f"q_g_{name}"]   = Iq_out * Vt
    grid.dae['h_dict'][f"Ip_{name}"]    = Ip
    grid.dae['h_dict'][f"Iq_{name}"]    = Iq
    grid.dae['h_dict'][f"V_flt_{name}"] = V_flt
    grid.dae['h_dict'][f"gain_{name}"]  = gain

    return p_W, q_var


# ------------------------------------------------------------------ #
# In-module test                                                       #
# ------------------------------------------------------------------ #
def test():
    import os
    import time
    from pydae.bps import BpsBuilder
    from pydae.core import Builder, Model

    hjson_path = os.path.join(os.path.dirname(__file__), 'regc_a.hjson')

    grid = BpsBuilder(hjson_path)
    grid.construct('temp_regc_a')
    bld = Builder(grid.sys_dict, target='cffi', sparse=False)
    bld.build()

    model = Model('temp_regc_a')
    t0 = time.perf_counter_ns()
    model.ini({}, 'temp_regc_a_xy_0.json')
    t1 = time.perf_counter_ns()
    print(f"ini time: {(t1-t0)/1e6:.2f} ms")

    Vt  = model.get_value(f'V_1')
    Ip  = model.get_value(f'Ip_1')
    Iq  = model.get_value(f'Iq_1')
    p_g = model.get_value(f'p_g_1')
    q_g = model.get_value(f'q_g_1')
    print(f"V_1={Vt:.4f}  Ip={Ip:.4f}  Iq={Iq:.4f}  p_g={p_g:.4f}  q_g={q_g:.4f}")

    assert abs(Ip - 0.8) < 0.01, f"Ip mismatch: {Ip}"
    assert abs(Iq - 0.0) < 0.01, f"Iq mismatch: {Iq}"
    print("test() PASSED")


if __name__ == '__main__':
    test()
