# -*- coding: utf-8 -*-
r"""
WECC Renewable Energy Generator/Converter Model — Type B (REGC_B).

Reference: WECC Second-Generation Renewable Energy Model Library (post-2015).

REGC_B is a simplified variant of :mod:`regc_a`.  It drops the Low Voltage
Power Logic (LVPL) and the low-voltage active current management gain,
replacing them with a **combined apparent-current limit**
(:math:`I_{pmax} = \sqrt{I_{max}^2 - I_q^2}`) applied directly on
:math:`I_{pcmd}`.  The active current up-ramp rate (:math:`r_{rpwr}`)
is still limited on the **upward direction only** (voltage recovery),
with the downward direction left unconstrained.

The high-voltage reactive current clamp and the current lags
(:math:`T_g`, :math:`I_{qr\min/\max}`) are unchanged from REGC_A.

**Differential equations**

.. math::

    \dot{V}_{flt} &= \frac{V_t - V_{flt}}{T_{fltr}}

    \dot{I}_p &= \mathrm{clip}\!\left(
                    \frac{I_{p,eff} - I_p}{T_g},\;
                    -999,\; r_{rpwr}\right)
                 \quad \text{(upward rate limit only)}

    \dot{I}_q &= \mathrm{clip}\!\left(
                    \frac{-I_{qcmd} - I_q}{T_g},\;
                    I_{qr\min},\; I_{qr\max}\right)

**Algebraic signals**

.. math::

    I_{pmax} &= \sqrt{\max(I_{max}^2 - I_q^2,\;\varepsilon)}

    I_{p,eff} &= \mathrm{clip}(I_{pcmd},\; 0,\; I_{pmax})

    I_{q,out} &= \mathrm{clip}\!\left(
                    I_q + K_{hv}\max(V_t - V_{olim},\,0),\;
                    I_{olim},\; +\infty\right)

**Power injection**

.. math::

    P = I_p\, V_t\, S_n, \qquad Q = I_{q,out}\, V_t\, S_n

Compared to REGC_A, REGC_B removes ``Lvplsw``, ``Lvpl1``, ``Zerox``,
``Brkpt``, ``lvpnt0``, ``lvpnt1`` and adds ``Imax``.
"""
import numpy as np


def descriptions():
    r"""Single source of truth for REGC_B parameters, inputs, states, and outputs."""
    d = []

    d += [{"type": "Parameter", "tex": "S_n",       "data": "S_n",    "model": "S_n",    "default": 100e6,  "description": "Converter MVA base",                               "units": "VA"}]
    d += [{"type": "Parameter", "tex": "T_{fltr}",  "data": "Tfltr",  "model": "Tfltr",  "default": 0.02,   "description": "Terminal voltage filter time constant",             "units": "s"}]
    d += [{"type": "Parameter", "tex": "T_g",       "data": "Tg",     "model": "Tg",     "default": 0.02,   "description": "Inverter current regulator lag time constant",      "units": "s"}]
    d += [{"type": "Parameter", "tex": "I_{max}",   "data": "Imax",   "model": "Imax",   "default": 1.1,    "description": "Maximum apparent current magnitude",               "units": "pu"}]
    d += [{"type": "Parameter", "tex": "r_{rpwr}",  "data": "rrpwr",  "model": "rrpwr",  "default": 10.0,   "description": "Active current up-ramp rate limit (voltage recovery)","units": "pu/s"}]
    d += [{"type": "Parameter", "tex": "V_{olim}",  "data": "Volim",  "model": "Volim",  "default": 1.2,    "description": "High-voltage clamp threshold",                     "units": "pu"}]
    d += [{"type": "Parameter", "tex": "I_{olim}",  "data": "Iolim",  "model": "Iolim",  "default": -1.3,   "description": "High-voltage clamp current lower bound (<0)",       "units": "pu"}]
    d += [{"type": "Parameter", "tex": "K_{hv}",    "data": "Khv",    "model": "Khv",    "default": 0.7,    "description": "High-voltage clamp gain",                          "units": "pu/pu"}]
    d += [{"type": "Parameter", "tex": "I_{qrmax}", "data": "Iqrmax", "model": "Iqrmax", "default": 999.0,  "description": "Maximum reactive current rate of change",           "units": "pu/s"}]
    d += [{"type": "Parameter", "tex": "I_{qrmin}", "data": "Iqrmin", "model": "Iqrmin", "default": -999.0, "description": "Minimum reactive current rate of change",           "units": "pu/s"}]

    d += [{"type": "Input", "tex": "I_{pcmd}", "data": "Ipcmd", "model": "Ipcmd", "default": 0.8, "description": "Active current command (from REEC_B or external)",   "units": "pu"}]
    d += [{"type": "Input", "tex": "I_{qcmd}", "data": "Iqcmd", "model": "Iqcmd", "default": 0.0, "description": "Reactive current command (from REEC_B or external)",  "units": "pu"}]

    d += [{"type": "Dynamic State", "tex": "V_{flt}", "data": "", "model": "V_flt", "default": "", "description": "Filtered terminal voltage",   "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "I_p",     "data": "", "model": "Ip",    "default": "", "description": "Actual active current",        "units": "pu"}]
    d += [{"type": "Dynamic State", "tex": "I_q",     "data": "", "model": "Iq",    "default": "", "description": "Actual reactive current",       "units": "pu"}]

    d += [{"type": "Output", "tex": "p_g", "data": "", "model": "p_g", "default": "", "description": "Active power injection (pu on S_n)",   "units": "pu"}]
    d += [{"type": "Output", "tex": "q_g", "data": "", "model": "q_g", "default": "", "description": "Reactive power injection (pu on S_n)", "units": "pu"}]

    return d


def regc_b(grid, name, bus_name, data_dict):
    """Build REGC_B and inject it into *grid*.

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

    meta = descriptions()
    default_map = {item['data']: item['default']
                   for item in meta if item.get('data')}

    def param(key):
        return data_dict.get(key, default_map.get(key, 0.0))

    # ── network bus ───────────────────────────────────────────────────────
    Vt = backend.symbols(f"V_{bus_name}")

    # ── inputs ────────────────────────────────────────────────────────────
    Ipcmd = backend.symbols(f"Ipcmd_{name}")
    Iqcmd = backend.symbols(f"Iqcmd_{name}")

    # ── dynamic states ────────────────────────────────────────────────────
    V_flt = backend.symbols(f"V_flt_{name}")
    Ip    = backend.symbols(f"Ip_{name}")
    Iq    = backend.symbols(f"Iq_{name}")

    # ── parameters ────────────────────────────────────────────────────────
    S_n    = backend.symbols(f"S_n_{name}")
    Tfltr  = backend.symbols(f"Tfltr_{name}")
    Tg     = backend.symbols(f"Tg_{name}")
    Imax   = backend.symbols(f"Imax_{name}")
    rrpwr  = backend.symbols(f"rrpwr_{name}")
    Volim  = backend.symbols(f"Volim_{name}")
    Iolim  = backend.symbols(f"Iolim_{name}")
    Khv    = backend.symbols(f"Khv_{name}")
    Iqrmax = backend.symbols(f"Iqrmax_{name}")
    Iqrmin = backend.symbols(f"Iqrmin_{name}")

    # ── combined apparent-current limit ───────────────────────────────────
    # Ipmax shares the Imax budget with the current reactive current.
    # Uses Iq (actual state) to avoid algebraic circular dependence.
    EPS    = 1e-4
    Ipmax  = backend.sqrt(backend.max(Imax**2 - Iq**2, EPS))
    Ipcmd_eff = backend.hard_limits(Ipcmd, 0.0, Ipmax)

    # ── high-voltage reactive current clamp ───────────────────────────────
    hv_correction = Khv * backend.max(Vt - Volim, 0.0)
    Iq_out = backend.hard_limits(Iq + hv_correction, Iolim, 999.0)

    # ── differential equations ────────────────────────────────────────────
    dV_flt = (Vt - V_flt) / Tfltr

    # Active current: upward rate limited to rrpwr, downward unconstrained.
    # Using a large negative bound (-999) as the effective "no limit" downward.
    dIp = backend.hard_limits(
        (Ipcmd_eff - Ip) / Tg, -999.0, rrpwr
    )

    # Reactive current lag with symmetric rate limits.
    dIq = backend.hard_limits(
        (-Iqcmd - Iq) / Tg, Iqrmin, Iqrmax
    )

    # ── power injection ───────────────────────────────────────────────────
    p_W   = Ip     * Vt * S_n
    q_var = Iq_out * Vt * S_n

    # ── assembly ──────────────────────────────────────────────────────────
    grid.dae['f'] += [dV_flt, dIp, dIq]
    grid.dae['x'] += [V_flt,  Ip,  Iq]

    # ── inputs ────────────────────────────────────────────────────────────
    Ipcmd_0 = param('Ipcmd')
    Iqcmd_0 = param('Iqcmd')
    grid.dae['u_ini_dict'].update({str(Ipcmd): Ipcmd_0})
    grid.dae['u_run_dict'].update({str(Ipcmd): Ipcmd_0})
    grid.dae['u_ini_dict'].update({str(Iqcmd): Iqcmd_0})
    grid.dae['u_run_dict'].update({str(Iqcmd): Iqcmd_0})

    # ── parameters ────────────────────────────────────────────────────────
    for key in ['S_n', 'Tfltr', 'Tg', 'Imax', 'rrpwr',
                'Volim', 'Iolim', 'Khv', 'Iqrmax', 'Iqrmin']:
        grid.dae['params_dict'][f"{key}_{name}"] = param(key)

    # ── initialization hints ──────────────────────────────────────────────
    V_0  = data_dict.get('V_ini', 1.0)
    Ip_0 = Ipcmd_0
    Iq_0 = -Iqcmd_0    # REGC_A/B sign convention: Iq_0 = -Iqcmd_0

    grid.dae['xy_0_dict'][str(V_flt)] = V_0
    grid.dae['xy_0_dict'][str(Ip)]    = Ip_0
    grid.dae['xy_0_dict'][str(Iq)]    = Iq_0

    # ── outputs ───────────────────────────────────────────────────────────
    grid.dae['h_dict'][f"p_g_{name}"]   = Ip * Vt
    grid.dae['h_dict'][f"q_g_{name}"]   = Iq_out * Vt
    grid.dae['h_dict'][f"Ip_{name}"]    = Ip
    grid.dae['h_dict'][f"Iq_{name}"]    = Iq
    grid.dae['h_dict'][f"Ipmax_{name}"] = Ipmax

    return p_W, q_var


# ======================================================================
# In-module test
# ======================================================================
def test():
    import os, time
    from pydae.bps import BpsBuilder
    from pydae.core import Builder, Model

    hjson_path = os.path.join(os.path.dirname(__file__), 'regc_b.hjson')
    grid = BpsBuilder(hjson_path)
    grid.construct('temp_regc_b')
    bld = Builder(grid.sys_dict, target='cffi', sparse=False)
    bld.build()

    model = Model('temp_regc_b')
    t0 = time.perf_counter_ns()
    model.ini({}, 'temp_regc_b_xy_0.json')
    t1 = time.perf_counter_ns()
    print(f"ini time: {(t1-t0)/1e6:.2f} ms")

    Vt    = model.get_value('V_1')
    Ip    = model.get_value('Ip_1')
    Iq    = model.get_value('Iq_1')
    Ipmax = model.get_value('Ipmax_1')
    p_g   = model.get_value('p_g_1')
    print(f"V={Vt:.4f}  Ip={Ip:.4f}  Iq={Iq:.4f}  Ipmax={Ipmax:.4f}  p_g={p_g:.4f}")

    assert abs(Ip - 0.8) < 0.02,   f"Ip={Ip:.4f} not close to 0.8 pu"
    assert abs(Iq - 0.0) < 0.01,   f"Iq={Iq:.4f} should be ~0"
    assert Ipmax > 0.8,             f"Ipmax={Ipmax:.4f} too small"

    # Ipmax = sqrt(Imax² - Iq²): with Iq≈0 → Ipmax ≈ Imax = 1.1
    Imax_val = 1.1
    assert abs(Ipmax - Imax_val) < 0.01, (
        f"Ipmax={Ipmax:.4f} should ≈ Imax={Imax_val} at zero reactive current"
    )
    print("test() PASSED")


if __name__ == '__main__':
    test()
