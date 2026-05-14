# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np


def descriptions():
    """Single source of truth for pod_2wo_3ll parameters, inputs, states, and outputs."""
    d = []
    d.append({"type": "Input", "tex": "$V_s$", "data": "V_{bus}",
              "model": "V", "description": "Bus voltage magnitude"})
    d.append({"type": "Input", "tex": r"$\theta_s$",
              "data": "theta_{bus}", "model": "theta",
              "description": "Bus voltage angle"})
    d.append({"type": "Input", "tex": r"$\omega_\mathrm{coi}$",
              "data": "omega_coi", "model": "omega_coi",
              "description": "Center of inertia speed"})
    d.append({"type": "Parameter", "tex": "$K_{p,pll}$", "data": "K_p_pll",
              "model": "K_p_pll", "default": 1.0,
              "description": "PLL proportional gain"})
    d.append({"type": "Parameter", "tex": "$K_{i,pll}$", "data": "K_i_pll",
              "model": "K_i_pll", "default": 1.0,
              "description": "PLL integral gain"})
    d.append({"type": "Parameter", "tex": "$T_{pll}$", "data": "T_pll",
              "model": "T_pll", "default": 0.01,
              "description": "PLL filter time constant"})
    d.append({"type": "Parameter", "tex": "$K_{stab}$", "data": "K_stab",
              "model": "K_stab", "default": 1.0,
              "description": "POD stabilizer gain"})
    d.append({"type": "Parameter", "tex": "$T_{lpf}$", "data": "T_lpf",
              "model": "T_lpf", "default": 0.02,
              "description": "Low-pass filter time constant"})
    d.append({"type": "Parameter", "tex": "$T_{wo1}$", "data": "T_wo1",
              "model": "T_wo1", "default": 2.0,
              "description": "Washout 1 time constant"})
    d.append({"type": "Parameter", "tex": "$T_{wo2}$", "data": "T_wo2",
              "model": "T_wo2", "default": 2.0,
              "description": "Washout 2 time constant"})
    d.append({"type": "Parameter", "tex": "$T_1$", "data": "T_1",
              "model": "T_1", "default": 0.1,
              "description": "Lead-lag 1 numerator time constant"})
    d.append({"type": "Parameter", "tex": "$T_2$", "data": "T_2",
              "model": "T_2", "default": 0.02,
              "description": "Lead-lag 1 denominator time constant"})
    d.append({"type": "Parameter", "tex": "$T_3$", "data": "T_3",
              "model": "T_3", "default": 0.1,
              "description": "Lead-lag 2 numerator time constant"})
    d.append({"type": "Parameter", "tex": "$T_4$", "data": "T_4",
              "model": "T_4", "default": 0.02,
              "description": "Lead-lag 2 denominator time constant"})
    d.append({"type": "Parameter", "tex": "$T_5$", "data": "T_5",
              "model": "T_5", "default": 0.1,
              "description": "Lead-lag 3 numerator time constant"})
    d.append({"type": "Parameter", "tex": "$T_6$", "data": "T_6",
              "model": "T_6", "default": 0.02,
              "description": "Lead-lag 3 denominator time constant"})
    d.append({"type": "Dynamic State", "tex": r"$\theta_{pll}$",
              "data": "theta_pll", "model": "theta_pll",
              "description": "PLL angle"})
    d.append({"type": "Dynamic State", "tex": r"$\xi_{pll}$",
              "data": "xi_pll", "model": "xi_pll",
              "description": "PLL integrator state"})
    d.append({"type": "Dynamic State", "tex": r"$\omega_{pll,f}$",
              "data": "omega_pll_f", "model": "omega_pll_f",
              "description": "Filtered PLL speed"})
    d.append({"type": "Dynamic State", "tex": r"$x_{lpf}$",
              "data": "x_lpf_pod", "model": "x_lpf_pod",
              "description": "Low-pass filter state"})
    d.append({"type": "Dynamic State", "tex": r"$x_{wo1}$",
              "data": "x_wo1_pod", "model": "x_wo1_pod",
              "description": "Washout 1 state"})
    d.append({"type": "Dynamic State", "tex": r"$x_{wo2}$",
              "data": "x_wo2_pod", "model": "x_wo2_pod",
              "description": "Washout 2 state"})
    d.append({"type": "Dynamic State", "tex": r"$x_{12}$",
              "data": "x_12_pod", "model": "x_12_pod",
              "description": "Lead-lag 1-2 state"})
    d.append({"type": "Dynamic State", "tex": r"$x_{34}$",
              "data": "x_34_pod", "model": "x_34_pod",
              "description": "Lead-lag 3-4 state"})
    d.append({"type": "Dynamic State", "tex": r"$x_{56}$",
              "data": "x_56_pod", "model": "x_56_pod",
              "description": "Lead-lag 5-6 state"})
    d.append({"type": "Algebraic State", "tex": r"$\mathrm{rocof}$",
              "data": "rocof", "model": "rocof",
              "description": "Rate of change of frequency"})
    d.append({"type": "Algebraic State", "tex": r"$\mathrm{pod\_out}$",
              "data": "pod_out", "model": "pod_out",
              "description": "POD output (saturated)"})
    d.append({"type": "Output", "tex": r"$\omega_{pll}$",
              "data": "omega_pll", "model": "omega_pll",
              "description": "PLL speed"})
    d.append({"type": "Output", "tex": r"$\omega_{pll,f}$",
              "data": "omega_pll_f", "model": "omega_pll_f",
              "description": "Filtered PLL speed"})
    d.append({"type": "Output", "tex": r"$\mathrm{rocof}$",
              "data": "rocof", "model": "rocof",
              "description": "Rate of change of frequency"})
    return d


def add_pod_2wo_3ll(dae, data, name=None, bus_name=None, backend=None):
    r"""
    POD with 2 washout blocks and 3 lead-lag blocks, plus a PLL.

    Works with both SymPy and CasADi backends.
    """

    if backend is None:
        import sympy as sym
        backend = type('Backend', (), {
            'use_casadi': False,
            'symbols': staticmethod(lambda n, **k: sym.symbols(n, real=True)),
            'hard_limit': staticmethod(lambda v, lo, hi: sym.Max(lo, sym.Min(hi, v))),
            'sin': staticmethod(sym.sin),
            'cos': staticmethod(sym.cos),
        })()

    if bus_name is None:
        bus_name = data['bus']

    if name is None:
        name = data.get('name', bus_name)

    sin = backend.sin
    cos = backend.cos

    # inputs
    V_s = backend.symbols(f"V_{bus_name}", real=True)
    theta_s = backend.symbols(f"theta_{bus_name}", real=True)
    omega_coi = backend.symbols("omega_coi", real=True)

    ############################################################################################################
    # PLL
    ############################################################################################################

    # dynamic states
    theta_pll, xi_pll, omega_pll_f = backend.symbols(f'theta_pll_{name},xi_pll_{name},omega_pll_f_{name}', real=True)

    # algebraic states
    rocof = backend.symbols(f'rocof_{name}', real=True)

    # parameters
    T_pll = backend.symbols(f'T_pll_{name}', real=True)
    K_p_pll, K_i_pll, K_theta_pll = backend.symbols(f'K_p_pll_{name},K_i_pll_{name},K_theta_pll_{name}', real=True)

    v_sD = V_s * sin(theta_s)
    v_sQ = V_s * cos(theta_s)

    v_sd_pll = v_sD * cos(theta_pll) - v_sQ * sin(theta_pll)
    Domega_pll = K_p_pll * v_sd_pll + K_i_pll * xi_pll
    omega_pll = Domega_pll + 1.0
    dtheta_pll = 2 * np.pi * 50 * (omega_pll - omega_coi) * K_theta_pll
    dxi_pll = v_sd_pll
    domega_pll_f = 1 / T_pll * (omega_pll - omega_pll_f)

    eq_rocof = rocof - domega_pll_f

    dae['f'] += [dtheta_pll, dxi_pll, domega_pll_f]
    dae['x'] += [theta_pll, xi_pll, omega_pll_f]
    dae['g'] += [eq_rocof]
    dae['y_ini'] += [rocof]
    dae['y_run'] += [rocof]
    dae['params_dict'].update({f'K_p_pll_{name}': data['K_p_pll']})
    dae['params_dict'].update({f'K_i_pll_{name}': data['K_i_pll']})
    dae['params_dict'].update({f'T_pll_{name}': data['T_pll']})
    dae['params_dict'].update({f'K_theta_pll_{name}': 1.0})

    dae['h_dict'].update({f"omega_pll_{name}": omega_pll})
    dae['h_dict'].update({f"omega_pll_f_{name}": omega_pll_f})
    dae['h_dict'].update({f"rocof_{name}": rocof})

    ############################################################################################################
    # POD
    ############################################################################################################

    # dynamic states
    x_lpf = backend.symbols(f"x_lpf_pod_{name}", real=True)
    x_wo1 = backend.symbols(f"x_wo1_pod_{name}", real=True)
    x_wo2 = backend.symbols(f"x_wo2_pod_{name}", real=True)
    x_12 = backend.symbols(f"x_12_pod_{name}", real=True)
    x_34 = backend.symbols(f"x_34_pod_{name}", real=True)
    x_56 = backend.symbols(f"x_56_pod_{name}", real=True)

    # algebraic states
    pod_out = backend.symbols(f"pod_out_{name}", real=True)

    # parameters
    T_wo1 = backend.symbols(f"T_wo1_pod_{name}", real=True)
    T_wo2 = backend.symbols(f"T_wo2_pod_{name}", real=True)
    T_lpf = backend.symbols(f"T_lpf_pod_{name}", real=True)
    T_1 = backend.symbols(f"T_1_pod_{name}", real=True)
    T_2 = backend.symbols(f"T_2_pod_{name}", real=True)
    T_3 = backend.symbols(f"T_3_pod_{name}", real=True)
    T_4 = backend.symbols(f"T_4_pod_{name}", real=True)
    T_5 = backend.symbols(f"T_5_pod_{name}", real=True)
    T_6 = backend.symbols(f"T_6_pod_{name}", real=True)
    K_stab = backend.symbols(f"K_stab_{name}", real=True)
    Limit = backend.symbols(f"Limit_pod_{name}", real=True)
    u_pll_probe = backend.symbols(f"u_pll_probe_{name}", real=True)

    # auxiliar
    omega = omega_pll_f
    u_lpf = K_stab * (omega - 1.0) + u_pll_probe

    z_lpf = x_lpf
    z_wo1 = z_lpf - x_wo1
    z_wo2 = z_wo1 - x_wo2

    u_wo1 = z_lpf
    u_wo2 = z_wo1
    z_12 = (z_wo2 - x_12) * T_1 / T_2 + x_12
    z_34 = (z_12 - x_34) * T_3 / T_4 + x_34
    z_56 = (z_34 - x_56) * T_5 / T_6 + x_56

    pod_out_nosat = z_56

    dx_lpf = (u_lpf - x_lpf) / T_lpf
    dx_wo1 = (u_wo1 - x_wo1) / T_wo1
    dx_wo2 = (u_wo2 - x_wo2) / T_wo2
    dx_12 = (z_wo2 - x_12) / T_2
    dx_34 = (z_12 - x_34) / T_4
    dx_56 = (z_34 - x_56) / T_6

    pod_out_limited = backend.hard_limit(pod_out_nosat, -Limit, Limit)

    g_pod_out = pod_out_limited - pod_out

    dae['f'] += [dx_lpf, dx_wo1, dx_wo2, dx_12, dx_34, dx_56]
    dae['x'] += [x_lpf, x_wo1, x_wo2, x_12, x_34, x_56]
    dae['g'] += [g_pod_out]
    dae['y_ini'] += [pod_out]
    dae['y_run'] += [pod_out]
    dae['params_dict'].update({str(T_lpf): data['T_lpf']})
    dae['params_dict'].update({str(T_wo1): data['T_wo1']})
    dae['params_dict'].update({str(T_wo2): data['T_wo2']})
    dae['params_dict'].update({str(T_1): data['T_1']})
    dae['params_dict'].update({str(T_2): data['T_2']})
    dae['params_dict'].update({str(T_3): data['T_3']})
    dae['params_dict'].update({str(T_4): data['T_4']})
    dae['params_dict'].update({str(T_5): data['T_5']})
    dae['params_dict'].update({str(T_6): data['T_6']})
    dae['params_dict'].update({str(K_stab): data['K_stab']})
    dae['params_dict'].update({str(Limit): 0.1})

    dae['u_ini_dict'].update({str(u_pll_probe): 0.0})
    dae['u_run_dict'].update({str(u_pll_probe): 0.0})


def test():
    import os
    import sys

    import matplotlib.pyplot as plt
    from pydae.bps import BpsBuilder
    from pydae.core import Model
    from pydae.core.builder import CasadiBuilder, CasadiModel

    module_dir = os.path.dirname(__file__)
    hjson_path = os.path.join(module_dir, 'pod_2wo_3ll.hjson')
    build_dir = os.path.join(module_dir, 'build')
    os.makedirs(build_dir, exist_ok=True)

    # SymPy / CFFI path
    grid = BpsBuilder(hjson_path)
    grid.checker()
    grid.uz_jacs = True
    grid.build('temp_pod')

    # Add build directory to sys.path for CFFI module import
    sys.path.insert(0, os.path.abspath(build_dir))

    model = Model('temp_pod', matrices_folder=build_dir)
    model.Dt = 0.001
    model.decimation = 1

    model.ini({'K_theta_pll_1': 10}, os.path.join(module_dir, 'temp_pod_xy_0.json'))
    model.report_x()
    model.report_y()
    model.report_z()
    model.report_params()

    model.run(1.0, {})
    model.run(10, {'v_ref_1': 1.05})

    model.post()

    fig, axes = plt.subplots()
    axes.plot(model.Time, model.get_values('omega_1'))
    axes.plot(model.Time, model.get_values('omega_pll_1'))
    axes.grid()
    fig.savefig(os.path.join(build_dir, 'pod_2wo_3ll_omegas.svg'))

    # CasADi path
    grid_c = BpsBuilder(hjson_path, use_casadi=True)
    grid_c.checker()
    grid_c.construct('temp_pod_casadi')

    bld_c = CasadiBuilder(grid_c.sys_dict)
    bld_c.build()

    model_c = CasadiModel(bld_c)
    model_c.Dt = 0.001
    model_c.decimation = 1

    model_c.ini({'K_theta_pll_1': 10}, os.path.join(module_dir, 'temp_pod_casadi_xy_0.json'))
    model_c.run(1.0, {})
    model_c.run(10, {'v_ref_1': 1.05})
    model_c.post()

    fig2, axes2 = plt.subplots()
    axes2.plot(model_c.Time, model_c.get_values('omega_1'))
    axes2.plot(model_c.Time, model_c.get_values('omega_pll_1'))
    axes2.grid()
    fig2.savefig(os.path.join(build_dir, 'pod_2wo_3ll_casadi_omegas.svg'))


if __name__ == '__main__':
    test()
