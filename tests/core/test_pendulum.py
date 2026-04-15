# tests/core/test_pendulum.py
"""
Smoke test: build and initialize the pendulum DAE model.
This verifies the full pipeline: parse → symbolics → codegen → compile → solve.
"""
import numpy as np
import sympy as sym
from pydae.core import Builder, Model


def make_pendulum_system(name='test_pendulum'):
    """Creates the pendulum DAE system dictionary."""
    L, G, M, K_d, K_lam = sym.symbols('L,G,M,K_d,K_lam', real=True)
    p_x, p_y, v_x, v_y = sym.symbols('p_x,p_y,v_x,v_y', real=True)
    lam, f_x, theta, u_dummy = sym.symbols('lam,f_x,theta,u_dummy', real=True)

    dp_x = v_x
    dp_y = v_y
    dv_x = (-2 * p_x * lam + f_x - K_d * v_x) / M
    dv_y = (-M * G - 2 * p_y * lam - K_d * v_y) / M

    g_1 = p_x**2 + p_y**2 - L**2 - lam * K_lam
    g_2 = -theta + sym.atan2(p_x, -p_y) + u_dummy

    return {
        'name': name,
        'params_dict': {'L': 5.21, 'G': 9.81, 'M': 10.0, 'K_d': 1e-3, 'K_lam': 1e-6},
        'f_list': [dp_x, dp_y, dv_x, dv_y],
        'g_list': [g_1, g_2],
        'x_list': [p_x, p_y, v_x, v_y],
        'y_ini_list': [lam, f_x],
        'y_run_list': [lam, theta],
        'u_ini_dict': {'theta': np.deg2rad(5.0), 'u_dummy': 0.0},
        'u_run_dict': {'f_x': 0, 'u_dummy': 0.0},
        'h_dict': {
            'E_p': M * G * (p_y + L),
            'E_k': 0.5 * M * (v_x**2 + v_y**2),
            'f_x': f_x,
            'lam': lam,
        },
    }


def test_build_pendulum():
    """Test that the pendulum model builds without errors."""
    sys_dict = make_pendulum_system()
    bld = Builder(sys_dict, target='ctypes')
    bld.build()


def test_ini_pendulum():
    """Test that the pendulum model initializes successfully."""
    sys_dict = make_pendulum_system()
    bld = Builder(sys_dict, target='ctypes')
    bld.build()

    model = Model('test_pendulum')
    deg = 10
    L = 5.21
    p_x_0 = L * np.sin(np.deg2rad(deg))
    p_y_0 = -L * np.cos(np.deg2rad(deg))

    success = model.ini(
        {'M': 30.0, 'L': L, 'K_lam': 1e-6, 'theta': np.deg2rad(deg), 'K_d': 0.0},
        xy_0={'p_x': p_x_0, 'p_y': p_y_0, 'lam': 0, 'f_x': 1},
    )
    assert success, "Pendulum initialization failed"
    assert model.ini_iterations < 50, f"Too many iterations: {model.ini_iterations}"
