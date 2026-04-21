# tests/core/test_compilation_modes.py
import pytest
import numpy as np
import sympy as sym
from pydae.core import Builder, Model
import os

@pytest.fixture
def pendulum_sys():
    """A minimal pendulum DAE for compilation tests."""
    L, G, M, K_d = sym.symbols("L,G,M,K_d", real=True)
    p_x, p_y, v_x, v_y = sym.symbols("p_x,p_y,v_x,v_y", real=True)
    lam, f_x, theta = sym.symbols("lam,f_x,theta", real=True)
    return {
        "name": "test_compilation",
        "params_dict": {"L": 5.21, "G": 9.81, "M": 10.0, "K_d": 1e-3},
        "f_list": [v_x, v_y, (-2 * p_x * lam + f_x - K_d * v_x) / M, (-M * G - 2 * p_y * lam - K_d * v_y) / M],
        "g_list": [p_x ** 2 + p_y ** 2 - L ** 2 - lam * 1e-6, -theta + sym.atan2(p_x, -p_y)],
        "x_list": [p_x, p_y, v_x, v_y],
        "y_ini_list": [lam, f_x],
        "y_run_list": [lam, theta],
        "u_ini_dict": {"theta": np.deg2rad(5.0)},
        "u_run_dict": {"f_x": 0},
        "h_dict": {"theta": theta},
    }

@pytest.mark.build
@pytest.mark.parametrize("target", ["cffi", "ctypes"])
@pytest.mark.parametrize("sparse", [False, "klu"])
def test_compilation_and_execution(pendulum_sys, target, sparse):
    # 1. Build
    bld = Builder(pendulum_sys, target=target, sparse=sparse)
    bld.build()
    
    # 2. Simulate
    model = Model(pendulum_sys["name"])
    
    # Ini
    success = model.ini(
        {"theta": np.deg2rad(30.0)},
        xy_0={"p_x": 0.9, "p_y": -5.1, "lam": 0.0, "f_x": 1.0},
    )
    assert success, f"Initialization failed for {target} sparse={sparse}"
    
    # SSA report (requires populated jacobians)
    model.A_eval()
    from pydae.ssa import damp
    # Input to ssa.damp() is a dense matrix A
    damp(model.A)
    assert model.A.shape == (4, 4)
    
    # Run
    model.run(0.1, {"f_x": 0.1})
    model.post()
    
    assert len(model.Time) > 1
    assert not np.isnan(model.get_values("theta")).any()
