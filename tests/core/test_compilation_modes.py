# tests/core/test_compilation_modes.py
import shutil
import pytest
import numpy as np
import sympy as sym
from pydae.core import Builder, Model
import os


def _pardiso_available():
    try:
        import ctypes
        for lib in ['libmkl_rt.so', 'libmkl_rt.dylib', 'mkl_rt.dll']:
            try:
                ctypes.CDLL(lib)
                return True
            except OSError:
                continue
        return False
    except Exception:
        return False

PARDISO = pytest.mark.skipif(
    not _pardiso_available(),
    reason="PARDISO requires Intel MKL not installed"
)


@pytest.fixture
def pendulum_sys(tmp_path):
    """A minimal pendulum DAE for compilation tests."""
    # Build in isolated temp directory to avoid conflicts
    orig_cwd = os.getcwd()
    os.chdir(tmp_path)

    L, G, M, K_d = sym.symbols("L,G,M,K_d", real=True)
    p_x, p_y, v_x, v_y = sym.symbols("p_x,p_y,v_x,v_y", real=True)
    lam, f_x, theta = sym.symbols("lam,f_x,theta", real=True)
    sys_dict = {
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

    yield sys_dict
    os.chdir(orig_cwd)

@pytest.mark.build
@pytest.mark.parametrize("target,sparse", [
    ("cffi", False),
    ("cffi", "klu"),
    ("ctypes", False),
    ("ctypes", "klu"),
])
def test_compilation_and_execution(pendulum_sys, target, sparse):
    """Test dense and KLU sparse backends."""
    # 1. Build
    bld = Builder(pendulum_sys, target=target, sparse=sparse)
    bld.build()
    
    # 2. Simulate
    model = Model(pendulum_sys["name"])
    
    # Ini
    # Physically stable initial guess:
    # - L = 5.21, theta = 30 deg: p_x = L*sin(30°) = 2.605, p_y = -L*cos(30°) = -4.516
    # - lam (tension) MUST NOT be 0.0 — causes singular Jacobian blocks (dFx/dlam = -p_x).
    #   Use rough tension estimate: T ≈ M*G/cos(30°) ≈ 113, or conservative 10.0.
    # - f_x = 1.0 keeps the pendulum moving slightly
    success = model.ini(
        {"theta": np.deg2rad(30.0)},
        xy_0={"p_x": 2.605, "p_y": -4.516, "lam": 10.0, "f_x": 1.0},
    )
    assert success, f"Initialization failed for {target} sparse={sparse}"
    
    # SSA report (requires populated jacobians)
    model.A_eval()
    from pydae.ssa import damp
    damp(model.A)
    assert model.A.shape == (4, 4)
    
    # Run
    model.run(0.1, {"f_x": 0.1})
    model.post()
    
    assert len(model.Time) > 1
    assert not np.isnan(model.get_values("theta")).any()


@pytest.mark.build
@pytest.mark.skipif(not _pardiso_available(), reason="PARDISO requires Intel MKL")
@pytest.mark.parametrize("target", ["cffi", "ctypes"])
def test_pardiso_compilation(pendulum_sys, target):
    """Test PARDISO sparse backend when Intel MKL is available."""
    _run_compilation_test(pendulum_sys, target, "pardiso")


def _run_compilation_test(pendulum_sys, target, sparse):
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
