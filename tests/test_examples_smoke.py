"""
Smoke tests mirroring the examples/ scripts: build, ini, run.
Assertions are on the final-time values of a few state variables.

If running all three at once hits a Windows DLL/heap issue, run them
one at a time:

    uv run pytest tests/test_examples_smoke.py::test_pendulum
    uv run pytest tests/test_examples_smoke.py::test_milano2ord
    uv run pytest tests/test_examples_smoke.py::test_milano4ord
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
import sympy as sym

from pydae.core import Builder, Model

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def here(tmp_path, monkeypatch):
    """Run each test from a clean tmp cwd and copy example fixtures in."""
    for p in EXAMPLES.glob("*"):
        if p.suffix in {".hjson", ".json"}:
            shutil.copy(p, tmp_path / p.name)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_pendulum(here):
    L, G, M, K_d = sym.symbols("L,G,M,K_d", real=True)
    p_x, p_y, v_x, v_y = sym.symbols("p_x,p_y,v_x,v_y", real=True)
    lam, f_x, theta = sym.symbols("lam,f_x,theta", real=True)

    sys_dict = {
        "name": "pendulum",
        "params_dict": {"L": 5.21, "G": 9.81, "M": 10.0, "K_d": 1e-3},
        "f_list": [v_x,
                   v_y,
                   (-2 * p_x * lam + f_x - K_d * v_x) / M,
                   (-M * G - 2 * p_y * lam - K_d * v_y) / M],
        "g_list": [p_x**2 + p_y**2 - L**2 - lam * 1e-6,
                   -theta + sym.atan2(p_x, -p_y)],
        "x_list": [p_x, p_y, v_x, v_y],
        "y_ini_list": [lam, f_x],
        "y_run_list": [lam, theta],
        "u_ini_dict": {"theta": np.deg2rad(5.0)},
        "u_run_dict": {"f_x": 0},
        "h_dict": {"E_p": M * G * (p_y + L),
                   "E_k": 0.5 * M * (v_x**2 + v_y**2)},
    }
    Builder(sys_dict, target="ctypes", sparse=False).build()

    model = Model("pendulum")
    model.ini({"theta": np.deg2rad(30), "K_d": 10},
              xy_0={"p_x": 0.9, "p_y": -5.1, "lam": 0, "f_x": 1})
    model.run(1.0, {})
    model.run(20.0, {"f_x": 0.0})
    model.post()

    theta_t = model.get_values("theta")
    assert np.all(np.isfinite(theta_t))
    # With K_d=10 damping over 21 s, theta should settle near 0
    assert abs(theta_t[-1]) < np.deg2rad(5)


def test_milano2ord(here):
    from pydae.bps import BpsBuilder

    grid = BpsBuilder("milano2ord.hjson")
    grid.checker()
    grid.uz_jacs = False
    grid.construct("milano2ord")
    Builder(grid.sys_dict, target="ctypes", sparse=False).build()

    model = Model("milano2ord")
    model.ini({"p_m_1": 0.5, "e1q_1": 1.5}, "xy_0.json")
    model.run(0.1, {})
    model.run(1.0, {"p_m_1": 1.0, "D_1": 20.0})
    model.post()

    omega = model.get_values("omega_1")
    assert np.all(np.isfinite(omega))
    assert 0.8 < omega[-1] < 1.2


def test_milano4ord(here):
    from pydae.bps import BpsBuilder

    grid = BpsBuilder("milano4ord.hjson")
    grid.checker()
    grid.uz_jacs = False
    grid.construct("temp_m4")
    Builder(grid.sys_dict, target="ctypes", sparse=False).build()

    model = Model("temp_m4")
    model.ini({"p_m_1": 0.5, "v_ref_1": 1.0}, "xy_0.json")
    model.run(1.0, {})
    model.run(10.0, {"p_m_1": 1.0, "v_ref_1": 1.05})
    model.post()

    omega = model.get_values("omega_1")
    e1q = model.get_values("e1q_1")
    assert np.all(np.isfinite(omega)) and np.all(np.isfinite(e1q))
    assert 0.8 < omega[-1] < 1.2
    assert e1q.max() - e1q.min() > 1e-3
