# tests/bps/test_milano_models.py
"""
Automated tests for pydae-bps Milano synchronous machine models.
Tests milano2ord, milano3ord, milano4ord, and milano6ord for steady-state coherence.

Run selectively:
    uv run pytest -m bps                              # all bps tests
    uv run pytest tests/bps/ -k "milano4ord"          # only 4th order
    uv run pytest tests/bps/test_milano_models.py -v  # verbose
"""

import os
import sys
import json
import shutil
import platform
import pytest


def make_milano_2bus(model_type):
    """Creates a 2-bus power system dict for a given Milano model type."""
    return {
        "system": {
            "name": f"test_{model_type}",
            "S_base": 100e6,
            "K_p_agc": 0.0, "K_i_agc": 0.0, "K_xif": 0.01,
        },
        "buses": [
            {"name": "1", "P_W": 0.0, "Q_var": 0.0, "U_kV": 20.0},
            {"name": "2", "P_W": 0.0, "Q_var": 0.0, "U_kV": 20.0},
        ],
        "lines": [
            {"bus_j": "1", "bus_k": "2", "X_pu": 0.05, "R_pu": 0.01,
             "Bs_pu": 1e-6, "S_mva": 200},
        ],
        "shunts": [],
        "syns": [
            {
                "bus": "1",
                "type": model_type,
                "S_n": 200e6, "F_n": 50.0,
                "X_d": 1.8, "X_q": 1.7, "X_l": 0.2,
                "X1d": 0.3, "X1q": 0.55,
                "X2d": 0.2, "X2q": 0.25,
                "T1d0": 8.0, "T1q0": 0.4,
                "T2d0": 0.03, "T2q0": 0.05,
                "T_AA": 0.0,
                "R_a": 0.01,
                "H": 5.0, "D": 0.0,
                "S_10": 0.0, "S_12": 0.0,
                "K_delta": 0.0, "K_sec": 0.0,
            }
        ],
        "sources": [{"type": "vsource", "bus": "2"}],
    }


@pytest.fixture
def work_dir(tmp_path, monkeypatch):
    """Run each test in an isolated temporary directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.bps
@pytest.mark.skipif(
    platform.system() != "Linux",
    reason="ctypes heap corruption on macOS/Windows with Python 3.13",
)
@pytest.mark.parametrize("model_type", [
    "milano2ord",
    "milano3ord",
    "milano4ord",
    "milano6ord",
])
def test_milano_steady_state(model_type, work_dir):
    """
    Builds a 2-bus system with the given Milano model, initializes,
    runs a 1s steady-state simulation, and checks physical validity.
    """
    from pydae.bps import BpsBuilder
    from pydae.core import Builder, Model

    sys_name = f"test_{model_type}"
    file_name = str(work_dir / f"{sys_name}.json")

    # Write the grid definition
    with open(file_name, 'w') as f:
        json.dump(make_milano_2bus(model_type), f, indent=4)

    # Build the DAE
    grid = BpsBuilder(file_name)
    grid.checker()
    grid.uz_jacs = False
    grid.construct(sys_name)

    bld = Builder(grid.sys_dict, target='ctypes')
    bld.build()

    # Initialize
    model = Model(sys_name)

    if model_type == "milano2ord":
        inputs = {'p_m_1': 0.5, 'e1q_1': 1.2}
    else:
        inputs = {'p_m_1': 0.5, 'v_f_1': 1.2}

    success = model.ini(inputs, xy_0=1)
    assert success, f"{model_type}: initialization failed"

    # Run 1 second of steady-state
    model.run(1.0, {})
    model.post()

    # Assertions
    omega = model.get_value('omega_1')
    v_gen = model.get_value('V_1')
    p_e = model.get_value('p_e_1')
    p_g = model.get_value('p_g_1')
    q_g = model.get_value('q_g_1')

    # Rotor speed: exactly 1.0 pu in steady state
    assert omega == pytest.approx(1.0, abs=1e-4), \
        f"{model_type}: omega = {omega}"

    # Terminal voltage: within ±10% of nominal
    assert 0.9 <= v_gen <= 1.1, \
        f"{model_type}: V_1 = {v_gen}"

    # Electrical power: matches mechanical input (0.5 pu) within stator losses
    assert p_e == pytest.approx(0.5, abs=0.025), \
        f"{model_type}: p_e = {p_e}"

    # Grid injection: matches electrical power
    assert p_g == pytest.approx(p_e, abs=0.1), \
        f"{model_type}: p_g={p_g} vs p_e={p_e}"

    # Reactive power: physically reasonable
    assert -1.0 <= q_g <= 1.0, \
        f"{model_type}: q_g = {q_g}"
