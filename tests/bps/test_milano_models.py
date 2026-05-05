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
import json
import subprocess
import sys
import textwrap

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


RUNNER_TEMPLATE = textwrap.dedent("""
    import json
    import pytest
    from pydae.bps import BpsBuilder
    from pydae.core import Builder, Model

    sys_name = {sys_name!r}
    grid = BpsBuilder({grid_file!r})
    grid.checker()
    grid.uz_jacs = False
    grid.construct(sys_name)

    bld = Builder(grid.sys_dict, target="cffi", sparse=False)
    bld.build()

    model = Model(sys_name)
    model_type = {model_type!r}
    if model_type == "milano2ord":
        inputs = {{"p_m_1": 0.5, "e1q_1": 1.2}}
    else:
        inputs = {{"p_m_1": 0.5, "v_f_1": 1.2}}

    success = model.ini(inputs, xy_0=1.0)
    assert success, "initialisation failed"

    omega = model.get_value('omega_1')
    v_gen = model.get_value('V_1')
    p_e = model.get_value('p_e_1')
    p_g = model.get_value('p_g_1')
    q_g = model.get_value('q_g_1')

    assert omega == pytest.approx(1.0, abs=1e-4), f"omega={{omega}}"
    assert 0.9 <= v_gen <= 1.1, f"V_1={{v_gen}}"
    assert p_e == pytest.approx(0.5, abs=0.025), f"p_e={{p_e}}"
    assert p_g == pytest.approx(p_e, abs=0.1), f"p_g={{p_g}} vs p_e={{p_e}}"
    assert -1.0 <= q_g <= 1.0, f"q_g={{q_g}}"

    model.run(1.0, {{}})
    model.post()
    print("RUNNER_OK")
""")


@pytest.mark.bps
@pytest.mark.parametrize("model_type", [
    "milano2ord",
    "milano3ord",
    "milano4ord",
    "milano6ord",
])
def test_milano_steady_state(model_type, work_dir):
    """Builds, inits, and runs a Milano machine in a fresh subprocess."""
    sys_name = f"test_{model_type}"
    grid_file = work_dir / f"{sys_name}.json"
    grid_file.write_text(json.dumps(make_milano_2bus(model_type), indent=2))

    runner = work_dir / "runner.py"
    runner.write_text(RUNNER_TEMPLATE.format(
        sys_name=sys_name,
        grid_file=str(grid_file).replace('\\', '/'),
        model_type=model_type,
    ))

    result = subprocess.run(
        [sys.executable, str(runner)],
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert "RUNNER_OK" in result.stdout, (
        f"{sys_name} runner did not reach completion\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
