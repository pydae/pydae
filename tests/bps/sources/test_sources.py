"""
Pytest coverage for pydae-bps vsource (ideal voltage source).

Tests:
- Build: DAE construction and compilation.
- Init: Newton solve at steady-state operating point.
- Run: Time-domain simulation with voltage step.

Run:
    uv run pytest tests/bps/sources/
    uv run pytest -m bps -k vsource
"""
import json
import subprocess
import sys
import textwrap

import pytest


def _grid_dict():
    """Two-bus grid: vsc_pq at bus 1, vsource at bus 2."""
    return {
        "system": {
            "name": "test_vsource",
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
        "vscs": [
            {"bus": "1", "type": "vsc_pq", "p_in": 0.5, "S_n": 10e6,
             "K_delta": 0.0},
        ],
        "sources": [{"type": "vsource", "bus": "2"}],
    }


RUNNER_TEMPLATE = textwrap.dedent("""
    import json
    from pydae.bps import BpsBuilder
    from pydae.core import Builder, Model

    sys_name = {sys_name!r}
    grid = BpsBuilder({grid_file!r})
    grid.checker()
    grid.uz_jacs = False
    grid.construct(sys_name)

    bld = Builder(grid.sys_dict, target="ctypes", sparse=False)
    bld.build()

    model = Model(sys_name)
    inputs = {inputs!r}

    success = model.ini(inputs, xy_0=1)
    assert success, "initialisation failed"

    V_2 = model.get_value('V_2')
    theta_2 = model.get_value('theta_2')

    assert abs(V_2 - 1.0) < 1e-4, f"V_2={{V_2}} not pinned to 1.0"
    assert abs(theta_2 - 0.0) < 1e-4, f"theta_2={{theta_2}} not pinned to 0.0"

    model.run(1.0, {{}})
    model.run(5.0, {step_inputs!r})
    model.post()

    V_2_final = model.get_values('V_2')[-1]
    assert abs(V_2_final - 1.05) < 1e-4, f"V_2={{V_2_final}} after step"

    print("RUNNER_OK")
""")


@pytest.fixture
def work_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.bps
def test_vsource_model(work_dir):
    """Builds, inits, and runs the vsource in a fresh Python subprocess."""
    sys_name = "test_vsource_model"
    grid = _grid_dict()
    grid["system"]["name"] = sys_name

    grid_file = work_dir / f"{sys_name}.json"
    grid_file.write_text(json.dumps(grid, indent=2))

    step_inputs = {"v_ref_2": 1.05}
    runner = work_dir / "runner.py"
    runner.write_text(RUNNER_TEMPLATE.format(
        sys_name=sys_name,
        grid_file=str(grid_file).replace("\\", "/"),
        inputs={"v_ref_2": 1.0, "theta_ref_2": 0.0},
        step_inputs=step_inputs,
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
