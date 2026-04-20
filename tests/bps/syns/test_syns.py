"""
Pytest coverage for pydae-bps Milano synchronous machine models.

Each machine order (2, 3, 4, 6) is tested in two configurations:

- **bare**: direct field-voltage / classical-EMF input, no controllers.
- **with controllers**: sexs AVR + ieeeg1 governor + pss_kundur_1 PSS
  attached to the machine. milano2ord is skipped in this case because
  the classical 2nd-order model has no v_f input for an AVR.

Each test writes a runner script to a tmp dir and invokes it in a
fresh Python subprocess so that CFFI shared-library state does not
leak between test cases.

Run:
    uv run pytest tests/bps/syns/
    uv run pytest -m bps -k syns
"""
from pathlib import Path
import json
import subprocess
import sys
import textwrap

import pytest


def _syn_params(model_type):
    """Common Milano-machine parameter block covering orders 2/3/4/6."""
    return {
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
        "K_delta": 0.001, "K_sec": 0.0,
        "Omega_b": 314.1592653589793, "omega_s": 1.0,
    }


def _grid_dict(model_type, with_controllers):
    """Two-bus grid: machine at bus 1, vsource at bus 2."""
    syn = _syn_params(model_type)
    if with_controllers:
        syn["avr"] = {
            "type": "sexs", "K_a": 200.0, "T_a": 0.015,
            "T_b": 10.0, "T_c": 1.0, "T_e": 0.1,
            "E_min": -5, "E_max": 5, "v_ref": 1.0,
        }
        syn["gov"] = {
            "type": "ieeeg1",
            "K": 20.0, "T_1": 0.0, "T_2": 0.0, "T_3": 0.1,
            "K_1": 0.3, "K_2": 0.0,
            "T_4": 0.3, "K_3": 0.3, "K_4": 0.0,
            "T_5": 7.0, "K_5": 0.4, "K_6": 0.0,
            "T_6": 0.6, "K_7": 0.0, "K_8": 0.0,
            "T_7": 0.0, "U_c": -0.5, "U_o": 0.5,
            "P_min": 0.0, "P_max": 2.0,
            "p_c": 0.5,
        }
        syn["pss"] = {
            "type": "pss_kundur_1",
            "K_stab": 20.0, "T_w": 10.0,
            "T_1": 0.05, "T_2": 0.02,
            "V_Smax": 0.1, "V_Smin": -0.1,
        }
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
        "syns": [syn],
        "sources": [{"type": "vsource", "bus": "2"}],
    }


def _ini_inputs(model_type, with_controllers):
    if with_controllers:
        return {"V_1": 1.0, "p_c_1": 0.5}
    if model_type == "milano2ord":
        return {"p_m_1": 0.5, "e1q_1": 1.2}
    return {"p_m_1": 0.5, "v_f_1": 1.5}


def _seed_overrides(with_controllers):
    """Non-unit initial guesses for controller states.

    The ieeeg1 servo-valve saturation block has a zero Jacobian diagonal
    at x_3_gov = 1.0 (inside limits, but antiwindup feedback cancels),
    so we seed each governor cascade state near the scheduled dispatch.
    """
    if not with_controllers:
        return {}
    return {
        "x_3_gov_1": 0.5, "x_4_gov_1": 0.5,
        "x_5_gov_1": 0.5, "x_6_gov_1": 0.5,
        "p_m_1": 0.5,
        "x_wo_1": 0.0, "x_lead_1": 0.0, "v_pss_1": 0.0,
    }


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
    inputs = {inputs!r}
    seed_overrides = {seed_overrides!r}

    # Seed xy_0: ones for every dynamic/algebraic var, then apply overrides
    # so controller states start near their steady-state values (the ieeeg1
    # saturation block has a zero Jacobian diagonal at the xy_0=1 trial guess).
    seed = {{name: 1.0 for name in model.x_list + model.y_ini_list}}
    seed.update(seed_overrides)
    success = model.ini(inputs, xy_0=seed)
    assert success, "initialisation failed"

    omega = model.get_value('omega_1')
    V = model.get_value('V_1')
    p_m = model.get_value('p_m_1')

    assert omega == pytest.approx(1.0, abs=1e-2), f"omega={{omega}}"
    assert 0.9 <= V <= 1.1, f"V_1={{V}}"
    assert p_m is not None, "p_m_1 not resolvable"

    model.run(1.0, {{}})
    model.post()
    print("RUNNER_OK")
""")


@pytest.fixture
def work_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


CASES = [
    ("milano2ord", False),
    ("milano3ord", False),
    ("milano4ord", False),
    ("milano6ord", False),
    pytest.param("milano2ord", True,
                 marks=pytest.mark.skip(
                     reason="milano2ord has no v_f input; AVR can't attach")),
    ("milano3ord", True),
    ("milano4ord", True),
    ("milano6ord", True),
]


@pytest.mark.bps
@pytest.mark.parametrize("model_type,with_controllers", CASES)
def test_syn_model(model_type, with_controllers, work_dir):
    """Builds, inits, and runs a Milano machine (bare or with full controller
    stack) in a fresh Python subprocess so CFFI state stays isolated."""
    sys_name = f"test_{model_type}_{'ctrls' if with_controllers else 'bare'}"
    grid = _grid_dict(model_type, with_controllers)
    grid["system"]["name"] = sys_name

    grid_file = work_dir / f"{sys_name}.json"
    grid_file.write_text(json.dumps(grid, indent=2))

    runner = work_dir / "runner.py"
    runner.write_text(RUNNER_TEMPLATE.format(
        sys_name=sys_name,
        grid_file=str(grid_file).replace('\\', '/'),
        inputs=_ini_inputs(model_type, with_controllers),
        seed_overrides=_seed_overrides(with_controllers),
    ))

    result = subprocess.run(
        [sys.executable, str(runner)],
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )

    # Windows CFFI teardown can emit a heap-corruption Traceback after the
    # runner has already finished. So ignore the exit code and the teardown
    # noise — success is the RUNNER_OK marker printed at the end of the
    # runner, after ini/run/post all completed without raising.
    assert "RUNNER_OK" in result.stdout, (
        f"{sys_name} runner did not reach completion\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
