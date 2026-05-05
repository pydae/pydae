"""
Pytest coverage for pydae-bps BESS VSC models.

Models tested:
- bess_pq_ss: BESS with state-space filtered PQ references.
- bess_dq_vrt: BESS with VRT capability and soft current saturation.

Tests:
- Build: DAE construction and compilation.
- Init: Newton solve at steady-state operating point.
- Run: Time-domain simulation with power reference steps.
- SSA: Small-signal eigenvalue analysis with the coupled state-space filter.

Run:
    uv run pytest tests/bps/vscs/
    uv run pytest -m bps -k bess
"""
import json
import subprocess
import sys
import textwrap

import pytest


def _bess_params(with_filter):
    """Parameter block for the bess_pq_ss inverter."""
    base = {
        "bus": "1",
        "type": "bess_pq_ss",
        "S_n": 2.5e6,
        "E_kWh": 5016.98,
        "K_delta": 0.01,
        "soc_ref": 0.5,
        "soc_min": 0.0,
        "soc_max": 1.0,
        "R_bat": 0.0,
        "A_loss": 0.0001,
        "B_loss": 0.0,
        "C_loss": 0.0001,
        "K_p": 1e-6,
        "K_i": 1e-6,
        "e_soc_order": 1,
        "socs": [
            0.000, 0.050, 0.100, 0.150, 0.200, 0.300, 0.400,
            0.500, 0.600, 0.700, 0.800, 0.850, 0.900, 0.950, 1.000,
        ],
        "es": [
            0.78, 0.954, 0.960, 0.968, 0.975, 0.986, 0.986,
            0.987, 0.997, 0.998, 0.998, 0.998, 0.998, 0.999, 1.011,
        ],
    }
    if with_filter:
        base["A_pq"] = [[-20.0, 0.0], [0.0, -20.0]]
        base["B_pq"] = [[20.0, 0.0], [0.0, 20.0]]
        base["C_pq"] = [[1.0, 0.0], [0.0, 1.0]]
        base["D_pq"] = [[0.0, 0.0], [0.0, 0.0]]
    return base


def _grid_dict(with_filter):
    """Two-bus grid: BESS at bus 1, vsource at bus 2."""
    return {
        "system": {
            "name": "test_bess_pq_ss",
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
        "vscs": [_bess_params(with_filter)],
        "sources": [{"type": "vsource", "bus": "2", "K_delta": 0.001}],
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

    bld = Builder(grid.sys_dict, target="ctypes", sparse=False)
    bld.build()

    model = Model(sys_name)
    inputs = {inputs!r}

    success = model.ini(inputs, xy_0=1)
    assert success, "initialisation failed"

    V = model.get_value('V_1')
    p_s = model.get_value('p_s_1')
    q_s = model.get_value('q_s_1')
    soc = model.get_value('soc_1')
    v_dc = model.get_value('v_dc_1')

    assert 0.9 <= V <= 1.1, f"V_1={{V}}"
    assert v_dc > 0, f"v_dc_1={{v_dc}} not positive"
    assert 0.0 < soc < 1.0, f"soc_1={{soc}} out of bounds"

    model.run(1.0, {{}})
    model.run(5.0, {step_inputs!r})
    model.post()

    p_s_final = model.get_values('p_s_1')[-1]
    soc_final = model.get_values('soc_1')[-1]
    assert p_s_final > 0, f"p_s_1 did not converge: {{p_s_final}}"
    assert soc_final > 0, f"soc_1 dropped to zero: {{soc_final}}"

    print("RUNNER_OK")
""")

SSA_RUNNER_TEMPLATE = textwrap.dedent("""
    import json
    import numpy as np
    from pydae.bps import BpsBuilder
    from pydae.core import Builder, Model
    from pydae.ssa import A_eval

    sys_name = {sys_name!r}
    grid = BpsBuilder({grid_file!r})
    grid.checker()
    grid.uz_jacs = False
    grid.construct(sys_name)

    bld = Builder(grid.sys_dict, target="ctypes", sparse=False)
    bld.build()

    model = Model(sys_name)
    success = model.ini({inputs!r}, xy_0=1)
    assert success, "initialisation failed"

    A = A_eval(model)
    N_x = len(model.x_list)
    assert A.shape == (N_x, N_x), f"A shape {{A.shape}} != ({{N_x}}, {{N_x}})"
    assert np.isfinite(A).all(), "A has non-finite entries"
    assert hasattr(model, 'A'), "A_eval did not set model.A"

    eig = np.linalg.eigvals(A)
    assert np.all(eig.real < 1e-6), f"unstable mode in eig={{eig}}"

    payload = dict(N_x=N_x,
                   eig_real=[float(v) for v in eig.real],
                   eig_imag=[float(v) for v in eig.imag],
                   x_list=list(model.x_list))
    with open({marker_file!r}, 'w') as _fh:
        json.dump(payload, _fh)

    try:
        from pydae.ssa import damp_report
        damp_report(model)
    except Exception:
        pass

    print("RUNNER_OK")
""")


@pytest.fixture
def work_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


CASES = [
    ("bare", False),
    ("with_filter", True),
]


@pytest.mark.bps
@pytest.mark.parametrize("config,with_filter", CASES)
def test_bess_pq_ss_model(config, with_filter, work_dir):
    """Builds, inits, and runs the bess_pq_ss inverter (bare or with
    state-space filter) in a fresh Python subprocess so ctypes state
    stays isolated."""
    sys_name = f"test_bess_pq_ss_{config}"
    grid = _grid_dict(with_filter)
    grid["system"]["name"] = sys_name

    grid_file = work_dir / f"{sys_name}.json"
    grid_file.write_text(json.dumps(grid, indent=2))

    step_inputs = {"p_s_ppc_1": 0.5, "q_s_ppc_1": 0.1}
    runner = work_dir / "runner.py"
    runner.write_text(RUNNER_TEMPLATE.format(
        sys_name=sys_name,
        grid_file=str(grid_file).replace("\\", "/"),
        inputs={"soc_ref_1": 0.5, "p_s_ppc_1": 0.0, "q_s_ppc_1": 0.0},
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


@pytest.mark.bps
def test_bess_pq_ss_ssa(work_dir):
    """Small-signal analysis on bess_pq_ss with coupled state-space filter:
    eval the reduced-state matrix A at the ini operating point and verify
    all modes are stable."""
    sys_name = "test_bess_pq_ss_ssa"
    grid = _grid_dict(with_filter=True)
    grid["system"]["name"] = sys_name

    grid_file = work_dir / f"{sys_name}.json"
    grid_file.write_text(json.dumps(grid, indent=2))

    marker_file = work_dir / "ssa_facts.json"
    runner = work_dir / "runner.py"
    runner.write_text(SSA_RUNNER_TEMPLATE.format(
        sys_name=sys_name,
        grid_file=str(grid_file).replace("\\", "/"),
        inputs={"soc_ref_1": 0.5, "p_s_ppc_1": 0.0, "q_s_ppc_1": 0.0},
        marker_file=str(marker_file).replace("\\", "/"),
    ), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(runner)],
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert marker_file.exists(), (
        f"{sys_name} SSA runner did not produce facts file\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    facts = json.loads(marker_file.read_text())
    assert facts["N_x"] == len(facts["x_list"])
    assert all(r < 1e-6 for r in facts["eig_real"]), (
        f"unstable mode in {facts}"
    )
    # With the 2-state filter + 2 battery states we should have >= 4 states.
    assert facts["N_x"] >= 4, f"expected >= 4 states, got {facts['N_x']}"


# =============================================================================
# bess_dq_vrt tests
# =============================================================================
def _bess_vrt_params(with_lvrt):
    """Parameter block for the bess_dq_vrt inverter."""
    base = {
        "bus": "1",
        "type": "bess_dq_vrt",
        "S_n": 2.5e6,
        "E_kWh": 5016.98,
        "K_delta": 0.01,
        "soc_ref": 0.5,
        "soc_min": 0.0,
        "soc_max": 1.0,
        "R_bat": 0.0,
        "A_loss": 0.0001,
        "B_loss": 0.0,
        "C_loss": 0.0001,
        "K_p": 1e-6,
        "K_i": 1e-6,
        "I_max": 1.2,
        "T_lvrt": 0.02,
        "Epsilon": 1e-8,
        "e_soc_order": 1,
        "socs": [
            0.000, 0.050, 0.100, 0.150, 0.200, 0.300, 0.400,
            0.500, 0.600, 0.700, 0.800, 0.850, 0.900, 0.950, 1.000,
        ],
        "es": [
            0.78, 0.954, 0.960, 0.968, 0.975, 0.986, 0.986,
            0.987, 0.997, 0.998, 0.998, 0.998, 0.998, 0.999, 1.011,
        ],
    }
    if with_lvrt:
        base["lvrt_ext"] = 1.0
    return base


def _vrt_grid_dict(with_lvrt):
    """Two-bus grid: BESS VRT at bus 1, vsource at bus 2."""
    return {
        "system": {
            "name": "test_bess_dq_vrt",
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
        "vscs": [_bess_vrt_params(with_lvrt)],
        "sources": [{"type": "vsource", "bus": "2", "K_delta": 0.001}],
    }


VRT_RUNNER_TEMPLATE = textwrap.dedent("""
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

    V = model.get_value('V_1')
    p_s = model.get_value('p_s_1')
    soc = model.get_value('soc_1')
    v_dc = model.get_value('v_dc_1')

    assert 0.9 <= V <= 1.1, f"V_1={{V}}"
    assert v_dc > 0, f"v_dc_1={{v_dc}} not positive"
    assert 0.0 < soc < 1.0, f"soc_1={{soc}} out of bounds"

    model.run(1.0, {{}})
    model.run(5.0, {step_inputs!r})
    model.post()

    p_s_final = model.get_values('p_s_1')[-1]
    soc_final = model.get_values('soc_1')[-1]
    assert p_s_final > 0, f"p_s_1 did not converge: {{p_s_final}}"
    assert soc_final > 0, f"soc_1 dropped to zero: {{soc_final}}"

    print("RUNNER_OK")
""")

VRT_SSA_RUNNER_TEMPLATE = textwrap.dedent("""
    import json
    import numpy as np
    from pydae.bps import BpsBuilder
    from pydae.core import Builder, Model
    from pydae.ssa import A_eval

    sys_name = {sys_name!r}
    grid = BpsBuilder({grid_file!r})
    grid.checker()
    grid.uz_jacs = False
    grid.construct(sys_name)

    bld = Builder(grid.sys_dict, target="ctypes", sparse=False)
    bld.build()

    model = Model(sys_name)
    success = model.ini({inputs!r}, xy_0=1)
    assert success, "initialisation failed"

    A = A_eval(model)
    N_x = len(model.x_list)
    assert A.shape == (N_x, N_x), f"A shape {{A.shape}} != ({{N_x}}, {{N_x}})"
    assert np.isfinite(A).all(), "A has non-finite entries"
    assert hasattr(model, 'A'), "A_eval did not set model.A"

    eig = np.linalg.eigvals(A)
    assert np.all(eig.real < 1e-6), f"unstable mode in eig={{eig}}"

    payload = dict(N_x=N_x,
                   eig_real=[float(v) for v in eig.real],
                   eig_imag=[float(v) for v in eig.imag],
                   x_list=list(model.x_list))
    with open({marker_file!r}, 'w') as _fh:
        json.dump(payload, _fh)

    try:
        from pydae.ssa import damp_report
        damp_report(model)
    except Exception:
        pass

    print("RUNNER_OK")
""")


VRT_CASES = [
    ("pq_mode", False),
    ("lvrt_mode", True),
]


@pytest.mark.bps
@pytest.mark.parametrize("config,with_lvrt", VRT_CASES)
def test_bess_dq_vrt_model(config, with_lvrt, work_dir):
    """Builds, inits, and runs the bess_dq_vrt inverter (PQ or LVRT mode)
    in a fresh Python subprocess so ctypes state stays isolated."""
    sys_name = f"test_bess_dq_vrt_{config}"
    grid = _vrt_grid_dict(with_lvrt)
    grid["system"]["name"] = sys_name

    grid_file = work_dir / f"{sys_name}.json"
    grid_file.write_text(json.dumps(grid, indent=2))

    step_inputs = {"p_s_ppc_1": 0.5, "q_s_ppc_1": 0.1}
    runner = work_dir / "runner.py"
    runner.write_text(VRT_RUNNER_TEMPLATE.format(
        sys_name=sys_name,
        grid_file=str(grid_file).replace("\\", "/"),
        inputs={"soc_ref_1": 0.5, "p_s_ppc_1": 0.0, "q_s_ppc_1": 0.0},
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


@pytest.mark.bps
def test_bess_dq_vrt_ssa(work_dir):
    """Small-signal analysis on bess_dq_vrt:
    eval the reduced-state matrix A at the ini operating point and verify
    all modes are stable."""
    sys_name = "test_bess_dq_vrt_ssa"
    grid = _vrt_grid_dict(with_lvrt=False)
    grid["system"]["name"] = sys_name

    grid_file = work_dir / f"{sys_name}.json"
    grid_file.write_text(json.dumps(grid, indent=2))

    marker_file = work_dir / "ssa_facts_vrt.json"
    runner = work_dir / "runner.py"
    runner.write_text(VRT_SSA_RUNNER_TEMPLATE.format(
        sys_name=sys_name,
        grid_file=str(grid_file).replace("\\", "/"),
        inputs={"soc_ref_1": 0.5, "p_s_ppc_1": 0.0, "q_s_ppc_1": 0.0},
        marker_file=str(marker_file).replace("\\", "/"),
    ), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(runner)],
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert marker_file.exists(), (
        f"{sys_name} SSA runner did not produce facts file\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    facts = json.loads(marker_file.read_text())
    assert facts["N_x"] == len(facts["x_list"])
    assert all(r < 1e-6 for r in facts["eig_real"]), (
        f"unstable mode in {facts}"
    )
    # With 2 battery states + 1 LVRT ramp state we should have >= 3 states.
    assert facts["N_x"] >= 3, f"expected >= 3 states, got {facts['N_x']}"
