# tests/core/test_pendulum.py
"""
Tests for the pydae.core pipeline using a pendulum DAE model.

Run selectively:
    uv run pytest -m parse          # only parser
    uv run pytest -m symbolic       # only Jacobian math
    uv run pytest -m codegen        # only C generation
    uv run pytest -m build          # full compile (needs gcc)
    uv run pytest -m model          # end-to-end simulation
    uv run pytest -k "test_parse"   # by name substring
"""
import os
import sys
import numpy as np
import sympy as sym
import pytest


# ─── Shared fixture ────────────────────────────────────────────────────

@pytest.fixture
def pendulum_sys():
    """The pendulum DAE system dictionary used across all tests."""
    L, G, M, K_d, K_lam = sym.symbols('L,G,M,K_d,K_lam', real=True)
    p_x, p_y, v_x, v_y = sym.symbols('p_x,p_y,v_x,v_y', real=True)
    lam, f_x, theta, u_dummy = sym.symbols('lam,f_x,theta,u_dummy', real=True)

    return {
        'name': 'test_pendulum',
        'params_dict': {'L': 5.21, 'G': 9.81, 'M': 10.0, 'K_d': 1e-3, 'K_lam': 1e-6},
        'f_list': [v_x, v_y,
                   (-2*p_x*lam + f_x - K_d*v_x)/M,
                   (-M*G - 2*p_y*lam - K_d*v_y)/M],
        'g_list': [p_x**2 + p_y**2 - L**2 - lam*K_lam,
                   -theta + sym.atan2(p_x, -p_y) + u_dummy],
        'x_list': [p_x, p_y, v_x, v_y],
        'y_ini_list': [lam, f_x],
        'y_run_list': [lam, theta],
        'u_ini_dict': {'theta': np.deg2rad(5.0), 'u_dummy': 0.0},
        'u_run_dict': {'f_x': 0, 'u_dummy': 0.0},
        'h_dict': {'E_p': M*G*(p_y+L), 'E_k': 0.5*M*(v_x**2+v_y**2), 'f_x': f_x, 'lam': lam},
    }


# ─── 1. Parser tests ──────────────────────────────────────────────────

@pytest.mark.parse
class TestParser:

    def test_check_system_detects_inirun(self, pendulum_sys):
        from pydae.core.builder.parser import check_system
        sys_out, inirun = check_system(pendulum_sys)
        assert inirun is True  # y_ini != y_run

    def test_check_system_no_inirun(self, pendulum_sys):
        from pydae.core.builder.parser import check_system
        # Make y_ini == y_run
        pendulum_sys['y_ini_list'] = pendulum_sys['y_run_list'].copy()
        _, inirun = check_system(pendulum_sys)
        assert inirun is False

    def test_process_dimensions(self, pendulum_sys):
        from pydae.core.builder.parser import check_system, process_system_dict
        sys_out, _ = check_system(pendulum_sys)
        sys_out = process_system_dict(sys_out)
        assert sys_out['N_x'] == 4
        assert sys_out['N_y'] == 2
        assert sys_out['N_z'] == 4

    def test_dummy_added_when_no_algebraic(self):
        """System with f but no g should get dummy algebraic eqs."""
        from pydae.core.builder.parser import check_system
        x, u = sym.symbols('x, u')
        sys_in = {
            'f_list': [u - x],
            'x_list': [x],
            'y_ini_list': [],
            'y_run_list': [],
            'u_ini_dict': {'u': 1.0},
            'u_run_dict': {'u': 1.0},
            'params_dict': {},
            'h_dict': {},
        }
        sys_out, _ = check_system(sys_in)
        assert len(sys_out['g_list']) == 2  # two dummies added


# ─── 2. Symbolic Jacobian tests ───────────────────────────────────────

@pytest.mark.symbolic
class TestSymbolic:

    def test_base_jacobians_shapes(self, pendulum_sys):
        from pydae.core.builder.parser import check_system, process_system_dict
        from pydae.core.builder.symbolic import compute_base_jacobians

        sys_out, inirun = check_system(pendulum_sys)
        sys_out = process_system_dict(sys_out)
        sys_out = compute_base_jacobians(sys_out, inirun)

        assert sys_out['Fx'].shape == (4, 4)
        assert sys_out['Fy_run'].shape == (4, 2)
        assert sys_out['Gx'].shape == (2, 4)
        assert sys_out['Gy_run'].shape == (2, 2)

    def test_jacobian_sparsity(self, pendulum_sys):
        """Fx should be sparse — v_x doesn't depend on v_y, etc."""
        from pydae.core.builder.parser import check_system, process_system_dict
        from pydae.core.builder.symbolic import compute_base_jacobians

        sys_out, inirun = check_system(pendulum_sys)
        sys_out = process_system_dict(sys_out)
        sys_out = compute_base_jacobians(sys_out, inirun)

        nnz = len(sys_out['Fx'].todok())
        total = 4 * 4
        assert nnz < total, f"Fx should be sparse, got {nnz}/{total} nonzeros"

    def test_large_jacobian_assembly(self, pendulum_sys):
        from pydae.core.builder.parser import check_system, process_system_dict
        from pydae.core.builder.symbolic import compute_base_jacobians, build_large_jacobians

        sys_out, inirun = check_system(pendulum_sys)
        sys_out = process_system_dict(sys_out)
        sys_out = compute_base_jacobians(sys_out, inirun)

        class FakeBuilder:
            jac_ini_list, jac_run_list, jac_trap_list = [], [], []
            Fu_list, Gu_list, Hx_list = [], [], []
            uz_jacs = True

        fb = FakeBuilder()
        fb.sys = sys_out
        build_large_jacobians(fb)

        N = 6  # 4 + 2
        assert sys_out['jac_ini'].shape == (N, N)
        assert sys_out['jac_trap'].shape == (N, N)
        assert len(fb.jac_ini_list) > 0


# ─── 3. Code generation tests ─────────────────────────────────────────

@pytest.mark.codegen
class TestCodegen:

    def test_sym2c_produces_ccode(self, pendulum_sys):
        from pydae.core.builder.parser import check_system, process_system_dict
        from pydae.core.builder.codegen.cffi_builder import sym2c

        sys_out, _ = check_system(pendulum_sys)
        sys_out = process_system_dict(sys_out)

        f_list = [{'sym': eq} for eq in sys_out['f']]
        sym2c(f_list)

        for i, item in enumerate(f_list):
            assert 'ccode' in item, f"ccode missing for f[{i}]"
            assert isinstance(item['ccode'], str)
            assert len(item['ccode']) > 0

    def test_sym2xyup_replaces_variables(self, pendulum_sys):
        from pydae.core.builder.parser import check_system, process_system_dict
        from pydae.core.builder.codegen.cffi_builder import sym2c, sym2xyup

        sys_out, _ = check_system(pendulum_sys)
        sys_out = process_system_dict(sys_out)

        f_list = [{'sym': eq} for eq in sys_out['f']]
        sym2c(f_list)
        sym2xyup(sys_out, f_list, 'run')

        for item in f_list:
            assert 'xyup' in item
            # Should contain array indexing, not symbolic names
            c_str = item['xyup']
            if c_str != '0':  # skip trivial zero equations
                assert 'x[' in c_str or 'y[' in c_str or 'u[' in c_str or 'p[' in c_str, \
                    f"Expected array syntax in: {c_str}"


# ─── 4. Full build tests ──────────────────────────────────────────────

@pytest.mark.build
class TestBuild:

    def test_ctypes_build(self, pendulum_sys):
        from pydae.core import Builder

        bld = Builder(pendulum_sys, target='ctypes')
        bld.build()

        lib_ext = '.dll' if sys.platform == 'win32' else '.so'
        lib_path = os.path.join('build', f"test_pendulum_ctypes{lib_ext}")
        assert os.path.exists(lib_path), f"Compiled library not found: {lib_path}"
        assert os.path.getsize(lib_path) > 0


# ─── 5. Model tests (end-to-end) ──────────────────────────────────────

@pytest.mark.model
class TestModel:

    @pytest.fixture(autouse=True)
    def _build_first(self, pendulum_sys):
        """Ensure the library is compiled before model tests."""
        from pydae.core import Builder
        bld = Builder(pendulum_sys, target='ctypes')
        bld.build()

    def test_initialization(self):
        from pydae.core import Model
        model = Model('test_pendulum')
        L = 5.21
        deg = 10
        success = model.ini(
            {'M': 30.0, 'L': L, 'K_lam': 1e-6, 'theta': np.deg2rad(deg)},
            xy_0={'p_x': L*np.sin(np.deg2rad(deg)),
                  'p_y': -L*np.cos(np.deg2rad(deg)),
                  'lam': 0, 'f_x': 1}
        )
        assert success
        assert model.ini_iterations < 50

    def test_simulation_runs(self):
        from pydae.core import Model
        model = Model('test_pendulum')
        L = 5.21
        deg = 10
        model.ini(
            {'M': 30.0, 'L': L, 'K_lam': 1e-6, 'theta': np.deg2rad(deg)},
            xy_0={'p_x': L*np.sin(np.deg2rad(deg)),
                  'p_y': -L*np.cos(np.deg2rad(deg)),
                  'lam': 0, 'f_x': 1}
        )
        model.run(1.0, {})
        model.run(5.0, {'f_x': 0.0})
        model.post()

        assert len(model.Time) > 100
        assert model.Time[-1] == pytest.approx(5.0, abs=0.02)
        assert not np.isnan(model.get_values('E_k')).any()

    def test_get_value(self):
        from pydae.core import Model
        model = Model('test_pendulum')
        model.ini(
            {'M': 30.0, 'L': 5.21, 'K_lam': 1e-6, 'theta': np.deg2rad(10)},
            xy_0={'p_x': 0.9, 'p_y': -5.1, 'lam': 0, 'f_x': 1}
        )
        # After init, parameters should be readable
        assert model.get_value('M') == pytest.approx(30.0)
        assert model.get_value('L') == pytest.approx(5.21)
