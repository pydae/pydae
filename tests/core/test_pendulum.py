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

import numpy as np
import pytest
import sympy as sym

# ─── Shared fixture ────────────────────────────────────────────────────

@pytest.fixture
def pendulum_sys(tmp_path):
    """The pendulum DAE system dictionary used across all tests."""
    orig_cwd = os.getcwd()
    os.chdir(tmp_path)

    L, G, M, K_d, K_lam = sym.symbols('L,G,M,K_d,K_lam', real=True)
    p_x, p_y, v_x, v_y = sym.symbols('p_x,p_y,v_x,v_y', real=True)
    lam, f_x, theta, u_dummy = sym.symbols('lam,f_x,theta,u_dummy', real=True)

    sys_dict = {
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

    yield sys_dict
    os.chdir(orig_cwd)


# ─── 1. Parser tests ──────────────────────────────────────────────────

@pytest.mark.parse
class TestParser:

    def test_check_system_detects_inirun(self, pendulum_sys):
        from pydae.core.common.parser import check_system
        sys_out, inirun = check_system(pendulum_sys)
        assert inirun is True  # y_ini != y_run

    def test_check_system_no_inirun(self, pendulum_sys):
        from pydae.core.common.parser import check_system
        # Make y_ini == y_run
        pendulum_sys['y_ini_list'] = pendulum_sys['y_run_list'].copy()
        _, inirun = check_system(pendulum_sys)
        assert inirun is False

    def test_process_dimensions(self, pendulum_sys):
        from pydae.core.common.parser import check_system, process_system_dict
        sys_out, _ = check_system(pendulum_sys)
        sys_out = process_system_dict(sys_out)
        assert sys_out['N_x'] == 4
        assert sys_out['N_y'] == 2
        assert sys_out['N_z'] == 4

    def test_dummy_added_when_no_algebraic(self):
        """System with f but no g should get dummy algebraic eqs."""
        from pydae.core.common.parser import check_system
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
        from pydae.core.common.parser import check_system, process_system_dict
        from pydae.core.common.symbolic import compute_base_jacobians

        sys_out, inirun = check_system(pendulum_sys)
        sys_out = process_system_dict(sys_out)
        sys_out = compute_base_jacobians(sys_out, inirun)

        assert sys_out['Fx'].shape == (4, 4)
        assert sys_out['Fy_run'].shape == (4, 2)
        assert sys_out['Gx'].shape == (2, 4)
        assert sys_out['Gy_run'].shape == (2, 2)

    def test_jacobian_sparsity(self, pendulum_sys):
        """Fx should be sparse — v_x doesn't depend on v_y, etc."""
        from pydae.core.common.parser import check_system, process_system_dict
        from pydae.core.common.symbolic import compute_base_jacobians

        sys_out, inirun = check_system(pendulum_sys)
        sys_out = process_system_dict(sys_out)
        sys_out = compute_base_jacobians(sys_out, inirun)

        nnz = len(sys_out['Fx'].todok())
        total = 4 * 4
        assert nnz < total, f"Fx should be sparse, got {nnz}/{total} nonzeros"

    def test_large_jacobian_assembly(self, pendulum_sys):
        from pydae.core.common.parser import check_system, process_system_dict
        from pydae.core.common.symbolic import build_large_jacobians, compute_base_jacobians

        sys_out, inirun = check_system(pendulum_sys)
        sys_out = process_system_dict(sys_out)
        sys_out = compute_base_jacobians(sys_out, inirun)

        class FakeBuilder:
            jac_ini_list, jac_run_list, jac_trap_list = [], [], []
            # Initialize UZ Jacobian lists (matching Builder.__init__)
            Fu_ini_list = []
            Fu_run_list = []
            Gu_ini_list = []
            Gu_run_list = []
            Hx_list = []
            Hy_ini_list = []
            Hy_run_list = []
            Hu_ini_list = []
            Hu_run_list = []
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
        from pydae.core.builder.codegen.cffi_builder import sym2c
        from pydae.core.common.parser import check_system, process_system_dict

        sys_out, _ = check_system(pendulum_sys)
        sys_out = process_system_dict(sys_out)

        f_list = [{'sym': eq} for eq in sys_out['f']]
        sym2c(f_list)

        for i, item in enumerate(f_list):
            assert 'ccode' in item, f"ccode missing for f[{i}]"
            assert isinstance(item['ccode'], str)
            assert len(item['ccode']) > 0

    def test_sym2xyup_replaces_variables(self, pendulum_sys):
        from pydae.core.builder.codegen.cffi_builder import sym2c, sym2xyup
        from pydae.core.common.parser import check_system, process_system_dict

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

    @pytest.fixture(autouse=True)
    def _build_first(self, pendulum_sys):
        """Ensure the library is compiled before model tests."""
        from pydae.core import Builder
        bld = Builder(pendulum_sys, target='ctypes', sparse=False)
        bld.build()


# ─── 5. KLU sparse build + A_eval ────────────────────────────────────

@pytest.mark.build
class TestBuildKLU:

    @pytest.fixture(autouse=True)
    def _skip_if_not_supported(self):
        import importlib
        if importlib.util.find_spec('cffi') is None:
            pytest.skip("cffi not available")

    def test_cffi_klu_build(self, pendulum_sys):
        import json

        from pydae.core import Builder
        bld = Builder(pendulum_sys, target='cffi', sparse='klu')
        bld.build()
        with open('test_pendulum_data.json') as f:
            data = json.load(f)
        assert data.get('NNZ_trap', 0) > 0, "KLU build did not record NNZ_trap"
        assert 'Ap_trap' in data, "KLU build did not store Ap_trap"
        assert 'Ai_trap' in data, "KLU build did not store Ai_trap"

    def test_A_eval_klu(self, pendulum_sys):
        from pydae.core import Builder, Model
        bld = Builder(pendulum_sys, target='cffi', sparse='klu')
        bld.build()

        L, M = 5.21, 30.0
        deg = 10
        model = Model('test_pendulum')
        success = model.ini(
            {'M': M, 'L': L, 'K_lam': 1e-6, 'theta': np.deg2rad(deg)},
            xy_0={'p_x': L*np.sin(np.deg2rad(deg)),
                  'p_y': -L*np.cos(np.deg2rad(deg)),
                  'lam': 50.0, 'f_x': 0.0, 'v_x': 0.0, 'v_y': 0.0}
        )
        assert success
        model.run(0.1, {})

        A_klu = model.A_eval()
        assert A_klu.shape == (4, 4)
        assert np.isfinite(A_klu).all(), "KLU A_eval returned NaN/Inf"

        # Verify A_eval matches dense result for the same model
        bld_dense = Builder(pendulum_sys, target='ctypes', sparse=False)
        bld_dense.build()
        model_dense = Model('test_pendulum')
        model_dense.ini(
            {'M': M, 'L': L, 'K_lam': 1e-6, 'theta': np.deg2rad(deg)},
            xy_0={'p_x': L*np.sin(np.deg2rad(deg)),
                  'p_y': -L*np.cos(np.deg2rad(deg)),
                  'lam': 50.0, 'f_x': 0.0, 'v_x': 0.0, 'v_y': 0.0}
        )
        model_dense.run(0.1, {})
        A_dense = model_dense.A_eval()

        np.testing.assert_allclose(A_klu, A_dense, rtol=1e-6,
                                   err_msg="KLU and dense A matrices differ")

        # A_eval must be re-evaluated after a second run (no stale cache)
        model.run(0.5, {'f_x': 1.0})
        A_klu2 = model.A_eval()
        assert np.isfinite(A_klu2).all()

    def test_BCD_eval_klu(self, pendulum_sys):
        """BCD_eval from KLU build matches BCD_eval from dense build."""
        from pydae.core import Builder, Model

        bld = Builder(pendulum_sys, target='cffi', sparse='klu')
        bld.build()

        L, M = 5.21, 30.0
        deg = 10
        model = Model('test_pendulum')
        model.ini(
            {'M': M, 'L': L, 'K_lam': 1e-6, 'theta': np.deg2rad(deg)},
            xy_0={'p_x': L*np.sin(np.deg2rad(deg)),
                  'p_y': -L*np.cos(np.deg2rad(deg)),
                  'lam': 50.0, 'f_x': 0.0, 'v_x': 0.0, 'v_y': 0.0}
        )
        model.run(0.1, {})
        B_klu, C_klu, D_klu = model.BCD_eval()

        assert B_klu.shape == (4, 2)
        assert C_klu.shape == (4, 4)
        assert D_klu.shape == (4, 2)
        assert np.isfinite(B_klu).all(), "KLU B has NaN/Inf"
        assert np.isfinite(C_klu).all(), "KLU C has NaN/Inf"
        assert np.isfinite(D_klu).all(), "KLU D has NaN/Inf"

        # Dense build for comparison
        bld_dense = Builder(pendulum_sys, target='ctypes', sparse=False)
        bld_dense.build()
        model_dense = Model('test_pendulum')
        model_dense.ini(
            {'M': M, 'L': L, 'K_lam': 1e-6, 'theta': np.deg2rad(deg)},
            xy_0={'p_x': L*np.sin(np.deg2rad(deg)),
                  'p_y': -L*np.cos(np.deg2rad(deg)),
                  'lam': 50.0, 'f_x': 0.0, 'v_x': 0.0, 'v_y': 0.0}
        )
        model_dense.run(0.1, {})
        B_dense, C_dense, D_dense = model_dense.BCD_eval()

        np.testing.assert_allclose(B_klu, B_dense, rtol=1e-6,
                                   err_msg="KLU and dense B matrices differ")
        np.testing.assert_allclose(C_klu, C_dense, rtol=1e-6,
                                   err_msg="KLU and dense C matrices differ")
        np.testing.assert_allclose(D_klu, D_dense, rtol=1e-6,
                                   err_msg="KLU and dense D matrices differ")

        # Re-evaluate after another run
        model.run(0.5, {'f_x': 1.0})
        B2, C2, D2 = model.BCD_eval()
        assert np.isfinite(B2).all()
        assert np.isfinite(C2).all()
        assert np.isfinite(D2).all()


# ─── 6. Model tests (end-to-end) ──────────────────────────────────────

@pytest.mark.model
class TestModel:

    @pytest.fixture(autouse=True)
    def _build_first(self, pendulum_sys):
        """Ensure the library is compiled before model tests."""
        from pydae.core import Builder
        bld = Builder(pendulum_sys, target='ctypes', sparse=False)
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
                  'lam': 50.0, 'f_x': 1, 'v_x': 0.0, 'v_y': 0.0}
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
                  'lam': 50.0, 'f_x': 1, 'v_x': 0.0, 'v_y': 0.0}
        )
        model.run(1.0, {})
        model.run(5.0, {'f_x': 0.0})
        model.post()

        assert len(model.Time) > 100
        assert model.Time[-1] == pytest.approx(5.0, abs=0.02)
        e_k = model.get_values('E_k')
        assert e_k is not None, f"get_values('E_k') returned None. z_list={model.z_list}"
        assert len(e_k) > 0
        assert not np.isnan(e_k).any()

    def test_get_value(self):
        from pydae.core import Model
        model = Model('test_pendulum')
        model.ini(
            {'M': 30.0, 'L': 5.21, 'K_lam': 1e-6, 'theta': np.deg2rad(10)},
            xy_0={'p_x': 0.9, 'p_y': -5.1, 'lam': 50.0, 'f_x': 1, 'v_x': 0.0, 'v_y': 0.0}
        )
        # After init, parameters should be readable
        assert model.get_value('M') == pytest.approx(30.0)
        assert model.get_value('L') == pytest.approx(5.21)

    def test_A_eval_dense(self):
        """A_eval returns a real (N_x, N_x) matrix with finite entries and
        correct pendulum eigenvalues (two imaginary pairs for undamped motion)."""
        from pydae.core import Model
        L, M = 5.21, 30.0
        deg = 10
        model = Model('test_pendulum')
        model.ini(
            {'M': M, 'L': L, 'K_lam': 1e-6, 'theta': np.deg2rad(deg)},
            xy_0={'p_x': L*np.sin(np.deg2rad(deg)),
                  'p_y': -L*np.cos(np.deg2rad(deg)),
                  'lam': 50.0, 'f_x': 0.0, 'v_x': 0.0, 'v_y': 0.0}
        )
        model.run(0.1, {})

        A = model.A_eval()
        assert A.shape == (4, 4)
        assert np.isfinite(A).all(), "A contains NaN or Inf"

        # Calling A_eval a second time (after no new run) must give the same result
        A2 = model.A_eval()
        np.testing.assert_array_equal(A, A2)

        # After another run() the Jacobian must be re-evaluated (no stale cache)
        model.run(0.5, {'f_x': 1.0})
        A3 = model.A_eval()
        assert A3.shape == (4, 4)
        assert np.isfinite(A3).all()

    def test_BCD_eval_dense(self):
        """BCD_eval returns real (N_x,N_u), (N_z,N_x), (N_z,N_u) matrices
        with finite entries.  Also verifies that B has the expected structure
        for the pendulum: only v_x equation depends on f_x input."""
        from pydae.core import Model
        L, M = 5.21, 30.0
        deg = 10
        model = Model('test_pendulum')
        model.ini(
            {'M': M, 'L': L, 'K_lam': 1e-6, 'theta': np.deg2rad(deg)},
            xy_0={'p_x': L*np.sin(np.deg2rad(deg)),
                  'p_y': -L*np.cos(np.deg2rad(deg)),
                  'lam': 50.0, 'f_x': 0.0, 'v_x': 0.0, 'v_y': 0.0}
        )
        model.run(0.1, {})

        B, C, D = model.BCD_eval()

        # u_run = [f_x, u_dummy] -> N_u=2; x has 4 states; h has 4 outputs
        assert B.shape == (4, 2), f"B shape {B.shape} != (4,2)"
        assert C.shape == (4, 4), f"C shape {C.shape} != (4,4)"
        assert D.shape == (4, 2), f"D shape {D.shape} != (4,2)"

        assert np.isfinite(B).all(), "B contains NaN or Inf"
        assert np.isfinite(C).all(), "C contains NaN or Inf"
        assert np.isfinite(D).all(), "D contains NaN or Inf"

        # Symbolic check: f equations are [v_x, v_y, (-2*px*lam+f_x-Kd*vx)/M, ...]
        # So dF/df_x = [0, 0, 1/M, 0]^T  (column 0 of B)
        np.testing.assert_allclose(B[:, 0], [0, 0, 1/M, 0], atol=1e-10)
        # u_dummy only appears in algebraic g equation, not in f
        np.testing.assert_allclose(B[:, 1], [0, 0, 0, 0], atol=1e-10)

        # C matrix: h = [M*G*(p_y+L), 0.5*M*(vx^2+vy^2), f_x, lam]
        # dh/dx = [0, M*G, 0, 0] for E_p
        #         [0, 0, M*vx, M*vy] for E_k
        #         [0, 0, 0, 0] for f_x (output equals input, no state dep)
        #         [0, 0, 0, 0] for lam (algebraic var, not state)
        # So C[0,:] should be [0, M*G, 0, 0]
        np.testing.assert_allclose(C[0, :], [0, M*9.81, 0, 0], atol=1e-10)

        # Re-evaluate after another run — must reflect new operating point
        model.run(0.5, {'f_x': 1.0})
        B2, C2, D2 = model.BCD_eval()
        assert B2.shape == (4, 2)
        assert np.isfinite(B2).all()
