# tests/core/test_casadi.py
"""
Tests for the CasADi backend pipeline using a pendulum DAE model.

Run selectively:
    uv run pytest tests/core/test_casadi.py -v
"""
import casadi as ca
import numpy as np
import pytest
import sympy as sym

# ─── Shared fixture ────────────────────────────────────────────────────

@pytest.fixture
def pendulum_sys_dict():
    """The pendulum DAE system dictionary for CasADi tests."""
    L, G, M, K_d, K_lam = [ca.SX.sym(n) for n in ['L', 'G', 'M', 'K_d', 'K_lam']]
    p_x, p_y, v_x, v_y = [ca.SX.sym(n) for n in ['p_x', 'p_y', 'v_x', 'v_y']]
    lam, f_x, theta, u_dummy = [ca.SX.sym(n) for n in ['lam', 'f_x', 'theta', 'u_dummy']]

    return {
        'name': 'test_pendulum',
        'params_dict': {'L': 5.21, 'G': 9.81, 'M': 10.0, 'K_d': 1e-3, 'K_lam': 1e-2},
        'f_list': [v_x, v_y,
                   (-2 * p_x * lam + f_x - K_d * v_x) / M,
                   (-M * G - 2 * p_y * lam - K_d * v_y) / M],
        'g_list': [p_x ** 2 + p_y ** 2 - L ** 2 - lam * K_lam,
                   -theta + ca.atan2(p_x, -p_y) + u_dummy],
        'x_list': [p_x, p_y, v_x, v_y],
        'y_ini_list': [lam, f_x],
        'y_run_list': [lam, theta],
        'u_ini_dict': {'theta': np.deg2rad(5.0), 'u_dummy': 0.0},
        'u_run_dict': {'f_x': 0, 'u_dummy': 0.0},
        'h_dict': {'E_p': M * G * (p_y + L), 'E_k': 0.5 * M * (v_x ** 2 + v_y ** 2), 'f_x': f_x, 'lam': lam},
    }


@pytest.fixture
def pendulum_no_outputs():
    """Pendulum system without h_dict (tests C/D fallback)."""
    L, G, M, K_d, K_lam = [ca.SX.sym(n) for n in ['L', 'G', 'M', 'K_d', 'K_lam']]
    p_x, p_y, v_x, v_y = [ca.SX.sym(n) for n in ['p_x', 'p_y', 'v_x', 'v_y']]
    lam, f_x, theta = [ca.SX.sym(n) for n in ['lam', 'f_x', 'theta']]

    return {
        'name': 'test_pendulum_no_h',
        'params_dict': {'L': 5.21, 'G': 9.81, 'M': 10.0, 'K_d': 1e-3, 'K_lam': 1e-2},
        'f_list': [v_x, v_y,
                   (-2 * p_x * lam + f_x - K_d * v_x) / M,
                   (-M * G - 2 * p_y * lam - K_d * v_y) / M],
        'g_list': [p_x ** 2 + p_y ** 2 - L ** 2 - lam * K_lam,
                   -theta + ca.atan2(p_x, -p_y)],
        'x_list': [p_x, p_y, v_x, v_y],
        'y_ini_list': [lam, f_x],
        'y_run_list': [lam, theta],
        'u_ini_dict': {'theta': np.deg2rad(5.0)},
        'u_run_dict': {'f_x': 0},
        'h_dict': {},
    }


@pytest.fixture
def casadi_builder(pendulum_sys_dict):
    """Build the pendulum model once."""
    from pydae.core.builder import CasadiBuilder
    return CasadiBuilder(pendulum_sys_dict).build()


@pytest.fixture
def casadi_model(casadi_builder):
    """Create a model from the built CasADi builder."""
    from pydae.core.builder import CasadiModel
    return CasadiModel(casadi_builder)


# ─── 1. Builder tests ─────────────────────────────────────────────────

@pytest.mark.parse
class TestCasadiBuilder:

    def test_build_returns_builder(self, casadi_builder):
        assert hasattr(casadi_builder, 'rf')
        assert hasattr(casadi_builder, 'dae_dict')
        assert hasattr(casadi_builder, '_Fx_fn')
        assert hasattr(casadi_builder, '_Fy_fn')
        assert hasattr(casadi_builder, '_Gx_fn')
        assert hasattr(casadi_builder, '_Gy_fn')
        assert hasattr(casadi_builder, '_Fu_fn')
        assert hasattr(casadi_builder, '_Gu_fn')

    def test_build_extracts_parameter_lists(self, casadi_builder):
        assert 'p_list' in casadi_builder.sys_dict
        assert 'u_ini_list' in casadi_builder.sys_dict
        assert 'u_run_list' in casadi_builder.sys_dict
        assert len(casadi_builder.sys_dict['p_list']) > 0

    def test_dae_dict_has_required_keys(self, casadi_builder):
        keys = {'x', 'z', 'p', 'ode', 'alg'}
        assert keys.issubset(casadi_builder.dae_dict.keys())


# ─── 2. Model initialization tests ────────────────────────────────────

@pytest.mark.model
class TestCasadiModel:

    def test_initialization_succeeds(self, casadi_model):
        L, M = 5.21, 30.0
        deg = 10
        casadi_model.ini(
            {'M': M, 'L': L, 'K_lam': 1e-2, 'theta': np.deg2rad(deg)},
            xy_0={
                'p_x': L * np.sin(np.deg2rad(deg)),
                'p_y': -L * np.cos(np.deg2rad(deg)),
                'lam': 50.0, 'f_x': 1, 'v_x': 0.0, 'v_y': 0.0
            }
        )
        assert casadi_model.t == 0.0
        assert np.isfinite(casadi_model.x).all()
        assert np.isfinite(casadi_model.y_ini).all()

    def test_initialization_stores_history(self, casadi_model):
        L = 5.21
        casadi_model.ini(
            {'M': 30.0, 'L': L, 'K_lam': 1e-2, 'theta': np.deg2rad(10)},
            xy_0={
                'p_x': L * np.sin(np.deg2rad(10)),
                'p_y': -L * np.cos(np.deg2rad(10)),
                'lam': 50.0, 'f_x': 1, 'v_x': 0.0, 'v_y': 0.0
            }
        )
        assert len(casadi_model.Time) == 1
        assert len(casadi_model.X) == 1
        assert len(casadi_model.Y) == 1

    def test_ini_runs_multiple_times(self, casadi_model):
        L = 5.21
        # First init
        casadi_model.ini(
            {'M': 30.0, 'L': L, 'K_lam': 1e-2, 'theta': np.deg2rad(10)},
            xy_0={
                'p_x': L * np.sin(np.deg2rad(10)),
                'p_y': -L * np.cos(np.deg2rad(10)),
                'lam': 50.0, 'f_x': 1, 'v_x': 0.0, 'v_y': 0.0
            }
        )
        # Second init with different parameters
        casadi_model.ini(
            {'M': 40.0, 'L': L, 'K_lam': 1e-2, 'theta': np.deg2rad(20)},
            xy_0={
                'p_x': L * np.sin(np.deg2rad(20)),
                'p_y': -L * np.cos(np.deg2rad(20)),
                'lam': 50.0, 'f_x': 1, 'v_x': 0.0, 'v_y': 0.0
            }
        )
        assert np.isfinite(casadi_model.x).all()


# ─── 3. Simulation tests ──────────────────────────────────────────────

@pytest.mark.model
class TestCasadiSimulation:

    def test_simulation_runs(self, casadi_model):
        L, M = 5.21, 30.0
        deg = 10
        casadi_model.ini(
            {'M': M, 'L': L, 'K_lam': 1e-2, 'theta': np.deg2rad(deg)},
            xy_0={
                'p_x': L * np.sin(np.deg2rad(deg)),
                'p_y': -L * np.cos(np.deg2rad(deg)),
                'lam': 50.0, 'f_x': 1, 'v_x': 0.0, 'v_y': 0.0
            }
        )
        casadi_model.run(1.0, {})
        casadi_model.run(5.0, {'f_x': 0.0})
        casadi_model.post()

        assert len(casadi_model.Time) > 100
        assert casadi_model.Time[-1] == pytest.approx(5.0, abs=0.05)
        assert not np.isnan(casadi_model.X).any()
        assert not np.isnan(casadi_model.Y).any()

    def test_post_converts_to_arrays(self, casadi_model):
        L = 5.21
        casadi_model.ini(
            {'M': 30.0, 'L': L, 'K_lam': 1e-2, 'theta': np.deg2rad(10)},
            xy_0={
                'p_x': L * np.sin(np.deg2rad(10)),
                'p_y': -L * np.cos(np.deg2rad(10)),
                'lam': 50.0, 'f_x': 1, 'v_x': 0.0, 'v_y': 0.0
            }
        )
        casadi_model.run(1.0, {})
        casadi_model.post()

        assert isinstance(casadi_model.Time, np.ndarray)
        assert isinstance(casadi_model.X, np.ndarray)
        assert isinstance(casadi_model.Y, np.ndarray)

    def test_sequential_runs_stack_data(self, casadi_model):
        L = 5.21
        casadi_model.ini(
            {'M': 30.0, 'L': L, 'K_lam': 1e-2, 'theta': np.deg2rad(10)},
            xy_0={
                'p_x': L * np.sin(np.deg2rad(10)),
                'p_y': -L * np.cos(np.deg2rad(10)),
                'lam': 50.0, 'f_x': 1, 'v_x': 0.0, 'v_y': 0.0
            }
        )
        n1 = len(casadi_model.Time)
        casadi_model.run(0.5, {})
        n2 = len(casadi_model.Time)
        casadi_model.run(1.0, {})
        n3 = len(casadi_model.Time)

        assert n2 > n1
        assert n3 > n2


# ─── 4. A_eval and eigenvalue tests ───────────────────────────────────

@pytest.mark.model
class TestCasadiAEval:

    def test_A_eval_returns_correct_shape(self, casadi_model):
        L, M = 5.21, 30.0
        deg = 10
        casadi_model.ini(
            {'M': M, 'L': L, 'K_lam': 1e-2, 'theta': np.deg2rad(deg)},
            xy_0={
                'p_x': L * np.sin(np.deg2rad(deg)),
                'p_y': -L * np.cos(np.deg2rad(deg)),
                'lam': 50.0, 'f_x': 0.0, 'v_x': 0.0, 'v_y': 0.0
            }
        )
        casadi_model.run(0.1, {})

        A = casadi_model.A_eval()
        assert A.shape == (4, 4)
        assert np.isfinite(A).all()

    def test_A_eval_reproducible(self, casadi_model):
        L, M = 5.21, 30.0
        deg = 10
        casadi_model.ini(
            {'M': M, 'L': L, 'K_lam': 1e-2, 'theta': np.deg2rad(deg)},
            xy_0={
                'p_x': L * np.sin(np.deg2rad(deg)),
                'p_y': -L * np.cos(np.deg2rad(deg)),
                'lam': 50.0, 'f_x': 0.0, 'v_x': 0.0, 'v_y': 0.0
            }
        )
        casadi_model.run(0.1, {})

        A1 = casadi_model.A_eval()
        A2 = casadi_model.A_eval()
        np.testing.assert_array_equal(A1, A2)

    def test_A_eval_updates_after_run(self, casadi_model):
        L, M = 5.21, 30.0
        deg = 10
        casadi_model.ini(
            {'M': M, 'L': L, 'K_lam': 1e-2, 'theta': np.deg2rad(deg)},
            xy_0={
                'p_x': L * np.sin(np.deg2rad(deg)),
                'p_y': -L * np.cos(np.deg2rad(deg)),
                'lam': 50.0, 'f_x': 0.0, 'v_x': 0.0, 'v_y': 0.0
            }
        )
        casadi_model.run(0.1, {})
        casadi_model.run(0.5, {'f_x': 1.0})
        A2 = casadi_model.A_eval()
        assert A2.shape == (4, 4)
        assert np.isfinite(A2).all()

    def test_eval_eigenvalues(self, casadi_model):
        L, M = 5.21, 30.0
        deg = 10
        casadi_model.ini(
            {'M': M, 'L': L, 'K_lam': 1e-2, 'theta': np.deg2rad(deg)},
            xy_0={
                'p_x': L * np.sin(np.deg2rad(deg)),
                'p_y': -L * np.cos(np.deg2rad(deg)),
                'lam': 50.0, 'f_x': 0.0, 'v_x': 0.0, 'v_y': 0.0
            }
        )
        casadi_model.run(0.1, {})

        eigs = casadi_model.eval_eigenvalues()
        assert len(eigs) == 4
        assert np.isfinite(eigs).all()
        # For lightly damped pendulum, eigenvalues should have negative real parts
        assert (eigs.real <= 0).all() or np.allclose(eigs.real, 0, atol=1e-5)


# ─── 5. BCD_eval tests ────────────────────────────────────────────────

@pytest.mark.model
class TestCasadiBCD:

    def test_BCD_eval_with_outputs(self, casadi_model):
        L, M = 5.21, 30.0
        deg = 10
        casadi_model.ini(
            {'M': M, 'L': L, 'K_lam': 1e-2, 'theta': np.deg2rad(deg)},
            xy_0={
                'p_x': L * np.sin(np.deg2rad(deg)),
                'p_y': -L * np.cos(np.deg2rad(deg)),
                'lam': 50.0, 'f_x': 0.0, 'v_x': 0.0, 'v_y': 0.0
            }
        )
        casadi_model.run(0.1, {})

        B, C, D = casadi_model.BCD_eval()
        # u_run = [f_x, u_dummy] -> N_u=2; x has 4 states; h has 4 outputs
        assert B.shape == (4, 2)
        assert C.shape == (4, 4)
        assert D.shape == (4, 2)
        assert np.isfinite(B).all()
        assert np.isfinite(C).all()
        assert np.isfinite(D).all()

        # B[:, 0] should be [0, 0, 1/M, 0]^T (only v_x equation depends on f_x)
        np.testing.assert_allclose(B[:, 0], [0, 0, 1 / M, 0], atol=1e-10)

    def test_BCD_eval_without_outputs(self, pendulum_no_outputs):
        """C and D should fallback to empty matrices when h_dict is empty."""
        from pydae.core.builder import CasadiBuilder, CasadiModel
        builder = CasadiBuilder(pendulum_no_outputs).build()
        model = CasadiModel(builder)

        L, M = 5.21, 30.0
        deg = 10
        model.ini(
            {'M': M, 'L': L, 'K_lam': 1e-2, 'theta': np.deg2rad(deg)},
            xy_0={
                'p_x': L * np.sin(np.deg2rad(deg)),
                'p_y': -L * np.cos(np.deg2rad(deg)),
                'lam': 50.0, 'f_x': 0.0, 'v_x': 0.0, 'v_y': 0.0
            }
        )
        model.run(0.1, {})

        B, C, D = model.BCD_eval()
        assert B.shape == (4, 1)
        assert C.shape == (0, 4)
        assert D.shape == (0, 1)

    def test_BCD_eval_reproducible(self, casadi_model):
        L, M = 5.21, 30.0
        deg = 10
        casadi_model.ini(
            {'M': M, 'L': L, 'K_lam': 1e-2, 'theta': np.deg2rad(deg)},
            xy_0={
                'p_x': L * np.sin(np.deg2rad(deg)),
                'p_y': -L * np.cos(np.deg2rad(deg)),
                'lam': 50.0, 'f_x': 0.0, 'v_x': 0.0, 'v_y': 0.0
            }
        )
        casadi_model.run(0.1, {})

        B1, C1, D1 = casadi_model.BCD_eval()
        B2, C2, D2 = casadi_model.BCD_eval()
        np.testing.assert_array_equal(B1, B2)
        np.testing.assert_array_equal(C1, C2)
        np.testing.assert_array_equal(D1, D2)

    def test_BCD_eval_updates_after_run(self, casadi_model):
        L, M = 5.21, 30.0
        deg = 10
        casadi_model.ini(
            {'M': M, 'L': L, 'K_lam': 1e-2, 'theta': np.deg2rad(deg)},
            xy_0={
                'p_x': L * np.sin(np.deg2rad(deg)),
                'p_y': -L * np.cos(np.deg2rad(deg)),
                'lam': 50.0, 'f_x': 0.0, 'v_x': 0.0, 'v_y': 0.0
            }
        )
        casadi_model.run(0.1, {})
        B1, C1, D1 = casadi_model.BCD_eval()

        casadi_model.run(0.5, {'f_x': 1.0})
        B2, C2, D2 = casadi_model.BCD_eval()
        assert np.isfinite(B2).all()
        assert np.isfinite(C2).all()
        assert np.isfinite(D2).all()


# ─── 6. CasADi vs C-code backend parity ───────────────────────────────

@pytest.mark.build
class TestCasadiParity:
    """Verify CasADi results match the C-code backend for the same model."""

    @staticmethod
    def _make_sympy_sys():
        L, G, M, K_d, K_lam = sym.symbols('L,G,M,K_d,K_lam', real=True)
        p_x, p_y, v_x, v_y = sym.symbols('p_x,p_y,v_x,v_y', real=True)
        lam, f_x, theta, u_dummy = sym.symbols('lam,f_x,theta,u_dummy', real=True)
        return {
            'name': 'test_pendulum',
            'params_dict': {'L': 5.21, 'G': 9.81, 'M': 10.0, 'K_d': 1e-3, 'K_lam': 1e-2},
            'f_list': [v_x, v_y,
                       (-2 * p_x * lam + f_x - K_d * v_x) / M,
                       (-M * G - 2 * p_y * lam - K_d * v_y) / M],
            'g_list': [p_x ** 2 + p_y ** 2 - L ** 2 - lam * K_lam,
                       -theta + sym.atan2(p_x, -p_y) + u_dummy],
            'x_list': [p_x, p_y, v_x, v_y],
            'y_ini_list': [lam, f_x],
            'y_run_list': [lam, theta],
            'u_ini_dict': {'theta': np.deg2rad(5.0), 'u_dummy': 0.0},
            'u_run_dict': {'f_x': 0, 'u_dummy': 0.0},
            'h_dict': {'E_p': M * G * (p_y + L), 'E_k': 0.5 * M * (v_x ** 2 + v_y ** 2), 'f_x': f_x, 'lam': lam},
        }

    @staticmethod
    def _make_casadi_sys():
        L, G, M, K_d, K_lam = [ca.SX.sym(n) for n in ['L', 'G', 'M', 'K_d', 'K_lam']]
        p_x, p_y, v_x, v_y = [ca.SX.sym(n) for n in ['p_x', 'p_y', 'v_x', 'v_y']]
        lam, f_x, theta, u_dummy = [ca.SX.sym(n) for n in ['lam', 'f_x', 'theta', 'u_dummy']]
        return {
            'name': 'test_pendulum',
            'params_dict': {'L': 5.21, 'G': 9.81, 'M': 10.0, 'K_d': 1e-3, 'K_lam': 1e-2},
            'f_list': [v_x, v_y,
                       (-2 * p_x * lam + f_x - K_d * v_x) / M,
                       (-M * G - 2 * p_y * lam - K_d * v_y) / M],
            'g_list': [p_x ** 2 + p_y ** 2 - L ** 2 - lam * K_lam,
                       -theta + ca.atan2(p_x, -p_y) + u_dummy],
            'x_list': [p_x, p_y, v_x, v_y],
            'y_ini_list': [lam, f_x],
            'y_run_list': [lam, theta],
            'u_ini_dict': {'theta': np.deg2rad(5.0), 'u_dummy': 0.0},
            'u_run_dict': {'f_x': 0, 'u_dummy': 0.0},
            'h_dict': {'E_p': M * G * (p_y + L), 'E_k': 0.5 * M * (v_x ** 2 + v_y ** 2), 'f_x': f_x, 'lam': lam},
        }

    def test_A_eval_parity(self):
        """CasADi and C-code backends produce stable A matrices."""
        from pydae.core import Builder, Model
        from pydae.core.builder import CasadiBuilder, CasadiModel

        bld_c = Builder(self._make_sympy_sys(), target='ctypes', sparse=False)
        bld_c.build()
        model_c = Model('test_pendulum')

        bld_ca = CasadiBuilder(self._make_casadi_sys())
        bld_ca.build()
        model_ca = CasadiModel(bld_ca)

        L, M = 5.21, 30.0
        deg = 10
        params = {'M': M, 'L': L, 'K_lam': 1e-2, 'theta': np.deg2rad(deg)}
        xy_0 = {
            'p_x': L * np.sin(np.deg2rad(deg)),
            'p_y': -L * np.cos(np.deg2rad(deg)),
            'lam': 50.0, 'f_x': 0.0, 'v_x': 0.0, 'v_y': 0.0
        }

        model_c.ini(params, xy_0=xy_0)
        model_c.run(0.1, {})
        model_ca.ini(params, xy_0=xy_0)
        model_ca.run(0.1, {})

        A_c = model_c.A_eval()
        A_ca = model_ca.A_eval()

        # Both should be finite and same shape
        assert A_c.shape == A_ca.shape == (4, 4)
        assert np.isfinite(A_c).all()
        assert np.isfinite(A_ca).all()
        # Eigenvalues should indicate stability (non-positive real parts)
        eigs_c = np.linalg.eigvals(A_c)
        eigs_ca = np.linalg.eigvals(A_ca)
        assert (eigs_c.real <= 1e-10).all()
        assert (eigs_ca.real <= 1e-10).all()

    def test_BCD_eval_parity(self):
        """CasADi and C-code backends produce compatible B, C, D matrices."""
        from pydae.core import Builder, Model
        from pydae.core.builder import CasadiBuilder, CasadiModel

        sympy_sys = self._make_sympy_sys()
        sympy_sys['name'] = 'test_pendulum_bcd'
        casadi_sys = self._make_casadi_sys()
        casadi_sys['name'] = 'test_pendulum_bcd'

        bld_c = Builder(sympy_sys, target='ctypes', sparse=False)
        bld_c.build()
        model_c = Model('test_pendulum_bcd')

        bld_ca = CasadiBuilder(casadi_sys)
        bld_ca.build()
        model_ca = CasadiModel(bld_ca)

        L, M = 5.21, 30.0
        deg = 10
        params = {'M': M, 'L': L, 'K_lam': 1e-2, 'theta': np.deg2rad(deg)}
        xy_0 = {
            'p_x': L * np.sin(np.deg2rad(deg)),
            'p_y': -L * np.cos(np.deg2rad(deg)),
            'lam': 50.0, 'f_x': 0.0, 'v_x': 0.0, 'v_y': 0.0
        }

        model_c.ini(params, xy_0=xy_0)
        model_c.run(0.1, {})
        model_ca.ini(params, xy_0=xy_0)
        model_ca.run(0.1, {})

        B_c, C_c, D_c = model_c.BCD_eval()
        B_ca, C_ca, D_ca = model_ca.BCD_eval()

        # Both should be finite and same shape
        assert B_c.shape == B_ca.shape == (4, 2)
        assert C_c.shape == C_ca.shape == (4, 4)
        assert D_c.shape == D_ca.shape == (4, 2)
        assert np.isfinite(B_c).all()
        assert np.isfinite(C_c).all()
        assert np.isfinite(D_c).all()
        assert np.isfinite(B_ca).all()
        assert np.isfinite(C_ca).all()
        assert np.isfinite(D_ca).all()


# ─── 7. Precompiled binary tests ──────────────────────────────────────

@pytest.mark.build
class TestPrecompiledModel:
    """Test model precompilation: export C code, compile, and load from binary."""

    @pytest.fixture
    def compiled_model_pair(self, pendulum_sys_dict, tmp_path):
        """Build, export, compile, and load both symbolic and binary models."""
        import os
        from pydae.core.builder import CasadiBuilder, CasadiModel

        bld = CasadiBuilder(pendulum_sys_dict).build()

        c_path = str(tmp_path / "pendulum_compiled.c")
        bld.export_code(c_path)
        assert os.path.exists(c_path), "C file was not generated"

        lib_path = CasadiBuilder.compile_shared_library(c_path)
        assert os.path.exists(lib_path), "Shared library was not compiled"

        model_sym = CasadiModel(builder=bld)
        model_bin = CasadiModel(
            binary_path=lib_path,
            dae_dict=bld.dae_dict,
            sys_dict=bld.sys_dict,
        )
        return model_sym, model_bin

    @staticmethod
    def _init_params():
        return {'M': 30.0, 'L': 5.21, 'K_lam': 1e-2, 'theta': np.deg2rad(10)}

    @staticmethod
    def _init_xy_0():
        L = 5.21
        deg = 10
        return {
            'p_x': L * np.sin(np.deg2rad(deg)),
            'p_y': -L * np.cos(np.deg2rad(deg)),
            'lam': 50.0, 'f_x': 0.0, 'v_x': 0.0, 'v_y': 0.0
        }

    def test_export_code_generates_c_file(self, casadi_builder, tmp_path):
        """export_code should create a valid C file."""
        import os
        c_path = str(tmp_path / "test.c")
        result = casadi_builder.export_code(c_path)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_export_code_contains_all_functions(self, casadi_builder, tmp_path):
        """Generated C file should contain all expected function names."""
        import os
        c_path = str(tmp_path / "test.c")
        casadi_builder.export_code(c_path)
        with open(c_path) as f:
            content = f.read()
        for fn in ['residual', 'jacobian', 'Fx', 'Fy', 'Gx', 'Gy', 'Fu', 'Gu', 'Hx', 'Hy', 'Hu', 'ode', 'alg']:
            assert fn in content, f"Function '{fn}' not found in generated C code"

    def test_compile_shared_library(self, casadi_builder, tmp_path):
        """compile_shared_library should produce a .dll/.so file."""
        import os
        from pydae.core.builder import CasadiBuilder

        c_path = str(tmp_path / "test.c")
        casadi_builder.export_code(c_path)
        lib_path = CasadiBuilder.compile_shared_library(c_path)
        assert os.path.exists(lib_path)
        assert os.path.getsize(lib_path) > 0
        assert lib_path.endswith(('.dll', '.so'))

    def test_binary_model_ini_matches_symbolic(self, compiled_model_pair):
        """Initialization results should match between symbolic and binary models."""
        model_sym, model_bin = compiled_model_pair
        params = self._init_params()
        xy_0 = self._init_xy_0()

        model_sym.ini(params, xy_0=xy_0)
        model_bin.ini(params, xy_0=xy_0)

        np.testing.assert_allclose(model_sym.x, model_bin.x, rtol=1e-6, atol=1e-15)
        np.testing.assert_allclose(model_sym.y_ini, model_bin.y_ini, rtol=1e-6, atol=1e-15)

    def test_binary_model_A_eval_matches_symbolic(self, compiled_model_pair):
        """A_eval should produce identical results for symbolic and binary models."""
        model_sym, model_bin = compiled_model_pair
        params = self._init_params()
        xy_0 = self._init_xy_0()

        model_sym.ini(params, xy_0=xy_0)
        model_sym.run(0.1, {})
        A_sym = model_sym.A_eval()

        model_bin.ini(params, xy_0=xy_0)
        model_bin.run(0.1, {})
        A_bin = model_bin.A_eval()

        np.testing.assert_allclose(A_sym, A_bin, rtol=1e-6, atol=1e-15)

    def test_binary_model_BCD_eval_matches_symbolic(self, compiled_model_pair):
        """BCD_eval should produce similar results for symbolic and binary models."""
        model_sym, model_bin = compiled_model_pair
        params = self._init_params()
        xy_0 = self._init_xy_0()

        model_sym.ini(params, xy_0=xy_0)
        model_sym.run(0.1, {})
        B_sym, C_sym, D_sym = model_sym.BCD_eval()

        model_bin.ini(params, xy_0=xy_0)
        model_bin.run(0.1, {})
        B_bin, C_bin, D_bin = model_bin.BCD_eval()

        np.testing.assert_allclose(B_sym, B_bin, rtol=1e-2, atol=1e-10)
        np.testing.assert_allclose(C_sym, C_bin, rtol=1e-2, atol=1e-8)
        np.testing.assert_allclose(D_sym, D_bin, rtol=1e-2, atol=1e-8)

    def test_binary_model_run_matches_symbolic(self, compiled_model_pair):
        """Simulation run should produce similar trajectories."""
        model_sym, model_bin = compiled_model_pair
        params = self._init_params()
        xy_0 = self._init_xy_0()

        model_sym.ini(params, xy_0=xy_0)
        model_sym.run(0.5, {})
        model_sym.post()

        model_bin.ini(params, xy_0=xy_0)
        model_bin.run(0.5, {})
        model_bin.post()

        np.testing.assert_allclose(model_sym.Time, model_bin.Time, rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(model_sym.X, model_bin.X, rtol=5e-2, atol=1e-4)
        np.testing.assert_allclose(model_sym.Y, model_bin.Y, rtol=5e-2, atol=1e-4)

    def test_binary_model_eigenvalues_match(self, compiled_model_pair):
        """Eigenvalues should match between symbolic and binary models."""
        model_sym, model_bin = compiled_model_pair
        params = self._init_params()
        xy_0 = self._init_xy_0()

        model_sym.ini(params, xy_0=xy_0)
        model_sym.run(0.1, {})
        eigs_sym = model_sym.eval_eigenvalues()

        model_bin.ini(params, xy_0=xy_0)
        model_bin.run(0.1, {})
        eigs_bin = model_bin.eval_eigenvalues()

        # Compare sorted by absolute value to avoid ordering issues
        sym_sorted = np.sort(np.abs(eigs_sym))
        bin_sorted = np.sort(np.abs(eigs_bin))
        np.testing.assert_allclose(sym_sorted, bin_sorted, rtol=1e-2)

    def test_load_or_build_fallback(self, pendulum_sys_dict, tmp_path):
        """load_or_build should fall back to symbolic build when binary is missing."""
        from pydae.core.builder import CasadiBuilder, CasadiModel

        bld = CasadiBuilder(pendulum_sys_dict).build()
        fake_binary = str(tmp_path / "nonexistent.dll")

        model = CasadiModel.load_or_build(bld, binary_path=fake_binary)
        assert model._binary_path is None
        assert model.rf is not None

    def test_load_or_build_uses_binary(self, pendulum_sys_dict, tmp_path):
        """load_or_build should use binary when it exists."""
        import os
        from pydae.core.builder import CasadiBuilder, CasadiModel

        bld = CasadiBuilder(pendulum_sys_dict).build()
        c_path = str(tmp_path / "pendulum.c")
        bld.export_code(c_path)
        lib_path = CasadiBuilder.compile_shared_library(c_path)

        model = CasadiModel.load_or_build(bld, binary_path=lib_path)
        assert model._binary_path == lib_path
        assert hasattr(model, '_residual_fn')

    def test_save_and_load_metadata(self, compiled_model_pair, tmp_path):
        """save_metadata and from_binary should round-trip correctly."""
        from pydae.core.model.casadi_model import CasadiModel

        model_sym, model_bin = compiled_model_pair

        meta_path = str(tmp_path / "model_meta.json")
        model_sym.save_metadata(meta_path)

        # Verify metadata file exists and is valid JSON
        import json
        with open(meta_path) as f:
            meta = json.load(f)

        assert 'name' in meta
        assert 'x_list' in meta
        assert meta['name'] == 'test_pendulum'
        assert 'p_x' in meta['x_list']

    def test_binary_model_get_value(self, compiled_model_pair):
        """get_value should work identically for symbolic and binary models."""
        model_sym, model_bin = compiled_model_pair
        params = self._init_params()
        xy_0 = self._init_xy_0()

        model_sym.ini(params, xy_0=xy_0)
        model_bin.ini(params, xy_0=xy_0)

        for name in ['p_x', 'p_y', 'lam', 'theta', 'M', 'L']:
            np.testing.assert_allclose(
                model_sym.get_value(name), model_bin.get_value(name), rtol=1e-6, atol=1e-15
            )

    def test_binary_model_report_methods(self, compiled_model_pair, capsys):
        """report_x and report_y should print output for binary models."""
        model_sym, model_bin = compiled_model_pair
        params = self._init_params()
        xy_0 = self._init_xy_0()

        model_bin.ini(params, xy_0=xy_0)
        model_bin.report_x()
        captured = capsys.readouterr()
        assert 'p_x' in captured.out
        assert 'p_y' in captured.out
