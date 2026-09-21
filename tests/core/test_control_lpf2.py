# tests/core/test_control_lpf2.py
"""
Tests for the second-order low-pass filter (LPF2) continuous control
block (``pydae.control.continuos.lpf2``), built and run on the CasADi
backend.

Run selectively:
    uv run pytest tests/core/test_control_lpf2.py -v
"""
import pytest

from pydae.control.continuos.lpf2 import descriptions, lpf2


@pytest.fixture
def lpf2_model():
    """Build a standalone CasadiModel containing a single LPF2 block."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    dae = {'name': 'test_lpf2',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {},
           'h_dict': {},
           'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    lpf2(dae, '1', backend, {'omega_n': 10.0, 'zeta': 0.5, 'u': 0.0})

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])

    return model


@pytest.mark.parse
def test_descriptions_schema():
    """descriptions() should list omega_n, zeta, u, x, v and z with required keys."""
    meta = descriptions()
    by_model = {item['model']: item for item in meta}

    assert set(by_model) == {'omega_n', 'zeta', 'u', 'x', 'v', 'z'}
    assert by_model['omega_n']['type'] == 'Parameter'
    assert by_model['zeta']['type'] == 'Parameter'
    assert by_model['u']['type'] == 'Input'
    assert by_model['x']['type'] == 'Dynamic State'
    assert by_model['v']['type'] == 'Dynamic State'
    assert by_model['z']['type'] == 'Algebraic State'

    for item in meta:
        assert {'type', 'tex', 'data', 'model', 'default', 'description', 'units'} <= set(item)


@pytest.mark.model
class TestLpf2:

    def test_steady_state_tracks_input(self, lpf2_model):
        """At steady state (du/dt = 0), the filter output z equals the
        input u and the rate state v is zero."""
        model = lpf2_model
        assert abs(model.get_value('z_lpf2_1') - 0.0) < 1e-6
        assert abs(model.get_value('x_lpf2_1') - 0.0) < 1e-6
        assert abs(model.get_value('v_lpf2_1') - 0.0) < 1e-6

    def test_underdamped_step_response_overshoots(self, lpf2_model):
        """For zeta < 1, a step in u produces an overshoot above the new
        steady-state value, which then settles back to u."""
        model = lpf2_model

        model.run(2.0, {'u_lpf2_1': 1.0})
        model.post()

        z = model.get_values('z_lpf2_1')

        assert max(z) > 1.05            # overshoot present
        assert max(z) < 1.30            # within expected range (~16% for zeta=0.5)
        assert abs(z[-1] - 1.0) < 0.01  # settled


@pytest.mark.model
def test_critically_damped_step_response_no_overshoot():
    """For zeta >= 1, a step in u produces a monotonic response with no
    overshoot, settling to u."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    dae = {'name': 'test_lpf2_critical',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {},
           'h_dict': {},
           'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    lpf2(dae, '1', backend, {'omega_n': 10.0, 'zeta': 1.0, 'u': 0.0})

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])

    model.run(2.0, {'u_lpf2_1': 1.0})
    model.post()

    z = model.get_values('z_lpf2_1')

    assert z[-1] == max(z)           # monotonic rise, no overshoot
    assert abs(z[-1] - 1.0) < 0.01   # settled
