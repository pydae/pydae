# tests/core/test_control_lpf1.py
"""
Tests for the first-order low-pass filter (LPF1) continuous control block
(``pydae.control.continuos.lpf1``), built and run on the CasADi backend.

Run selectively:
    uv run pytest tests/core/test_control_lpf1.py -v
"""
import numpy as np
import pytest

from pydae.control.continuos.lpf1 import descriptions, lpf1


@pytest.fixture
def lpf1_model():
    """Build a standalone CasadiModel containing a single LPF1 block."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    dae = {'name': 'test_lpf1',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {},
           'h_dict': {},
           'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    lpf1(dae, '1', {'T': 0.5, 'u': 0.0}, backend)

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])

    return model


@pytest.mark.parse
def test_descriptions_schema():
    """descriptions() should list T, u, x and z with required keys."""
    meta = descriptions()
    by_model = {item['model']: item for item in meta}

    assert set(by_model) == {'T', 'u', 'x', 'z'}
    assert by_model['T']['type'] == 'Parameter'
    assert by_model['u']['type'] == 'Input'
    assert by_model['x']['type'] == 'Dynamic State'
    assert by_model['z']['type'] == 'Algebraic State'

    for item in meta:
        assert {'type', 'tex', 'data', 'model', 'default', 'description', 'units'} <= set(item)


@pytest.mark.model
class TestLpf1:

    def test_steady_state_tracks_input(self, lpf1_model):
        """At steady state (du/dt = 0), the filter output z equals the input u."""
        model = lpf1_model
        assert abs(model.get_value('z_lpf1_1') - 0.0) < 1e-6
        assert abs(model.get_value('x_lpf1_1') - 0.0) < 1e-6

    def test_step_response_time_constant(self, lpf1_model):
        """A step in u produces the classic first-order response: ~63.2% of
        the step amplitude after one time constant T, settling to the new
        input value at steady state (t >> T).
        """
        model = lpf1_model
        T = 0.5

        model.run(5 * T, {'u_lpf1_1': 1.0})
        model.post()

        t = model.Time
        z = model.get_values('z_lpf1_1')

        idx_T = np.argmin(np.abs(t - T))
        assert abs(z[idx_T] - (1 - np.exp(-1))) < 0.02  # ~63.2% after one T
        assert abs(z[-1] - 1.0) < 0.01                   # settled after 5T
