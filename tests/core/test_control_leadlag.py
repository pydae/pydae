# tests/core/test_control_leadlag.py
"""
Tests for the first-order lead-lag compensator (LEADLAG) continuous
control block (``pydae.control.continuos.leadlag``), built and run on
the CasADi backend.

Run selectively:
    uv run pytest tests/core/test_control_leadlag.py -v
"""
import numpy as np
import pytest

from pydae.control.continuos.leadlag import descriptions, leadlag


@pytest.fixture
def leadlag_model():
    """Build a standalone CasadiModel containing a single LEADLAG block."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    dae = {'name': 'test_leadlag',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {},
           'h_dict': {},
           'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    leadlag(dae, '1', {'T_num': 1.0, 'T_den': 0.5, 'u': 0.0}, backend)

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])

    return model


@pytest.mark.parse
def test_descriptions_schema():
    """descriptions() should list T_num, T_den, u, x and z with required keys."""
    meta = descriptions()
    by_model = {item['model']: item for item in meta}

    assert set(by_model) == {'T_num', 'T_den', 'u', 'x', 'z'}
    assert by_model['T_num']['type'] == 'Parameter'
    assert by_model['T_den']['type'] == 'Parameter'
    assert by_model['u']['type'] == 'Input'
    assert by_model['x']['type'] == 'Dynamic State'
    assert by_model['z']['type'] == 'Algebraic State'

    for item in meta:
        assert {'type', 'tex', 'data', 'model', 'default', 'description', 'units'} <= set(item)


@pytest.mark.model
class TestLeadLag:

    def test_steady_state_is_transparent(self, leadlag_model):
        """At steady state (du/dt = 0), the output z equals the input u
        regardless of T_num and T_den."""
        model = leadlag_model
        assert abs(model.get_value('z_leadlag_1') - 0.0) < 1e-6
        assert abs(model.get_value('x_leadlag_1') - 0.0) < 1e-6

    def test_step_response_lead_then_decay(self, leadlag_model):
        """A step in u produces an initial jump of size T_num/T_den (the
        lead action), which then decays with time constant T_den to the
        new steady-state value u (z = u).
        """
        model = leadlag_model
        T_num, T_den = 1.0, 0.5

        model.run(5 * T_den, {'u_leadlag_1': 1.0})
        model.post()

        z = model.get_values('z_leadlag_1')

        assert abs(max(z) - T_num / T_den) < 0.05  # initial lead overshoot
        assert abs(z[-1] - 1.0) < 0.02              # settled after 5*T_den


@pytest.mark.model
def test_zero_T_num_matches_lpf1():
    """With T_num = 0 the block degenerates into a plain first-order
    low-pass filter with time constant T_den (no lead overshoot)."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    T_den = 0.5
    dae = {'name': 'test_leadlag_pure_lag',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {},
           'h_dict': {},
           'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    leadlag(dae, '1', {'T_num': 0.0, 'T_den': T_den, 'u': 0.0}, backend)

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])

    model.run(T_den, {'u_leadlag_1': 1.0})
    model.post()

    z = model.get_values('z_leadlag_1')

    # No lead overshoot: monotonic rise, max value at the end of the run.
    assert z[-1] == max(z)
