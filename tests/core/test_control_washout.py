# tests/core/test_control_washout.py
"""
Tests for the washout (high-pass) filter continuous control block
(``pydae.control.continuos.washout``), built and run on the CasADi
backend.

Run selectively:
    uv run pytest tests/core/test_control_washout.py -v
"""
import pytest

from pydae.control.continuos.washout import descriptions, washout


@pytest.fixture
def washout_model():
    """Build a standalone CasadiModel containing a single WASHOUT block."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    dae = {'name': 'test_washout',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {},
           'h_dict': {},
           'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    washout(dae, '1', {'T_wo': 2.0, 'u': 0.0}, backend)

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])

    return model


@pytest.mark.parse
def test_descriptions_schema():
    """descriptions() should list T_wo, u, x and z with required keys."""
    meta = descriptions()
    by_model = {item['model']: item for item in meta}

    assert set(by_model) == {'T_wo', 'u', 'x', 'z'}
    assert by_model['T_wo']['type'] == 'Parameter'
    assert by_model['u']['type'] == 'Input'
    assert by_model['x']['type'] == 'Dynamic State'
    assert by_model['z']['type'] == 'Algebraic State'

    for item in meta:
        assert {'type', 'tex', 'data', 'model', 'default', 'description', 'units'} <= set(item)


@pytest.mark.model
class TestWashout:

    def test_steady_state_output_is_zero(self, washout_model):
        """At steady state (du/dt = 0), the washout output z is zero,
        regardless of the operating-point value of u."""
        model = washout_model
        assert abs(model.get_value('z_washout_1') - 0.0) < 1e-6
        assert abs(model.get_value('x_washout_1') - 0.0) < 1e-6

    def test_step_response_decays_to_zero(self, washout_model):
        """A step in u produces an immediate unity-gain jump in z, which
        then decays to zero with time constant T_wo.
        """
        model = washout_model
        T_wo = 2.0

        model.run(5 * T_wo, {'u_washout_1': 1.0})
        model.post()

        z = model.get_values('z_washout_1')

        assert abs(max(z) - 1.0) < 0.02  # initial unity-gain jump
        assert abs(z[-1] - 0.0) < 0.01   # washed out after 5*T_wo
