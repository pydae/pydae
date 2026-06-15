# tests/core/test_control_stab_1lpf_2wo_3ll.py
"""
Tests for the 2-washout / 3-lead-lag stabilizer filter chain
(``pydae.control.continuos.stab_1lpf_2wo_3ll``), built and run on the
CasADi backend.

This block is assembled from the basic ``lpf1``, ``washout`` and
``leadlag`` blocks: ``lpf1 -> washout -> washout -> leadlag -> leadlag
-> leadlag -> saturation``. The two washout stages give the composite
block zero steady-state gain (any constant input settles to z = 0),
while the low-pass filter and lead-lag stages shape the transient
response, which is finally clipped to +-Limit.

Run selectively:
    uv run pytest tests/core/test_control_stab_1lpf_2wo_3ll.py -v
"""
import pytest

from pydae.control.continuos.stab_1lpf_2wo_3ll import descriptions, stab_1lpf_2wo_3ll


def _build_stab_1lpf_2wo_3ll_model(name='1', T_lpf=0.02, T_wo1=2.0, T_wo2=2.0,
                               T_1=0.3, T_2=0.05, T_3=0.3, T_4=0.05,
                               T_5=0.3, T_6=0.05, Limit=10.0, u_0=0.0):
    """Build a standalone CasadiModel containing a single STAB_1LPF_2WO_3LL block."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    dae = {'name': f'test_stab_1lpf_2wo_3ll_loop_{name}',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {},
           'h_dict': {},
           'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    stab_1lpf_2wo_3ll(dae, name, backend, {'T_lpf': T_lpf, 'T_wo1': T_wo1, 'T_wo2': T_wo2,
                              'T_1': T_1, 'T_2': T_2, 'T_3': T_3, 'T_4': T_4,
                              'T_5': T_5, 'T_6': T_6, 'Limit': Limit,
                              'u': u_0})

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])

    return model


@pytest.mark.parse
def test_descriptions_schema():
    """descriptions() should list the 11 parameters, the input u, the
    six internal states and the output z with required keys."""
    meta = descriptions()
    by_model = {item['model']: item for item in meta}

    assert set(by_model) == {
        'T_lpf', 'T_wo1', 'T_wo2', 'T_1', 'T_2', 'T_3', 'T_4', 'T_5', 'T_6', 'K_stab', 'Limit',
        'u', 'x_lpf', 'x_wo1', 'x_wo2', 'x_ll1', 'x_ll2', 'x_ll3', 'z',
    }
    assert by_model['u']['type'] == 'Input'
    for state in ['x_lpf', 'x_wo1', 'x_wo2', 'x_ll1', 'x_ll2', 'x_ll3']:
        assert by_model[state]['type'] == 'Dynamic State'
    assert by_model['z']['type'] == 'Algebraic State'
    for param in ['T_lpf', 'T_wo1', 'T_wo2', 'T_1', 'T_2', 'T_3', 'T_4', 'T_5', 'T_6', 'Limit']:
        assert by_model[param]['type'] == 'Parameter'

    for item in meta:
        assert {'type', 'tex', 'data', 'model', 'default', 'description', 'units'} <= set(item)


@pytest.mark.model
class TestStab2wo3ll:

    def test_steady_state_output_is_zero(self):
        """At steady state (du/dt = 0), the two washout stages drive
        the output z to zero, regardless of the operating-point value
        of u."""
        model = _build_stab_1lpf_2wo_3ll_model(name='1', u_0=0.7)

        assert abs(model.get_value('z_stab_1lpf_2wo_3ll_1') - 0.0) < 1e-6

    def test_step_response_transient_then_decays_to_zero(self):
        """A step in u excites a transient (shaped by the low-pass
        filter and the three lead-lag stages) which then decays back
        to zero as the washout stages remove the new DC component.
        """
        name = '1'
        T_wo1, T_wo2 = 2.0, 2.0
        model = _build_stab_1lpf_2wo_3ll_model(name=name, T_wo1=T_wo1, T_wo2=T_wo2, u_0=0.0)

        model.run(10 * max(T_wo1, T_wo2), {f'u_stab_1lpf_2wo_3ll_{name}': 1.0})
        model.post()

        z = model.get_values(f'z_stab_1lpf_2wo_3ll_{name}')

        assert max(abs(z)) > 0.1         # transient excitation
        assert abs(z[-1] - 0.0) < 0.01   # washed out at steady state

    def test_saturation_limits_output(self):
        """A large step in u drives the unsaturated chain output
        beyond +-Limit; the output z stays clipped at the saturation
        limit during the transient.
        """
        name = '1'
        Limit = 0.05
        model = _build_stab_1lpf_2wo_3ll_model(name=name, Limit=Limit, u_0=0.0)

        model.run(0.5, {f'u_stab_1lpf_2wo_3ll_{name}': 1.0})
        model.post()

        z = model.get_values(f'z_stab_1lpf_2wo_3ll_{name}')

        assert abs(max(abs(z)) - Limit) < 1e-3
        assert (abs(z) <= Limit + 1e-6).all()
