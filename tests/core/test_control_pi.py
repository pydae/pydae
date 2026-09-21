# tests/core/test_control_pi.py
"""
Tests for the PI controller continuous control block
(``pydae.control.continuos.pi``), built and run on the CasADi backend.

A bare integrator has no equilibrium of its own (any constant value of
``xi`` satisfies ``dxi/dt = 0`` whenever ``u = 0``), so initializing an
*isolated* PI block is structurally singular. These tests close a
unity-negative-feedback loop around a reference ``r``
(``u_pi = r - z_pi``) — exactly how a PI block is always used in
practice — which makes initialization well-posed and exercises the
integral action.

Run selectively:
    uv run pytest tests/core/test_control_pi.py -v
"""
import casadi as ca
import pytest

from pydae.control.continuos.pi import descriptions, pi


def _build_pi_loop_model(name='1', K_p=2.0, K_i=0.5, K_aw=2.0,
                          z_min=-10.0, z_max=10.0, r_0=0.0):
    """Build a standalone CasadiModel containing a single PI block in
    unity negative feedback: ``u_pi = r - z_pi``."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    dae = {'name': f'test_pi_loop_{name}',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {},
           'h_dict': {},
           'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    pi(dae, name, backend, {'K_p': K_p, 'K_i': K_i, 'K_aw': K_aw,
                            'z_min': z_min, 'z_max': z_max,
                            'u': 0.0, 'xi_0': 0.0})

    xi = dae['x_list'][-1]
    z = dae['h_dict'][f'z_pi_{name}']
    u = ca.symvar(dae['f_list'][-1])[0]

    dae['u_ini_dict'].pop(str(u))
    dae['u_run_dict'].pop(str(u))

    r = backend.symbols(f'r_pi_{name}')
    g_loop = u - (r - z)

    dae['g_list'] += [g_loop]
    dae['y_ini_list'] += [u]
    dae['y_run_list'] += [u]
    dae['u_ini_dict'].update({f"{r}": r_0})
    dae['u_run_dict'].update({f"{r}": r_0})
    dae['xy_0_dict'].update({f"{u}": 0.0, f"{xi}": 0.0})

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])

    return model


@pytest.mark.parse
def test_descriptions_schema():
    """descriptions() should list K_p, K_i, K_aw, z_min, z_max, u, xi and
    z with required keys."""
    meta = descriptions()
    by_model = {item['model']: item for item in meta}

    assert set(by_model) == {'K_p', 'K_i', 'K_aw', 'z_min', 'z_max', 'u', 'xi', 'z'}
    assert by_model['K_p']['type'] == 'Parameter'
    assert by_model['K_i']['type'] == 'Parameter'
    assert by_model['K_aw']['type'] == 'Parameter'
    assert by_model['z_min']['type'] == 'Parameter'
    assert by_model['z_max']['type'] == 'Parameter'
    assert by_model['u']['type'] == 'Input'
    assert by_model['xi']['type'] == 'Dynamic State'
    assert by_model['z']['type'] == 'Algebraic State'

    for item in meta:
        assert {'type', 'tex', 'data', 'model', 'default', 'description', 'units'} <= set(item)


@pytest.mark.model
class TestPi:

    def test_closed_loop_zero_steady_state_error(self):
        """At steady state, integral action drives the error to zero, so
        the PI output z equals the reference r and xi = r / K_i."""
        K_p, K_i, r_0 = 2.0, 0.5, 0.0
        model = _build_pi_loop_model(name='1', K_p=K_p, K_i=K_i, r_0=r_0)

        assert abs(model.get_value('z_pi_1') - r_0) < 1e-6
        assert abs(model.get_value('u_pi_1') - 0.0) < 1e-6
        assert abs(model.get_value('xi_pi_1') - r_0 / K_i) < 1e-6

    def test_step_response_proportional_jump_then_zero_error(self):
        """A step in the reference r produces an immediate jump in z
        driven by K_p, then settles with zero steady-state error driven
        by K_i: z(0+) = K_p/(1+K_p) * r, z(inf) = r, xi(inf) = r / K_i.
        """
        name, K_p, K_i, r_1 = '1', 2.0, 0.5, 1.0
        model = _build_pi_loop_model(name=name, K_p=K_p, K_i=K_i, r_0=0.0)

        model.run(30.0, {f'r_pi_{name}': r_1})
        model.post()

        z = model.get_values(f'z_pi_{name}')
        xi = model.get_values(f'xi_pi_{name}')

        assert abs(z[1] - K_p / (1 + K_p) * r_1) < 0.05  # immediate jump
        assert abs(z[-1] - r_1) < 0.02                    # zero steady-state error
        assert abs(xi[-1] - r_1 / K_i) < 0.05

    def test_anti_windup_limits_integrator_at_saturation(self):
        """When the reference drives the unsaturated output beyond
        [z_min, z_max], the back-calculation term K_aw*(z - v) keeps
        the integrator state xi bounded at a finite value instead of
        winding up without limit, and z stays clamped at z_max.
        """
        name = '1'
        K_p, K_i, K_aw = 2.0, 0.5, 2.0
        z_min, z_max = -1.0, 1.0
        r_1 = 2.0  # outside [z_min, z_max] -> forces saturation

        model = _build_pi_loop_model(name=name, K_p=K_p, K_i=K_i, K_aw=K_aw,
                                      z_min=z_min, z_max=z_max, r_0=0.0)

        model.run(5.0, {f'r_pi_{name}': r_1})
        model.post()

        z = model.get_values(f'z_pi_{name}')
        xi = model.get_values(f'xi_pi_{name}')

        assert abs(z[1] - z_max) < 1e-3    # output saturates immediately
        assert abs(z[-1] - z_max) < 1e-3   # remains saturated
        assert abs(xi[-1] - (-1.0)) < 0.02  # integrator settles at a finite value
