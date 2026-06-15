# tests/core/test_control_pi_lpf1_loop.py
"""
Tests for a small feedback control system built from two
``pydae.control.continuos`` blocks: a PI controller
(``pydae.control.continuos.pi``) driving a first-order plant modelled
as a low-pass filter (``pydae.control.continuos.lpf1``), closed in
unity negative feedback around a reference ``r``:

    r --> [+ -] --> PI --> plant (LPF1) --> y
           ^                                |
           |________________________________|

Equations added on top of the two blocks:
    u_lpf1 = z_pi          (PI output drives the plant input)
    u_pi   = r - z_lpf1    (unity negative feedback: error = r - y)

This exercises the integral action of the PI together with the genuine
dynamics of a controlled plant (as opposed to the algebraic feedback
used in ``test_control_pi.py``), so the closed-loop response is a real
second-order system.

Run selectively:
    uv run pytest tests/core/test_control_pi_lpf1_loop.py -v
"""
import casadi as ca
import pytest

from pydae.control.continuos.pi import pi
from pydae.control.continuos.lpf1 import lpf1


def _build_pi_lpf1_loop_model(name='1', K_p=2.0, K_i=1.0, T=0.5, K_aw=2.0,
                               z_min=-10.0, z_max=10.0, r_0=0.0):
    """Build a standalone CasadiModel with a PI controller driving a
    first-order (LPF1) plant in unity negative feedback."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    dae = {'name': f'test_pi_lpf1_loop_{name}',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {},
           'h_dict': {},
           'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)

    # Controller: PI
    pi(dae, name, {'K_p': K_p, 'K_i': K_i, 'K_aw': K_aw,
                    'z_min': z_min, 'z_max': z_max,
                    'u': 0.0, 'xi_0': 0.0}, backend)

    # Plant: first-order low-pass filter
    lpf1(dae, name, {'T': T, 'u': 0.0}, backend)

    xi_pi = dae['x_list'][-2]
    x_plant = dae['x_list'][-1]
    z_pi = dae['h_dict'][f'z_pi_{name}']
    z_plant = dae['h_dict'][f'z_lpf1_{name}']

    u_pi = ca.symvar(dae['f_list'][0])[0]
    u_plant = ca.symvar(dae['f_list'][1])[0]

    # PI output drives the plant input: u_lpf1 = z_pi
    dae['u_ini_dict'].pop(str(u_plant))
    dae['u_run_dict'].pop(str(u_plant))
    g_connect = u_plant - z_pi
    dae['g_list'] += [g_connect]
    dae['y_ini_list'] += [u_plant]
    dae['y_run_list'] += [u_plant]

    # Unity negative feedback: u_pi = r - y (y = plant output)
    dae['u_ini_dict'].pop(str(u_pi))
    dae['u_run_dict'].pop(str(u_pi))
    r = backend.symbols(f'r_pi_{name}')
    g_loop = u_pi - (r - z_plant)
    dae['g_list'] += [g_loop]
    dae['y_ini_list'] += [u_pi]
    dae['y_run_list'] += [u_pi]
    dae['u_ini_dict'].update({f"{r}": r_0})
    dae['u_run_dict'].update({f"{r}": r_0})

    dae['xy_0_dict'].update({f"{u_pi}": 0.0, f"{u_plant}": 0.0,
                              f"{xi_pi}": 0.0, f"{x_plant}": 0.0})

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])

    return model


@pytest.mark.model
class TestPiLpf1Loop:

    def test_steady_state_zero_error(self):
        """At steady state, integral action drives the error to zero, so
        the plant output (filter output z_lpf1) equals the reference r
        and xi_pi = r / K_i."""
        K_p, K_i, r_0 = 2.0, 1.0, 0.0
        model = _build_pi_lpf1_loop_model(name='1', K_p=K_p, K_i=K_i, r_0=r_0)

        assert abs(model.get_value('z_lpf1_1') - r_0) < 1e-6
        assert abs(model.get_value('u_pi_1') - 0.0) < 1e-6
        assert abs(model.get_value('xi_pi_1') - r_0 / K_i) < 1e-6

    def test_step_response_settles_to_reference(self):
        """A step in the reference r drives the plant output y = z_lpf1
        gradually (through the plant's first-order dynamics) to r, with
        zero steady-state error and xi_pi(inf) = r / K_i.
        """
        name, K_p, K_i, T, r_1 = '1', 2.0, 1.0, 0.5, 1.0
        model = _build_pi_lpf1_loop_model(name=name, K_p=K_p, K_i=K_i, T=T, r_0=0.0)

        model.run(20.0, {f'r_pi_{name}': r_1})
        model.post()

        y = model.get_values(f'z_lpf1_{name}')
        xi = model.get_values(f'xi_pi_{name}')

        assert y[1] < 0.5 * r_1            # plant smooths the step, no instant jump
        assert y[1] < y[-1]                 # response builds up gradually
        assert abs(y[-1] - r_1) < 0.01      # zero steady-state error
        assert abs(xi[-1] - r_1 / K_i) < 0.02

    def test_anti_windup_with_saturated_controller(self):
        """If the controller output is saturated (z_pi clamped), the
        plant output cannot reach the reference and a steady-state error
        remains -- but the anti-windup term K_aw*(z_pi - v) keeps xi_pi
        bounded at a finite value instead of winding up.
        """
        name = '1'
        K_p, K_i, K_aw, T = 2.0, 1.0, 2.0, 0.5
        z_min, z_max = -1.0, 1.0
        r_1 = 2.0  # would require z_pi = 2 at steady state -> saturates at z_max=1

        model = _build_pi_lpf1_loop_model(name=name, K_p=K_p, K_i=K_i, T=T, K_aw=K_aw,
                                           z_min=z_min, z_max=z_max, r_0=0.0)

        model.run(20.0, {f'r_pi_{name}': r_1})
        model.post()

        z_pi = model.get_values(f'z_pi_{name}')
        y = model.get_values(f'z_lpf1_{name}')
        xi = model.get_values(f'xi_pi_{name}')

        assert abs(z_pi[-1] - z_max) < 1e-3   # controller output saturated
        assert abs(y[-1] - z_max) < 1e-3      # plant tracks the saturated input
        assert y[-1] < r_1                     # persistent steady-state error
        assert abs(xi[-1]) < 10.0              # integrator stays bounded (no windup)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 6))

    # --- Unsaturated step response -------------------------------------
    name, K_p, K_i, T, r_1 = '1', 2.0, 1.0, 0.5, 1.0
    model = _build_pi_lpf1_loop_model(name=name, K_p=K_p, K_i=K_i, T=T, r_0=0.0)
    model.run(20.0, {f'r_pi_{name}': r_1})
    model.post()

    ax = axes[0]
    ax.plot(model.Time, model.get_values(f'z_lpf1_{name}'), label='$y$ (plant output)')
    ax.plot(model.Time, model.get_values(f'z_pi_{name}'), label='$z_{pi}$ (controller output)')
    ax.plot(model.Time, model.get_values(f'xi_pi_{name}'), label=r'$\xi_{pi}$ (integrator state)')
    ax.axhline(r_1, color='k', ls='--', lw=0.8, label='$r$ (reference)')
    ax.set_title('Unsaturated step response')
    ax.set_ylabel('Output')
    ax.legend()

    # --- Saturated / anti-windup step response --------------------------
    K_aw, z_min, z_max, r_2 = 2.0, -1.0, 1.0, 2.0
    model_sat = _build_pi_lpf1_loop_model(name=name, K_p=K_p, K_i=K_i, T=T, K_aw=K_aw,
                                           z_min=z_min, z_max=z_max, r_0=0.0)
    model_sat.run(20.0, {f'r_pi_{name}': r_2})
    model_sat.post()

    ax = axes[1]
    ax.plot(model_sat.Time, model_sat.get_values(f'z_lpf1_{name}'), label='$y$ (plant output)')
    ax.plot(model_sat.Time, model_sat.get_values(f'z_pi_{name}'), label='$z_{pi}$ (controller output)')
    ax.plot(model_sat.Time, model_sat.get_values(f'xi_pi_{name}'), label=r'$\xi_{pi}$ (integrator state)')
    ax.axhline(r_2, color='k', ls='--', lw=0.8, label='$r$ (reference)')
    ax.axhline(z_max, color='gray', ls=':', lw=0.8, label='$z_{max}$')
    ax.set_title('Saturated step response (anti-windup)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output')
    ax.legend()

    fig.tight_layout()
    fig.savefig('pi_lpf1_loop_response.svg')
    print("Plot saved as 'pi_lpf1_loop_response.svg'.")
