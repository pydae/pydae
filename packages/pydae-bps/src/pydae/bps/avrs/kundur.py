# -*- coding: utf-8 -*-
r"""
Kundur simple AVR with PV-bus initialisation.

The model is a minimal textbook excitation system (Kundur, *Power
System Stability and Control*, chapter 8): a first-order
terminal-voltage sensor followed by a proportional gain with a fixed
bias and hard field limits. It has a single dynamic state and one
algebraic equation, with no artificial initialisation integrator — the
initialisation is handled by the ``ini``/``run`` variable swap
described below.

**Signal path**

The terminal voltage is sensed through a first-order lag with time
constant $T_r$:

$$\frac{d v_1}{dt} = \frac{V - v_1}{T_r}$$

where $V$ is the regulated (terminal or remote) bus voltage. The
voltage error feeds a proportional gain with a bias of $1.5$ pu, which
positions the no-load operating point near $v_f \approx 1.5$:

$$e_{fd}^{nosat} = K_a \left( v^{\star} - v_1 + v_s \right) + 1.5$$

The output is passed through a hard limiter and returned as the
algebraic variable $v_f$ via the residual

$$0 = \mathrm{sat}\!\left(e_{fd}^{nosat},\; E_{fmin},\; E_{fmax}\right) - v_f$$

with $\mathrm{sat}(x, a, b) = \min\{b,\, \max\{a,\, x\}\}$.

**The ini/run variable swap**

For a generator bus held at a voltage setpoint, the initialisation
problem is PV (active power and voltage magnitude specified, reactive
power and voltage reference unknown) while the subsequent time-domain
simulation is reference-driven (the reference is an input and the
voltage magnitude is solved from the network). ``pydae`` supports this
by allowing the ``y`` (algebraic) and ``u`` (input) partitions to
differ between the ``ini`` and ``run`` phases while sharing the same
set of residual equations $g$.

The table below shows how each quantity is classified in each phase:

                    ini                 run
    v_f             y_ini               y_run      (algebraic, always solved)
    v_ref           y_ini               u_run      (unknown in ini, input in run)
    V_bus           u_ini               y_run      (pinned in ini, solved in run)

List cardinalities are preserved: ``|y_ini| == |y_run|`` and ``|g|`` is
unchanged. The swap is performed in place at the existing ``y_ini``
index of ``V_bus`` (added earlier by the bus builder) so that
downstream components that reference ``y_ini`` by integer index — for
example ``vsource``'s ``g[idx_V] = ...`` override — continue to target
the correct equations.

This replaces the earlier ``xi_v`` dummy-integrator approach (a fake
state with $d\xi_v/dt = v^{\star} - v_1$ and a tiny gain $K_{ai}$) that
was used to pin $V$ to $v^{\star}$ during ``ini``. The swap is cleaner
(one fewer state, one fewer equation, no spurious near-zero eigenvalue)
and strictly equivalent at the steady state.

**Value transfer between phases**

After ``ini()`` converges, ``ini2run()`` automatically copies solved
values to the run-phase state:

- ``v_ref`` is in ``y_ini`` and appears in ``u_run``: its solved value
  becomes the run-phase input, so the operating point is preserved
  without the user having to pass ``v_ref`` manually.
- ``V_bus`` is in ``u_ini`` and appears in ``y_run``: its pinned
  setpoint is used as the starting value of the run-phase solver.

**Configuration**

Example data entry::

    "avr": {"type": "kundur", "K_a": 100.0, "T_r": 0.02,
            "E_fmin": -5.0, "E_fmax": 5.0, "v_ref": 1.0}

The ``v_ref`` field serves two roles: it is the **bus voltage setpoint
during ini()** (since ``v_ref`` itself is unknown there) and the
**initial run-phase input** (overwritten by the value solved during
ini via ``ini2run``). The optional ``bus`` key selects a remote bus
whose voltage is regulated; if omitted the generator bus is used.
"""


def descriptions():
    """Single source of truth for kundur parameters, inputs, states, and outputs."""
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "K_a", "data": "K_a",
                           "model": "K_a", "default": 100.0,
                           "description": "AVR proportional gain",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_r", "data": "T_r",
                           "model": "T_r", "default": 0.02,
                           "description": "Terminal-voltage sensor time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "E_{fmin}",
                           "data": "E_fmin", "model": "E_fmin", "default": -5.0,
                           "description": "Lower field-voltage limit",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "E_{fmax}",
                           "data": "E_fmax", "model": "E_fmax", "default": 5.0,
                           "description": "Upper field-voltage limit",
                           "units": "pu"}]

    # Inputs (run phase). During ini, v_ref is solved and V_bus is pinned
    # instead — see the module docstring.
    descriptions_list += [{"type": "Input", "tex": "v^{\\star}",
                           "data": "v_ref", "model": "v_ref",
                           "ieee": "V_ref", "default": 1.0,
                           "description": ("Voltage reference. Acts as the "
                                           "PV-bus setpoint during ini (where "
                                           "it is solved for) and as an input "
                                           "during run."),
                           "units": "pu"}]
    descriptions_list += [{"type": "Input", "tex": "v_s",
                           "data": "v_pss", "model": "v_pss",
                           "ieee": "V_s", "default": 0.0,
                           "description": ("Supplementary stabilising input "
                                           "(PSS output), added to the "
                                           "voltage error."),
                           "units": "pu"}]

    # Dynamic states
    descriptions_list += [{"type": "Dynamic State", "tex": "v_1",
                           "data": "", "model": "v_1", "default": "",
                           "description": "Sensed terminal voltage (first-order lag output)",
                           "units": "pu"}]

    # Algebraic state — solved in both ini and run phases.
    descriptions_list += [{"type": "Algebraic State", "tex": "v_f",
                           "data": "", "model": "v_f", "default": "",
                           "description": ("Field-voltage command sent to the "
                                           "synchronous machine exciter "
                                           "(saturated)."),
                           "units": "pu"}]

    return descriptions_list


def kundur(dae, data, name, bus_name, backend=None):
    """
    Example data entry::

        "avr": {"type": "kundur", "K_a": 100.0, "T_r": 0.02,
                 "E_fmin": -5.0, "E_fmax": 5.0, "v_ref": 1.0}

    The ``v_ref`` value supplied in data is used as the **bus voltage
    setpoint during ini()** (since v_ref is unknown there) and as the
    **initial guess / starting input during run()**.
    """
    if backend is None:
        import sympy as sym
        backend = type('Backend', (), {
            'symbols': lambda _, n, **k: sym.Symbol(n, real=True),
            'Piecewise': sym.Piecewise,
            'sin': sym.sin,
            'cos': sym.cos,
            'sqrt': sym.sqrt,
            'exp': sym.exp,
        })()

    avr_data = data['avr']
    remote_bus_name = bus_name
    if 'bus' in avr_data:
        remote_bus_name = avr_data['bus']

    v_t = backend.symbols(f"V_{remote_bus_name}", real=True)

    v_1 = backend.symbols(f"v_1_{name}", real=True)
    v_f = backend.symbols(f"v_f_{name}", real=True)

    K_a = backend.symbols(f"K_a_{name}", real=True)
    T_r = backend.symbols(f"T_r_{name}", real=True)
    E_fmin = backend.symbols(f"E_fmin_{name}", real=True)
    E_fmax = backend.symbols(f"E_fmax_{name}", real=True)

    v_ref = backend.symbols(f"v_ref_{name}", real=True)
    v_pss = backend.symbols(f"v_pss_{name}", real=True)

    v_s = v_pss

    # Signal path — no artificial xi_v / v_ini integrator.
    epsilon_v = v_ref - v_1 + v_s
    v_f_nosat = K_a * epsilon_v + 1.5

    # Sensor dynamics.
    dv_1 = (v_t - v_1) / T_r

    # Algebraic: v_f = saturated command.
    g_v_f = backend.Piecewise((E_fmin, v_f_nosat < E_fmin),
                          (E_fmax, v_f_nosat > E_fmax),
                          (v_f_nosat, True)) - v_f

    # --- ini/run variable partition ------------------------------------
    # Swap V_bus <-> v_ref at the same y_ini position. Replacing in place
    # keeps downstream index-based code (e.g. vsource's g[idx_V]=...) from
    # targeting the wrong equation when V_bus was just removed.
    #   ini: V_bus is u_ini (pinned), v_ref is y_ini (unknown)
    #   run: V_bus is y_run (unknown), v_ref is u_run (input, seeded from ini)

    v_setpoint = avr_data['v_ref']
    v_t_str = str(v_t)
    if v_t_str in [str(y) for y in dae['y_ini']]:
        idx_V = [str(y) for y in dae['y_ini']].index(v_t_str)
        dae['y_ini'][idx_V] = v_ref
    else:
        dae['y_ini'] += [v_ref]

    dae['f'] += [dv_1]
    dae['x'] += [v_1]
    dae['g'] += [g_v_f]

    dae['y_ini'] += [v_f]
    dae['y_run'] += [v_f]

    dae['params_dict'].update({str(K_a): avr_data['K_a']})
    dae['params_dict'].update({str(T_r): avr_data['T_r']})
    dae['params_dict'].update({str(E_fmin): avr_data['E_fmin']})
    dae['params_dict'].update({str(E_fmax): avr_data['E_fmax']})

    # During ini, V_bus is pinned at the setpoint; v_ref is solved for.
    dae['u_ini_dict'].update({str(v_t): v_setpoint})
    dae['u_ini_dict'].update({str(v_pss): 0.0})

    # During run, v_ref is the input (value auto-transferred from ini by
    # ini2run because v_ref is in y_ini and u_run).
    dae['u_run_dict'].update({str(v_ref): v_setpoint})
    dae['u_run_dict'].update({str(v_pss): 0.0})

    dae['h_dict'].update({str(v_pss): v_pss})
    dae['h_dict'].update({str(v_ref): v_ref})

    # Steady-state relation: v_1 = V_setpoint and v_f = K_a*(v_ref - v_1) + 1.5,
    # so v_f_guess ≈ 1.5 implies v_ref ≈ V_setpoint.
    v_f_guess = 1.5
    dae['xy_0_dict'].update({str(v_f): v_f_guess})
    dae['xy_0_dict'].update({str(v_1): v_setpoint})
    dae['xy_0_dict'].update({str(v_ref): v_setpoint
                             + (v_f_guess - 1.5) / avr_data['K_a']})


def test():
    from pydae.core import Builder, Model
    from pydae.bps import BpsBuilder
    import pytest

    grid = BpsBuilder('kundur.hjson')
    grid.uz_jacs = False
    grid.construct('temp_kundur')

    bld = Builder(grid.sys_dict, target="ctypes", sparse=False)
    bld.build()

    model = Model('temp_kundur')

    v_set = 1.05
    model.ini({'p_m_1': 1.0, 'V_1': v_set}, 'xy_0.json')
    model.report_x()
    model.report_y()
    model.report_u()

    assert model.get_value('V_1') == pytest.approx(v_set, rel=1e-3)

    v_ref_solved = model.get_value('v_ref_1')

    model.ini({'p_m_1': 0.5, 'V_1': 1.0}, 'xy_0.json')
    model.run(1.0, {})
    model.run(5.0, {'v_ref_1': v_ref_solved})
    model.post()

    string = f'{model.Time[0]:0.2f}, '
    string += f"{model.get_values('V_1')[0]:0.2f}"
    print(string)

    string = f'{model.Time[-1]:0.2f}, '
    string += f"{model.get_values('V_1')[-1]:0.2f}"
    print(string)


if __name__ == '__main__':
    test()
