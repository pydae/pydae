# -*- coding: utf-8 -*-
"""
Pure-algebraic automatic voltage regulator with PV-bus behaviour.

The controller is a single static-gain equation with no internal state and
no output limits. The field-voltage command $v_f$ is driven from the
voltage error at the regulated bus:

$$v_f = K_a \left( v^{\star} - V + v_s \right)$$

written as the residual

$$0 = v_f - K_a \left( v^{\star} - V + v_s \right)$$

where $V$ is the terminal (or a remote) bus voltage magnitude,
$v^{\star}$ is the voltage reference, $v_s$ is a supplementary PSS
input, and $v_f$ is the field-voltage command fed to the synchronous
machine. The large gain $K_a$ provides effectively stiff voltage control.

**The ini/run variable swap**

For a generator bus held at a voltage setpoint, the initialization
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
index of ``V_bus`` (added earlier by the bus builder) so that downstream
components that reference ``y_ini`` by integer index — for example
``vsource``'s ``g[idx_V] = ...`` override — continue to target the
correct equations.

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

    "avr": {"type": "avr_1", "K_a": 100.0, "v_ref": 1.0, "bus": "2"}

The ``v_ref`` field serves two roles: it is the **bus voltage setpoint
during ini()** (since ``v_ref`` itself is unknown there) and the
**initial run-phase input** (overwritten by the value solved during
ini via ``ini2run``). The optional ``bus`` key selects a remote bus
whose voltage is regulated; if omitted the generator bus is used.
"""


def descriptions():
    """Single source of truth for avr_1 parameters, inputs, and outputs."""
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "K_a", "data": "K_a",
                           "model": "K_a", "default": 100.0,
                           "description": "AVR proportional gain",
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
                                           "(PSS output), added to the voltage "
                                           "error."),
                           "units": "pu"}]

    # Algebraic state — solved in both ini and run phases.
    descriptions_list += [{"type": "Algebraic State", "tex": "v_f",
                           "data": "", "model": "v_f",
                           "default": "",
                           "description": ("Field-voltage command sent to the "
                                           "synchronous machine exciter."),
                           "units": "pu"}]

    return descriptions_list


def avr_1(dae, data, name, bus_name, backend=None):
    """
    Example data entry::

        "avr": {"type": "avr_1", "K_a": 100.0, "v_ref": 1.0}

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
    v_f = backend.symbols(f"v_f_{name}", real=True)
    K_a = backend.symbols(f"K_a_{name}", real=True)
    v_ref = backend.symbols(f"v_ref_{name}", real=True)
    v_pss = backend.symbols(f"v_pss_{name}", real=True)

    v_s = v_pss

    # Single algebraic equation — pure static gain, no dynamics, no limits.
    g_v_f = v_f - K_a * (v_ref - v_t + v_s)

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

    dae['g'] += [g_v_f]
    dae['x'] += []
    dae['f'] += []

    dae['y_ini'] += [v_f]              # v_f is algebraic in ini
    dae['y_run'] += [v_f]              # V_bus stays in y_run via the bus code

    dae['params_dict'].update({str(K_a): avr_data['K_a']})

    # During ini, V_bus is pinned at the setpoint; v_ref is solved for.
    dae['u_ini_dict'].update({str(v_t): v_setpoint})
    dae['u_ini_dict'].update({str(v_pss): 0.0})

    # During run, v_ref is the input (value auto-transferred from ini by
    # ini2run because v_ref is in y_ini and u_run).
    dae['u_run_dict'].update({str(v_ref): v_setpoint})
    dae['u_run_dict'].update({str(v_pss): 0.0})

    dae['h_dict'].update({str(v_pss): v_pss})
    dae['h_dict'].update({str(v_ref): v_ref})

    # v_f must start well above 1.0 for a loaded machine; with K_a large,
    # (v_ref - V) is tiny, so v_ref's initial guess should be ~V_setpoint
    # plus v_f_guess/K_a.
    v_f_guess = 1.5
    dae['xy_0_dict'].update({str(v_f): v_f_guess})
    dae['xy_0_dict'].update({str(v_ref): v_setpoint + v_f_guess / avr_data['K_a']})


def test():
    import hjson
    from pydae.core import Builder, Model
    from pydae.bps import BpsBuilder
    import pytest


    grid = BpsBuilder('avr_1.hjson')
    grid.uz_jacs = False
    grid.construct('temp_avr1')
    print(grid.sys_dict)

    for it_x, item in enumerate(grid.sys_dict['f_list']):
        print(it_x, item)

    for it_y, item in enumerate(grid.sys_dict['g_list']):
        print(it_y, it_y+it_x, item)


    bld = Builder(grid.sys_dict, target="cffi", sparse=False)
    bld.build()

    

    model = Model('temp_avr1')

    dict_xy0 = {
    "V_1": 1.0,
    "theta_1": 0.0,
    "V_2": 1.0,
    "theta_2": 0.0,
    "omega_coi": 1.0,
    "omega_1": 1.0,
    "e1q_1": 1.0,
    "e1d_1": 0.5,
    "i_q_1": 0.5,
    "v_f_1": 1.5,
    "v_ref_1": 1.005
        }
    v_set = 1.05
    model.ini({'p_m_1': 1.0, 'V_1': v_set}, dict_xy0)
    model.report_x()
    model.report_y()

    assert model.get_value('V_1') == pytest.approx(v_set, rel=1e-3)

    v_ref_solved = model.get_value('v_ref_1')

    model.ini({'p_m_1': 0.5, 'V_1': 1.0}, 'xy_0.json')
    model.run(1.0, {})
    model.run(15.0, {'v_ref_1': v_ref_solved})
    model.post()

    


if __name__ == '__main__':
    test()
