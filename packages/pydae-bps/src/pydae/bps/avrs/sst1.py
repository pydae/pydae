# -*- coding: utf-8 -*-
r"""
ST1-style static excitation system with PV-bus initialisation.

The model implements a lightweight IEEE ST1 / NTSST1 variant: a
first-order terminal-voltage sensor, a lead-lag voltage regulator, a
proportional main-regulator gain, a hard field-voltage limiter, and a
first-order output filter. There are three dynamic states and no
algebraic output — the field-voltage command ``v_f`` is filtered, not
algebraic. Initialisation is handled by the ``ini``/``run`` variable
swap described below; no artificial ``xi_v`` integrator is used.

**Signal path**

The sensed terminal voltage follows $V$ with a first-order lag of time
constant $T_r$:

$$\frac{d v_r}{dt} = \frac{V - v_r}{T_r}$$

(The sensed state ``v_r`` is exported via ``h_dict`` for diagnostics;
the summing junction uses the instantaneous $V$, matching the
historical ST1 implementation in this codebase.)

The voltage error feeds a lead-lag $(1 + s T_c)/(1 + s T_b)$ with
internal state $x_{cb}$:

$$v_1 = v^{\star} - V + v_s$$
$$\frac{d x_{cb}}{dt} = \frac{v_1 - x_{cb}}{T_b}$$
$$z_{cb} = (v_1 - x_{cb})\frac{T_c}{T_b} + x_{cb}$$

The proportional gain $K_a$ produces the unsaturated field command,
which is passed through a hard limiter:

$$e_{fd}^{nosat} = K_a \, z_{cb}$$
$$e_{fd} = \mathrm{sat}\!\left(e_{fd}^{nosat},\; V_{fmin},\; V_{fmax}\right)$$

with $\mathrm{sat}(x, a, b) = \min\{b,\, \max\{a,\, x\}\}$.

The output filter with time constant $T_r$ produces the field-voltage
command seen downstream:

$$\frac{d v_f}{dt} = \frac{e_{fd} - v_f}{T_r}$$

(The original ST1 implementation reuses $T_r$ for both the sensor lag
and the output filter; that choice is preserved here.)

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
    v_ref           y_ini               u_run      (unknown in ini, input in run)
    V_bus           u_ini               y_run      (pinned in ini, solved in run)

List cardinalities are preserved: ``|y_ini| == |y_run|`` and ``|g|`` is
unchanged. The swap is performed in place at the existing ``y_ini``
index of ``V_bus`` (added earlier by the bus builder) so that
downstream components that reference ``y_ini`` by integer index — for
example ``vsource``'s ``g[idx_V] = ...`` override — continue to target
the correct equations. Because ``v_f`` is a dynamic state (not
algebraic), this module contributes no new entries to ``g`` / ``y``
beyond the swap itself.

This replaces the earlier ``xi_v`` dummy-integrator approach (a fake
state with $d\xi_v/dt = v^{\star} - V$ and a tiny gain $K_{ai}$) and
the ``k_sat`` ini/run saturation blend. The swap is cleaner (one fewer
state, one fewer input, no spurious near-zero eigenvalue) and strictly
equivalent at the steady state.

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

    "avr": {"type": "sst1", "K_a": 200.0, "T_r": 0.02, "T_b": 10.0,
            "T_c": 1.0, "V_f_min": -100.0, "V_f_max": 100.0,
            "v_ref": 1.0}

The ``v_ref`` field serves two roles: it is the **bus voltage setpoint
during ini()** (since ``v_ref`` itself is unknown there) and the
**initial run-phase input** (overwritten by the value solved during
ini via ``ini2run``). The optional ``bus`` key selects a remote bus
whose voltage is regulated; if omitted the generator bus is used.
``V_f_min`` / ``V_f_max`` default to ±100 (effectively unbounded) to
match the original ``sst1`` defaults.
"""


def descriptions():
    """Single source of truth for sst1 parameters, inputs, states, and outputs."""
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "K_a", "data": "K_a",
                           "model": "K_a", "default": 200.0,
                           "description": "AVR main proportional gain",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_r", "data": "T_r",
                           "model": "T_r", "default": 0.02,
                           "description": ("Terminal-voltage sensor and "
                                           "output-filter time constant "
                                           "(shared)"),
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_b", "data": "T_b",
                           "model": "T_b", "default": 10.0,
                           "description": "Lead-lag denominator time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_c", "data": "T_c",
                           "model": "T_c", "default": 1.0,
                           "description": "Lead-lag numerator time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{fmin}",
                           "data": "V_f_min", "model": "V_f_min",
                           "default": -100.0,
                           "description": "Lower field-voltage limit",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{fmax}",
                           "data": "V_f_max", "model": "V_f_max",
                           "default": 100.0,
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
    descriptions_list += [{"type": "Dynamic State", "tex": "v_r",
                           "data": "", "model": "v_r", "default": "",
                           "description": ("Sensed terminal voltage "
                                           "(first-order lag output, "
                                           "diagnostic only)"),
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{cb}",
                           "data": "", "model": "x_cb", "default": "",
                           "description": "Lead-lag internal state",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "v_f",
                           "data": "", "model": "v_f", "default": "",
                           "description": ("Filtered field-voltage command "
                                           "sent to the synchronous machine "
                                           "exciter (post-saturation)."),
                           "units": "pu"}]

    return descriptions_list


def sst1(dae, data, name, bus_name, backend=None):
    """
    Example data entry::

        "avr": {"type": "sst1", "K_a": 200.0, "T_r": 0.02, "T_b": 10.0,
                 "T_c": 1.0, "V_f_min": -100.0, "V_f_max": 100.0,
                 "v_ref": 1.0}

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

    v_r = backend.symbols(f"v_r_{name}", real=True)
    x_cb = backend.symbols(f"x_cb_{name}", real=True)
    v_f = backend.symbols(f"v_f_{name}", real=True)

    K_a = backend.symbols(f"K_a_{name}", real=True)
    T_r = backend.symbols(f"T_r_{name}", real=True)
    T_b = backend.symbols(f"T_b_{name}", real=True)
    T_c = backend.symbols(f"T_c_{name}", real=True)
    V_f_min = backend.symbols(f"V_f_min_{name}", real=True)
    V_f_max = backend.symbols(f"V_f_max_{name}", real=True)

    v_ref = backend.symbols(f"v_ref_{name}", real=True)
    v_pss = backend.symbols(f"v_pss_{name}", real=True)

    v_s = v_pss

    # Summing junction — no artificial v_ini / xi_v integrator.
    v_1 = v_ref - v_t + v_s

    # Sensor dynamics (diagnostic; not fed back into the summing junction).
    dv_r = (v_t - v_r) / T_r

    # Lead-lag block.
    dx_cb = (v_1 - x_cb) / T_b
    z_cb = (v_1 - x_cb) * T_c / T_b + x_cb

    # Main regulator gain with hard field-voltage limits.
    v_f_nosat = K_a * z_cb
    v_f_sat = backend.Piecewise((V_f_max, v_f_nosat > V_f_max),
                            (V_f_min, v_f_nosat < V_f_min),
                            (v_f_nosat, True))

    # First-order output filter — T_r reused to match the original sst1.
    dv_f = (v_f_sat - v_f) / T_r

    # --- ini/run variable partition ------------------------------------
    # Swap V_bus <-> v_ref at the same y_ini position. Replacing in place
    # keeps downstream index-based code (e.g. vsource's g[idx_V]=...) from
    # targeting the wrong equation when V_bus was just removed.
    #   ini: V_bus is u_ini (pinned), v_ref is y_ini (unknown)
    #   run: V_bus is y_run (unknown), v_ref is u_run (input, seeded from ini)
    # No new algebraic equation is added — v_f is dynamic.

    v_setpoint = avr_data['v_ref']
    if v_t in dae['y_ini']:
        idx_V = dae['y_ini'].index(v_t)
        dae['y_ini'][idx_V] = v_ref
    else:
        dae['y_ini'] += [v_ref]

    dae['f'] += [dv_r, dx_cb, dv_f]
    dae['x'] += [v_r, x_cb, v_f]

    dae['params_dict'].update({str(K_a): avr_data['K_a']})
    dae['params_dict'].update({str(T_r): avr_data['T_r']})
    dae['params_dict'].update({str(T_b): avr_data['T_b']})
    dae['params_dict'].update({str(T_c): avr_data['T_c']})
    dae['params_dict'].update({str(V_f_min): avr_data.get('V_f_min', -100.0)})
    dae['params_dict'].update({str(V_f_max): avr_data.get('V_f_max', 100.0)})

    # During ini, V_bus is pinned at the setpoint; v_ref is solved for.
    dae['u_ini_dict'].update({str(v_t): v_setpoint})
    dae['u_ini_dict'].update({str(v_pss): 0.0})

    # During run, v_ref is the input (value auto-transferred from ini by
    # ini2run because v_ref is in y_ini and u_run).
    dae['u_run_dict'].update({str(v_ref): v_setpoint})
    dae['u_run_dict'].update({str(v_pss): 0.0})

    dae['h_dict'].update({str(v_pss): v_pss})
    dae['h_dict'].update({str(v_ref): v_ref})
    dae['h_dict'].update({str(v_r): v_r})

    # Steady-state relations: v_r = V_setpoint; x_cb = v_1 = v_ref - V_setpoint
    # (since v_s = 0 at ini); z_cb = x_cb; v_f_nosat = K_a * x_cb; v_f = v_f_sat.
    # For v_f_guess ≈ 1.5, v_ref sits a hair above V_setpoint.
    v_f_guess = 1.5
    dae['xy_0_dict'].update({str(v_f): v_f_guess})
    dae['xy_0_dict'].update({str(v_r): v_setpoint})
    dae['xy_0_dict'].update({str(x_cb): v_f_guess / avr_data['K_a']})
    dae['xy_0_dict'].update({str(v_ref): v_setpoint + v_f_guess / avr_data['K_a']})


def test():
    from pydae.core import Builder, Model
    from pydae.bps import BpsBuilder
    import pytest

    grid = BpsBuilder('sst1.hjson')
    grid.uz_jacs = False
    grid.construct('temp_sst1')

    bld = Builder(grid.sys_dict, target="cffi", sparse=False)
    bld.build()

    model = Model('temp_sst1')

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
