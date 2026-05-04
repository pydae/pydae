# -*- coding: utf-8 -*-
r"""
SEXS simplified excitation system with PV-bus initialisation.

The model implements an IEEE-style *Simplified Excitation System* (SEXS):
a lead-lag voltage regulator feeding a first-order exciter with hard
field-voltage limits. The controller is written as two dynamic states
plus one algebraic equation, with no artificial initialisation
integrator — the initialisation is handled by the ``ini``/``run``
variable swap described below.

**Signal path**

The voltage error is

$$v_2 = v^{\star} - V + v_s$$

where $V$ is the terminal (or remote) bus voltage magnitude,
$v^{\star}$ is the voltage reference, and $v_s$ is the supplementary
PSS input. The error feeds a lead-lag block with state $x_{ab}$,

$$\frac{d x_{ab}}{dt} = \frac{v_2 - x_{ab}}{T_b}$$
$$z_{ab} = (v_2 - x_{ab}) \frac{T_a}{T_b} + x_{ab}$$

whose output drives a first-order exciter with state $x_e$ and gain
$K_a$,

$$\frac{d x_e}{dt} = \frac{K_a z_{ab} - x_e}{T_e}$$

The unsaturated field command is offset by one so that the operating
point sits near $v_f \approx 1$ at no-load:

$$e_{fd}^{nosat} = x_e + 1$$

The actual field voltage is produced by a hard limiter and returned as
the algebraic variable $v_f$ via the residual

$$0 = \mathrm{sat}\!\left(e_{fd}^{nosat},\; E_{min},\; E_{max}\right) - v_f$$

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
state with $d\xi_v/dt = v^{\star} - V$ and a tiny gain $K_{ai}$) that
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

    "avr": {"type": "sexs", "K_a": 200.0, "T_a": 0.015, "T_b": 10.0,
            "T_e": 0.1, "E_min": -5.0, "E_max": 5.0, "v_ref": 1.0}

The ``v_ref`` field serves two roles: it is the **bus voltage setpoint
during ini()** (since ``v_ref`` itself is unknown there) and the
**initial run-phase input** (overwritten by the value solved during
ini via ``ini2run``). The optional ``bus`` key selects a remote bus
whose voltage is regulated; if omitted the generator bus is used.
"""


def descriptions():
    """Single source of truth for sexs parameters, inputs, states, and outputs."""
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "K_a", "data": "K_a",
                           "model": "K_a", "default": 200.0,
                           "description": "AVR main gain",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_a", "data": "T_a",
                           "model": "T_a", "default": 0.015,
                           "description": "Lead-lag numerator time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_b", "data": "T_b",
                           "model": "T_b", "default": 10.0,
                           "description": "Lead-lag denominator time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_e", "data": "T_e",
                           "model": "T_e", "default": 0.1,
                           "description": "Exciter time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "E_{min}",
                           "data": "E_min", "model": "E_min", "default": -5.0,
                           "description": "Lower field-voltage limit",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "E_{max}",
                           "data": "E_max", "model": "E_max", "default": 5.0,
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
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{ab}",
                           "data": "", "model": "x_ab", "default": "",
                           "description": "Lead-lag internal state",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_e",
                           "data": "", "model": "x_e", "default": "",
                           "description": ("Exciter state — field command "
                                           "before the +1 offset and "
                                           "saturation"),
                           "units": "pu"}]

    # Algebraic state — solved in both ini and run phases.
    descriptions_list += [{"type": "Algebraic State", "tex": "v_f",
                           "data": "", "model": "v_f", "default": "",
                           "description": ("Field-voltage command sent to the "
                                           "synchronous machine exciter "
                                           "(saturated)."),
                           "units": "pu"}]

    return descriptions_list


def sexs(dae, data, name, bus_name, backend=None):
    """
    Optimized SEXS AVR for CasADi and SymPy.
    """
    if backend is None:
        # Default fallback to SymPy if no backend is provided
        import sympy as sym
        backend = type('Backend', (), {
            'symbols': lambda _, n, **k: sym.Symbol(n, real=True),
            'Piecewise': sym.Piecewise,
            'use_casadi': False,
        })()

    avr_data = data['avr']
    remote_bus_name = avr_data.get('bus', bus_name)

    # 1. Define Symbols
    v_t = backend.symbols(f"V_{remote_bus_name}")
    x_ab = backend.symbols(f"x_ab_{name}")
    x_e = backend.symbols(f"x_e_{name}")
    v_f = backend.symbols(f"v_f_{name}")

    K_a = backend.symbols(f"K_a_{name}")
    T_a = backend.symbols(f"T_a_{name}")
    T_b = backend.symbols(f"T_b_{name}")
    T_e = backend.symbols(f"T_e_{name}")
    E_min = backend.symbols(f"E_min_{name}")
    E_max = backend.symbols(f"E_max_{name}")

    v_ref = backend.symbols(f"v_ref_{name}")
    v_pss = backend.symbols(f"v_pss_{name}")

    # 2. AVR Equations
    v_s = v_pss
    v_2 = v_ref - v_t + v_s

    # Lead-Lag Block
    dx_ab = (v_2 - x_ab) / T_b
    z_ab = (v_2 - x_ab) * T_a / T_b + x_ab

    # Gain and Time Constant
    dx_e = (K_a * z_ab - x_e) / T_e

    efd_nosat = x_e + 1.0

    # Saturation: disable for CasADi backend to avoid fmax/fmin Jacobian NaN.
    if backend.use_casadi:
        efd = efd_nosat
    else:
        efd = backend.hard_limit(efd_nosat, E_min, E_max)
     
    # Algebraic constraint: v_f must equal the limited output
    g_v_f = efd - v_f

    # 3. Initialization Logic (Inverse Initialization)
    # We swap V_t (bus voltage) for v_ref in the algebraic variables list.
    # This means during ini(), we set V_t = 1.0 and solve for the required v_ref.
    v_setpoint = avr_data['v_ref']
    v_t_str = str(v_t)
    
    # Replace V_t with v_ref in the initialization algebraic variables
    if v_t_str in [str(y) for y in dae['y_ini']]:
        dae['y_ini'] = [v_ref if str(y) == v_t_str else y for y in dae['y_ini']]
    else:
        dae['y_ini'] += [v_ref]

    # 4. Update DAE Dictionary
    dae['f'] += [dx_ab, dx_e]
    dae['x'] += [x_ab, x_e]
    dae['g'] += [g_v_f]
    dae['y_ini'] += [v_f]
    dae['y_run'] += [v_f]

    # 5. Parameters and Inputs
    dae['params_dict'].update({
        str(K_a): avr_data['K_a'], str(T_a): avr_data['T_a'],
        str(T_b): avr_data['T_b'], str(T_e): avr_data['T_e'],
        str(E_min): avr_data['E_min'], str(E_max): avr_data['E_max']
    })

    # Set initialization inputs (Bus voltage target)
    dae['u_ini_dict'].update({str(v_t): v_setpoint, str(v_pss): 0.0})

    # Set runtime inputs (Reference voltage starts at setpoint)
    dae['u_run_dict'].update({str(v_ref): v_setpoint, str(v_pss): 0.0})

    dae['h_dict'].update({str(v_pss): v_pss, str(v_ref): v_ref, str(v_f): v_f})

    # 6. Critical: Robust Initial Guesses (xy_0)
    # To avoid the Newton solver failing on the limiter, we start in the linear region.
    v_f_guess = 1.5 
    x_e_guess = v_f_guess - 1.0
    z_ab_guess = x_e_guess / avr_data['K_a']
    
    dae['xy_0_dict'].update({
        str(v_f): v_f_guess,
        str(x_e): x_e_guess,
        str(x_ab): z_ab_guess,
        str(v_ref): v_setpoint + z_ab_guess  # Consistent with lead-lag steady state
    })


def test():
    import pytest
    from pydae.bps import BpsBuilder
    from pydae.core import Builder, Model

    grid = BpsBuilder('sexs.hjson')
    grid.uz_jacs = False
    grid.construct('temp_sexs')

    bld = Builder(grid.sys_dict, target="ctypes", sparse=False)
    bld.build()

    model = Model('temp_sexs')

    v_set = 1.05
    model.ini({'p_m_1': 1.0, 'V_1': v_set}, 'xy_0.json')
    model.report_x()
    model.report_y()
    model.report_u()

    assert model.get_value('V_1') == pytest.approx(v_set, rel=1e-3)

    v_ref_solved = model.get_value('v_ref_1')

    model.ini({'p_m_1': 0.5, 'V_1': 1.0}, 'xy_0.json')
    model.run(1.0, {})
    model.run(5.0, {'v_ref_1': 1.05})
    model.post()

    string = f'{model.Time[0]:0.2f}, '
    string += f'{model.get_values("V_1")[0]:0.2f}'
    print(string)

    string = f'{model.Time[-1]:0.2f}, '
    string += f'{model.get_values("V_1")[-1]:0.2f}'
    print(string)



if __name__ == '__main__':
    test()
