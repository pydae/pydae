# -*- coding: utf-8 -*-
r"""
IEEE 421.5 ST1A static excitation system (REE NTS parameter set).

This controller implements the bus-fed static excitation system IEEE
type ST1A (IEEE Std 421.5). The model is a lead-lag compensator
followed by a high-gain amplifier with hard output limits; the field
voltage is $E_{fd} = V_R$ (no rectifier loading since $K_C = 0$). With
the REE NTS parameter set the amplifier time constant $T_A = 0$ (pure
gain, no state) and the rate-feedback stabiliser is disabled
($K_F = 0$, no washout state), so the dynamic order is reduced to two.

**Signal path**

The terminal voltage passes through a first-order transducer with
state $v_c$,

$$\frac{d v_c}{dt} = \frac{V - v_c}{T_R}$$

The summing junction forms the voltage error

$$\varepsilon = v^{\star} - v_c + v_s$$

where $v^{\star}$ is the voltage reference and $v_s$ is the
supplementary PSS input. (UEL/OEL branches are omitted.) The error
is clipped by the input limiter,

$$\varepsilon_{lim} = \mathrm{sat}(\varepsilon,\; V_{Imin},\; V_{Imax})$$

and fed to the lead-lag compensator $(1 + s T_C)/(1 + s T_B)$ with
internal state $x_{lead}$,

$$\frac{d x_{lead}}{dt} = \frac{\varepsilon_{lim} - x_{lead}}{T_B}$$
$$y_{lead} = (\varepsilon_{lim} - x_{lead}) \frac{T_C}{T_B} + x_{lead}$$

With $T_A = 0$ the amplifier stage reduces to a pure gain,

$$V_R^{nosat} = K_A \, y_{lead}$$

and the field voltage command is produced by the output limiter,

$$V_R = \mathrm{sat}(V_R^{nosat},\; V_{Rmin},\; V_{Rmax})$$

The field voltage is returned as the algebraic variable $v_f$ via the
residual

$$0 = V_R - v_f$$

with $\mathrm{sat}(x, a, b) = \min\{b,\, \max\{a,\, x\}\}$.

**Simplifications under the REE parameter set**

- $T_A = 0$ — amplifier is a pure gain; no state $V_A$.
- $K_F = 0$ — rate-feedback stabiliser disabled; no washout state.
- $K_C = 0$ — no rectifier loading; upper ceiling $V_T V_{Rmax} - K_C I_{FD}$
  collapses to the constant $V_{Rmax}$.
- $V_{Imax} = V_{Rmax} = 999$ and $V_{Imin} = V_{Rmin} = -999$ — limits
  effectively inactive, but kept as ``Piecewise`` residuals so the model
  is structurally complete if a future fixture tightens them.

**The ini/run variable swap**

For a generator bus held at a voltage setpoint, the initialisation
problem is PV (active power and voltage magnitude specified, reactive
power and voltage reference unknown) while the subsequent time-domain
simulation is reference-driven (the reference is an input and the
voltage magnitude is solved from the network). ``pydae`` supports this
by allowing the ``y`` and ``u`` partitions to differ between phases
while sharing the same residual equations $g$.

                    ini                 run
    v_f             y_ini               y_run      (algebraic, always solved)
    v_ref           y_ini               u_run      (unknown in ini, input in run)
    V_bus           u_ini               y_run      (pinned in ini, solved in run)

The swap is performed in place at the existing ``y_ini`` index of
``V_bus`` so downstream components that reference ``y_ini`` by integer
index — for example ``vsource``'s ``g[idx_V] = ...`` override —
continue to target the correct equations.

**Value transfer between phases**

After ``ini()`` converges, ``ini2run()`` automatically copies solved
values to the run-phase state: ``v_ref`` (in ``y_ini`` and ``u_run``)
becomes the run input, and ``V_bus`` (in ``u_ini`` and ``y_run``) seeds
the run-phase solver.

**Configuration**

Example data entry (REE NTS defaults)::

    "avr": {"type": "st1",
            "T_R": 0.01, "T_B": 10.0, "T_C": 1.0,
            "K_A": 200.0, "T_A": 0.0,
            "V_Imax": 999.0, "V_Imin": -999.0,
            "V_Rmax": 999.0, "V_Rmin": -999.0,
            "v_ref": 1.0}

The ``v_ref`` field serves two roles: it is the **bus voltage setpoint
during ini()** (since ``v_ref`` itself is unknown there) and the
**initial run-phase input** (overwritten by the value solved during
ini via ``ini2run``). The optional ``bus`` key selects a remote bus
whose voltage is regulated; if omitted the generator bus is used.
"""


def descriptions():
    """Single source of truth for st1 parameters, inputs, states, and outputs."""
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "T_R", "data": "T_R",
                           "model": "T_R", "default": 0.01,
                           "description": "Terminal voltage transducer time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_B", "data": "T_B",
                           "model": "T_B", "default": 10.0,
                           "description": "Lead-lag denominator time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_C", "data": "T_C",
                           "model": "T_C", "default": 1.0,
                           "description": "Lead-lag numerator time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_A", "data": "K_A",
                           "model": "K_A", "default": 200.0,
                           "description": "AVR amplifier gain",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_A", "data": "T_A",
                           "model": "T_A", "default": 0.0,
                           "description": ("Amplifier time constant. Under the "
                                           "REE NTS spec T_A = 0 so the stage "
                                           "is a pure gain; kept as a parameter "
                                           "only for documentation — no state "
                                           "is instantiated."),
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{Imax}",
                           "data": "V_Imax", "model": "V_Imax", "default": 999.0,
                           "description": "Upper input-error limit",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{Imin}",
                           "data": "V_Imin", "model": "V_Imin", "default": -999.0,
                           "description": "Lower input-error limit",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{Rmax}",
                           "data": "V_Rmax", "model": "V_Rmax", "default": 999.0,
                           "description": "Upper regulator output limit",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{Rmin}",
                           "data": "V_Rmin", "model": "V_Rmin", "default": -999.0,
                           "description": "Lower regulator output limit",
                           "units": "pu"}]

    # Inputs (run phase). During ini, v_ref is solved and V_bus is pinned.
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
    descriptions_list += [{"type": "Dynamic State", "tex": "v_c",
                           "data": "", "model": "v_c", "default": "",
                           "description": "Terminal voltage transducer state",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{lead}",
                           "data": "", "model": "x_lead", "default": "",
                           "description": "Lead-lag compensator internal state",
                           "units": "pu"}]

    # Algebraic state — solved in both ini and run phases.
    descriptions_list += [{"type": "Algebraic State", "tex": "v_f",
                           "data": "", "model": "v_f", "default": "",
                           "description": ("Field-voltage command sent to the "
                                           "synchronous machine exciter "
                                           "(saturated)."),
                           "units": "pu"}]

    return descriptions_list


def st1(dae, data, name, bus_name, backend=None):
    """
    Example data entry (REE NTS parameter set)::

        "avr": {"type": "st1",
                 "T_R": 0.01, "T_B": 10.0, "T_C": 1.0,
                 "K_A": 200.0, "T_A": 0.0,
                 "V_Imax": 999.0, "V_Imin": -999.0,
                 "V_Rmax": 999.0, "V_Rmin": -999.0,
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
            'hard_limits': staticmethod(lambda x, xmin, xmax: sym.Min(sym.Max(x, xmin), xmax)),
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

    v_c = backend.symbols(f"v_c_{name}", real=True)
    x_lead = backend.symbols(f"x_lead_{name}", real=True)
    v_f = backend.symbols(f"v_f_{name}", real=True)

    T_R = backend.symbols(f"T_R_{name}", real=True)
    T_B = backend.symbols(f"T_B_{name}", real=True)
    T_C = backend.symbols(f"T_C_{name}", real=True)
    K_A = backend.symbols(f"K_A_{name}", real=True)
    V_Imax = backend.symbols(f"V_Imax_{name}", real=True)
    V_Imin = backend.symbols(f"V_Imin_{name}", real=True)
    V_Rmax = backend.symbols(f"V_Rmax_{name}", real=True)
    V_Rmin = backend.symbols(f"V_Rmin_{name}", real=True)

    v_ref = backend.symbols(f"v_ref_{name}", real=True)
    v_pss = backend.symbols(f"v_pss_{name}", real=True)

    v_s = v_pss

    # Terminal voltage transducer.
    dv_c = (v_t - v_c) / T_R

    # Summing junction and input limiter.
    epsilon = v_ref - v_c + v_s
    epsilon_lim = backend.hard_limits(epsilon, V_Imin, V_Imax)

    # Lead-lag (1 + s T_C) / (1 + s T_B).
    dx_lead = (epsilon_lim - x_lead) / T_B
    y_lead = (epsilon_lim - x_lead) * T_C / T_B + x_lead

    # Amplifier — pure gain since T_A = 0 under the REE spec.
    v_r_nosat = K_A * y_lead

    # Output limiter.
    v_r = backend.hard_limits(v_r_nosat, V_Rmin, V_Rmax)

    # Field voltage (E_fd = V_R since K_C = 0).
    g_v_f = v_r - v_f

    # --- ini/run variable partition ------------------------------------
    # Swap V_bus <-> v_ref at the same y_ini position. Replacing in place
    # keeps downstream index-based code (e.g. vsource's g[idx_V]=...) from
    # targeting the wrong equation when V_bus was just removed.
    v_setpoint = avr_data['v_ref']
    v_t_str = str(v_t)
    if v_t_str in [str(y) for y in dae['y_ini']]:
        idx_V = [str(y) for y in dae['y_ini']].index(v_t_str)
        dae['y_ini'][idx_V] = v_ref
    else:
        dae['y_ini'] += [v_ref]

    dae['f'] += [dv_c, dx_lead]
    dae['x'] += [v_c, x_lead]
    dae['g'] += [g_v_f]

    dae['y_ini'] += [v_f]
    dae['y_run'] += [v_f]

    dae['params_dict'].update({str(T_R): avr_data['T_R']})
    dae['params_dict'].update({str(T_B): avr_data['T_B']})
    dae['params_dict'].update({str(T_C): avr_data['T_C']})
    dae['params_dict'].update({str(K_A): avr_data['K_A']})
    dae['params_dict'].update({str(V_Imax): avr_data['V_Imax']})
    dae['params_dict'].update({str(V_Imin): avr_data['V_Imin']})
    dae['params_dict'].update({str(V_Rmax): avr_data['V_Rmax']})
    dae['params_dict'].update({str(V_Rmin): avr_data['V_Rmin']})

    # During ini, V_bus is pinned at the setpoint; v_ref is solved for.
    dae['u_ini_dict'].update({str(v_t): v_setpoint})
    dae['u_ini_dict'].update({str(v_pss): 0.0})

    # During run, v_ref is the input (auto-transferred from ini by ini2run).
    dae['u_run_dict'].update({str(v_ref): v_setpoint})
    dae['u_run_dict'].update({str(v_pss): 0.0})

    dae['h_dict'].update({str(v_pss): v_pss})
    dae['h_dict'].update({str(v_ref): v_ref})

    # Steady-state relations: dv_c = 0 → v_c = v_t = v_setpoint;
    # dx_lead = 0 → x_lead = epsilon_lim = epsilon; y_lead = epsilon;
    # v_r = K_A * epsilon = v_f. So epsilon = v_f / K_A and
    # v_ref = v_c - v_s + epsilon = v_setpoint + v_f / K_A.
    v_f_guess = 1.0
    epsilon_guess = v_f_guess / avr_data['K_A']
    dae['xy_0_dict'].update({str(v_f): v_f_guess})
    dae['xy_0_dict'].update({str(v_c): v_setpoint})
    dae['xy_0_dict'].update({str(x_lead): epsilon_guess})
    dae['xy_0_dict'].update({str(v_ref): v_setpoint + epsilon_guess})


def test():
    from pydae.core import Builder, Model
    from pydae.bps import BpsBuilder
    import pytest

    grid = BpsBuilder('st1.hjson')
    grid.uz_jacs = False
    grid.construct('temp_st1')

    bld = Builder(grid.sys_dict, target="cffi", sparse=False)
    bld.build()

    model = Model('temp_st1')

    v_set = 1.05
    model.ini({'p_m_1': 1.0, 'V_1': v_set}, 'xy_0.json')
    model.report_x()
    model.report_y()
    model.report_u()

    assert model.get_value('V_1') == pytest.approx(v_set, rel=1e-3)

    model.ini({'p_m_1': 0.5, 'V_1': 1.0}, 'xy_0.json')
    model.run(1.0, {})
    model.run(5.0, {'v_ref_1': 1.05})
    model.post()

    string = f'{model.Time[0]:0.2f}, '
    string += f"{model.get_values('V_1')[0]:0.2f}"
    print(string)

    string = f'{model.Time[-1]:0.2f}, '
    string += f"{model.get_values('V_1')[-1]:0.2f}"
    print(string)


if __name__ == '__main__':
    test()
