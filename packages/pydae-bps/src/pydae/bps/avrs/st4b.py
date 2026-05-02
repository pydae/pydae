# -*- coding: utf-8 -*-
r"""
IEEE Std 421.5 type ST4B excitation system with PV-bus initialisation.

The model follows the block diagram of IEEE Std 421.5 type **ST4B**.
Default parameters match those mandated by the REE *Normas Técnicas
de Supervisión* (NTS) for grid-code compliance simulations.

**Signal path**

The compensated terminal voltage passes through a first-order sensor
with time constant $T_R$:

$$\frac{d v_c}{dt} = \frac{V - v_c}{T_R}$$

The voltage error feeds a proportional-integral regulator with hard
limits $V_{RMIN}$, $V_{RMAX}$. The PI is written so that the
state $\xi_r$ carries the *integral contribution to the output*
(rather than the time integral of the error itself), so setting the
integral gain to zero freezes the state cleanly:

$$\frac{d \xi_r}{dt} = K_{IR}\,(v^{\star} - v_c + v_s) - \epsilon_{leak}\,\xi_r$$
$$v_r^{nosat} = K_{PR}\,(v^{\star} - v_c + v_s) + \xi_r$$
$$v_r = \mathrm{sat}(v_r^{nosat}, V_{RMIN}, V_{RMAX})$$

The $\epsilon_{leak} = 10^{-6}$ term is a tiny self-decay that
guarantees a unique equilibrium for $\xi_r$ when $K_{IR} = 0$;
with any reasonable $K_{IR} > 0$ it is negligible (time constant
$10^6$ s).

A first-order lag of time constant $T_A$ separates the outer and
inner loops:

$$\frac{d x_a}{dt} = \frac{v_r - x_a}{T_A}$$

The inner PI regulates field current; its error $\varepsilon_m = x_a
- K_G v_f$ reduces to $x_a$ in the REE configuration ($K_G = 0$).
Same integral-state convention (with the same leakage term) as the
outer loop:

$$\frac{d \xi_m}{dt} = K_{IM}\,\varepsilon_m - \epsilon_{leak}\,\xi_m$$
$$v_m^{nosat} = K_{PM}\,\varepsilon_m + \xi_m$$
$$v_m = \mathrm{sat}(v_m^{nosat}, V_{MMIN}, V_{MMAX})$$

The leakage term matters here: the REE default $K_{IM} = 0$ would
otherwise leave $\xi_m$ as an unconstrained state (zero row in the
ini Jacobian, effectively singular). The leakage pins $\xi_m$ to 0
at steady state when $K_{IM} = 0$.

The exciter voltage $V_E$ and rectifier function $F_{EX}$ are given
in the standard as

$$V_E = \left|\, j K_P \bar{V}_T + \left(K_I + j K_P X_L e^{j\theta_P}\right)\,\bar{I}_T\,\right|$$
$$F_{EX}(I_N) = \text{piecewise}, \quad I_N = \frac{K_C I_{FD}}{V_E}$$

For the REE default parameters ($K_I = 0$, $X_L = 0$, $\theta_P = 0$)
the first expression collapses to $V_E = K_P V$; and $K_C = -0.08 < 0$
makes $I_N \le 0$ so $F_{EX} \equiv 1$ for all physical operating
points. This module implements those simplifications:

$$V_E = K_P V, \qquad F_{EX} = 1$$

Using the full vector $V_E$ and a non-trivial $F_{EX}$ requires
wiring the machine's field current $I_{FD}$ and the bus complex
current $\bar{I}_T$ into the AVR — not needed for the REE compliance
simulations this model targets.

The field-voltage command is produced with a hard upper limit
$V_{BMAX}$ (the rectifier ceiling) and returned as the algebraic
variable $v_f$:

$$e_{fd}^{nosat} = v_m V_E$$
$$0 = \min(V_{BMAX},\; e_{fd}^{nosat}) - v_f$$

The lower bound on $v_f$ is set indirectly by $V_{MMIN}$: with $v_m
\ge V_{MMIN}$ and $V_E > 0$, $e_{fd}^{nosat} \ge V_{MMIN} V_E$.

**The ini/run variable swap**

For a generator bus held at a voltage setpoint, the initialisation
problem is PV (active power and voltage magnitude specified, reactive
power and voltage reference unknown) while the subsequent time-domain
simulation is reference-driven (the reference is an input and the
voltage magnitude is solved from the network). ``pydae`` supports
this by allowing the ``y`` (algebraic) and ``u`` (input) partitions
to differ between the ``ini`` and ``run`` phases while sharing the
same set of residual equations $g$.

The table below shows how each quantity is classified in each phase:

                    ini                 run
    v_f             y_ini               y_run      (algebraic, always solved)
    v_ref           y_ini               u_run      (unknown in ini, input in run)
    V_bus           u_ini               y_run      (pinned in ini, solved in run)

List cardinalities are preserved: ``|y_ini| == |y_run|`` and ``|g|``
is unchanged. The swap is performed in place at the existing
``y_ini`` index of ``V_bus`` (added earlier by the bus builder) so
that downstream components that reference ``y_ini`` by integer
index — for example ``vsource``'s ``g[idx_V] = ...`` override —
continue to target the correct equations.

No artificial ``xi_v`` integrator is used: the PV-bus initialisation
is handled entirely by the swap.

**Value transfer between phases**

After ``ini()`` converges, ``ini2run()`` automatically copies solved
values to the run-phase state:

- ``v_ref`` is in ``y_ini`` and appears in ``u_run``: its solved
  value becomes the run-phase input, so the operating point is
  preserved without the user having to pass ``v_ref`` manually.
- ``V_bus`` is in ``u_ini`` and appears in ``y_run``: its pinned
  setpoint is used as the starting value of the run-phase solver.

**Configuration**

Example data entry (REE NTS ST4B defaults)::

    "avr": {"type": "st4b",
            "T_R": 0.02,
            "K_PR": 3.15, "K_IR": 3.15,
            "V_RMAX": 1.0, "V_RMIN": -0.87,
            "T_A": 0.02,
            "K_PM": 1.0, "K_IM": 0.0,
            "V_MMAX": 1.0, "V_MMIN": -0.87,
            "K_G": 0.0, "K_P": 6.5,
            "V_BMAX": 8.0,
            "v_ref": 1.0}

The ``v_ref`` field serves two roles: it is the **bus voltage setpoint
during ini()** (since ``v_ref`` itself is unknown there) and the
**initial run-phase input** (overwritten by the value solved during
ini via ``ini2run``). The optional ``bus`` key selects a remote bus
whose voltage is regulated; if omitted the generator bus is used.

Parameters $K_I$, $X_L$, $\theta_P$, $K_C$ from the full IEEE ST4B
spec are currently fixed to the simplified form documented above; if
needed in the future, pass them through ``avr_data`` and extend the
signal path.
"""


_EPS_LEAK = 1e-6


def descriptions():
    """Single source of truth for st4b parameters, inputs, states, and outputs."""
    descriptions_list = []

    # Parameters (REE NTS defaults)
    descriptions_list += [{"type": "Parameter", "tex": "T_R", "data": "T_R",
                           "model": "T_R", "default": 0.02,
                           "description": "Terminal-voltage sensor time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_{PR}", "data": "K_PR",
                           "model": "K_PR", "default": 3.15,
                           "description": "Outer voltage-regulator proportional gain",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_{IR}", "data": "K_IR",
                           "model": "K_IR", "default": 3.15,
                           "description": "Outer voltage-regulator integral gain",
                           "units": "pu/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{RMAX}",
                           "data": "V_RMAX", "model": "V_RMAX",
                           "default": 1.0,
                           "description": "Upper limit on outer-regulator output",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{RMIN}",
                           "data": "V_RMIN", "model": "V_RMIN",
                           "default": -0.87,
                           "description": "Lower limit on outer-regulator output",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_A", "data": "T_A",
                           "model": "T_A", "default": 0.02,
                           "description": "Inter-stage lag time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_{PM}", "data": "K_PM",
                           "model": "K_PM", "default": 1.0,
                           "description": "Inner field-current regulator proportional gain",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_{IM}", "data": "K_IM",
                           "model": "K_IM", "default": 0.0,
                           "description": ("Inner field-current regulator integral "
                                           "gain (REE doc writes K_IN; interpreted "
                                           "as K_IM per IEEE 421.5 ST4B)"),
                           "units": "pu/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{MMAX}",
                           "data": "V_MMAX", "model": "V_MMAX",
                           "default": 1.0,
                           "description": "Upper limit on inner-regulator output",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{MMIN}",
                           "data": "V_MMIN", "model": "V_MMIN",
                           "default": -0.87,
                           "description": "Lower limit on inner-regulator output",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_G", "data": "K_G",
                           "model": "K_G", "default": 0.0,
                           "description": ("Field-voltage feedback gain into inner "
                                           "loop. Default 0 disables the feedback "
                                           "(REE NTS default)."),
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_P", "data": "K_P",
                           "model": "K_P", "default": 6.5,
                           "description": ("Exciter voltage gain. With X_L = K_I = "
                                           "theta_P = 0 (REE default), V_E = K_P V."),
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{BMAX}",
                           "data": "V_BMAX", "model": "V_BMAX",
                           "default": 8.0,
                           "description": "Rectifier ceiling on field-voltage command",
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
    descriptions_list += [{"type": "Dynamic State", "tex": "v_c",
                           "data": "", "model": "v_c", "default": "",
                           "description": "Sensed terminal voltage (T_R lag output)",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "\\xi_r",
                           "data": "", "model": "xi_r", "default": "",
                           "description": ("Outer-PI integrator — integral "
                                           "contribution to v_r^{nosat}."),
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_a",
                           "data": "", "model": "x_a", "default": "",
                           "description": "Inter-stage lag output",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "\\xi_m",
                           "data": "", "model": "xi_m", "default": "",
                           "description": ("Inner-PI integrator — integral "
                                           "contribution to v_m^{nosat}. "
                                           "Pinned to 0 by leakage when K_IM = 0."),
                           "units": "pu"}]

    # Algebraic state — solved in both ini and run phases.
    descriptions_list += [{"type": "Algebraic State", "tex": "v_f",
                           "data": "", "model": "v_f", "default": "",
                           "description": ("Field-voltage command sent to the "
                                           "synchronous machine exciter "
                                           "(post rectifier ceiling)."),
                           "units": "pu"}]

    return descriptions_list


def st4b(dae, data, name, bus_name, backend=None):
    """
    Example data entry::

        "avr": {"type": "st4b",
                 "T_R": 0.02,
                 "K_PR": 3.15, "K_IR": 3.15,
                 "V_RMAX": 1.0, "V_RMIN": -0.87,
                 "T_A": 0.02,
                 "K_PM": 1.0, "K_IM": 0.0,
                 "V_MMAX": 1.0, "V_MMIN": -0.87,
                 "K_G": 0.0, "K_P": 6.5,
                 "V_BMAX": 8.0,
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

    v_c = backend.symbols(f"v_c_{name}", real=True)
    xi_r = backend.symbols(f"xi_r_{name}", real=True)
    x_a = backend.symbols(f"x_a_{name}", real=True)
    xi_m = backend.symbols(f"xi_m_{name}", real=True)
    v_f = backend.symbols(f"v_f_{name}", real=True)

    T_R = backend.symbols(f"T_R_{name}", real=True)
    K_PR = backend.symbols(f"K_PR_{name}", real=True)
    K_IR = backend.symbols(f"K_IR_{name}", real=True)
    V_RMAX = backend.symbols(f"V_RMAX_{name}", real=True)
    V_RMIN = backend.symbols(f"V_RMIN_{name}", real=True)
    T_A = backend.symbols(f"T_A_{name}", real=True)
    K_PM = backend.symbols(f"K_PM_{name}", real=True)
    K_IM = backend.symbols(f"K_IM_{name}", real=True)
    V_MMAX = backend.symbols(f"V_MMAX_{name}", real=True)
    V_MMIN = backend.symbols(f"V_MMIN_{name}", real=True)
    K_G = backend.symbols(f"K_G_{name}", real=True)
    K_P = backend.symbols(f"K_P_{name}", real=True)
    V_BMAX = backend.symbols(f"V_BMAX_{name}", real=True)

    v_ref = backend.symbols(f"v_ref_{name}", real=True)
    v_pss = backend.symbols(f"v_pss_{name}", real=True)

    v_s = v_pss

    # Sensor lag — no artificial v_ini / xi_v integrator.
    dv_c = (v_t - v_c) / T_R

    # Outer PI (state = integral contribution to output; tiny self-decay
    # anchors the state when K_IR = 0 — negligible otherwise).
    epsilon_v = v_ref - v_c + v_s
    v_r_nosat = K_PR * epsilon_v + xi_r
    v_r = backend.Piecewise((V_RMAX, v_r_nosat > V_RMAX),
                        (V_RMIN, v_r_nosat < V_RMIN),
                        (v_r_nosat, True))
    dxi_r = K_IR * epsilon_v - _EPS_LEAK * xi_r

    # Inter-stage lag.
    dx_a = (v_r - x_a) / T_A

    # Inner PI (same convention; leakage is what makes K_IM = 0 non-singular).
    epsilon_m = x_a - K_G * v_f
    v_m_nosat = K_PM * epsilon_m + xi_m
    v_m = backend.Piecewise((V_MMAX, v_m_nosat > V_MMAX),
                        (V_MMIN, v_m_nosat < V_MMIN),
                        (v_m_nosat, True))
    dxi_m = K_IM * epsilon_m - _EPS_LEAK * xi_m

    # Exciter voltage (simplified: V_E = K_P * V, valid when X_L=K_I=theta_P=0).
    v_e = K_P * v_t

    # Field-voltage command with rectifier ceiling (F_EX = 1 assumption).
    v_f_nosat = v_m * v_e
    efd = backend.Piecewise((V_BMAX, v_f_nosat > V_BMAX),
                        (v_f_nosat, True))
    g_v_f = efd - v_f

    # --- ini/run variable partition ------------------------------------
    # Swap V_bus <-> v_ref at the same y_ini position. Replacing in place
    # keeps downstream index-based code (e.g. vsource's g[idx_V]=...) from
    # targeting the wrong equation when V_bus was just removed.
    #   ini: V_bus is u_ini (pinned), v_ref is y_ini (unknown)
    #   run: V_bus is y_run (unknown), v_ref is u_run (input, seeded from ini)

    v_setpoint = avr_data['v_ref']
    if v_t in dae['y_ini']:
        idx_V = dae['y_ini'].index(v_t)
        dae['y_ini'][idx_V] = v_ref
    else:
        dae['y_ini'] += [v_ref]

    dae['f'] += [dv_c, dxi_r, dx_a, dxi_m]
    dae['x'] += [v_c, xi_r, x_a, xi_m]
    dae['g'] += [g_v_f]

    dae['y_ini'] += [v_f]
    dae['y_run'] += [v_f]

    dae['params_dict'].update({str(T_R): avr_data.get('T_R', 0.02)})
    dae['params_dict'].update({str(K_PR): avr_data.get('K_PR', 3.15)})
    dae['params_dict'].update({str(K_IR): avr_data.get('K_IR', 3.15)})
    dae['params_dict'].update({str(V_RMAX): avr_data.get('V_RMAX', 1.0)})
    dae['params_dict'].update({str(V_RMIN): avr_data.get('V_RMIN', -0.87)})
    dae['params_dict'].update({str(T_A): avr_data.get('T_A', 0.02)})
    dae['params_dict'].update({str(K_PM): avr_data.get('K_PM', 1.0)})
    dae['params_dict'].update({str(K_IM): avr_data.get('K_IM', 0.0)})
    dae['params_dict'].update({str(V_MMAX): avr_data.get('V_MMAX', 1.0)})
    dae['params_dict'].update({str(V_MMIN): avr_data.get('V_MMIN', -0.87)})
    dae['params_dict'].update({str(K_G): avr_data.get('K_G', 0.0)})
    dae['params_dict'].update({str(K_P): avr_data.get('K_P', 6.5)})
    dae['params_dict'].update({str(V_BMAX): avr_data.get('V_BMAX', 8.0)})

    # During ini, V_bus is pinned at the setpoint; v_ref is solved for.
    dae['u_ini_dict'].update({str(v_t): v_setpoint})
    dae['u_ini_dict'].update({str(v_pss): 0.0})

    # During run, v_ref is the input (value auto-transferred from ini by
    # ini2run because v_ref is in y_ini and u_run).
    dae['u_run_dict'].update({str(v_ref): v_setpoint})
    dae['u_run_dict'].update({str(v_pss): 0.0})

    dae['h_dict'].update({str(v_pss): v_pss})
    dae['h_dict'].update({str(v_ref): v_ref})

    # Steady-state relations (REE defaults, K_G = K_IM = 0):
    #   v_c = V_setpoint;  epsilon_v = 0 -> v_ref = V_setpoint;
    #   v_r = xi_r;  x_a = v_r;  epsilon_m = x_a;  v_m = K_PM * x_a + xi_m;
    #   xi_m -> 0 via leakage;  v_e = K_P * V_setpoint;
    #   v_f = v_m * v_e = K_PM * x_a * K_P * V_setpoint.
    v_f_guess = 1.5
    K_PM_val = avr_data.get('K_PM', 1.0)
    K_P_val = avr_data.get('K_P', 6.5)
    x_a_guess = v_f_guess / (K_PM_val * K_P_val * v_setpoint)
    dae['xy_0_dict'].update({str(v_c): v_setpoint})
    dae['xy_0_dict'].update({str(xi_r): x_a_guess})
    dae['xy_0_dict'].update({str(x_a): x_a_guess})
    dae['xy_0_dict'].update({str(xi_m): 0.0})
    dae['xy_0_dict'].update({str(v_f): v_f_guess})
    dae['xy_0_dict'].update({str(v_ref): v_setpoint})


def test():
    from pydae.core import Builder, Model
    from pydae.bps import BpsBuilder
    import pytest

    grid = BpsBuilder('st4b.hjson')
    grid.uz_jacs = False
    grid.construct('temp_st4b')

    bld = Builder(grid.sys_dict, target="cffi", sparse=False)
    bld.build()

    model = Model('temp_st4b')

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
