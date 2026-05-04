# -*- coding: utf-8 -*-
r"""
Automatic Generation Control (AGC) for a single designated generator.

Overview
--------

AGC is the second-level frequency controller in a hierarchical power-system
control architecture.  Its role is to drive the system frequency back to
nominal (typically 50 Hz or 60 Hz) after a load or generation imbalance by
adjusting one generator's active-power setpoint.

The three control levels that interact here are:

1. **Primary control — governor** (milliseconds to ~10 s):
   The synchronous machine's governor responds to frequency deviations
   through a speed-droop characteristic.  It arrests frequency excursions but
   leaves a residual steady-state error proportional to the droop setting.

2. **Secondary control — AGC** (seconds to ~2 min):
   AGC eliminates the residual frequency error by integrating the speed
   deviation and feeding a corrective setpoint increment back to the generator.
   At steady state the integrator forces :math:`\omega \to 1` pu exactly.

3. **Tertiary control — Load Controller (LC)** (minutes):
   When an LC is also present (see :mod:`pydae.bps.miscellaneous.load_controller`),
   the slow LC integrator maintains the correct scheduled active-power injection
   :math:`p_g = p_{c,lc}` over long timescales, compensating for armature
   losses automatically.

Mathematical model
------------------

Let :math:`\omega_{gen}` be the rotor speed (pu) of the designated generator
and :math:`p_{ctrl}` the algebraic variable that AGC drives.  AGC is a
proportional-integral controller on the speed error
:math:`\varepsilon = 1 - \omega_{gen}`:

.. math::

   \dot{\xi}_{agc} &= 1 - \omega_{gen} \\
   0 &= -p_{ctrl} + K_p\,(1 - \omega_{gen}) + K_i\,\xi_{agc}

At steady state (:math:`\omega_{gen} = 1`):

.. math::

   \dot{\xi}_{agc} = 0, \qquad p_{ctrl} = K_i\,\xi_{agc}

so the integrator state :math:`\xi_{agc}` winds to the value that holds the
power balance.

**Initialisation**:  because :math:`\omega_{gen} = 1` trivially satisfies
:math:`\dot{\xi}_{agc} = 0`, Newton–Raphson is free to choose :math:`\xi_{agc}`
and :math:`p_{ctrl}` from the network equations alone.  No prior knowledge of
armature losses is required — the correct operating-point power is found
automatically.

Control variable priority
--------------------------

``add_agc`` must be called **after** ``add_syns`` (and any governor/LC
builders), so the controlled variable already exists in ``u_ini_dict``.
The function resolves the controlled variable in the following order:

+----------------+-----------------------------------------------------+
| Key present    | What AGC drives                                     |
+================+=====================================================+
| ``dp_lc_{gen}``| **LC fast channel** — AGC signal reaches the        |
|                | governor/machine immediately, bypassing the slow    |
|                | LC integrator.  This is the preferred mode when an  |
|                | LC is used alongside AGC; it preserves both the fast|
|                | secondary-frequency response and the slow loss-      |
|                | compensating base setpoint.                         |
+----------------+-----------------------------------------------------+
| ``p_c_{gen}``  | Governor load-reference (governor present, no LC).  |
+----------------+-----------------------------------------------------+
| ``p_m_{gen}``  | Direct mechanical power (no governor, no LC).       |
+----------------+-----------------------------------------------------+

Two-level LC + AGC architecture
---------------------------------

When both LC and AGC are present the net governor/machine setpoint is::

    ctrl_sym  =  x_lc          +  dp_lc
                 ^^^^              ^^^^^
                 slow base         AGC output
                 (K_i_lc ~ 0.01)   (K_i_agc ~ 1–10)

- ``x_lc`` drifts slowly (τ ≈ 100 s) to hold :math:`p_g = p_{c,lc}`.
  It absorbs the armature-loss correction and tracks long-term dispatch
  changes.
- ``dp_lc`` is driven by AGC at its own PI timescale (seconds) to cancel
  frequency deviations.  Because it bypasses the LC integrator, the
  generator sees the full AGC step without the ~100 s delay.

Signal-flow diagram::

    p_c_lc (slow setpoint, minutes)
        │
        ▼
    x_lc ─── LC integrator  K_i_lc ≈ 0.01  (τ ≈ 100 s)
        │     dx_lc/dt = K_i_lc * (p_c_lc − p_g)
        │
        +────── dp_lc ◄── AGC algebraic output  (τ ≈ 1–30 s)
        │                  0 = −dp_lc + K_p·ε + K_i·ξ_agc
        ▼
    ctrl_sym = x_lc + dp_lc   →  governor p_c  →  p_m  →  p_g

Configuration
-------------

System-level key in the HJSON/JSON data dict::

    agc: {gen: "2", K_p_agc: 0.0, K_i_agc: 2.0}

``gen`` must match the generator's ``name`` (or bus name when no explicit
name is given).  ``K_p_agc`` defaults to 0.0 (pure integral); adding a small
proportional term can improve damping during large frequency excursions.

Must be placed after all ``syns`` entries in the data file (and after ``lc``
keys in those entries) because ``add_agc`` is called last in
:meth:`BpsBuilder.construct`.

See also
--------

:mod:`pydae.bps.miscellaneous.load_controller` — the per-machine LC that
provides the slow base setpoint and the ``dp_lc`` fast channel.
"""




def add_agc(grid):
    r"""
    Attach the system-level AGC integrator to *grid.dae* for the designated
    generator.

    The controlled variable is determined by the generator's configuration:

    - **With governor, no LC**: AGC drives ``p_c_{gen}`` (governor load
      reference). ``p_c`` is removed from inputs and becomes an algebraic
      variable solved by the AGC equation.
    - **With governor AND LC**: AGC drives ``dp_lc_{gen}`` (LC fast channel).
      The LC already owns ``p_c``; AGC provides the fast frequency-response
      signal that bypasses the slow LC integrator.
    - **Without governor**: AGC drives ``p_m_{gen}`` (mechanical power).
      ``p_m`` is removed from inputs and becomes an algebraic variable.

    The PI control law is:

    .. math::

        0 = -p_{ctrl} + K_p\,(1-\omega) + K_i\,\xi_{agc}

    Must be called **after** ``add_syns`` so the controlled variable already
    exists.  ``BpsBuilder.construct`` calls it last, after all component
    builders.
    """

    backend = grid.backend
    agc_data = grid.data['agc']
    gen_name  = agc_data['gen']
    K_p_val   = agc_data.get('K_p_agc', 0.0)
    K_i_val   = agc_data.get('K_i_agc', 1.0)

    # omega   = backend.symbols(f"omega_{gen_name}")
    omega   = backend.symbols(f"omega_coi")

    K_p     = backend.symbols(f"K_p_agc")
    K_i     = backend.symbols(f"K_i_agc")
    xi_agc  = backend.symbols(f"xi_agc")

    # Determine the controlled variable based on what is available.
    dp_lc_key = f'dp_lc_{gen_name}'
    p_c_key   = f'p_c_{gen_name}'
    p_m_key   = f'p_m_{gen_name}'

    if dp_lc_key in grid.dae['u_ini_dict']:
        # LC is present — AGC drives the fast channel.
        ctrl_key = dp_lc_key
    elif p_c_key in grid.dae['u_ini_dict']:
        # Governor present, no LC — AGC drives p_c directly.
        ctrl_key = p_c_key
    else:
        # No governor — AGC drives p_m directly.
        ctrl_key = p_m_key

    ctrl_sym = backend.symbols(ctrl_key)
    grid.dae['u_ini_dict'].pop(ctrl_key)
    grid.dae['u_run_dict'].pop(ctrl_key)

    # Remove from xy_0_dict too (it was an input, now it's algebraic).
    grid.dae['xy_0_dict'].pop(ctrl_key, None)

    epsilon = 1 - omega

    grid.dae['f']     += [epsilon]
    grid.dae['x']     += [xi_agc]
    grid.dae['g']     += [-ctrl_sym + K_p * epsilon + K_i * xi_agc]
    grid.dae['y_ini'] += [ctrl_sym]
    grid.dae['y_run'] += [ctrl_sym]

    print("agc: ctrl_key=", ctrl_key)
    print("grid.dae['g'] AGC", grid.dae['g'][-1]) 

    grid.dae['params_dict'].update({str(K_p): K_p_val, str(K_i): K_i_val})

    # Auto-compute consistent initial guesses for AGC integrator and governor states.
    # When AGC drives p_c directly (no LC), we need:
    #   xi_agc ≈ p_c_expected / K_i_agc  (from steady-state AGC equation)
    #   x_1_gov = x_2_gov = p_m = p_c_expected  (governor steady state)
    # We infer p_c_expected from existing xy_0_dict seeds set by tgov1/LC.
    if ctrl_key == p_c_key:
        # Look for any existing power guess for this generator.
        p_c_guess = grid.dae['u_ini_dict'].get(ctrl_key, 0.5)
        # Also check xy_0_dict for governor states that may have better seeds.
        x_1_key = f'x_1_gov_{gen_name}'
        x_2_key = f'x_2_gov_{gen_name}'
        p_m_key_full = f'p_m_{gen_name}'
        existing_gov_guess = max(
            grid.dae['xy_0_dict'].get(x_1_key, 0.0),
            grid.dae['xy_0_dict'].get(x_2_key, 0.0),
            grid.dae['xy_0_dict'].get(p_m_key_full, 0.0),
            p_c_guess,
        )
        if existing_gov_guess > 0:
            p_c_expected = existing_gov_guess
        else:
            p_c_expected = 0.5
        xi_0 = p_c_expected / K_i_val if K_i_val != 0.0 else 0.0
        grid.dae['xy_0_dict'].update({str(xi_agc): xi_0, ctrl_key: p_c_expected})
        # Ensure governor states match expected power at steady state.
        grid.dae['xy_0_dict'].update({x_1_key: p_c_expected, x_2_key: p_c_expected, p_m_key_full: p_c_expected})

    grid.dae['h_dict'].update({
        'p_agc':  ctrl_sym,
        'xi_agc': xi_agc,
    })


def test():
    from pydae.core import Builder, Model
    from pydae.bps import BpsBuilder

    grid = BpsBuilder('agc.hjson')
    grid.uz_jacs = False
    grid.construct('temp_agc')

    bld = Builder(grid.sys_dict, target="cffi", sparse=False)
    bld.build()

    model = Model('temp_agc')

    v_set = 1.02
    model.ini({'V_1': v_set}, 'xy_0.json')

    print('\nx states:\n');  model.report_x()
    print('\ny states:\n');  model.report_y()
    print('\nu inputs:\n');  model.report_u()


if __name__ == '__main__':
    test()
