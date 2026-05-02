# -*- coding: utf-8 -*-
r"""
Ideal voltage source — infinite-bus equivalent for pydae-bps.

A ``vsource`` connected to a bus pins that bus's voltage magnitude and
angle to their reference values, acting as a **stiff Thévenin source**
with zero internal impedance.  It is used to model:

- An **infinite bus** (slack node) in single-machine or multi-machine
  studies where one terminal is an ideal grid equivalent.
- The **New York system equivalent** in the IEEE 39-bus benchmark, where
  the New England area is connected to an external, very stiff grid.

**What the vsource does**

It replaces the standard network bus equations for bus :math:`k` with:

.. math::

   0 = V_k - v_{ref}
   \qquad
   0 = \theta_k - \theta_{ref}

so the power-flow and time-domain solver treats :math:`V_k` and
:math:`\theta_k` as fixed inputs rather than unknown algebraic states.

A dummy dynamic state :math:`V_{dummy}` is added with

.. math::

   \dot{V}_{dummy} = v_{ref} - V_{dummy}

so that the state vector has a consistent dimension.  This state has no
physical meaning; it simply settles to :math:`v_{ref}` at steady state.

**COI contribution**

The vsource contributes :math:`H = 10^6` s to the centre-of-inertia
(COI) computation.  This makes it the dominant term, pinning
:math:`\omega_{COI} \approx 1` pu and establishing the absolute angle
reference for the rest of the network.

**Inputs (runtime settable)**

+----------------------+--------+--------------------------------------------+
| Symbol               | Default| Description                                |
+======================+========+============================================+
| ``v_ref_{name}``     | 1.0 pu | Bus voltage magnitude setpoint             |
+----------------------+--------+--------------------------------------------+
| ``theta_ref_{name}`` | 0.0 rad| Bus voltage angle setpoint                 |
+----------------------+--------+--------------------------------------------+

**HJSON configuration**

Minimal (slack at bus "6" with default V = 1 pu, θ = 0 rad)::

    sources: [{type: "vsource", bus: "6"}]

With explicit setpoints::

    sources: [{"type": "vsource", "bus": "39",
               "S_n": 10000e9, "F_n": 60,
               "X_v": 0.0001, "R_v": 0.0,
               "K_delta": 0.01, "K_alpha": 0.01}]

Note: ``S_n``, ``F_n``, ``X_v``, ``R_v``, ``K_delta``, ``K_alpha`` are
accepted in the HJSON for documentation purposes but are **not currently
used** by this builder — the source is always ideal (zero impedance).
Use ``v_ref_{name}`` and ``theta_ref_{name}`` in ``model.ini()`` to
override the operating-point setpoints at runtime.

**Usage**

.. code-block:: python

    # Pin bus "39" at V = 1.03 pu, θ = 0 (default)
    model.ini({"v_ref_39": 1.03}, "xy_0.json")

    # Step the source voltage during a simulation
    model.run(1.0, {})
    model.run(10.0, {"v_ref_39": 1.05})
"""



def vsource(grid, name, bus_name, data_dict):
    """
    Attach an ideal voltage source to *bus_name* inside *grid*.

    Replaces the bus V and θ equations with algebraic pin constraints and
    adds a dummy state ``V_dummy_{name}`` for consistent vector sizing.
    """

    sin = grid.backend.sin
    cos = grid.backend.cos

    # inputs
    V = grid.backend.symbols(f"V_{bus_name}")
    theta = grid.backend.symbols(f"theta_{bus_name}")
    v_ref = grid.backend.symbols(f"v_ref_{name}")
    theta_ref = grid.backend.symbols(f"theta_ref_{name}")
    V_dummy = grid.backend.symbols(f"V_dummy_{name}")


    # dynamic states

    # algebraic states
    # V = sym.Symbol(f"V_{name}", real=True)
    # theta = sym.Symbol(f"theta_{name}", real=True)


    # parameters
    params_list = []

    # auxiliar

    # dynamic equations
    grid.dae['f'] += [v_ref - V_dummy]
    grid.dae['x'] += [V_dummy]

    # algebraic equations
    g_V = V - v_ref
    g_theta = theta - theta_ref


    # dae

    H = 1e6
    grid.H_total += H
    grid.omega_coi_numerator += H
    grid.omega_coi_denominator += H

    idx_V = next(i for i, y in enumerate(grid.dae['y_ini']) if str(y) == str(V))
    idx_theta = next(i for i, y in enumerate(grid.dae['y_ini']) if str(y) == str(theta))

    grid.dae['g'][idx_V] = g_V
    grid.dae['g'][idx_theta] = g_theta

    # grid.dae['y_ini'] += [V, theta]
    # grid.dae['y_run'] += [V, theta]

    grid.dae['u_ini_dict'].update({f'{str(v_ref)}':1.0})
    grid.dae['u_run_dict'].update({f'{str(v_ref)}':1.0})

    grid.dae['u_ini_dict'].update({f'{str(theta_ref)}':0.0})
    grid.dae['u_run_dict'].update({f'{str(theta_ref)}':0.0})

    grid.dae['xy_0_dict'].update({str(V):1.0})
    grid.dae['xy_0_dict'].update({str(theta):0.0})

    grid.dae['h_dict'].update({f"V_dummy_{name}":V_dummy})

    # outputs


def test_mkl():


    from pydae.bps import BpsBuilder

    grid = BpsBuilder('vsource.hjson')
    grid.construct('temp')
    grid.compile_mkl('temp')

    import temp

    model = temp.model()
    model.ini({},'xy_0.json')
    model.report_y()


def test():

    from pydae.bps import BpsBuilder

    grid = BpsBuilder('vsource.hjson')
    grid.checker()
    grid.verbose = True
    grid.build('temp')

    import temp

    model = temp.model()
    model.ini({},'xy_0.json')
    model.report_y()



if __name__=='__main__':
    test_mkl()





