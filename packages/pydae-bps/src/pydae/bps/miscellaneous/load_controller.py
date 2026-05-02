# -*- coding: utf-8 -*-
r"""
Per-machine active-power load controller (LC).

Problem statement
-----------------

Every synchronous machine has armature resistance :math:`R_a`.  Because of
this, the mechanical power :math:`p_m` and the electrical power injected into
the grid :math:`p_g` are not equal at steady state:

.. math::

   p_m = p_g + R_a\,(i_d^2 + i_q^2)

The loss term :math:`R_a(i_d^2 + i_q^2)` depends on the operating point and
is not known before solving.  This means that a user who sets
``p_c = 0.8 pu`` as the *governor load reference* gets
:math:`p_g \approx 0.8 - \text{losses} < 0.8` injected into the grid.
For small :math:`R_a` the error is typically 1–3 %, but it breaks the
one-to-one correspondence between dispatch setpoint and grid injection that
power-system operators expect.

Solution: slow integrator
--------------------------

The load controller wraps the governor (or direct :math:`p_m`) with a **pure
integrator** whose steady-state condition automatically compensates for losses:

.. math::

   \dot{x}_{lc} &= K_{i,lc}\,(p_{c,lc} - p_g) \\
   0 &= x_{lc} + \Delta p_{lc} - p_{ctrl}

where :math:`p_{ctrl}` is either the governor's ``p_c`` (when a governor is
present) or the synchronous machine's ``p_m`` (standalone), and
:math:`\Delta p_{lc}` is the fast additive channel described below.

**At ini()** the steady-state condition :math:`\dot{x}_{lc} = 0` forces

.. math::

   p_g = p_{c,lc}

regardless of :math:`K_{i,lc}`.  Newton–Raphson finds the :math:`x_{lc}`
(and thus :math:`p_{ctrl}`) that achieves this, automatically absorbing
the loss correction.

**During run()** the integrator continuously adjusts the base setpoint so
that :math:`p_g \to p_{c,lc}` on the slow timescale :math:`1/K_{i,lc}`.
With :math:`K_{i,lc} = 0.01\ \text{pu/s/pu}` the time constant is
:math:`\tau_{lc} = 100\ \text{s}`, which is one to two orders of magnitude
slower than primary governor dynamics (:math:`\tau_{gov} \approx 1`–
:math:`10\ \text{s}`) and swing dynamics (:math:`\tau_{swing} \approx 1`–
:math:`5\ \text{s}`).  The load controller therefore does not interfere with
fault-ride-through or inter-area oscillation studies.

Fast additive channel for AGC
-------------------------------

A slow LC creates a problem when AGC is also present: if all power setpoint
changes must pass through the 100 s integrator, the AGC response at the grid
level appears sluggish.

This is solved by splitting the controlled variable into **two additive
terms**:

.. math::

   p_{ctrl} = x_{lc} + \Delta p_{lc}

+------------------+--------------------------------------------------+
| Term             | Role                                             |
+==================+==================================================+
| :math:`x_{lc}`  | **Slow base setpoint** (τ ≈ 100 s).             |
|                  | Ensures :math:`p_g = p_{c,lc}` at long-term      |
|                  | steady state and absorbs armature-loss correction.|
+------------------+--------------------------------------------------+
| :math:`\Delta p` | **Fast additive channel** (default 0).           |
| :math:`_{lc}`    | Driven by AGC at the AGC's own PI timescale      |
|                  | (seconds).  Bypasses the slow integrator so AGC  |
|                  | signals reach the governor/machine immediately.  |
+------------------+--------------------------------------------------+

:func:`pydae.bps.miscellaneous.agc.add_agc` automatically detects the
``dp_lc_{name}`` input and connects to it in preference to ``p_c`` or
``p_m``.

Three-level control hierarchy
------------------------------

With a governor, LC, and AGC all active the complete signal flow is::

    p_c_lc  ─── user dispatch setpoint (minutes / economic dispatch)
        │
        ▼
    x_lc  ─── LC integrator   K_i_lc ≈ 0.01  (τ ≈ 100 s)
        │      dx_lc/dt = K_i_lc*(p_c_lc − p_g)
        │
        +────── dp_lc  ◄── AGC PI   K_i_agc ≈ 1–10  (τ ≈ 1–30 s)
        │                    dξ_agc/dt = 1 − ω
        │                    0 = −dp_lc + K_p·(1−ω) + K_i·ξ_agc
        ▼
    ctrl_sym = x_lc + dp_lc
        │
        ▼  (governor load reference p_c, or direct p_m if no governor)
    governor ─── primary droop  (milliseconds–seconds)
        │
        ▼
    p_m  →  synchronous machine  →  p_g (grid injection)

**Timescale separation** means each level acts on a different part of the
frequency-restoration process:

- Governor (primary): arrests the frequency drop within seconds.
- AGC (secondary): eliminates the residual frequency error within minutes by
  driving ``dp_lc``.  Because ``dp_lc`` bypasses ``x_lc``, the generator
  sees the AGC signal at full speed.
- LC (tertiary): over tens of minutes it shifts ``x_lc`` to absorb the AGC
  offset, restoring ``dp_lc → 0`` and ensuring :math:`p_g = p_{c,lc}` once
  the system has re-dispatched.

Control variable priority
--------------------------

``add_lc`` resolves the controlled variable at construction time:

1. If ``p_c_{name}`` exists in ``u_ini_dict`` (governor present): drives the
   governor's load reference.
2. Otherwise drives ``p_m_{name}`` directly (no governor).

This is the same pattern used by :func:`pydae.bps.miscellaneous.agc.add_agc`.

Variables added to the DAE
---------------------------

+------------------------+-------+---------+------------------------------------------+
| Symbol                 | Phase | Type    | Description                              |
+========================+=======+=========+==========================================+
| ``x_lc_{name}``        | both  | state   | Integrator state (= base setpoint)       |
+------------------------+-------+---------+------------------------------------------+
| ``ctrl_sym``           | both  | alg.    | Controlled variable (gov. p_c or p_m)   |
+------------------------+-------+---------+------------------------------------------+
| ``p_c_lc_{name}``      | both  | input   | Desired p_g (user setpoint)              |
+------------------------+-------+---------+------------------------------------------+
| ``dp_lc_{name}``       | both  | input   | Fast additive channel (AGC or manual)    |
+------------------------+-------+---------+------------------------------------------+

Configuration
-------------

**Explicit ``lc:`` block** — full control over ``K_i``::

    gov: {type: "tgov1", ..., p_c: 0.8},
    lc:  {K_i: 0.01,  p_c_lc: 0.8}

**Bare ``p_c_lc:`` shorthand** — ``add_syns`` auto-creates
``lc: {K_i: 0.001, p_c_lc: ...}`` and calls ``add_lc``::

    gov: {type: "tgov1", ..., p_c: 0.8},
    p_c_lc: 0.8

If ``p_m`` is present at the syn level instead of ``p_c_lc``, the LC is
**not** added and ``p_m`` is used directly as the mechanical power input.

The ``p_c_lc`` value is the **desired grid injection** :math:`p_g`, not the
mechanical power.  The ``gov.p_c`` field serves only as an initial guess for
the governor's internal states and is replaced by the LC integrator output.

``add_lc`` must be called after the governor builder; :func:`add_syns` in
:mod:`pydae.bps.syns.syns` ensures this ordering.

Outputs available after ``model.ini()``
-----------------------------------------

- ``p_c_lc_{name}`` — the desired :math:`p_g` setpoint (echo of input).
- ``dp_lc_{name}``  — the fast additive increment (0 at rest, driven by AGC
  during a frequency event).
- ``x_lc_{name}``   — the integrator state (≈ the true :math:`p_m` at steady
  state when ``dp_lc = 0``).

See also
--------

:mod:`pydae.bps.miscellaneous.agc` — the system-level AGC that connects to
the ``dp_lc`` fast channel.
"""

import sympy as sym


def add_lc(dae, syn_data, name, _bus_name, backend=None):
    r"""
    Attach the per-machine load controller to *dae* for generator *name*.

    Wraps the controlled variable (governor ``p_c`` or direct ``p_m``) with
    a slow integrator and exposes a fast additive channel ``dp_lc`` for AGC:

    .. math::

        \dot{x}_{lc} &= K_{i,lc}\,(p_{c,lc} - p_g) \\
        0 &= x_{lc} + \Delta p_{lc} - p_{ctrl}

    Called by :func:`pydae.bps.syns.syns.add_syns` **after** the optional
    governor builder has run, so the governor's ``p_c_lc`` is already in
    ``u_ini_dict`` when this function executes.
    """
    if backend is None:
        backend = type('Backend', (), {
            'symbols': lambda _, n, **k: sym.Symbol(n, real=True),
            'use_casadi': False,
        })()

    lc_data = syn_data['lc']
    p_c_ini = lc_data.get('p_c_lc', 0.5)   # desired p_g setpoint
    K_i_val = lc_data.get('K_i', 0.01)  # integrator gain (pu/s per pu error)

    p_g = backend.symbols(f"p_g_{name}")
    x_lc = backend.symbols(f"x_lc_{name}")
    p_c_lc = backend.symbols(f"p_c_lc_{name}")
    dp_lc = backend.symbols(f"dp_lc_{name}")  # fast additive channel (AGC)
    K_i_lc = backend.symbols(f"K_i_lc_{name}")

    # Priority: governor p_c > direct p_m.
    p_c_key = f'p_c_{name}'
    p_m_key = f'p_m_{name}'
    if p_c_key in dae['u_ini_dict']:
        ctrl_sym = backend.symbols(p_c_key)
        dae['u_ini_dict'].pop(p_c_key)
        dae['u_run_dict'].pop(p_c_key, None)
    else:
        ctrl_sym = backend.symbols(p_m_key)
        dae['u_ini_dict'].pop(p_m_key, None)
        dae['u_run_dict'].pop(p_m_key, None)

    # Integrator: at steady state dx_lc/dt = 0  →  p_g = p_c_lc.
    dx_lc  = K_i_lc * (p_c_lc - p_g)
    # Algebraic: ctrl_sym = x_lc (slow base) + dp_lc (fast AGC increment).
    # dp_lc bypasses the integrator so AGC signals reach the governor/machine
    # at their own timescale, not filtered through the slow LC.
    g_ctrl = x_lc + dp_lc - ctrl_sym

    dae['f']     += [dx_lc]
    dae['x']     += [x_lc]
    dae['g']     += [g_ctrl]
    dae['y_ini'] += [ctrl_sym]
    dae['y_run'] += [ctrl_sym]

    dae['params_dict'].update({str(K_i_lc): K_i_val})

    dae['u_ini_dict'].update({str(p_c_lc): p_c_ini})
    dae['u_run_dict'].update({str(p_c_lc): p_c_ini})
    dae['u_ini_dict'].update({str(dp_lc): 0.0})
    dae['u_run_dict'].update({str(dp_lc): 0.0})

    # Initial guess: start at desired p_g, not the old governor setpoint.
    # p_c_ini is a better seed than p_ctrl_ini because x_lc ≈ p_m ≈ p_c_ini + losses.
    dae['xy_0_dict'].update({str(x_lc): p_c_ini, str(ctrl_sym): p_c_ini})

    dae['h_dict'].update({
        f'p_c_lc_{name}': p_c_lc,
        f'dp_lc_{name}':  dp_lc,
        f'x_lc_{name}':   x_lc,
    })


def test():
    import pytest
    from pydae.bps import BpsBuilder
    from pydae.core import Builder, Model

    # --- governor + LC ---
    grid = BpsBuilder('lc.hjson')
    grid.uz_jacs = False
    grid.construct('temp_lc')

    bld = Builder(grid.sys_dict, target='ctypes', sparse=False)
    bld.build()

    model = Model('temp_lc')

    v_set   = 1.02
    p_c_set = 0.8
    model.ini({'V_1': v_set, 'p_c_lc_1': p_c_set}, 'xy_0.json')

    print('\nx states:\n'); model.report_x()
    print('\ny states:\n'); model.report_y()
    print('\nu inputs:\n'); model.report_u()

    p_g_ini = model.get_value('p_g_1')
    p_m_ini = model.get_value('p_m_1')
    print(f"p_g_1 = {p_g_ini:.5f}  (target {p_c_set})")
    print(f"p_m_1 = {p_m_ini:.5f}  (> p_g due to R_a losses)")

    # p_g must match the setpoint exactly at steady state.
    assert p_g_ini == pytest.approx(p_c_set, rel=1e-3), \
        f"p_g {p_g_ini:.5f} != p_c_lc {p_c_set}"
    # p_m must be strictly larger than p_g (losses > 0 when R_a > 0).
    assert p_m_ini > p_g_ini, "p_m should exceed p_g (armature losses)"
    assert model.get_value('V_1') == pytest.approx(v_set, rel=1e-3)

    model.run(1.0, {})
    model.run(30.0, {'p_c_lc_1': 0.6})
    model.post()

    p_g_end = model.get_values('p_g_1')[-1]
    print(f"p_g_1 at t=31 s: {p_g_end:.4f}  (target 0.6)")
    assert p_g_end == pytest.approx(0.6, rel=5e-3), \
        f"p_g did not settle to new setpoint: {p_g_end:.4f}"


if __name__ == '__main__':
    test()
