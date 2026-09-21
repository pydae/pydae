# -*- coding: utf-8 -*-
r"""Branch current from the branch admittance and the bus voltages.

One source of truth for the :math:`\pi`-section current, shared by the
monitoring block in :class:`~pydae.bps.bps_builder.BpsBuilder` and by the
dynamic-thermal-rating models, which need the current *inside* their heat
balance rather than merely as a reported quantity.

Why not :math:`|S|/V`
---------------------
The older form computed the branch current as the apparent power divided by
the terminal voltage. That needs ``p_line``/``q_line`` solved first, puts a
voltage that collapses during a fault in the **denominator**, and cannot
represent the line charging current at all -- it returns the same magnitude at
both ends of a :math:`\pi` section. Ohm's law on the branch admittance has the
voltage in the numerator and needs no power flow:

.. math::
    I_{jk} = y\,(V_j - V_k) + j\,\frac{b_s}{2}\,V_j

Why the primitive and not the Ybus
----------------------------------
The per-branch primitive (``g_``, ``b_``, ``bs_``) is used rather than the
Ybus off-diagonal ``Y[j,k]``. The latter is summed through the incidence
matrix, so two parallel circuits between the same pair of buses would report
their combined current as the current of *each* circuit.

Why per unit
------------
Everything here is per unit. Amperes are applied only at the point of use: a
current in amperes runs 3-5 orders above the per-unit voltages and powers
beside it, and as an algebraic variable it puts entries of ~1e5 into a ``Gy``
row whose own diagonal is 1. On a 14-branch PV plant that alone was
``cond(Gy) = 1.2e10`` against ``1.4e5`` without it.
"""


def line_currents_pu(backend, bus_j, bus_k, line_name):
    r"""Rectangular :math:`\pi`-section currents at both ends, in per unit.

    .. math::
        I_{jk} &= y\,(V_j - V_k) + j\,\tfrac{b_s}{2}\,V_j \\
        I_{kj} &= y\,(V_k - V_j) + j\,\tfrac{b_s}{2}\,V_k

    with :math:`y = g + jb` taken from the branch primitive.

    Parameters
    ----------
    backend : MathBackend
        Symbol/function factory; symbols are resolved **by name**, so this
        works whether ``g_``/``b_`` are parameters (an ordinary line) or
        algebraic variables (a DTR line, where they track ``R(T)``).
    bus_j, bus_k : str
        Bus names, used to reach ``V_{bus}`` and ``theta_{bus}``.
    line_name : str
        Branch name, used to reach ``g_``/``b_``/``bs_``.

    Returns
    -------
    tuple
        ``((i_jk_r, i_jk_i), (i_kj_r, i_kj_i))`` in per unit of the system
        base.
    """
    V_j = backend.symbols(f"V_{bus_j}")
    V_k = backend.symbols(f"V_{bus_k}")
    theta_j = backend.symbols(f"theta_{bus_j}")
    theta_k = backend.symbols(f"theta_{bus_k}")

    g_jk = backend.symbols(f"g_{line_name}")
    b_jk = backend.symbols(f"b_{line_name}")
    bs_jk = backend.symbols(f"bs_{line_name}")

    v_jr = V_j*backend.cos(theta_j)
    v_ji = V_j*backend.sin(theta_j)
    v_kr = V_k*backend.cos(theta_k)
    v_ki = V_k*backend.sin(theta_k)

    dv_r = v_jr - v_kr
    dv_i = v_ji - v_ki

    i_jk = ( g_jk*dv_r - b_jk*dv_i - bs_jk/2*v_ji,
             g_jk*dv_i + b_jk*dv_r + bs_jk/2*v_jr)
    i_kj = (-g_jk*dv_r + b_jk*dv_i - bs_jk/2*v_ki,
            -g_jk*dv_i - b_jk*dv_r + bs_jk/2*v_kr)
    return i_jk, i_kj


def current_magnitude_sq(i_r, i_i, i_base=1.0):
    r"""Squared current magnitude, :math:`|I|^2`, with no square root.

    Preferred wherever the consumer squares the current anyway -- notably the
    Joule term of the thermal models, :math:`W = I^2 R`. Forming
    :math:`\sqrt{\cdot}` only to square it again would introduce a :math:`0/0`
    derivative at exactly zero flow for nothing; this expression is polynomial
    in the voltage rectangular components and smooth everywhere.
    """
    return (i_r**2 + i_i**2)*i_base**2


def current_magnitude(i_r, i_i, i_base=1.0, eps=1e-18):
    r"""Current magnitude :math:`|I|`, scaled by ``i_base``.

    ``eps`` keeps :math:`\partial |I| / \partial I_r` bounded at exactly zero
    current, which is reachable in normal operation -- a grid-following
    converter commanded to zero current leaves its branch at exactly zero
    flow -- and would otherwise put a NaN in the output Jacobian. At 1e-18
    pu**2 it is ~1e-9 pu, nine orders below anything meaningful.

    Use :func:`current_magnitude_sq` instead when the caller squares the
    result.
    """
    return (i_r**2 + i_i**2 + eps)**0.5*i_base
