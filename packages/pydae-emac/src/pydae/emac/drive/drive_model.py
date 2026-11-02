# -*- coding: utf-8 -*-
"""The switched drive as a CasADi model -- machine, inverter and DSP.

The multirate solver of pydae-emac: what a mean-value DAE (``machines``)
cannot show -- sampling, computation delay, the carrier, each leg's
switching edge, dead time and the FEA's spatial ripple -- for the price of
one compiled function per control period.  The machine is built from
:func:`pydae.emac.luts.build_flux_tables`.

A real drive runs at two rates, and so does this model:
  * the DSP, every T_s = 50 us: it samples the currents, runs the current
    controller and writes three duty cycles (one per inverter leg);
  * the PWM hardware, continuously: it compares those duties with a
    triangular carrier and switches each leg exactly where they cross.

One "control period" (T_s) of all that is a single CasADi function, and
CasADi's mapaccum chains thousands of them in one call.

The pieces, in the order they are built:
  build_machine()         the machine's ODE, with the fluxes as states
  build_window()          the machine over one T_s with the voltage changing
                          at the 3 switching instants (4 constant pieces,
                          one Runge-Kutta step each)
  PIController            the default controller -- replace it to prototype
                          your own (see "THE CONTROLLER" below)
  build_control_period()  DSP + PWM + window: one complete T_s
  simulate()              run N control periods
"""
import casadi as ca
import numpy as np

SQ3 = np.sqrt(3.0)


def build_machine(tables, R, J):
    """The machine's differential equations.

    states   x = [lam_d, lam_q, omega_m, theta_m]   fluxes [Wb], mechanical
                                                    speed [rad/s] and angle [rad]
    inputs   u = [v_alpha, v_beta, T_load, dt]      stator voltage in the FIXED
                                                    (alpha-beta) frame [V], load
                                                    torque [N.m], duration [s]

    The voltage is given in the fixed frame because that is what the inverter
    holds constant between two switching edges; in the rotor's dq frame it
    turns with the rotor, so the Park transform is done inside the equations.

    The equations are written in "scaled time" tau = t / dt (tau goes 0 -> 1):
        dx/dtau = dt * dx/dt
    so ONE integrator handles a piece of any length, even zero.

    Returns the ODE (for the integrator) and a function x -> [i_d, i_q, T].
    """
    # The B-spline gives ALL Fourier coefficients of i_d, i_q and T at once,
    # as a function of the two fluxes.  Cubic = smooth to 2nd derivatives.
    spline = ca.interpolant("lut", "bspline", [tables["grid_d"], tables["grid_q"]],
                            tables["data"], {"degree": [3, 3]})

    lam_d, lam_q = ca.SX.sym("lam_d"), ca.SX.sym("lam_q")
    omega, theta = ca.SX.sym("omega_m"), ca.SX.sym("theta_m")
    v_al, v_be = ca.SX.sym("v_alpha"), ca.SX.sym("v_beta")
    T_load, dt = ca.SX.sym("T_load"), ca.SX.sym("dt")
    x = ca.vertcat(lam_d, lam_q, omega, theta)
    u = ca.vertcat(v_al, v_be, T_load, dt)

    p = tables["p"]
    theta_e = p * theta                             # electrical angle
    omega_e = p * omega                             # electrical speed

    # Fourier basis [1, cos(h1 th), sin(h1 th), cos(h2 th), ...]
    h = ca.DM(tables["harmonics"])
    cos_sin = ca.horzcat(ca.cos(h * theta_e), ca.sin(h * theta_e)).T
    basis = ca.vertcat(1, ca.reshape(cos_sin, -1, 1))

    # Currents and torque: coefficients (3 x n_basis) times the basis
    coeffs = ca.reshape(spline(ca.vertcat(lam_d, lam_q)), 3, tables["n_basis"])
    i_d, i_q, T_e = ca.vertsplit(ca.mtimes(coeffs, basis))

    # Park transform of the stator voltage into the rotor frame
    v_d = v_al * ca.cos(theta_e) + v_be * ca.sin(theta_e)
    v_q = -v_al * ca.sin(theta_e) + v_be * ca.cos(theta_e)

    dxdt = ca.vertcat(
        v_d - R * i_d + omega_e * lam_q,            # stator d-axis
        v_q - R * i_q - omega_e * lam_d,            # stator q-axis
        (T_e - T_load) / J,                         # rotor speed
        omega)                                      # rotor angle
    ode = dict(x=x, p=u, ode=dt * dxdt)             # in scaled time
    currents = ca.Function("currents", [x], [ca.vertcat(i_d, i_q, T_e)])
    return ode, currents


def build_window(ode, integrator="rk4"):
    """The machine over one control period: 4 constant-voltage pieces in a row.

    window(x0, U) -> (x_end, X)
        U  4 x 4: one column per piece, [v_alpha, v_beta, T_load, dt]
        X  4 x 4: the state at the END of every piece, i.e. at every
           switching instant -- the samples used for averaging and plots

    integrator "rk4": ONE classical Runge-Kutta step per piece, written out
                      as plain CasADi expressions.  A piece lasts at most
                      50 us and the machine's electrical time constants are
                      milliseconds, so one step is enough: against 8 steps
                      per piece it moves the mean torque by 0.013 % and the
                      flux by ~1e-6 Wb, and runs 4x faster.  Written out (not
                      CasADi's "rk" plugin) so the model can also be compiled
                      to C, which the plugin cannot.
               "cvodes": adaptive, error-controlled; the precise reference,
                      for checking the rk4 result (several times slower).
    """
    x0 = ca.MX.sym("x0", 4)
    U = ca.MX.sym("U", 4, 4)
    if integrator == "rk4":
        # dx/dtau as a function; tau runs 0 -> 1 over a piece (scaled time)
        f = ca.Function("f", [ode["x"], ode["p"]], [ode["ode"]])

        def piece(x, u):                            # one RK4 step, tau 0 -> 1
            k1 = f(x, u)
            k2 = f(x + 0.5 * k1, u)
            k3 = f(x + 0.5 * k2, u)
            k4 = f(x + k3, u)
            return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    else:
        I = ca.integrator("piece", integrator, ode, 0.0, 1.0,
                          dict(abstol=1e-8, reltol=1e-6))

        def piece(x, u):                            # adaptive, tau 0 -> 1
            return I(x0=x, p=u)["xf"]

    x, samples = x0, []
    for j in range(4):                              # the 4 pieces, one after another
        x = piece(x, U[:, j])                       # the next piece starts here
        samples.append(x)
    return ca.Function("window", [x0, U], [x, ca.horzcat(*samples)])


def leg_voltages_to_alpha_beta(s, V_dc):
    """Inverter leg states (0 = low, 1 = high; or duties) -> v_alpha, v_beta.

    Phase voltages to the star point: v_a = V_dc (2 s_a - s_b - s_c) / 3, etc.
    Then the Clarke transform (amplitude-invariant, like the dq model).
    """
    v_a = V_dc * (2 * s[0] - s[1] - s[2]) / 3
    v_b = V_dc * (2 * s[1] - s[2] - s[0]) / 3
    v_c = V_dc * (2 * s[2] - s[0] - s[1]) / 3
    return ca.vertcat((2 * v_a - v_b - v_c) / 3, (v_b - v_c) / SQ3)


# =============================================================================
# THE CONTROLLER -- the part to replace when prototyping a new one
# =============================================================================
# A controller is any object with
#     n_state   size of its memory carried from one period to the next
#     n_ref     how many references it takes per period
#     step(meas, ref, xc, T_s) -> (v_ab, xc_next)
# called ONCE per control period, at the sampling instant.  It returns the
# voltage it wants, in the fixed (alpha-beta) frame, for the NEXT period (the
# duties it leads to are applied one period later: the DSP's computation
# delay).  Everything is CasADi symbolic, so the controller becomes part of
# the same compiled graph and costs no Python per period.
#
# meas: what the DSP measures at the sampling instant
#     "i_dq"     dq currents [A] (2)        "i_abc"    phase currents [A] (3)
#     "theta_e"  electrical angle [rad]     "omega_e"  electrical speed [rad/s]
#     "V_dc"     DC-link voltage [V]        "v_max"    largest voltage the
#                                                      SVPWM can give, V_dc/sqrt(3)
#     "lam_dq"   the machine's TRUE fluxes [Wb] (2) -- NOT measurable on a
#                real drive; there for decoupling and observer studies
# ref: the n_ref references of this period (from simulate's i_ref / refs)
# xc:  the controller's memory (n_state), returned updated as xc_next


def dq_to_alpha_beta(v_dq, theta):
    """Inverse Park: rotor-frame (d, q) -> fixed-frame (alpha, beta)."""
    c, s = ca.cos(theta), ca.sin(theta)
    return ca.vertcat(v_dq[0] * c - v_dq[1] * s, v_dq[0] * s + v_dq[1] * c)


def limit_voltage(v_ab, v_max):
    """Scale v_ab into the inverter's circle |v| <= v_max.
    Returns (v_ab_limited, limited) with limited = 1 when it had to scale."""
    size = ca.sqrt(ca.sumsqr(v_ab))
    return v_ab * ca.fmin(1, v_max / ca.fmax(size, 1e-12)), size > v_max


class PIController:
    """The default: dq current PI with cross-coupling decoupling.

    references  [i_d_ref, i_q_ref]
    memory      the integral of the current error (2)
    """
    n_state = 2
    n_ref = 2

    def __init__(self, Kp, Ki):
        self.Kp, self.Ki = ca.DM(Kp), ca.DM(Ki)     # per axis [d, q]

    def step(self, meas, ref, xc, T_s):
        e = ref - meas["i_dq"]                      # current error
        lam, w = meas["lam_dq"], meas["omega_e"]
        v_dq = (self.Kp * e + self.Ki * xc          # PI
                + ca.vertcat(-w * lam[1], w * lam[0]))   # decoupling
        # to the fixed frame, at the angle the rotor will have in the MIDDLE
        # of the period these duties are applied in (one period later: 1.5 T_s)
        v_ab = dq_to_alpha_beta(v_dq, meas["theta_e"] + 1.5 * T_s * w)
        v_ab, limited = limit_voltage(v_ab, meas["v_max"])
        # anti-windup: stop integrating while the voltage is limited
        xc_next = xc + ca.if_else(limited, 0, 1) * e * T_s
        return v_ab, xc_next


def dead_duty(rising, i_abc, t_dead, T_s):
    """Extra high time the dead time gives each leg over one carrier ramp, as
    a duty (+ or -): on a rising ramp a leg whose current flows INTO it stays
    high t_dead longer, on a falling ramp one whose current flows OUT stays
    low t_dead longer.  rising = 1 on a rising ramp."""
    return t_dead / T_s * (rising * (i_abc < 0) - (1 - rising) * (i_abc > 0))


def pwm_pieces(d, rising, i_abc, T_load, V_dc, T_s, mode="switched", t_dead=0.0):
    """The PWM hardware and the inverter over one control period.

    d       the three duties in [0, 1] (what the DSP wrote to the compare registers)
    rising  1 if this period is the carrier's rising ramp, 0 if falling
    i_abc   the phase currents at the period start: they decide, during the
            dead time, which diode sets each leg's voltage
    T_load  load torque [N.m], carried in each piece for the machine window

    Returns (pieces, dts): four columns [v_alpha, v_beta, T_load, dt], the
    constant-voltage pieces the machine window integrates, and their lengths.
    mode "switched" places each leg's single edge where duty and carrier
    cross; "averaged" holds the period-mean voltage in four equal quarters.
    """
    out_of_leg = i_abc > 0                          # 1 where i > 0, else 0
    into_leg = i_abc < 0                            # 1 where i < 0, else 0
    if mode == "switched":
        # The carrier is a triangle; this period is one ramp of it (rising or
        # falling, alternating).  A leg is high while carrier < duty, so it
        # switches once in the period, at
        #   rising ramp:  t = d * T_s        (high -> low)
        #   falling ramp: t = (1 - d) * T_s  (low -> high)
        t_sw = (rising * d + (1 - rising) * (1 - d)) * T_s
        # dead time: delay the edges the diodes hold back (see dead_duty);
        # an edge pushed past the period end is cut there (only when a duty
        # is within t_dead of 0 or 1, near the voltage peak)
        t_sw = ca.fmin(t_sw + t_dead * (rising * into_leg
                                        + (1 - rising) * out_of_leg), T_s)
        # sort the three switching instants: first, middle, last
        first = ca.fmin(t_sw[0], ca.fmin(t_sw[1], t_sw[2]))
        last = ca.fmax(t_sw[0], ca.fmax(t_sw[1], t_sw[2]))
        middle = t_sw[0] + t_sw[1] + t_sw[2] - first - last
        edges = [0, first, middle, last, T_s]
        pieces, dts = [], []
        for j in range(4):                          # the 4 constant pieces
            mid = 0.5 * (edges[j] + edges[j + 1])   # a time inside the piece
            # each leg's state in this piece: high before its edge on a rising
            # ramp, high after its edge on a falling ramp
            legs = rising * (mid < t_sw) + (1 - rising) * (mid > t_sw)
            dt_j = edges[j + 1] - edges[j]
            pieces.append(ca.vertcat(leg_voltages_to_alpha_beta(legs, V_dc),
                                     T_load, dt_j))
            dts.append(dt_j)
    else:
        # averaged: the mean voltage over the period, held for all of it, as
        # four equal quarter-pieces.  The window always takes four RK4 steps,
        # so quarters cost nothing extra and give four samples per period
        # (one per period would miss the peaks of the 3.6 kHz slot ripple).
        # with dead time: the volt-seconds of the switched case, as a duty
        d_eff = ca.fmin(ca.fmax(d + dead_duty(rising, i_abc, t_dead, T_s), 0), 1)
        v_mean = leg_voltages_to_alpha_beta(d_eff, V_dc)
        dts = [T_s / 4] * 4
        pieces = [ca.vertcat(v_mean, T_load, dt_j) for dt_j in dts]

    return pieces, dts


def build_control_period(window, currents, p, V_dc, T_s, controller,
                         mode="switched", t_dead=0.0, compensate=False):
    """One complete control period: DSP (controller + modulator), PWM, machine.

    period(z, q) -> (z_next, X, dt_pieces)
        z = [x (4), controller memory (n_state), duties waiting (3)]
            everything carried from one period to the next
        q = [references (n_ref), rising]
            per period: the controller's references and the carrier direction
    controller: see "THE CONTROLLER" above (PIController is the default)
    The load torque always equals the motor torque, so the speed is held.

    mode "switched": the real switching pattern (4 pieces per period)
         "averaged": the duty-averaged voltage held for the whole period

    t_dead      dead time [s]: after every switching command both transistors
                of the leg stay off for t_dead, and meanwhile the phase
                current's diode decides the leg voltage.  So each leg's one
                edge in the period is delayed by t_dead, or not at all:
                  rising ramp,  high -> low:  delayed if the current flows
                                              INTO the leg (i < 0: upper
                                              diode keeps it high)
                  falling ramp, low -> high:  delayed if the current flows
                                              OUT of the leg (i > 0: lower
                                              diode keeps it low)
                Over a carrier period this loses V_dc t_dead f_sw of voltage
                per leg (10.8 V at 540 V, 2 us, 10 kHz) against the current.
                The current's sign is the one sampled at the period start.
    compensate  the DSP adds the opposite correction to the duties it
                writes, for the ramp they will be applied on.
    """
    nc, nr = controller.n_state, controller.n_ref
    z = ca.MX.sym("z", 4 + nc + 3)
    q = ca.MX.sym("q", nr + 1)
    x, xc, d_waiting = z[0:4], z[4:4 + nc], z[4 + nc:]
    ref, rising = q[0:nr], q[nr]

    # ---- DSP: sample ---------------------------------------------------------
    i_d, i_q, T_e = ca.vertsplit(currents(x))
    theta_e, omega_e = p * x[3], p * x[2]
    # phase currents (inverse Park), for the dead time: which diode conducts
    ang = theta_e + ca.DM([0, -2 * np.pi / 3, 2 * np.pi / 3])
    i_abc = i_d * ca.cos(ang) - i_q * ca.sin(ang)

    # ---- DSP: the controller (replaceable) -----------------------------------
    meas = {"i_dq": ca.vertcat(i_d, i_q), "i_abc": i_abc,
            "theta_e": theta_e, "omega_e": omega_e,
            "V_dc": V_dc, "v_max": V_dc / SQ3,
            "lam_dq": x[0:2]}                       # ideal: not measurable
    v_ab, xc_next = controller.step(meas, ref, xc, T_s)

    # ---- DSP: space-vector modulation -> three duties in [0, 1] -------------
    v_a = v_ab[0]
    v_b = -0.5 * v_ab[0] + SQ3 / 2 * v_ab[1]
    v_c = -0.5 * v_ab[0] - SQ3 / 2 * v_ab[1]
    # min-max injection: shift all three so the largest and smallest are
    # symmetric about zero (equivalent to space-vector PWM)
    shift = -0.5 * (ca.fmax(v_a, ca.fmax(v_b, v_c)) + ca.fmin(v_a, ca.fmin(v_b, v_c)))
    d_new = ca.fmin(ca.fmax(0.5 + (ca.vertcat(v_a, v_b, v_c) + shift) / V_dc, 0), 1)
    if compensate:
        # take the dead time's extra high time off the duties now, for the
        # ramp they will run on next period (the ramps alternate: next = 1 - rising)
        d_new = ca.fmin(ca.fmax(d_new - dead_duty(1 - rising, i_abc, t_dead, T_s), 0), 1)
    # computation delay: THIS period applies the duties computed LAST period
    d = d_waiting

    # ---- PWM hardware + inverter: the duties on this carrier ramp ------------
    pieces, dts = pwm_pieces(d, rising, i_abc, T_e, V_dc, T_s, mode, t_dead)

    # ---- the machine over the period -----------------------------------------
    x_end, X = window(x, ca.horzcat(*pieces))
    z_next = ca.vertcat(x_end, xc_next, d_new)
    return ca.Function("period", [z, q], [z_next, X, ca.vertcat(*dts)])


def simulate(period, currents, x0, T_s, n_periods, i_ref):
    """Run n_periods control periods in ONE CasADi call (mapaccum).

    i_ref : the controller's references: a tuple held constant (e.g. the
            PI's (i_d_ref, i_q_ref)), or an array (n_ref x n_periods) with one
            column per period
    Returns t [s], x (4 x N) states and y (3 x N) = [i_d, i_q, T_e] at the
    end of every switching piece (the switching instants), plus the duties
    applied in each control period (n_periods x 3).
    """
    n_z = period.size1_in(0)                         # 4 + n_state + 3
    n_ref = period.size1_in(1) - 1
    refs = np.asarray(i_ref, float)
    if refs.ndim == 1:                               # constant: repeat it
        refs = np.tile(refs[:, None], (1, n_periods))
    assert refs.shape == (n_ref, n_periods), "references: n_ref x n_periods"
    # per-period inputs: references and carrier direction (rising, falling, ...)
    Q = np.vstack([refs, np.arange(n_periods) % 2 == 0])
    z0 = np.r_[x0, np.zeros(n_z - 7), 0.5, 0.5, 0.5]   # controller memory 0,
    run = period.mapaccum("run", n_periods)              # zero voltage
    Z, X, DT = run(z0, Q)
    X, DT = np.array(X), np.array(DT)               # 4 x 4N states, 4 x N durations
    # duties APPLIED in each period: those computed the period before
    # (computation delay), 0.5 (zero voltage) in the first
    duties = np.vstack([np.full(3, 0.5), np.array(Z)[-3:, :-1].T])

    # time of every sample: start of its period + end of its piece
    t = (np.arange(n_periods) * T_s + np.cumsum(DT, axis=0)).T.ravel()
    keep = DT.T.ravel() > 1e-12 * T_s               # drop zero-length pieces
    X = X[:, keep]
    Y = np.array(currents.map(X.shape[1])(X))
    return dict(t=t[keep], x=X, y=Y, duties=duties)


def time_average(t, values):
    """Time-weighted mean of samples that are NOT evenly spaced."""
    w = np.gradient(t)
    return (values * w).sum(axis=-1) / w.sum()
