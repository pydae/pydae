# -*- coding: utf-8 -*-
"""The processor's side of the modulation, in NumPy, for a controller written
as a plain Python loop around :class:`~pydae.emac.drive.pwm_plant.PwmPlant`.

The same operations as the CasADi drive (``drive_model``), so a Python
controller built from these reproduces ``build_control_period``.
"""
import numpy as np

SQ3 = np.sqrt(3.0)


def abc_to_dq(x_abc, theta_e):
    """Clarke + Park, amplitude-invariant: phase quantities -> rotor frame (d, q).

    The inverse of the plant's own convention, x_k = x_d cos(theta_e - k 2pi/3)
    - x_q sin(theta_e - k 2pi/3) for phases k = a, b, c.
    """
    ang = theta_e + np.array([0.0, -2 * np.pi / 3, 2 * np.pi / 3])
    return 2.0 / 3.0 * np.array([np.dot(x_abc, np.cos(ang)), -np.dot(x_abc, np.sin(ang))])


def dq_to_alpha_beta(v_dq, theta_e):
    """Inverse Park: rotor frame (d, q) -> fixed frame (alpha, beta)."""
    c, s = np.cos(theta_e), np.sin(theta_e)
    return np.array([v_dq[0] * c - v_dq[1] * s, v_dq[0] * s + v_dq[1] * c])


def limit_voltage(v_ab, v_max):
    """Scale v_ab into the circle |v| <= v_max.  Returns (v_ab, limited)."""
    size = np.hypot(v_ab[0], v_ab[1])
    return v_ab * min(1.0, v_max / max(size, 1e-12)), size > v_max


def svpwm_duties(v_ab, V_dc):
    """Space-vector modulation: the voltage wanted (alpha, beta) -> the three
    duties in [0, 1] the PWM modulator compares with its carrier.

    Min-max injection: the three phase voltages are shifted so the largest and
    smallest are symmetric about zero, which is equivalent to space-vector PWM
    and reaches |v| = V_dc / sqrt(3) before saturating.
    """
    v = np.array([v_ab[0],
                  -0.5 * v_ab[0] + SQ3 / 2 * v_ab[1],
                  -0.5 * v_ab[0] - SQ3 / 2 * v_ab[1]])
    shift = -0.5 * (v.max() + v.min())
    return np.clip(0.5 + (v + shift) / V_dc, 0.0, 1.0)


def sine_duties(v_ab, V_dc):
    """Sinusoidal PWM: the voltage wanted (alpha, beta) -> the three duties,
    each the phase voltage itself, d_k = 0.5 + v_k / V_dc, clipped to [0, 1].

    No common-mode shift, so the duties are sinusoids -- and the largest phase
    voltage it reaches before clipping is V_dc / 2, 13 % less than
    :func:`svpwm_duties`' V_dc / sqrt(3).
    """
    v = np.array([v_ab[0],
                  -0.5 * v_ab[0] + SQ3 / 2 * v_ab[1],
                  -0.5 * v_ab[0] - SQ3 / 2 * v_ab[1]])
    return np.clip(0.5 + v / V_dc, 0.0, 1.0)


def compensate_dead_time(duties, i_abc, rising, t_dead, T_s):
    """The processor's dead-time compensation: the duties it writes now, minus
    the extra high time the dead time will give each leg.

    rising  1 if the period these duties will run on is a rising carrier ramp
            (the period after the current one: the ramps alternate)
    i_abc   the phase currents sampled now: their signs say which diode holds
            each leg during the dead time

    The same correction as ``build_control_period(..., compensate=True)``
    (``drive_model.dead_duty``): on a rising ramp a leg whose current flows
    into it stays high t_dead longer, on a falling ramp one whose current
    flows out stays low t_dead longer.
    """
    dead = t_dead / T_s * (rising * (np.asarray(i_abc) < 0) - (1 - rising) * (np.asarray(i_abc) > 0))
    return np.clip(duties - dead, 0.0, 1.0)
