# -*- coding: utf-8 -*-
"""Controllers for the switched drive, beyond the default ``PIController``.

A controller is any object with ``n_state``, ``n_ref`` and
``step(meas, ref, xc, T_s) -> (v_ab, xc_next)``, written with CasADi
operations so it becomes part of the compiled period function (see "THE
CONTROLLER" in :mod:`pydae.emac.drive.drive_model`).
"""
import casadi as ca
from pydae.emac.drive.drive_model import PIController, dq_to_alpha_beta, limit_voltage


class PredictiveController:
    """Deadbeat-type current control with one-period delay compensation.

    The duties computed now are applied one period later, so it first
    predicts where the current will be when they start, using the voltage it
    already sent (kept in its memory), then asks for the voltage that would
    bring the current to the reference by the end of that period.  ``alpha``
    < 1 takes only part of the step (robustness to the model error).

    references  [i_d_ref, i_q_ref]
    memory      the alpha-beta voltage sent last period (2): it is the one
                being applied NOW, so the prediction needs it
    """
    n_state = 2
    n_ref = 2

    def __init__(self, R, L_d, L_q, alpha=0.5):
        self.R, self.L, self.alpha = R, ca.DM([L_d, L_q]), alpha

    def step(self, meas, ref, xc, T_s):
        i, lam, w, th = meas["i_dq"], meas["lam_dq"], meas["omega_e"], meas["theta_e"]
        back_emf = ca.vertcat(-w * lam[1], w * lam[0])          # omega x lambda
        # 1. the voltage being applied now, in dq (mid-period angle)
        c, s = ca.cos(th + 0.5 * T_s * w), ca.sin(th + 0.5 * T_s * w)
        v_now = ca.vertcat(c * xc[0] + s * xc[1], -s * xc[0] + c * xc[1])
        # 2. predicted current when the new duties start (one period ahead)
        i_next = i + T_s * (v_now - self.R * i - back_emf) / self.L
        # 3. voltage that closes alpha of the remaining error in one period
        v_dq = self.R * i_next + back_emf + self.alpha * self.L * (ref - i_next) / T_s
        v_ab = dq_to_alpha_beta(v_dq, th + 1.5 * T_s * w)       # applied at +1.5 T_s
        v_ab, _ = limit_voltage(v_ab, meas["v_max"])
        return v_ab, v_ab                                        # remember what was sent


__all__ = ["PIController", "PredictiveController"]
