# -*- coding: utf-8 -*-
"""Post-processing shared by both solvers: means over unevenly spaced samples,
phase-current harmonics, switching ripple, and the incremental inductances a
controller is tuned on."""
import numpy as np
from pydae.emac.drive.drive_model import time_average
from pydae.emac.luts.flux_tables import ForwardMap


def trapezoid_mean(t, v):
    """Time mean of v(t) by the trapezoidal rule (samples need not be even)."""
    return float(np.trapezoid(v, t) / (t[-1] - t[0]))


def harmonics(t, i, f_e, n_periods, n_orders=40, n_fft=8192):
    """Harmonics of a phase current, % of the fundamental, over the last
    ``n_periods`` whole electrical periods of frequency ``f_e``.

    The samples need not be evenly spaced (the switched drive's sit at the
    switching instants); they are resampled to a uniform grid first.  With a
    whole number of periods, order k is FFT bin k * n_periods.  Returns
    ``{k: %}`` for k = 1..n_orders-1, plus ``"THD"`` [%] and ``"peak"`` (the
    fundamental's amplitude, in the current's units).
    """
    tu = np.linspace(t[-1] - n_periods / f_e, t[-1], n_fft, endpoint=False)
    X = np.abs(np.fft.rfft(np.interp(tu, t, i))) * 2 / len(tu)
    fund = X[n_periods]
    h = {k: float(100 * X[k * n_periods] / fund) for k in range(1, n_orders)}
    h["THD"] = float(100 * np.sqrt(sum(X[k * n_periods] ** 2 for k in range(2, n_orders))) / fund)
    h["peak"] = float(fund)
    return h


def switching_ripple(t, i, T_c, n_sub=200):
    """rms of i minus its own carrier-period (T_c) moving average."""
    dt = T_c / n_sub
    tu = np.arange(t[0], t[-1], dt)
    iu = np.interp(tu, t, i)
    avg = np.convolve(iu, np.ones(n_sub) / n_sub, mode="same")
    return float(np.sqrt(np.mean((iu - avg)[n_sub:-n_sub] ** 2)))


def incremental_inductances(lut, i, rel_step=1e-3):
    """dlam_d/di_d and dlam_q/di_q [H] of the theta-averaged FEA map at the
    currents i = (i_d, i_q) [A peak]: what a current controller is tuned on."""
    mean = {k: ForwardMap(lut["i_d"], lut["i_q"], lut[k].mean(axis=2)) for k in ("lam_d", "lam_q")}
    i = np.asarray(i, float)
    h = rel_step * max(np.linalg.norm(i), 1e-3 * np.abs(np.r_[lut["i_d"], lut["i_q"]]).max())
    return np.array([(mean["lam_d"](i + [h, 0])[0] - mean["lam_d"](i - [h, 0])[0]) / (2 * h),
                     (mean["lam_q"](i + [0, h])[0] - mean["lam_q"](i - [0, h])[0]) / (2 * h)])


__all__ = ["time_average", "trapezoid_mean", "harmonics", "switching_ripple",
           "incremental_inductances"]
