# -*- coding: utf-8 -*-
r"""Skew a position-resolved FEA table: the multi-slice (2.5D) model, applied to
an unskewed (2D) map instead of re-solving the FEA.

A skew of total angle $\alpha_{sk}$ (mechanical) is modelled as $N$ slices at
the centres of equal axial layers.  Slice $k$ sees the stator turned by
$\alpha_k = \alpha_{sk}\,((k + \tfrac12)/N - \tfrac12)$ against the rotor
(pyleecan's ``SliceModel`` with a linear pattern does the same, turning the
rotor).  In that slice's own rotor frame:

* the stator current vector is rotated by $-p\alpha_k$:
  $\mathbf i_k = R(-p\alpha_k)\,\mathbf i$;
* the table's rotor angle is $\theta + s\,\alpha_k$, where the sign $s$
  depends on the table's angle convention against the slice rotation.

So, with $\lambda_{2D}$, $T_{2D}$ the unskewed map,

$$\boldsymbol\lambda(\mathbf i, \theta) = \frac1N \sum_k R(p\alpha_k)\,
  \boldsymbol\lambda_{2D}\big(R(-p\alpha_k)\,\mathbf i,\ \theta + s\,\alpha_k\big),
  \qquad T(\mathbf i, \theta) = \frac1N \sum_k T_{2D}\big(R(-p\alpha_k)\,\mathbf i,\ \theta + s\,\alpha_k\big)$$

**The sign was settled by FEA, not derived.**  On srmpma3600w20pp (pyleecan
map, 7 slices over one slot pitch, rated current) $s = -1$ reproduces the
2.5D FEA torque waveform of ``skew_rated.json`` to 0.18 N.m rms with no
shift (mean 213.19 vs 213.14 N.m, ripple 2.83 vs 2.91 %); $s = +1$ gives the
same mean but a different waveform (1.7 N.m rms even at its best shift).
The likely reason: pyleecan's rotor turns in the negative direction, so a
positive slice rotation of the rotor is an EARLIER table angle.  The overall
sign of the skew does not matter (the slices come in +/- pairs); the
relative sign $s$ does.

The angle shift is exact Fourier interpolation (the table holds one period of
the dq quantities, uniformly sampled); the current rotation is the table's
cubic interpolation, continued linearly outside it (rotating a node near the
table's edge can leave it).  The result is a table of the same layout and
grid, ready for :func:`~pydae.emac.luts.flux_tables.build_flux_tables`.

This is what the 2.5D FEA computes, minus what a slice model cannot hold
either (axial flux, end effects), and minus the FEA's re-solve at the
rotated currents: interpolation stands in for it.
"""
import numpy as np
from pydae.emac.luts.flux_tables import ForwardMap


def _shift_angle(F, shift):
    """F(theta + shift) on the same uniform grid, along the last axis; shift in
    units of the grid's own period fraction: radians of the table's angle."""
    n = F.shape[-1]
    k = np.fft.fftfreq(n, 1.0 / n)                  # harmonic number per bin
    return np.real(np.fft.ifft(np.fft.fft(F, axis=-1) * np.exp(1j * k * shift), axis=-1))


def skew_fea_lut(lut, skew_mech, n_slices=7, angle_sign=-1):
    """The skewed version of an FEA table (layout of ``load_fea_lut``).

    skew_mech   total skew angle [rad mechanical], e.g. one slot pitch 2 pi / Z
    n_slices    number of slices (the project's 2.5D uses 7)
    angle_sign  s above: -1 for pyleecan maps (validated against 2.5D FEA)
    """
    p = lut["p"]
    th = lut["theta_e"]
    period = len(th) * (th[1] - th[0])              # [rad elec] one period of the dq quantities
    base = 2 * np.pi / period                       # bin m = harmonic m * base
    ID, IQ = np.meshgrid(lut["i_d"], lut["i_q"], indexing="ij")
    out = {k: np.zeros_like(lut[k]) for k in ("lam_d", "lam_q", "T")}
    for k in range(n_slices):
        a = p * skew_mech * ((k + 0.5) / n_slices - 0.5)       # [rad elec] slice k
        shifted = {name: _shift_angle(lut[name], angle_sign * base * a) for name in out}
        c, s = np.cos(a), np.sin(a)
        pts = np.c_[(c * ID + s * IQ).ravel(), (-s * ID + c * IQ).ravel()]   # R(-a) i
        for j in range(len(th)):
            ld, lq, T = (ForwardMap(lut["i_d"], lut["i_q"], shifted[name][:, :, j])(pts).reshape(ID.shape)
                         for name in ("lam_d", "lam_q", "T"))
            out["lam_d"][:, :, j] += (c * ld - s * lq) / n_slices        # R(a) lambda
            out["lam_q"][:, :, j] += (s * ld + c * lq) / n_slices
            out["T"][:, :, j] += T / n_slices
    return dict(lut, **out, skew_mech=skew_mech, n_slices=n_slices)
