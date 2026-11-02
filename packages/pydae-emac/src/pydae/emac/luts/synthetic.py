# -*- coding: utf-8 -*-
r"""Synthetic FEA tables with a known answer, for tests and examples.

:func:`linear_fea_lut` writes a linear machine (constant $L_d$, $L_q$,
$\Psi_{pm}$) in exactly the layout :func:`~pydae.emac.luts.flux_tables.load_fea_lut`
returns, optionally with spatial ripple.  Every theta-mean is then known in
closed form, so the whole chain -- inversion, Fourier series, B-spline, the
DAE machine and the switched drive -- can be checked against algebra rather
than against itself:

$$\bar\lambda_d = L_d i_d + \Psi_{pm}, \qquad \bar\lambda_q = L_q i_q,
\qquad \bar T = \tfrac{3}{2} p \left(\Psi_{pm} i_q + (L_d - L_q) i_d i_q\right)$$

The ripple terms (order 6, the first a 3-phase machine has) average to zero
over the table's 60 deg electrical period, so they leave those means exact.
"""
import numpy as np


def linear_fea_lut(L_d, L_q, Psi_pm, p, i_max, n_i=13, n_th=24,
                   torque_ripple=0.0, flux_ripple=0.0):
    """A linear PMSM's position-resolved table.

    L_d, L_q [H], Psi_pm [Wb] (on the d axis), p pole pairs, currents swept
    over [-i_max, i_max] A peak on both axes with n_i points, n_th angles
    over one 60 deg electrical period.  torque_ripple is the 6th-order
    torque ripple amplitude as a fraction of 1.5 p Psi_pm i_max; flux_ripple
    the 6th-order flux ripple as a fraction of Psi_pm.
    """
    i = np.linspace(-i_max, i_max, n_i)
    th = np.arange(n_th) * (np.pi / 3) / n_th
    ID, IQ, TH = np.meshgrid(i, i, th, indexing="ij")
    lam_d = L_d * ID + Psi_pm + flux_ripple * Psi_pm * np.cos(6 * TH)
    lam_q = L_q * IQ - flux_ripple * Psi_pm * np.sin(6 * TH)
    T = (1.5 * p * (lam_d * IQ - lam_q * ID)
         + torque_ripple * 1.5 * p * Psi_pm * i_max * np.cos(6 * TH))
    return dict(i_d=i, i_q=i, theta_e=th, lam_d=lam_d, lam_q=lam_q, T=T, p=int(p))


def linear_mean_torque(L_d, L_q, Psi_pm, p, i_d, i_q):
    """The theta-mean torque of :func:`linear_fea_lut` at (i_d, i_q) [N.m]."""
    return 1.5 * p * (Psi_pm * i_q + (L_d - L_q) * i_d * i_q)
