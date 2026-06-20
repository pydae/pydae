"""Single-frequency sine fitting for frequency-response (Bode) extraction."""
from __future__ import annotations

import numpy as np


def fit_sine(t, y, f):
    """Least-squares fit ``A·sin(ωt)+B·cos(ωt)+C``.

    Returns ``(amp, phase_rad, mean)`` with ``amp = hypot(A, B)`` and
    ``phase = atan2(B, A)``.
    """
    t = np.asarray(t, float); y = np.asarray(y, float)
    w = 2.0 * np.pi * f
    M = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)])
    (a, b, c), *_ = np.linalg.lstsq(M, y, rcond=None)
    return float(np.hypot(a, b)), float(np.arctan2(b, a)), float(c)


def gain_phase(t, ref, out, f):
    """Closed-loop gain (|out|/|ref|) and phase lag (∠out − ∠ref, deg) at ``f``."""
    ar, pr, _ = fit_sine(t, ref, f)
    ao, po, _ = fit_sine(t, out, f)
    gain  = ao / ar if ar > 1e-9 else float("nan")
    phase = np.degrees(np.angle(np.exp(1j * (po - pr))))  # wrapped to (-180,180]
    return gain, phase
