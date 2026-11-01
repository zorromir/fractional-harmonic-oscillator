"""Exact solution of the *simple* fractional oscillator (thesis Section 6.1).

The Caputo initial-value problem is D_C^alpha x + omega**2 x = 0,
where 1 < alpha <= 2, x(0)=x0 and x'(0)=v0.
"""
from __future__ import annotations
import numpy as np
from .special import mittag_leffler


def fractional_oscillator_exact(t, *, alpha: float, omega: float = 1.0,
                                x0: float = 1.0, v0: float = 0.0):
    r"""Return x0 E_{alpha,1}(-omega^2 t^alpha) + v0*t E_{alpha,2}(...).

    For alpha=2 the elementary cosine/sine solution is used. This helps
    avoid unnecessary cancellation and makes the classical limit explicit.
    t is nonnegative; use mpmath-backed Mittag-Leffler summation otherwise.
    """
    if not (1.0 < alpha <= 2.0):
        raise ValueError('Expected 1 < alpha <= 2 for a two-initial-condition model.')
    if not (np.isfinite(omega) and omega > 0 and np.isfinite(x0) and np.isfinite(v0)):
        raise ValueError('omega must be positive and all parameters finite.')
    values = np.asarray(t, dtype=float)
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError('t must be finite and nonnegative.')
    if alpha == 2:
        out = x0*np.cos(omega*values) + (v0/omega)*np.sin(omega*values)
    else:
        z = -omega*omega*np.power(values, alpha)
        out = x0*mittag_leffler(z, alpha, 1.0)
        if v0 != 0:
            out += v0*values*mittag_leffler(z, alpha, 2.0)
    return float(out) if np.ndim(out) == 0 else out
