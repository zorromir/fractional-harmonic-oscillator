"""Mittag-Leffler function evaluated by a high-precision power series."""
from __future__ import annotations
import numpy as np
import mpmath as mp


def _ml_scalar(z: float, alpha: float, beta: float, dps: int, max_terms: int) -> float:
    with mp.workdps(dps):
        zz = mp.mpf(str(z))
        aa = mp.mpf(str(alpha))
        bb = mp.mpf(str(beta))
        total = mp.mpf('0')
        power = mp.mpf('1')
        tol = mp.power(10, -dps + 8)
        for k in range(max_terms):
            term = power * mp.rgamma(aa*k + bb)
            total += term
            power *= zz
            # A nonzero z may have initially small terms before its series peaks.
            if k >= 12 and abs(term) <= tol * max(1, abs(total)):
                break
        else:
            raise ArithmeticError('Mittag-Leffler series did not converge: increase max_terms or dps.')
        return float(total)


def mittag_leffler(z, alpha: float, beta: float = 1.0, *, dps: int = 55, max_terms: int = 500):
    """Evaluate the real-valued two-parameter Mittag-Leffler function."""
    if not (np.isfinite(alpha) and np.isfinite(beta) and alpha > 0 and beta > 0):
        raise ValueError('Requires finite alpha > 0 and beta > 0.')
    values = np.asarray(z, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError('z must be finite and real.')
    flat = [_ml_scalar(float(v), alpha, beta, dps, max_terms) for v in values.flat]
    out = np.asarray(flat, dtype=float).reshape(values.shape)
    return float(out) if out.ndim == 0 else out
