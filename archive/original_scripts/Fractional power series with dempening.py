# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 13:01:29 2025

@author: kelvi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import pandas as pd

def compute_fps_terms(n_terms: int, alpha: float, beta: float, gamma_coeff: float, delta: float, x0: float, x1: float = 0):
    """
    Compute the first n_terms of the fractional power series for the damped oscillator:
    D^alpha x + gamma D^beta x + delta x = 0

    Parameters:
        n_terms (int): number of terms to compute
        alpha (float): order of main derivative (Caputo)
        beta (float): order of damping derivative (Caputo)
        gamma_coeff (float): damping coefficient
        delta (float): stiffness coefficient
        x0 (float): initial position
        x1 (float): initial velocity (optional for Caputo D^1 x)

    Returns:
        List[float]: coefficients c_0, ..., c_{n-1}
    """
    # Choose sigma = smallest rational step (e.g. 0.5 if alpha and beta are 1 and 0.5)
    sigma = np.gcd(int(alpha * 100), int(beta * 100)) / 100
    m = int(alpha / sigma)
    n = int(beta / sigma)

    c = [0.0] * n_terms
    c[0] = x0
    if n_terms > 1:
        c[1] = 2 * x1 if sigma == 0.5 else x1  # estimate for fractional power

    for r in range(n_terms - m):
        idx_r = r
        idx_rn = r + n
        idx_rm = r + m
        if idx_rn >= n_terms or idx_rm >= n_terms:
            break

        gamma_term = (
            gamma_coeff * c[idx_rn] * gamma(sigma * idx_rn + 1) / gamma(sigma * idx_rn + beta)
            if c[idx_rn] != 0 else 0
        )
        delta_term = delta * c[idx_r]

        c[idx_rm] = - (gamma_term + delta_term) * gamma(sigma * idx_rm) / gamma(sigma * idx_rm + alpha)

    return c, sigma

# === Parameters ===
n_terms =   400
alpha = 1.2
beta = 0.4
gamma_coeff = 1.0
delta = 2.0
x0 = 1.0
x1 = 0.0

# === Compute Coefficients ===
coeffs, sigma = compute_fps_terms(n_terms, alpha, beta, gamma_coeff, delta, x0, x1)

# === Evaluate Series ===
t = np.linspace(0, 10, 400)
x_approx = sum(c * t**(i * sigma) for i, c in enumerate(coeffs))

# === Display Coefficients Table ===
df = pd.DataFrame({'c_k': coeffs})
print(df)

# === Plotting ===
plt.figure(figsize=(8, 5))
plt.plot(t, x_approx, label=f'FPS Approximation ({n_terms} terms)', color='green')
plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
plt.title(f"Damped Fractional Oscillator Approximation\n($\\alpha={alpha}$, $\\beta={beta}$, $\\gamma={gamma_coeff}$, $\\delta={delta}$)")
plt.xlabel("Time $t$")
plt.ylabel("$x(t)$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
