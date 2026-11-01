# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 14:07:33 2025

@author: kelvi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Lotka-Volterra system definition
def lotka_volterra(t, z, alpha, beta, delta, gamma):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# Parameters
alpha = 1.1
beta = 0.4
delta = 0.1
gamma = 0.4

# Time span
t_span = (0, 50)
t_eval = np.linspace(*t_span, 1000)

# Initial conditions
initial_conditions = [10, 5]  # Prey, Predator

# Solve the system
sol = solve_ivp(lotka_volterra, t_span, initial_conditions, args=(alpha, beta, delta, gamma), t_eval=t_eval)

# Phase plot
plt.figure(figsize=(8, 6))
plt.plot(sol.y[0], sol.y[1], color='purple', lw=1.5)
plt.xlabel('Prey Population (x)')
plt.ylabel('Predator Population (y)')
plt.grid(True)
plt.tight_layout()
plt.savefig('lotka_volterra_phase_plot.png', dpi=300, bbox_inches='tight')
plt.show()
