# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 14:02:39 2025

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

# Combined x(t) and y(t) plot
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='Prey (x)', color='tab:blue')
plt.plot(sol.t, sol.y[1], label='Predator (y)', color='tab:orange')
plt.xlabel('Time t')
plt.ylabel('Population')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('lotka_volterra_xt_combined_plot.png', dpi=300, bbox_inches='tight')
plt.show()
