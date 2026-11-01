# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 16:27:15 2025

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the Lotka-Volterra (predator-prey) equations
def lotka_volterra(t, z, alpha, beta, delta, gamma):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# Parameters for Lotka-Volterra model
alpha = 1.1  # Prey growth rate
beta = 0.4   # Predation rate coefficient
delta = 0.1  # Growth rate of predator per prey eaten
gamma = 0.4  # Predator death rate
z0 = [10, 5]  # Initial conditions [prey0, predators0]
t_span_lv = (0, 50)  # Time span for the solution
t_eval_lv = np.linspace(t_span_lv[0], t_span_lv[1], 1000)  # Time points at which to store the solution

# Solve the differential equations
solution_lv = solve_ivp(lotka_volterra, t_span_lv, z0, args=[alpha, beta, delta, gamma], t_eval=t_eval_lv, method='RK45')

# Plotting the populations over time
plt.figure(figsize=(12, 5))
plt.plot(solution_lv.t, solution_lv.y[0], label='Prey Population (x)')
plt.plot(solution_lv.t, solution_lv.y[1], label='Predator Population (y)')
plt.title('Lotka-Volterra Model: Population Dynamics')
plt.xlabel('Time')
plt.ylabel('Population')
plt.grid(True)
plt.legend()
plt.show()

# Plotting the phase portrait
plt.figure(figsize=(8, 6))
plt.plot(solution_lv.y[0], solution_lv.y[1], label='Phase Trajectory')
plt.title('Phase Portrait of Lotka-Volterra Model')
plt.xlabel('Prey Population (x)')
plt.ylabel('Predator Population (y)')
plt.grid(True)
plt.legend()
plt.show()
