# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:39:44 2025

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Generate a random polynomial U(x)
np.random.seed(67)  # For reproducibility
coefficients = np.random.uniform(-2, 2, 4)  # Random coefficients for a cubic polynomial
U = np.poly1d(coefficients)  # Define the polynomial U(x)

# Compute the force as F(x) = -dU/dx
dU_dx = np.polyder(U)  # Derivative of U(x)
def force(x):
    return -dU_dx(x)

# Define the system of ODEs: dx/dt = v, dv/dt = F(x)/m
def oscillator(t, y, m):
    x, v = y
    dxdt = v
    dvdt = force(x) / m  # Newton's second law
    return [dxdt, dvdt]

# Parameters
m = 1.0  # Mass
x0 = 1.0  # Initial position
v0 = 0.0  # Initial velocity
t_span = (0, 30)  # Time span
t_eval = np.linspace(*t_span, 1000)  # Time points for solution

# Solve the system numerically using Runge-Kutta (RK45)
sol = solve_ivp(oscillator, t_span, [x0, v0], t_eval=t_eval, args=(m,))

# Plot the results
fig, ax = plt.subplots(2, 1, figsize=(8, 8))

# Plot the potential function
x_vals = np.linspace(-2, 2, 100)
U_vals = U(x_vals)
ax[0].plot(x_vals, U_vals, label=r'$U(x)$', color='b')
ax[0].set_xlabel("Position (x)")
ax[0].set_ylabel("Potential Energy U(x)")
ax[0].set_title("Random Potential Function U(x)")
ax[0].legend()
ax[0].grid()

# Plot the motion x(t)
ax[1].plot(sol.t, sol.y[0], label=r'$x(t)$', color='r')
ax[1].set_xlabel("Time (t)")
ax[1].set_ylabel("Position (x)")
ax[1].set_title("Motion of the Particle in U(x)")
ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.show()

# Display the random polynomial
U
