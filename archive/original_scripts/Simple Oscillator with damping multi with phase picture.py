# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:17:12 2025

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Generate a random polynomial U(x)
np.random.seed(42)  # For reproducibility
coefficients = np.random.uniform(-2, 2, 4)  # Random coefficients for a cubic polynomial
U = np.poly1d(coefficients)  # Define the polynomial U(x)

# Compute the force as F(x) = -dU/dx
dU_dx = np.polyder(U)  # Derivative of U(x)
def force(x):
    return -dU_dx(x)

# Define the system of ODEs: dx/dt = v, dv/dt = F(x)/m

def nonlinear_oscillator(t, y, m, gamma):
    x, v = y
    dxdt = v
    dvdt = (force(x) - gamma * v) / m  # Newton's second law with damping
    return [dxdt, dvdt]

# Parameters
m = 1.0  # Mass
x0 = -1.5  # Initial position
v0 = 0.0  # Initial velocity
t_span = (0, 50)  # Time span
t_eval = np.linspace(*t_span, 1000)  # Time points for solution

# Different damping values for comparison
damping_values = [0.01, 1.0, 2.0]  # Underdamped to overdamped

# Generate x values for plotting the polynomial
x_vals = np.linspace(-2, 2, 100)
U_vals = U(x_vals)  # Compute U(x)

# Create a dual plot for multiple damping values
fig, ax = plt.subplots(2, 1, figsize=(8, 10))

# Plot the potential function U(x)
ax[0].plot(x_vals, U_vals, label=r'$U(x)$', color='b')
ax[0].set_xlabel("Position (x)")
ax[0].set_ylabel("Potential Energy U(x)")
ax[0].set_title("Random Potential Function U(x)")
ax[0].legend()
ax[0].grid()

# Solve and plot oscillator motion for different damping values
for gamma in damping_values:
    sol = solve_ivp(nonlinear_oscillator, t_span, [x0, v0], t_eval=t_eval, args=(m, gamma))
    ax[1].plot(sol.t, sol.y[0], label=rf"$\gamma = {gamma}$")

# Format the oscillator motion plot
ax[1].set_xlabel("Time (t)")
ax[1].set_ylabel("Displacement (x)")
ax[1].set_title("Oscillator Motion for Different Damping Coefficients")
ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.show()

# Create a phase space plot (velocity vs. position)
plt.figure(figsize=(8, 6))

# Solve and plot phase space for different damping values
for gamma in damping_values:
    sol = solve_ivp(nonlinear_oscillator, t_span, [x0, v0], t_eval=t_eval, args=(m, gamma))
    plt.plot(sol.y[0], sol.y[1], label=rf"$\gamma = {gamma}$")

# Formatting the phase space plot
plt.xlabel("Position (x)")
plt.ylabel("Velocity (v)")
plt.title("Phase Space of the Nonlinear Oscillator")
plt.legend()
plt.grid()

plt.show()
