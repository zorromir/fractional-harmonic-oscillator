# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:25:31 2025

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
k = 1  # Spring constant
m = 1  # Mass
omega = np.sqrt(k / m)  # Angular frequency

# Define damping coefficient
b = 1.95 # Damping coefficient

# Define the system of ODEs for the damped harmonic oscillator
def damped_oscillator(x, v, m, b, k):
    dxdt = v
    dvdt = (-b * v - k * x) / m
    return dxdt, dvdt

# Generate a grid of initial conditions
x_vals = np.linspace(-2, 2, 20)
v_vals = np.linspace(-2, 2, 20)
X, V = np.meshgrid(x_vals, v_vals)

# Compute derivatives at each grid point
dX, dV = damped_oscillator(X, V, m, b, k)

# Plot the phase portrait
plt.figure(figsize=(8, 6))
plt.quiver(X, V, dX, dV, color='blue', angles='xy', scale_units='xy', scale=10)

# Plot some example trajectories
t = np.linspace(0, 20, 1000)

for x0, v0 in [(-1, 1), (1, -1), (-2, 0), (0, 2)]:
    gamma = b / (2 * m)
    omega_d = np.sqrt(abs(k / m - gamma**2))  # Damped frequency

    if k / m > gamma**2:  # Underdamped case
        x_traj = np.exp(-gamma * t) * (x0 * np.cos(omega_d * t) + (v0 + gamma * x0) / omega_d * np.sin(omega_d * t))
        v_traj = np.exp(-gamma * t) * (-x0 * omega_d * np.sin(omega_d * t) + (v0 + gamma * x0) * np.cos(omega_d * t))
    else:  # Overdamped case
        r1 = -gamma + np.sqrt(gamma**2 - k / m)
        r2 = -gamma - np.sqrt(gamma**2 - k / m)
        C1, C2 = x0, (v0 - r1 * x0) / (r2 - r1)
        x_traj = C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t)
        v_traj = C1 * r1 * np.exp(r1 * t) + C2 * r2 * np.exp(r2 * t)

    plt.plot(x_traj, v_traj, 'r')

# Labels and title
plt.xlabel('Position (x)')
plt.ylabel('Velocity (v)')
plt.title('Phase Portrait of the Damped Harmonic Oscillator')
plt.grid()
plt.show()
