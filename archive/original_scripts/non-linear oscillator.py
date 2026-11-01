# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 11:57:42 2025

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the differential equation
def nonlinear_oscillator(t, y):
    return [y[1], -np.sin(y[0])]

# Initial conditions [x(0), dx/dt(0)]
y0 = [np.pi / 2, 0]

# Time span for integration
t_span = (0, 20)
t_eval = np.linspace(*t_span, 1000)

# Solve the ODE
solution = solve_ivp(nonlinear_oscillator, t_span, y0, t_eval=t_eval)

# Plotting the solution
plt.figure(figsize=(10, 6))
plt.plot(solution.t, solution.y[0], label='x(t)')
plt.title('Solution of Nonlinear Oscillator ODE: $d^2x/dt^2 + \sin(x) = 0$')
plt.xlabel('Time (t)')
plt.ylabel('x(t)')
plt.grid(True)
plt.legend()
plt.show()
# Phase portrait from numerical solution
x_numerical = solution.y[0]
dx_dt_numerical = solution.y[1]

# Create a grid for vector field visualization
X, Y = np.meshgrid(np.linspace(-2*np.pi, 2*np.pi, 20), np.linspace(-3, 3, 20))
U = Y
V = -np.sin(X)

# Plotting phase portrait with flow field
plt.figure(figsize=(10, 8))
plt.streamplot(X, Y, U, V, color='blue', density=1.2, linewidth=1, arrowsize=1.5)

# Overlay the numerical solution trajectory
plt.plot(x_numerical, dx_dt_numerical, color='red', linewidth=2, label='Numerical Trajectory')

plt.xlabel('x(t) postition')
plt.ylabel('v(t) velocity')
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.show()

# Define the classical harmonic oscillator
def harmonic_oscillator(t, y, omega=1.0):
    return [y[1], -omega**2 * y[0]]

# Initial conditions
y0_classic = [np.pi / 2, 0]

# Solve the classical harmonic oscillator equation
solution_classic = solve_ivp(harmonic_oscillator, t_span, y0_classic, t_eval=t_eval)

# Create a grid for vector field visualization (harmonic oscillator)
X_c, Y_c = np.meshgrid(np.linspace(-2*np.pi, 2*np.pi, 20), np.linspace(-3, 3, 20))
U_c = Y_c
V_c = -X_c  # for omega = 1

# Plotting phase portrait with flow field for the harmonic oscillator
plt.figure(figsize=(10, 8))
plt.streamplot(X_c, Y_c, U_c, V_c, color='green', density=1.2, linewidth=1, arrowsize=1.5)

# Overlay numerical solution trajectory
plt.plot(solution_classic.y[0], solution_classic.y[1], color='purple', linewidth=2, label='Numerical Trajectory')


plt.xlabel('x(t) position')
plt.ylabel('v(t) velocity')
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.show()

# Phase portrait from numerical solution
x_numerical = solution.y[0]
dx_dt_numerical = solution.y[1]

# Plotting the phase portrait
plt.figure(figsize=(8, 8))
plt.plot(x_numerical, dx_dt_numerical, label='Phase Portrait (Numerical Solution)')
plt.title('Phase Portrait of the Nonlinear Oscillator\n$d^2x/dt^2 + \sin(x) = 0$')
plt.xlabel('x(t)')
plt.ylabel('dx/dt')
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.show()

