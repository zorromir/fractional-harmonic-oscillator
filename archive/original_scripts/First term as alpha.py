# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 14:51:42 2025

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def harmonic_oscillator(t, y, omega):
    """
    Defines the system of ODEs for the classical harmonic oscillator.
    
    Parameters:
      t     : Time variable (not used explicitly as the system is autonomous)
      y     : A list or array with y[0]=x and y[1]=x'
      omega : Angular frequency
      
    Returns:
      A list containing the derivatives [x', x'']
    """
    x, v = y  # unpack the state variables: x and its velocity v
    dxdt = v
    dvdt = -omega**2 * x
    return [dxdt, dvdt]

if __name__ == '__main__':
    # ---------------------
    # Set parameters:
    # ---------------------
    omega = 1.0    # angular frequency
    x0 = 1.0       # initial displacement x(0)
    v0 = 0.0       # initial velocity x'(0)
    T = 20.0       # final time

    # Initial condition vector: y(0) = [x0, v0]
    y0 = [x0, v0]

    # Define the time span and evaluation points:
    t_span = (0, T)
    t_eval = np.linspace(0, T, 1000)

    # ---------------------
    # Solve the ODE:
    # ---------------------
    sol = solve_ivp(fun=lambda t, y: harmonic_oscillator(t, y, omega),
                    t_span=t_span, y0=y0, t_eval=t_eval, method='RK45')

    # ---------------------
    # Plot the results:
    # ---------------------
    plt.figure(figsize=(8, 5))
    plt.plot(sol.t, sol.y[0], label='x(t) - Displacement')
    plt.plot(sol.t, sol.y[1], label="x'(t) - Velocity", linestyle='--')
    plt.xlabel('Time t')
    plt.ylabel('Solution')
    plt.title('Classical Harmonic Oscillator')
    plt.legend()
    plt.grid(True)
    plt.show()
