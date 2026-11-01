# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 13:50:53 2025

@author: kelvi
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Time and grids
t_span = (0, 20)
t_eval = np.linspace(*t_span, 1000)

# Van der Pol Interactive Plot
def interactive_van_der_pol():
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    
    x_vals = np.linspace(-3, 3, 20)
    y_vals = np.linspace(-5, 5, 20)
    X, Y = np.meshgrid(x_vals, y_vals)

    mu_init = 1.0
    Z0 = [1.0, 0.0]

    def system(mu):
        return Y, mu * (1 - X**2) * Y - X

    U, V = system(mu_init)
    quiv = ax.quiver(X, Y, U, V, color='blue', alpha=0.3)
    
    sol = solve_ivp(lambda t, z: [z[1], mu_init * (1 - z[0]**2) * z[1] - z[0]], t_span, Z0, t_eval=t_eval)
    traj, = ax.plot(sol.y[0], sol.y[1], 'r')
    
    ax.set_title("Van der Pol Oscillator")
    ax.set_xlabel("x")
    ax.set_ylabel("dx/dt")
    ax.grid()

    ax_mu = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider_mu = Slider(ax_mu, 'μ', 0.1, 2.0, valinit=mu_init, valstep=0.1)

    def update(val):
        mu = slider_mu.val
        U, V = system(mu)
        quiv.set_UVC(U, V)
        sol = solve_ivp(lambda t, z: [z[1], mu * (1 - z[0]**2) * z[1] - z[0]], t_span, Z0, t_eval=t_eval)
        traj.set_data(sol.y[0], sol.y[1])
        fig.canvas.draw_idle()

    slider_mu.on_changed(update)
    plt.show()

# Lotka-Volterra Interactive Plot
def interactive_lotka_volterra():
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.3, bottom=0.35)

    x_vals = np.linspace(0, 20, 20)
    y_vals = np.linspace(0, 20, 20)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Initial parameter values
    α, β, δ, γ = 1.0, 0.4, 0.1, 0.4
    Z0 = [10, 5]

    def system(alpha, beta, delta, gamma):
        U = alpha * X - beta * X * Y
        V = delta * X * Y - gamma * Y
        return U, V

    U, V = system(α, β, δ, γ)
    quiv = ax.quiver(X, Y, U, V, color='green', alpha=0.3)

    sol = solve_ivp(lambda t, z: [α * z[0] - β * z[0] * z[1],
                                  δ * z[0] * z[1] - γ * z[1]],
                    t_span, Z0, t_eval=t_eval)
    traj, = ax.plot(sol.y[0], sol.y[1], 'g')

    ax.set_title("Lotka-Volterra System")
    ax.set_xlabel("Prey (x)")
    ax.set_ylabel("Predator (y)")
    ax.grid()

    # Sliders
    ax_alpha = plt.axes([0.3, 0.25, 0.6, 0.02])
    ax_beta  = plt.axes([0.3, 0.20, 0.6, 0.02])
    ax_delta = plt.axes([0.3, 0.15, 0.6, 0.02])
    ax_gamma = plt.axes([0.3, 0.10, 0.6, 0.02])

    s_alpha = Slider(ax_alpha, 'α', 0.1, 2.0, valinit=α, valstep=0.1)
    s_beta  = Slider(ax_beta,  'β', 0.1, 2.0, valinit=β, valstep=0.1)
    s_delta = Slider(ax_delta, 'δ', 0.1, 2.0, valinit=δ, valstep=0.1)
    s_gamma = Slider(ax_gamma, 'γ', 0.1, 2.0, valinit=γ, valstep=0.1)

    def update(val):
        alpha = s_alpha.val
        beta = s_beta.val
        delta = s_delta.val
        gamma = s_gamma.val

        U, V = system(alpha, beta, delta, gamma)
        quiv.set_UVC(U, V)

        sol = solve_ivp(lambda t, z: [alpha * z[0] - beta * z[0] * z[1],
                                      delta * z[0] * z[1] - gamma * z[1]],
                        t_span, Z0, t_eval=t_eval)
        traj.set_data(sol.y[0], sol.y[1])
        fig.canvas.draw_idle()

    s_alpha.on_changed(update)
    s_beta.on_changed(update)
    s_delta.on_changed(update)
    s_gamma.on_changed(update)
    plt.ion()
    plt.show()
    


# Run interactives (run one at a time in scripts or notebooks)
# interactive_van_der_pol()
interactive_lotka_volterra()
