# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 14:17:23 2025

@author: kelvi
"""

# Interactive time-domain plot for Van der Pol and Lotka-Volterra systems using sliders

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp

# Time setup
t_span = (0, 20)
t_eval = np.linspace(*t_span, 1000)

# --- VAN DER POL SYSTEM ---
def interactive_van_der_pol_time():
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    mu_init = 1.0
    Z0 = [1.0, 0.0]

    def solve_vdp(mu):
        return solve_ivp(lambda t, z: [z[1], mu * (1 - z[0]**2) * z[1] - z[0]],
                         t_span, Z0, t_eval=t_eval)

    sol = solve_vdp(mu_init)
    [line] = ax.plot(sol.t, sol.y[0], 'r')
    ax.set_title("Van der Pol Oscillator: Time Domain")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Displacement x(t)")
    ax.grid()

    ax_mu = plt.axes([0.25, 0.1, 0.65, 0.03])
    s_mu = Slider(ax_mu, 'μ', 0.0, 5.0, valinit=mu_init, valstep=0.1)

    def update(val):
        mu = s_mu.val
        sol = solve_vdp(mu)
        line.set_data(sol.t, sol.y[0])
        ax.set_ylim(np.min(sol.y[0]) - 0.5, np.max(sol.y[0]) + 0.5)
        fig.canvas.draw_idle()

    s_mu.on_changed(update)
    plt.show()

# --- LOTKA-VOLTERRA SYSTEM ---
def interactive_lotka_volterra_time():
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.3, bottom=0.35)

    alpha, beta, delta, gamma = 1.1, 0.4, 0.1, 0.4
    Z0 = [10, 5]

    def solve_lv(a, b, d, g):
        return solve_ivp(lambda t, z: [a * z[0] - b * z[0] * z[1], d * z[0] * z[1] - g * z[1]],
                         t_span, Z0, t_eval=t_eval)

    sol = solve_lv(alpha, beta, delta, gamma)
    line1, = ax.plot(sol.t, sol.y[0], 'b', label="Prey x(t)")
    line2, = ax.plot(sol.t, sol.y[1], 'g', label="Predator y(t)")

    ax.set_title("Lotka-Volterra: Time Domain")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Population")
    ax.legend()
    ax.grid()

    ax_alpha = plt.axes([0.3, 0.25, 0.6, 0.02])
    ax_beta  = plt.axes([0.3, 0.20, 0.6, 0.02])
    ax_delta = plt.axes([0.3, 0.15, 0.6, 0.02])
    ax_gamma = plt.axes([0.3, 0.10, 0.6, 0.02])

    s_alpha = Slider(ax_alpha, 'α', 0.1, 2.0, valinit=alpha, valstep=0.1)
    s_beta  = Slider(ax_beta,  'β', 0.1, 2.0, valinit=beta, valstep=0.1)
    s_delta = Slider(ax_delta, 'δ', 0.1, 2.0, valinit=delta, valstep=0.1)
    s_gamma = Slider(ax_gamma, 'γ', 0.1, 2.0, valinit=gamma, valstep=0.1)

    def update(val):
        a = s_alpha.val
        b = s_beta.val
        d = s_delta.val
        g = s_gamma.val
        sol = solve_lv(a, b, d, g)
        line1.set_data(sol.t, sol.y[0])
        line2.set_data(sol.t, sol.y[1])
        ax.set_ylim(0, max(np.max(sol.y[0]), np.max(sol.y[1])) + 5)
        fig.canvas.draw_idle()

    s_alpha.on_changed(update)
    s_beta.on_changed(update)
    s_delta.on_changed(update)
    s_gamma.on_changed(update)
    plt.show()

# Run either one:
interactive_van_der_pol_time()
#interactive_lotka_volterra_time()

