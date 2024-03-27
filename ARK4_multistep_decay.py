# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 05:58:54 2024

@author: ahmet
"""

import matplotlib.pyplot as plt
import numpy as np
import timeit
import numpy as np
import timeit



def ARK4_multistep_decay(tinc=0.01, tend=4000, alpha=0.05, num_updates=3):
    start = timeit.default_timer()
    
    # Constants
    m = 25
    g = 9.81  # m/s^2
    l = 1.5   # length of pendulum in meters

    def f(t, y):
        """
        Defines the system of first-order ODEs for the physical pendulum.
        y[0] = phi, y[1] = omega
        """
        return np.array([y[1], -1.5 * (g / l) * np.sin(y[0])])

    # Initial conditions
    phi0 = np.pi / 2  # initial angle (in radians)
    omega0 = 0.0      # initial angular velocity

    # Time parameters
    t0 = 0.0          # initial time
    tf = tend         # final time
    h = tinc          # step size

    # Number of steps
    num_steps = int((tf - t0) / h)

    # Arrays to store results
    t_values = np.linspace(t0, tf, num_steps)
    phi_values = np.zeros_like(t_values)
    omega_values = np.zeros_like(t_values)
    w_values = []  # Store w values during the simulation

    # Set initial conditions
    phi_values[0] = phi0
    omega_values[0] = omega0

    E_kin_list = np.zeros(num_steps)
    E_pot_list = np.zeros(num_steps)
    E_summe_list = np.zeros(num_steps)

    E_kin_list[0] = (1/6) * m * (l**2) * (omega0**2)
    E_pot_list[0] = -m * g * l * np.cos(phi0) / 2
    E_summe_list[0] = E_kin_list[0] + E_pot_list[0]
    
    w = 1
    alpha_decay_rate = 0.9 # Decay rate

    # Solve using RK4 method
    for i in range(num_steps - 1):
        t = t_values[i]
        y = np.array([phi_values[i], omega_values[i]])

        # Perform multiple updates to w within each time step
        for _ in range(num_updates):
            k1 = h * f(t, y)
            k2 = h * f(t + h/2, y + k1/2)
            k3 = h * f(t + h/2, y + k2/2)
            k4 = h * f(t + h, y + k3)

            # Compute next value of y
            phi_next = phi_values[i] + (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6
            phi_dot_next = omega_values[i] + (k1[1] + 2*k2[1] + 2*k3[1] + w*k4[1]) / 6  # you should only use w in the second ODE

            E_kin_next = (1/6) * m * (l**2) * (phi_dot_next**2)
            E_pot_next = -m * g * l * np.cos(phi_next) / 2
            E_summe_next = E_kin_next + E_pot_next

            delta_w = (E_summe_next - E_summe_list[i])
            w -= alpha * delta_w

            # Decay alpha
            alpha *= alpha_decay_rate

        # Update phi, phi_dot, and energy lists after multiple w updates
        phi_values[i+1] = phi_next
        omega_values[i+1] = phi_dot_next
        E_kin_list[i+1] = E_kin_next
        E_pot_list[i+1] = E_pot_next
        E_summe_list[i+1] = E_summe_next

        
    stop = timeit.default_timer()
    sim_time = stop - start

    return phi_values, omega_values, t_values, E_summe_list, sim_time

    
