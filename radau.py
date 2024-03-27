# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 06:17:00 2024

@author: ahmet
"""

import math
import numpy as np
from scipy.integrate import solve_ivp

def radau(tinc=0.01, tend=4000, method="Radau"):
    
    def DGL(t, state, param):
        phi, phi_dot = state
        g, l = param

        rhs = [phi_dot, 
               -(3/2)*(g/l) * math.sin(phi)]
        return rhs
    
    m = 25
    g = 9.81
    l= 1.5

    phi_0 = math.pi/2
    phi_dot_0 = 0

    param = [g, l]
    initval = [phi_0, phi_dot_0]

    tstop = tend
    tlist = np.arange(0.0, tstop, tinc)

    sol = solve_ivp(DGL, (0.0, tstop), initval, method=method, t_eval=tlist, args=(param,))

    t = sol.t
    philist = sol.y[0]
    phidotlist = sol.y[1]
    
    n = len(t)
    E_kin_list = np.zeros(n)
    E_pot_list = np.zeros(n)
    E_summe_list = np.zeros(n)
    
    
    for i in range(0, n):
        E_kin_list[i] = (1/6)*m*(l**2)*(phidotlist[i]**2)
        E_pot_list[i] = -m*g*l*np.cos((philist[i]))/2
        E_summe_list[i] = E_kin_list[i] + E_pot_list[i]
    
    return philist, phidotlist, t, E_summe_list

