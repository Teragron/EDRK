import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator




def rk4_default_solve_ivp(tinc=0.01, tend=4000, method="RK45"):
    
    def DGL(t, state, param):
        phi, phi_dot = state
        g, l = param

        rhs = [phi_dot, 
               -(3/2)*(g/l) * math.sin(phi)]
        return rhs
    
    m = 25
    g = 9.8
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
    
    return philist

