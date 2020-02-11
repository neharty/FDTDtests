import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

def n(z):
    return 2-0.5*np.exp(z/100)

def dndz(z):
    return -0.5/100 * np.exp(z/100)

def odes(t, y):
    return [-dndz(y[1])/n(y[1]), 1/np.tan(y[0])]

sol=solve_ivp(odes, [0, 100], [np.pi+0.001, -1000], method='RK45', max_step=0.1)

plt.plot(sol.t, sol.y[1])
plt.xlabel('r')
plt.ylabel('z')
plt.show()
#iplt.clf()
#plt.plot(sol.y[0], sol.y[1])
#plt.show()
