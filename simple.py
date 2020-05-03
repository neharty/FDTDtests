import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def ode(t,y):
    return [y[1], -y[0]]

sol = solve_ivp(ode, [0,int(2e4)], [0, 1], method='RK45')

err = np.array(np.abs(sol.y[0] - np.sin(sol.t)))

#plt.plot(sol.t, sol.y[0])
#plt.show()
#plt.clf()
plt.grid()
plt.semilogy(sol.t, err, '.')
plt.show()

