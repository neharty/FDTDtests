import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

L = np.pi
dx = 0.1
x = np.linspace(0, L, num=int(L/dx)+1)

u = np.zeros(len(x))
u[0] = 10
u[-1] = 10

coeffs = np.diag([-2 for a in range(len(x))]) + np.diag([1 for b in range(len(x)-1)], k=-1) + np.diag([1 for c in range(len(x)-1)], k=1)
coeffs[0,0] = 0
coeffs[0,1] = 0
coeffs[-1,-1] = 0
coeffs[-1, -2] = 0
coeffs = np.matrix((1/(dx**2))*coeffs)

def odesys(t, y):
    y = np.transpose(y)
    dydt = np.dot(coeffs, y)
    #print(dydt)
    return dydt

sol = solve_ivp(odesys, [0, 10], u, max_step=0.1, dense_output=True, method='RK45')
#index = min(range(len(sol.t)), key=lambda i: abs(sol.t[i]-1))
for i in range(10):
    plt.plot(x, sol.sol(i/2))
plt.show()
plt.clf()
#plt.plot(sol.t, sol.y[10])
#plt.show()
