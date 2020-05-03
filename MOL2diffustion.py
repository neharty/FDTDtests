from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import quad
import helper as h

L = 10
L=float(L)
numints = 100
dx = float(L)/numints
print(int(L/dx))
xx = np.linspace(0, L, num = numints+1)
N = len(xx)

uinit = h.initial(xx,L)

a1=np.zeros(N-1)
uxx = np.diag(a1+(1.0), k=1) + np.diag(a1+(1.0), k=-1)+np.diag(-2*np.ones(N), k=0)
uxx[0,:] = 0
uxx[-1,:] = 0

uxx = 1/(dx)**2 *uxx

def D(z):
    #return 1+z/(L)
    return 1

def odesys(t, y):
    dydt = np.zeros(N)
    y[0] = 0
    
    y[N-1] = (4./3)*y[N-2] - (1./3)*y[N-3]

    dydt = np.dot(uxx, y)

    return dydt

T = 10

sol = solve_ivp(odesys, [0, T], uinit, method='RK45', dense_output=True)

fig, ax = plt.subplots(nrows = 1, ncols=1,figsize=(16,9))

for i in range(0, T+1):
    if(i%(1) == 0):
        y = sol.sol(i)

        ax.plot(xx, y)
        ax.set_ylabel('u')

        fig.suptitle('t='+str(i))
        plt.tight_layout()

plt.show()

