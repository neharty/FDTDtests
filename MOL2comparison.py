from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import quad
import helper as h
import time
import sys

#for q in range(qnum):
t0 = time.time()
L = 1
numints = 1000
dx = float(L)/numints
print(int(L/dx))
xx = np.linspace(0, L, num = numints+1)
N = len(xx)

Einit = h.initial(xx,L)
Hinit = h.initial(xx,L)

H00 = h.D0(xx, L)

a1=0.5*np.ones(N-1)

Dx = np.diag(a1, k=1) + np.diag(-a1, k=-1)
np.set_printoptions(precision=3,threshold=10)

Dx[0,0] = -3/2
Dx[0,1] = 2
Dx[0,2] = -1/2

Dx[-1,-1] = 3/2
Dx[-1,-2] = -2
Dx[-1,-3] = 1/2

Dx = (1/dx)*Dx

def odesys(t, y):
    Etmp=np.zeros(N)

    Etmp[1:-1]=y[:N-2]

    Htmp=np.zeros(N)
    Htmp[:]=y[N-2:]

    dHdt = Dx @ Etmp
    dEdt = Dx @ Htmp

    dydt = np.hstack([dEdt[1:-1], dHdt])
    return dydt
    
T = 100

sol = solve_ivp(odesys, [0, T], np.block([Einit[1:-1], Hinit]), method='RK45')
Herr=np.zeros(len(xx))
Eerr=np.zeros(len(xx))

Herr = np.zeros(len(sol.t))
Eerr=np.zeros(len(sol.t))
ctr = 0
#Herr = np.zeros(len(sol.t))
#Eerr = np.zeros(len(sol.t))
H = H00 + np.zeros(N)
E = np.zeros(N)
t = sol.t
XX, TT = np.meshgrid(xx, t)
for k in range(1,150):
        H = H + h.Hn(XX, TT, k, L)
        E = E + h.En(XX, TT, k, L)
Herr = np.amax(np.abs(sol.y[N-2:] - np.transpose(H)), axis=0)
Eerr = np.amax(np.abs(sol.y[:N-2] - np.transpose(E[:,1:-1])), axis=0)


plt.semilogy(sol.t, Herr, '.', label='max H error')
plt.semilogy(sol.t, Eerr, '.', label='max E error')
plt.grid()
plt.xlabel('t')
plt.ylabel('error')
plt.title('MOL2, dx=' + str(dx))
plt.legend()
plt.tight_layout()
plt.show()

