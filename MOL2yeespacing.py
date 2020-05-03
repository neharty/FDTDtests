from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import quad
import helper as h
import time
import sys

def odesys(t, y):
    dydt = np.zeros(2*N)
    y = np.transpose(y)
    dydt[:N] = np.dot(Em, y[N:])
    dydt[N:] = np.dot(Hm, y[:N])
    return dydt

ints = np.linspace(100,int(1e4), 10)
dxs = np.zeros(len(ints))

tout = np.zeros((3,len(ints)))

for k in range(len(ints)):
    L = 10
    numints=int(ints[k])
    dx = float(L)/numints
    print(int(L/dx))
    xx = np.linspace(0, L, num = numints+1)
    N = len(xx)

    dxs[k] = dx

    Einit = h.initial(xx)
    Einit[0] = 0
    Einit[-1] = 0

    Hinit = h.initial(xx)
    Hinit[0] = 0
    Hinit[-1] = 0

    H00 = h.D0(xx, L)

    a1 = np.zeros(N-1)

    initial = np.concatenate((Hinit, Einit))
    A = np.diag(a1+(1.0/2), k=1) + np.diag(a1-(1.0/2), k=-1)

    Hm = np.copy(A)

    Hm[0,:] = 0.0
    Hm[-1,:] = 0.0

    Hm[1, 0 ] = 0.
    Hm[1, 1] = -2.0/3
    Hm[1, 2] = 2.0/3

    Hm[-2, -1] = 0.
    Hm[-2, -2] = 2.0/3
    Hm[-2, -3] = -2.0/3
    vals, vects = np.linalg.eig(Hm)
    Hm = 1/(dx)*Hm

    Em = np.copy(A)

    Em[0, 1] = 2.
    Em[0, 2] = -1./2

    Em[-1, -2] = -2.
    Em[-1, -3] = 1./2

    Em[:, 0] = 0.
    Em[:, -1] = 0.

    vals, vects = np.linalg.eig(Em)
    Em = 1/(dx)*Em

    #times = np.linspace(1, 50, num = 10)
    #tout = np.zeros((4,len(ints)))
    #tout = np.zeros((3,len(ints)))

    T = 20

    t0RK45 = time.time()
    sol = solve_ivp(odesys, [0, T], initial, method='RK45')
    tout[0, k] = time.time() - t0RK45


    t0DOP853 = time.time()
    sol = solve_ivp(odesys, [0, T], initial, method='DOP853')
    tout[1, k] = time.time() - t0DOP853
    
    #t0LSODA = time.time()
    #sol = solve_ivp(odesys, [0, T], initial, method='LSODA')
    #tout[2, k] = time.time() - t0LSODA

    dt = dx/2
    s = dt/(2*dx)

    m = int(round(T/dt))
    
    Eyee = h.initial(xx)
    Eyee[0] = 0
    Eyee[-1] = 0

    Hyee = h.initial(xx)
    Hyee[0] = 0
    Hyee[-1] = 0

    t0yee = time.time()
    for j in range(m):
        Hyee[1:-1] = Hyee[1:-1] + s*(Eyee[2:] - Eyee[:-2])
        Hyee[0]=Hyee[1]
        Hyee[-1]=Hyee[-2]
        Eyee[1:-1] = Eyee[1:-1] + s*(Hyee[2:] - Hyee[:-2])
    tout[2, k] = time.time() - t0yee

#print(dxs)
#print(tout.T)

plt.plot(dxs, tout.T)
plt.legend(('RK45', 'DOP853', 'FDTD'))
plt.xlabel('dx')
plt.ylabel('computation time [s]')
plt.title('T='+str(T))
plt.savefig('testresults/timing/MOL2yeespacing.pdf')
plt.clf()

plt.semilogy(dxs, tout.T,'.')
plt.legend(('RK45', 'DOP853', 'FDTD'))
plt.xlabel('dx')
plt.ylabel('computation time [s]')
plt.title('T='+str(T))
plt.savefig('testresults/timing/MOL2yeespacingsemilog.pdf')
plt.clf()

