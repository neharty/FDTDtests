from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import quad
import helper as h
import time
import sys

qs = np.logspace(-2, -1, 5)
qnum = len(qs)
Herrmax = np.zeros(qnum)
Eerrmax = np.zeros(qnum)
times = np.zeros(qnum)

#for q in range(qnum):
t0 = time.time()
L = 10
numints =500
dx = float(L)/numints
#dx = 0.1
print(int(L/dx))
xx = np.linspace(0, L, num = numints+1)
N = len(xx)

Einit = h.initial(xx)
Einit[0] = 0
Einit[-1] = 0

Hinit = h.initial(xx)
#Hinit = np.zeros(N)
Hinit[0] = 0
Hinit[-1] = 0

H00 = h.D0(xx, L)

a1 = np.zeros(N-1)

initial = np.concatenate((Hinit, Einit))
A = np.diag(a1+(1.0/2), k=1) + np.diag(a1-(1.0/2), k=-1)

np.set_printoptions(precision=3,threshold=10)
#print(A)
Hm = np.copy(A)

Hm[0,:] = 0.0
Hm[-1,:] = 0.0

Hm[1, 0 ] = 0.
Hm[1, 1] = -2.0/3
Hm[1, 2] = 2.0/3

Hm[-2, -1] = 0.
Hm[-2, -2] = 2.0/3
Hm[-2, -3] = -2.0/3
print(Hm)
vals, vects = np.linalg.eig(Hm)
print(vals[np.abs(np.real(vals)) > 1])
Hm = 1/(dx)*Hm

Em = np.copy(A)

Em[0, 1] = 2.
Em[0, 2] = -1./2

Em[-1, -2] = -2.
Em[-1, -3] = 1./2

Em[:, 0] = 0.
Em[:, -1] = 0.

print(Em)
vals, vects = np.linalg.eig(Em)
print(vals[np.abs(np.real(vals)) > 1])
Em = 1/(dx)*Em

def odesys(t, y):
    dydt = np.zeros(2*N)
    y = np.transpose(y)
    dydt[:N] = np.dot(Em, y[N:])
    dydt[N:] = np.dot(Hm, y[:N])
    return dydt

times = np.logspace(1,3, num=10)
tout = np.zeros(len(times))

for k in range(len(times)):
    t0 = time.time()
    T = times[k]
    sol = solve_ivp(odesys, [0, T], initial, method='DOP853')
    tout[k] = time.time()-t0

'''
Herr=np.zeros(len(xx))
Eerr=np.zeros(len(xx))

for i in range(1, T+1):
    idx = (np.abs(sol.t - i)).argmin()
    H = H00 + np.zeros(N)
    E = np.zeros(N)
    t = sol.t[idx]
    for k in range(1,151):
        H = H + h.Hn(xx, t, k, L)
        E = E + h.En(xx, t, k, L)

    Herr = np.abs(sol.y[:N, idx] - H)
    Eerr = np.abs(sol.y[N:, idx] - E)
    
    fig, ((ax11,ax12), (ax21, ax22)) = plt.subplots(nrows = 2, ncols=2,figsize=(16,9))
    ax11.plot(xx, sol.y[N:, idx])
    ax11.set_ylabel('E')
    ax11.set_ylim([-1,1])

    ax12.plot(xx, sol.y[:N,idx])
    ax12.set_ylabel('H')
    ax12.set_ylim([-1, 1])

    ax22.semilogy(xx, Herr, '.')
    ax21.semilogy(xx, Eerr, '.')
    fig.suptitle('t='+str(t))
    plt.tight_layout()
    plt.grid()
    plt.show()
'''
''''
    plt.semilogy(sol.t, Herr, '.', label='max H error')
    plt.semilogy(sol.t, Eerr, '.', label='max E error')
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('error')
    plt.title('MOL2, dx=' + str(dx))
    plt.legend()
    plt.tight_layout()
    plt.show()
'''
plt.plot(times, tout)
plt.show()
