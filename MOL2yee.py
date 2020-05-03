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
    #y[N] = (12.0/25)*(4*y[N+1]-3*y[N+2]+(4.0/3)*y[N+3]-(1.0/4)*y[N+4])
    y = np.transpose(y)
    dydt[:N] = np.dot(Em, y[N:])
    dydt[N:] = np.dot(Hm, y[:N])
    return dydt

T = 1000

sol = solve_ivp(odesys, [0, T], initial, rtol = 1e-6, atol=1e-8,method='RK45')
#sol = solve_ivp(odesys, [0, T], initial, max_step=dx, method='RK45')
Herr=np.zeros(len(sol.t))
Eerr=np.zeros(len(sol.t))
Herrmin=np.zeros(len(sol.t))
Eerrmin=np.zeros(len(sol.t))

H = H00 + np.array([0.0 for i in range(N)])
E = np.array([0.0 for i in range(N)])
t = sol.t
XX, TT = np.meshgrid(xx, t)
for k in range(1,151):
    H = H + h.Hn(XX, TT, k, L)
    E = E + h.En(XX, TT, k, L)

Herr = np.amax(np.abs(sol.y[:N] - np.transpose(H)), axis=0)
Eerr = np.amax(np.abs(sol.y[N:] - np.transpose(E)), axis=0)   

Eyee = h.initial(xx)
Eyee[0] = 0
Eyee[-1] = 0

Hyee = h.initial(xx)
Hyee[0] = 0
Hyee[-1] = 0

dt = dx/2

s = dt/(2*dx)

m = int(round(T/dt))
Herryee = np.zeros(m)
Eerryee = np.zeros(m)

for j in range(m):
    Hyee[1:-1] = Hyee[1:-1] + s*(Eyee[2:] - Eyee[:-2])
    Hyee[0]=Hyee[1]
    Hyee[-1]=Hyee[-2]
    Eyee[1:-1] = Eyee[1:-1] + s*(Hyee[2:] - Hyee[:-2])

    t = (j+1)*dt

    H = H00 + np.array([0.0 for i in range(N)])
    E = np.array([0.0 for i in range(N)])
    for k in range(1,151):
        H = H + h.Hn(xx, t, k, L)
        E = E + h.En(xx, t, k, L)
    Herryee[j] = np.max(np.abs(Hyee - H))
    Eerryee[j] = np.max(np.abs(Eyee-E))

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(16,9))
ax1.semilogy(sol.t, Herr, '.', label='max H error')
ax1.semilogy(sol.t, Eerr, '.', label='max E error')
ax1.grid()
ax1.set_xlabel('t')
ax1.set_ylabel('error')
ax1.set_title('MOL2, dx=' + str(dx))
ax1.legend()

ax2.semilogy(dt*np.arange(m), Herryee, '.', label='max H error')
ax2.semilogy(dt*np.arange(m), Eerryee, '.', label='max E error')
ax2.grid()
ax2.set_xlabel('t')
ax2.set_ylabel('error')
ax2.set_title('FDTD2, dx=' + str(dx))
ax2.legend()
plt.tight_layout()
plt.savefig('MOL2vsyee.pdf')


