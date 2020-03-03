import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import quad
import helper as h

L = 10
dx = 0.01
xx = np.linspace(0, L, num = (L/dx) + 1)
N = len(xx)

Einit = h.initial(xx)
Einit[0] = 0
Einit[-1] = 0

Hinit = h.initial(xx)
Hinit[1] = Hinit[0]
Hinit[-2] = Hinit[-1]

H00 = h.D0(xx, L)

initial = np.concatenate((Hinit, Einit))

A = np.diag([1 for p in range(N-1)], k=1) + np.diag([-1 for c in range(N-1)], k=-1)

Hm = np.copy(A)

Hm[0,:] = 0
Hm[-1,:] = 0

Hm[1, :] = 0
Hm[1,0] = -1
Hm[1,1] = 1

Hm[-2, :] = 0
Hm[-2,-1] = 1
Hm[-2, -2] = -1

Hm = 1/(2*dx)*Hm

Em = np.copy(A)
Em[0, :] = 0
Em[0,1] = 4
Em[0,2] = -1

Em[1,:] = 0
Em[1, 2] = 1

Em[-2, :] = 0
Em[-2, -3] = -1

Em[-1, :] = 0
Em[-1, -2] = -4
Em[-1, -3] = 1
Em = 1/(2*dx)*Em

coeffs = np.block([
    [np.zeros((N, N)), Em],
    [Hm, np.zeros((N, N))]])

def odesys(t, y):
    dydt = np.zeros(2*N)
    y = np.transpose(y)
    dydt[:N] = np.dot(Em, y[N:])
    dydt[N:] = np.dot(Hm, y[:N])
    return dydt

T = 10

sol = solve_ivp(odesys, [0, T], initial, max_step=0.01, dense_output=True, method='RK45')

Herr = np.zeros(len(sol.t))
Eerr=np.zeros(len(sol.t))
ctr = 0

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

ctr = 0
for j in range(m):
    Hyee[1:-1] = Hyee[1:-1] + s*(Eyee[2:] - Eyee[:-2])
    Hyee[0]=Hyee[1]
    Hyee[-1] = Hyee[-2]

    Eyee[1:-1] = Eyee[1:-1] + s*(Hyee[2:] - Hyee[:-2])

    t = j*dt

    H = H00 + np.array([0.0 for i in range(N)])
    E = np.array([0.0 for i in range(N)])
    for k in range(1,101):
        H = H + h.Hn(xx, t, k, L)
        E = E + h.En(xx, t, k, L)
    Herryee[ctr] = np.max(np.abs(Hyee - H))
    Eerryee[ctr] = np.max(np.abs(Eyee - E))
    ctr = ctr+1

Herrmol = np.zeros(len(sol.t))
Eerrmol = np.zeros(len(sol.t))
ctr = 0
for j in range(len(sol.t)):
    H = H00 + np.array([0.0 for i in range(N)])
    E = np.array([0.0 for i in range(N)])
    t = sol.t[j]
    for k in range(1,101):
        H = H + h.Hn(xx, t, k, L)
        E = E + h.En(xx, t, k, L)
    
    Herrmol[ctr] = np.max(np.abs(sol.y[:N, j] - H))
    Eerrmol[ctr] = np.max(np.abs(sol.y[N:, j] - E))
    ctr = ctr+1

plt.semilogy(Herryee, '.', label='H error yee')
plt.semilogy(Eerryee, '.', label='E error yee')
plt.semilogy(Herrmol, '.', label='H error mol')
plt.semilogy(Eerrmol, '.', label='E error mol')
plt.legend()
plt.show()

