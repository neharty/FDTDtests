import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import quad
import helper as h
import time

np.set_printoptions(precision=3,threshold=10)

qs = np.logspace(-1, -1, 1)
qnum = len(qs)
Herrmax = np.zeros(qnum)
Eerrmax = np.zeros(qnum)
times = np.zeros(qnum)

for q in range(qnum):
    t0 = time.time()
    L = 10
    dx = 0.01
    xx = np.linspace(0, L, num = (L/dx) + 1)
    N = len(xx)

    Einit = h.initial(xx)
    Einit[0] = 0
    Einit[-1] = 0

    Hinit = h.initial(xx)
    Hinit[0] = 0
    Hinit[-1] = 0

    H00 = h.D0(xx, L)

    initial = np.concatenate((Hinit, Einit))
    A = np.zeros((N,N))
    
    a1 = np.array([0.0 for n in range(N-1)])
    A = np.diag(a1+1, k=1) + np.diag(a1-1, k=-1)

    Hm = np.copy(A)

    Hm[0,:] = 0
    Hm[-1,:] = 0

    Hm[1, :] = 0
    Hm[1,1] = -4.0/3
    Hm[1,2] = 4.0/3

    Hm[-2, :] = 0
    Hm[-2,-2] = 4.0/3
    Hm[-2, -3] = -4.0/3

    Hm = 1/(2*dx)*Hm

    Em = np.copy(A)
    Em[0, :] = 0
    Em[0,1] = 4
    Em[0,2] = -1

    Em[1,0] = 0
    Em[-2,-1] = 0

    Em[-1, :] = 0
    Em[-1, -2] = -4
    Em[-1, -3] = 1
    Em = 1/(2*dx)*Em
    
    print(Em)
    print(Hm)
    #coeffs = np.block([
    #   [np.zeros((N, N)), Em],
    #   [Hm, np.zeros((N, N))]])

    def odesys(t, y):
        dydt = np.zeros(2*N)
        y = np.transpose(y)
        dydt[:N] = np.dot(Em, y[N:])
        dydt[N:] = np.dot(Hm, y[:N])
        return dydt

    T = 100

    sol = solve_ivp(odesys, [0, T], initial, max_step=dx, method='RK45')

    Herr = np.zeros(len(sol.t))
    Eerr=np.zeros(len(sol.t))
    H = H00 + np.array([0.0 for i in range(N)])
    E = np.array([0.0 for i in range(N)])
    t = sol.t
    XX, TT = np.meshgrid(xx, t)
    for k in range(1,100):
        H = H + h.Hn(XX, TT, k, L)
        E = E + h.En(XX, TT, k, L)
    axes = tuple(i for i in range(N))
    Herr = np.amax(np.abs(sol.y[:N] - np.transpose(H)), axis=0)
    Eerr = np.amax(np.abs(sol.y[N:] - np.transpose(E)), axis=0)

    print(q)
    Herrmax[q] = np.max(Herr)
    Eerrmax[q] = np.max(Eerr)
    #times[q] = time.time() - t
'''
fig, (ax1, ax2) = plt.subplots(1,2)
for i in range(T+1):
    index = np.abs(sol.t - i).argmin()
    #fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.plot(xx,sol.y[:N, index])
    ax1.set_ylabel('H')
    ax2.plot(xx,sol.y[N:, index])
    ax2.set_ylabel('E')
    ax2.set_title('t='+str(sol.t[index]))
    plt.tight_layout()
    
plt.clf()
'''
plt.semilogy(sol.t, Herr, '.', label='H error')
plt.semilogy(sol.t, Eerr, '.', label='E error')
#plt.loglog(qs, qs**2, '--')
#plt.legend()
plt.show()

