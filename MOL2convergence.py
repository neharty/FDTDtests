import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import helper as h

qs= np.logspace(2, 3, num=21)
qs = np.floor(qs).astype(int)
qnum = len(qs)
hs = np.zeros(qnum)
Herrmax = np.zeros(qnum)
Eerrmax = np.zeros(qnum)

for q in range(qnum):

    L = 1
    dx = L/qs[q]
    hs[q] = dx
    xx = np.linspace(0, L, num = qs[q] + 1)
    N = len(xx)

    Einit = h.initial(xx,L)
    Hinit = h.initial(xx,L)

    H00 = h.D0(xx, L)

    initial = np.concatenate((Einit[1:-1], Hinit))

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


    T = 4
    sol = solve_ivp(odesys, [0, T], initial, method='RK45')

    Herr = np.zeros(len(sol.t))
    Eerr=np.zeros(len(sol.t))
    ctr = 0
    Herr = np.zeros(len(sol.t))
    Eerr=np.zeros(len(sol.t))
    H = H00 + np.array([0.0 for i in range(N)])
    E = np.array([0.0 for i in range(N)])
    t = sol.t
    XX, TT = np.meshgrid(xx, t)
    for k in range(1,150):
            H = H + h.Hn(XX, TT, k, L)
            E = E + h.En(XX, TT, k, L)
    Herr = np.amax(np.abs(sol.y[N-2:] - np.transpose(H)), axis=0)
    Eerr = np.amax(np.abs(sol.y[:N-2] - np.transpose(E[:,1:-1])), axis=0)
    print(q)
    Herrmax[q] = np.max(Herr)
    Eerrmax[q] = np.max(Eerr)

#np.savetxt('txtfiles/mol4compar.out', (hs,Herrmax,Eerrmax,hs**4))

plt.loglog(hs, Herrmax,'.-', label='H error')
plt.loglog(hs, Eerrmax,'.-',label='E error')
plt.loglog(hs, 1e4*hs**2, '--', label=r'$(\Delta x)^2$')
plt.legend()
plt.xlabel(r'$\Delta x$')
plt.title('MOL convergence')
#plt.savefig('./testresults/moltest/MOL2/MOL2convergence.pdf')
plt.show()
