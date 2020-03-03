from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import quad
import helper as h

qs = np.logspace(-2,-1,7)
qnum = len(qs)
Herrmax = np.zeros(qnum)
Eerrmax = np.zeros(qnum)

for q in range(qnum):

    dx = qs[q]
    L = 10
    xx = np.linspace(0, L, num = (L/dx) + 1)
    N = len(xx)

    T = 10

    H00 = h.D0(xx, L)

    Eyee = h.initial(xx)
    Eyee[0] = 0
    Eyee[-1] = 0

    Hyee = h.initial(xx)
    Hyee[0] = 0
    Hyee[-1] = 0

    dt = dx/2

    s = dt/(2*dx)

    m = int(round(T/dt))
    Herr = np.zeros(m)
    Eerr=np.zeros(m)
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
        Herr[ctr] = np.max(np.abs(Hyee - H)) 
        Eerr[ctr] = np.max(np.abs(Eyee-E))
        ctr = ctr+1
    print(q)         
    Herrmax[q] = np.max(Herr)
    Eerrmax[q] = np.max(Eerr)

plt.loglog(qs, Herrmax, '.-')
plt.loglog(qs, Eerrmax, '.-')
plt.legend()
plt.show()
