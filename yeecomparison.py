from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import quad
import helper as h


L = 10.0
numints = 500
dx  =  L/numints
xx = np.linspace(0, L, num = numints + 1)
N = len(xx)

T = 100

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
    Herr[j] = np.max(np.abs(Hyee - H)) 
    Eerr[j] = np.max(np.abs(Eyee-E))
  
plt.semilogy(dt*np.arange(m), Herr, '.', label='H error')
plt.semilogy(dt*np.arange(m), Eerr, '.', label='E error')
plt.legend()
plt.show()
