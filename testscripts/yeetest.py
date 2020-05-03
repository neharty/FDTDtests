from matplotlib import pyplot as plt
import numpy as np
import helper as h
#from scipy.integrate import solve_ivp

numints = 1000
L = 10.
dx = L/numints
x = np.linspace(0, L, num = numints + 1)
N = len(x)

Ea = h.initial(x)
Ea[0] = 0
Ea[-1] = 0

Ha = h.initial(x)
Ha[0] = 0
Ha[-1] = 0

T = 10
dt = dx/2

s = dt/(2*dx)

m = int(round(T/dt))

Herr = np.zeros(N)
Eerr = np.zeros(N)

H00 = h.D0(L)

def n(z):
    return 1+z/float(L)

for j in range(m):
    #Ea[1:-1] = Ea[1:-1] + s*(Ha[2:] - Ha[:-2])
    
    Ha[0] = (4./3)*Ha[1] - (1./3)*Ha[2]
    Ha[-1] = (4./3)*Ha[-2] - (1./3)*Ha[-3]

    Ha[1:-1] = Ha[1:-1] + s*(Ea[2:] - Ea[:-2])
    #Ha[0] = (4./3)*Ha[1] - (1./3)*Ha[2]
    #Ha[-1] = -(4./3)*Ha[-2] + (1./3)*Ha[-3]

    Ea[1:-1] = Ea[1:-1] + s*(Ha[2:] - Ha[:-2])
    
    t = (j+1)*dt
    
    if(t==int(t)):
        fig, ((ax11,ax12), (ax21, ax22)) = plt.subplots(nrows = 2, ncols=2,figsize=(16,9))

        ax11.plot(x, Ea)
        ax11.set_ylabel('E')
        ax11.set_ylim([-1,1])
                
        ax12.plot(x, Ha)
        ax12.set_ylabel('H')
        ax12.set_ylim([-1, 1])

        H = H00 + np.zeros(N)
        E = np.zeros(N)
        for k in range(1,151):
            H = H + h.Hn(x, t, k, L)
            E = E + h.En(x, t, k, L)
        Herr = np.abs(Ha - H)
        Eerr = np.abs(Ea - E)
    
        ax22.semilogy(x, Herr, '.')
        ax21.semilogy(x, Eerr, '.')
        fig.suptitle('t='+str(t))
        plt.tight_layout()
        plt.show()

