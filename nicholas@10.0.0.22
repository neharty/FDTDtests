from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import quad
import helper as h

L = 10
L=float(L)
numints = 1000
dx = float(L)/numints
print(int(L/dx))
xx = np.linspace(0, L, num = numints+1)
N = len(xx)

Einit = h.initial(xx,L)
Einit[0] = 0
Einit[-1] = 0

Hinit = h.initial(xx,L)
Hinit[0] = 0
Hinit[-1] = 0

a1=np.zeros(N-1)

initial = np.concatenate((Hinit[1:-1], Einit[1:-1]))
A = np.diag(a1+(1.0/2), k=1) + np.diag(a1-(1.0/2), k=-1)

np.set_printoptions(precision=3,threshold=100)
Hm = np.copy(A)

Hm[-1, :] = 0
Hm[0, :] = 0
print(Hm)
Hm = 1/(dx)*Hm
Em = np.copy(A)

Em[-1, :] = 0
Em[0, :] = 0
print(Em)
Em = 1/(dx)*Em

def n(z):
    return 1+0.1*z/L

def odesys(t, y):
    c = 1/n(xx)

    dydt = np.zeros(2*(N-2))
    Etmp=np.zeros(N)
    Etmp[1:-1]=y[N-2:]

    Htmp = np.zeros(N)
    Htmp[1:-1]=y[:N-2]
    Htmp[0] = ((4./3)*Htmp[1] - (1./3)*Htmp[2])
    Htmp[-1] = ((4./3)*Htmp[-2] - (1./3)*Htmp[-3])
    
    #Htmp[0] = Htmp[1]
    #Htmp[-1] = Htmp[-2]

    dydt[:N-2] = np.dot(Em, Etmp)[1:-1]
    tmp = (c*np.dot(Hm, Htmp))
    #print(tmp[0], tmp[-1])
    dydt[N-2:] = (np.dot(Hm, Htmp))[1:-1]
    #print(dydt[0], dydt[N-1],dydt[N-2], dydt[-1])

    return dydt

T = 500

sol = solve_ivp(odesys, [0, T], initial, method='RK45', dense_output=True)
Herr=np.zeros(len(xx))
Eerr=np.zeros(len(xx))

fig, (ax11,ax12) = plt.subplots(nrows = 1, ncols=2,figsize=(16,9))

for i in range(0, T+1):
    #if(i%(3*L) == 0):
        y = sol.sol(i)
        
        Etmp=np.zeros(N)
        Etmp[1:-1]=y[N-2:]
        ax11.plot(xx, Etmp)
        ax11.set_ylabel('E')

        Htmp = np.zeros(N)
        Htmp[1:-1]=y[:N-2]
        Htmp[0] = (4./3)*Htmp[1] - (1./3)*Htmp[2]
        Htmp[-1] = (4./3)*Htmp[-2] - (1./3)*Htmp[-3]
        ax12.plot(xx, Htmp)
        ax12.set_ylabel('H')

        fig.suptitle('t='+str(i))
        plt.tight_layout()

plt.show()

