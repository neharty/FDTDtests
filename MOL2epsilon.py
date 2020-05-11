from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import quad
import helper as h
import diffcheb as d

L = 1
L=float(L)
numints = 1000
dx = float(L)/numints
print(int(L/dx), dx)
xx = np.linspace(0, L, num = numints+1)
N = len(xx)
print(N)

#xx, Dxcheb = d.diffcheb(numints, [0, 1])

Einit = h.initial(xx,L)
Hinit = h.initial(xx,L)

a1=0.5*np.ones(N-1)

Dx = np.diag(a1, k=1) + np.diag(-a1, k=-1)
np.set_printoptions(precision=3,threshold=10)

Dx[0,0] = -3/2
Dx[0,1] = 2
Dx[0,2] = -1/2

Dx[-1,-1] = 3/2
Dx[-1,-2] = -2
Dx[-1,-3] = 1/2

print(Dx)

Dx = (1/dx)*Dx

def n(z):
    return 1+0.5*z/L

def odesys(t, y):
    c2 = (1/n(xx))**2
    #if(t>0.5):
    #    plt.plot(y)
    #    plt.show()
    Etmp=np.zeros(N)

    Etmp[1:-1]=y[:N-2]

    Htmp=np.zeros(N)
    Htmp[:]=y[N-2:]

    dHdt = Dx @ Etmp
    dEdt = c2*Dx @ Htmp

    dydt = np.hstack([dEdt[1:-1], dHdt])
        
    return dydt

T = 6
Ts = np.linspace(0,T,200)

sol = solve_ivp(odesys, [0,T], np.block([Einit[1:-1], Hinit]), method='RK45', dense_output=True)

Herr=np.zeros(len(xx))
Eerr=np.zeros(len(xx))

fig, (ax11,ax12) = plt.subplots(nrows = 1, ncols=2,figsize=(16,9))

#y = sol.sol(T)
print(sol.t)
for i in range(len(Ts)):
    #print(T[i]-sol.t)
    idx = (np.abs(Ts[i]-sol.t)).argmin()
    #print(sol.y.shape)
    y=sol.y[:,idx]
    Etmp=np.zeros(N)
    Etmp[1:-1]=y[:N-2]
    ax11.plot(xx, Etmp)
    ax11.set_ylabel('E')

    Htmp = np.zeros(N)
    Htmp[:]=y[N-2:]
    ax12.plot(xx, Htmp)
    ax12.set_ylabel('H')

    #fig.suptitle('t='+str(i))
    plt.tight_layout()

plt.show()

