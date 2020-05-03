from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import quad
import helper as h

L = 50
L=float(L)
numints = 1000
dx = float(L)/numints
print(int(L/dx))
xx = np.linspace(0, L, num = numints+1)
N = len(xx)

Einit = h.initial(xx,L)
Einit[0] = 0
Einit[-1] = 0

Hinit = -h.initial(xx,L)
Hinit[0] = 0
Hinit[-1] = 0

a1=np.zeros(N-3)

initial = np.concatenate((Hinit, Einit))
A = np.diag(a1+(1.0/2), k=1) + np.diag(a1-(1.0/2), k=-1)

np.set_printoptions(precision=3,threshold=100)
Hm = np.copy(A)
'''
Hm[0,:] = 0.0
Hm[-1,:] = 0.0

Hm[1, 1] = -2.0/3
Hm[1, 2] = 2.0/3

Hm[-2, -2] = 2.0/3
Hm[-2, -3] = -2.0/3

Hm[:,0] = 0
Hm[:,-1] = 0
'''
Hm[0, 0] = -3/2
Hm[0, 1] = 2
Hm[0, 2] = -1/2

Hm[-1, -1] = 3/2
Hm[-1, -2] = -2
Hm[-1, -3] = 1/2


print(Hm)
Hm = 1/(dx)*Hm
Em = np.copy(A)
'''
Em[0, 1] = 2.
Em[0, 2] = -1./2

Em[-1, -2] = -2.
Em[-1, -3] = 1./2

Em[:, 0] = 0.
Em[:, -1] = 0.
'''
print(Em)
Em = 1/(dx)*Em

Ea = h.initial(xx, L)
Ea[0] = 0
Ea[-1] = 0

Ha = -h.initial(xx, L)
Ha[0] = 0
Ha[-1] = 0

T = 60

dt = dx/2

s = dt/(2*dx)

m = int(round(T/dt))

def n(z):
    return 1+z/(L)

def odesys(t, y):
    dydt = np.zeros(2*N)
    y[0] = (4./3)*y[1] - (1./3)*y[2]
    y[N-1] = (4./3)*y[N-2] - (1./3)*y[N-3]

    #H'(t) = E_x interior pts
    dydt[1:N-1] = np.dot(Em, y[N+1:-1])
    #Boundary conds

    y[0] = (4./3)*y[1] - (1./3)*y[2]
    y[N-1] = (4./3)*y[N-2] - (1./3)*y[N-3]

    #E'(t) = H_x
    y[N] = 0
    y[-1] = 0
    
    dydt[N+1:-1] = (1./n(xx[1:-1]))**2*np.dot(Hm, y[1:N-1])
    #y[N] = 0
    #y[-1] = 0
    #dydt[N+1:-1] = np.dot(Hm, y[1:N-1])
    #Dirichlet conds
    dydt[N] = 0
    dydt[-1] = 0

    return dydt

sol = solve_ivp(odesys, [0, T], initial, method='LSODA', dense_output=True)

Herr=np.zeros(T)
Eerr=np.zeros(T)

#fig, (ax11,ax12) = plt.subplots(nrows = 1, ncols=2,figsize=(16,9))

for j in range(m):
    Ha[1:-1] = Ha[1:-1] + s*(Ea[2:] - Ea[:-2])
    Ha[0] = (4./3)*Ha[1] - (1./3)*Ha[2]
    Ha[-1] = (4./3)*Ha[-2] - (1./3)*Ha[-3]
    
    Ea[1:-1] = Ea[1:-1] + (1/(n(xx[1:-1]))**2)*s*(Ha[2:] - Ha[:-2])
    
    t = (j+1)*dt

    if(t==int(t)):
        y = sol.sol(t)
         
        Herr[int(t)-1] = np.max(np.abs(Ha - y[:N]))
        Eerr[int(t)-1] = np.max(np.abs(Ea - y[N:]))
'''
ax11.semilogy(np.arange(1,T+1), Herr, '.')
ax11.set_ylabel('H error')

ax12.semilogy(np.arange(1,T+1), Eerr, '.')
ax12.set_ylabel('E error')
'''
plt.semilogy(np.arange(1,T+1), Herr, '.', label='H error')
plt.semilogy(np.arange(1,T+1), Eerr, '.', label='E error')
plt.xlabel('t')
plt.ylabel('max error')
plt.title('t='+str(T+1))
plt.tight_layout()

plt.show()

