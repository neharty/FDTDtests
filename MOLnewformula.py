from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import quad
import helper as h
import time
import sys

qs = np.logspace(-2, -1, 5)
qnum = len(qs)
Herrmax = np.zeros(qnum)
Eerrmax = np.zeros(qnum)
times = np.zeros(qnum)

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
a2 = np.zeros(N-2)

initial = np.concatenate((Hinit, Einit))
#A = np.zeros((N,N))
#print(type(A[0,0]))
A = np.diag(a1+(2.0/3), k=1) + np.diag(a1-(2.0/3), k=-1) + np.diag(a2+(1.0/12), k=-2) + np.diag(a2-(1.0/12), k=2)

np.set_printoptions(precision=3,threshold=10)
#print(A)
Hm = np.copy(A)

Hm[0,:] = 0.0
Hm[-1,:] = 0.0

Hm[1, -1] = 1.0/12
Hm[-2,0] = -1.0/12
#print(Hm)
vals, vects = np.linalg.eig(Hm)
print(vals[np.abs(np.real(vals)) > 1])
Hm = 1/(dx)*Hm

Em = np.copy(A)


Em[0, -3] = -2./3
Em[0, -2] =1./12 
Em[-1, 1] = 2./3
Em[-1, 2] = -1./12

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

T = 500

sol = solve_ivp(odesys, [0, T], initial, max_step=dx,rtol=1e-5,atol=1e-8, method='RK45')
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

plt.clf()
plt.semilogy(sol.t, Herr, '.', label='max H error')
plt.semilogy(sol.t, Eerr, '.', label='max E error')
plt.grid()
plt.xlabel('t')
plt.ylabel('error')
plt.title('dx=' + str(dx))
plt.legend()
plt.tight_layout()
plt.show()
#plt.savefig('testresults/molNFstability1k.pdf')
