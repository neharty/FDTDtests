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

a1 = np.array([0.0 for n in range(N-1)])
a2 = np.array([0.0 for n in range(N-2)])

initial = np.concatenate((Hinit, Einit))
#A = np.zeros((N,N))
#print(type(A[0,0]))
A = np.diag(a1+(2.0/3), k=1) + np.diag(a1-(2.0/3), k=-1) + np.diag(a2+(1.0/12), k=-2) + np.diag(a2-(1.0/12), k=2)

np.set_printoptions(precision=3,threshold=10)
#print(A)
Hm = np.copy(A)

Hm[0,:] = 0.0
Hm[-1,:] = 0.0

Hm[1, :] = 0.0
Hm[1,1] = -197.0/150
Hm[1,2] = 93.0/50
Hm[1,3] = -33.0/50
Hm[1,4] = 17.0/150

Hm[2,:] = 0.0
Hm[2,1] = -38.0/75
Hm[2,2] = -3.0/25
Hm[2,3] = 18.0/25
Hm[2,4] = -7.0/75

Hm[-2, :] = 0.0
Hm[-2,-2] = 197.0/150
Hm[-2,-3] = -93.0/50
Hm[-2,-4] = 33.0/50
Hm[-2,-5] = -17.0/150

Hm[2,0] = 0.0
Hm[-3,:] = 0.0
Hm[-3,-2] = 38.0/75
Hm[-3,-3] = 3.0/25
Hm[-3,-4] = -18.0/25
Hm[-3,-5] = 7.0/75

Hm = 1/(dx)*Hm

Em = np.copy(A)
Em[0, :] = 0.0
Em[0,1] = 4.0
Em[0,2] = -3.0
Em[0,3] = 4.0/3
Em[0,4] = -1.0/4

Em[1,:] = 0.0
Em[1,1] = -5.0/6
Em[1,2] = 3.0/2
Em[1,3] = -1.0/2
Em[1,4] = 1.0/12

Em[-2, :] = 0.0
Em[-2,-2] = 5.0/6
Em[-2,-3] = -3.0/2
Em[-2,-4] = 1.0/2
Em[-2,-5] = -1.0/12

Em[-1, :] = 0.0
Em[-1,-2] = -4.0
Em[-1,-3] = 3.0
Em[-1,-4] = -4.0/3
Em[-1,-5] = 1.0/4

Em[2,0] = 0.0
Em[-3,-1] = 0.0

Em = 1/(dx)*Em

print(Em)
print(Hm)

#coeffs = np.block([
#    [np.zeros((N, N)), Em],
#    [Hm, np.zeros((N, N))]])

def odesys(t, y):
    dydt = np.zeros(2*N)
    #y[N] = (12.0/25)*(4*y[N+1]-3*y[N+2]+(4.0/3)*y[N+3]-(1.0/4)*y[N+4])
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
#print(q)
#Herrmax[q] = np.max(Herr)
#Eerrmax[q] = np.max(Eerr)
#times[q] = time.time() - t0
#print(times)
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
    
    H = H00 + np.array([0.0 for i in range(N)])
    E = np.array([0.0 for i in range(N)])
    t = sol.t[index]
    for k in range(1,100):
        H = H + h.Hn(xx, t, k, L)
        E = E + h.En(xx, t, k, L)
    #print(np.max(np.abs(sol.y[:N, index] - H)))
    #print(np.max(np.abs(sol.y[N:, index] - E)))
    #plt.show()
plt.show()
'''
plt.semilogy(sol.t, Herr, '.', label='H error')
plt.semilogy(sol.t, Eerr, '.', label='E error')
plt.grid()
#plt.loglog(qs, qs**2, '--')
plt.legend()
plt.show()

