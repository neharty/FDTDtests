#actually 4th order accurate
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import quad
import helper as h
import time

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
A = np.diag(a1+(2.0/3), k=1) + np.diag(a1-(2.0/3), k=-1) + np.diag(a2+(1.0/12), k=-2) + np.diag(a2-(1.0/12), k=2)

np.set_printoptions(precision=3,threshold=10)
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

vals, vects = np.linalg.eig(Hm)
print(vals[np.abs(np.real(vals)) > 1])
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
#Herrmin = np.amin(np.abs(sol.y[:N] - np.transpose(H)), axis=0)
#Eerrmin = np.amin(np.abs(sol.y[N:] - np.transpose(E)), axis=0)

'''
Herr2=np.zeros(len(sol.t))
Eerr2=np.zeros(len(sol.t))
for j in range(len(sol.t)):
    H = H00 + np.array([0.0 for i in range(N)])
    E = np.array([0.0 for i in range(N)])
    for k in range(1,101):
        H = H + h.Hn(xx, sol.t[j], k, L)
        E = E + h.En(xx, sol.t[j], k, L)
    Herr2[j] = np.max(np.abs(sol.y[:N,j] - H))
    Eerr2[j] = np.max(np.abs(sol.y[N:,j] - E)) 
'''
'''
fig = plt.figure(figsize=[16,9])
ax = fig.add_subplot(221)
ax.matshow(H,extent=(0,10,0,T))
ax.set_xlabel('x')
ax.set_ylabel('t')
#ax.set_zlabel('H')

ax = fig.add_subplot(222)
ax.matshow(E,extent=(0,10,0,T))
ax.set_xlabel('x')
ax.set_ylabel('t')
#ax.set_zlabel('E')
'''
'''
fig = plt.figure(figsize=[16,9])

Hsol = np.transpose(sol.y[:N,:])
ax = fig.add_subplot(121)
mt = ax.matshow(Hsol, extent=(0,10,0,T))
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_title('H')
cbar = fig.colorbar(mt, ax=ax)
#ax.set_zlabel('H')

Esol = np.transpose(sol.y[N:, :])
ax = fig.add_subplot(122)
mt = ax.matshow(Esol, extent=(0,10,0,T))
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_title('E')
cbar = fig.colorbar(mt, ax=ax)
#ax.set_zlabel('E')

plt.tight_layout()
plt.savefig('testresults/moltest/plots/rk45/HzeroEgaussian.pdf')
#plt.show()
#plt.clf()
'''

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
    for k in range(1,500):
        H = H + h.Hn(xx, t, k, L)
        E = E + h.En(xx, t, k, L)
    #print(np.max(np.abs(sol.y[:N, index] - H)))
    #print(np.max(np.abs(sol.y[N:, index] - E)))
    #plt.show()
plt.show()
'''
plt.clf()
plt.semilogy(sol.t, Herr, '.', label='max H error')
plt.semilogy(sol.t, Eerr, '.', label='max E error')
#plt.semilogy(sol.t, Eerrmin, '.', label='min E error')
#plt.semilogy(sol.t, Herrmin, '.', label='min H error')
plt.grid()
plt.xlabel('t')
plt.ylabel('error')
plt.title('dx=' + str(dx))
#plt.loglog(qs, qs**2, '--')
plt.legend()
plt.tight_layout()
plt.show()
#plt.savefig('testresults/moltest/plots/MOL4errorvstime300.pdf')
'''
plt.clf()
plt.semilogy(sol.t, Herr2, '.', label='H error')
plt.semilogy(sol.t, Eerr2, '.', label='E error')
plt.grid()
#plt.loglog(qs, qs**2, '--')
plt.legend()
plt.show()
'''
