import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import helper as h

qs=np.logspace(2, 3, num=5)
qs=np.floor(qs).astype(int)
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

    H00 = h.D0(xx, L)
    
    dt = dx/2
    s = dt/(2*dx)
    
    T=2
    m = int(round(T/dt))
    
    Ha = np.zeros((N,m+1))
    Ha[:,0]=h.initial(xx,L)

    Ea = np.zeros((N,m+1))
    Ea[:,0]=h.initial(xx,L)
    Ea[0,0]=0
    Ea[-1,0]=0

    for j in range(1,m+1):
        Ha[1:-1,j] = Ha[1:-1,j-1] + s*(Ea[2:,j-1] - Ea[:-2,j-1])
        Ha[0,j] = (4./3)*Ha[1,j] - (1./3)*Ha[2,j]
        Ha[-1,j] = (4./3)*Ha[-2,j] - (1./3)*Ha[-3,j]

        Ea[1:-1,j] = Ea[1:-1,j-1] + s*(Ha[2:,j] - Ha[:-2,j])
        Ea[0,j] = 0
        Ea[-1,j] = 0
        
        t=dt*j
        '''
        if(t==int(t)):
            plt.plot(xx, Ea[:, j])
            plt.show()
        '''

    t = dt*np.arange(0, m+1)
    H = H00 + np.zeros(N)
    E = np.zeros(N)
    XX, TT = np.meshgrid(xx, t)
    for k in range(1,150):
            H = H + h.Hn(XX, TT, k, L)
            E = E + h.En(XX, TT, k, L)
    #plt.plot(np.abs(Ha[:,m] - np.transpose(H)[:,m]))
    #plt.show()
    
    Herr = np.amax(np.abs(Ha - np.transpose(H)), axis=0)
    Eerr = np.amax(np.abs(Ea - np.transpose(E)), axis=0)
    print(q)
    Herrmax[q] = np.max(Herr)
    Eerrmax[q] = np.max(Eerr)

#np.savetxt('txtfiles/mol4compar.out', (hs,Herrmax,Eerrmax,hs**4))

plt.loglog(hs, Herrmax,'.-', label='H error')
plt.loglog(hs, Eerrmax,'.-',label='E error')
plt.loglog(hs, 1e4*hs**2, '--', label=r'$(\Delta x)^2$')
plt.legend()
plt.xlabel(r'$\Delta x$')
plt.title('FDTD convergence')
#plt.savefig('./testresults/moltest/MOL2/MOL2convergence.pdf')
plt.show()
