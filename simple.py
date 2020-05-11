import numpy as np
#from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import helper as h
import helperints as hi

L=1
x = np.linspace(0,1, 1001)

T = np.linspace(0,1.5,7)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,9))

Herr = np.zeros(len(T))
Eerr = np.zeros(len(T))

for i in T:
    H = np.zeros(len(x))
    E = np.zeros(len(x))

    H = H+h.D0(x,L)

    #Hi = np.zeros(len(x))
    #Ei = np.zeros(len(x))

    #Hi = H+hi.D0(10)

    for j in range(1, 150):
        E=E+h.En(x,i,j,L)
        H=H+h.Hn(x,i,j,L)

        #Ei=E+hi.En(x,i,j,10)
        #Hi=H+hi.Hn(x,i,j,10)
    
    #Herr[i] = np.max(np.abs(H-Hi))
    #Eerr[i] = np.max(np.abs(E-Ei))

    ax1.plot(x, E, label='t='+str(i))
    ax2.plot(x, H, label='t='+str(i))
    

ax1.set_xlabel('$x$')
ax2.set_xlabel('$x$')

ax1.set_ylabel('$E_z$')
ax2.set_ylabel('$H_y$')

plt.legend()
plt.tight_layout()
#plt.savefig('analyticEH.pdf')
plt.show()
'''
plt.clf()

plt.semilogy(T, Eerr, '.', label='E error')
plt.semilogy(T, Herr, '.', label='H error')
plt.xlabel('t')
plt.ylabel('max abs error')
plt.legend()
plt.show()
'''
