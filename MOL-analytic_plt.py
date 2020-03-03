import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import quad
import helper as h

L = 10
dx = 0.1
xx = np.linspace(0, L, num = (L/dx) + 1)
N = len(xx)

Einit = h.initial(xx)
#Einit = np.sin(np.pi*x/L)
Einit[0] = 0
Einit[-1] = 0

Hinit = h.initial(xx)
#Hinit = np.sin(np.pi*x/L)
Hinit[0] = Hinit[1]
Hinit[-1] = Hinit[-2]

initial = np.concatenate((Hinit, Einit))

A = np.diag([1 for b in range(N-1)], k=1) + np.diag([-1 for c in range(N-1)], k=-1)

Hm = np.copy(A)

Hm[0,:] = 0
Hm[-1,:] = 0

Hm[1, :] = 0
Hm[1,0] = -1
Hm[1,1] = 1

Hm[-2, :] = 0
Hm[-2,-1] = 1
Hm[-2, -2] = -1

Hm = 1/(2*dx)*Hm

Em = np.copy(A)
Em[0, :] = 0
Em[0,1] = 2

Em[1,:] = 0
Em[1, 2] = 1

Em[-2, :] = 0
Em[-2, -3] = -1

Em[-1, :] = 0
Em[-1, -2] = -2
Em = 1/(2*dx)*Em

print(Hm)
print(Em)

coeffs = np.block([
    [np.zeros((N, N)), Em],
    [Hm, np.zeros((N, N))]])

def odesys(t, y):
    y = np.transpose(y)
    dydt = np.dot(coeffs, y)
    #print(dydt[len(x)-2])
    #dydt[-1] = dydt[-2]
    return dydt

T=5

sol = solve_ivp(odesys, [0, T], initial, max_step=0.1, dense_output=True, method='LSODA')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

H00 = h.D0(xx, L)

for j in range(len(sol.t)):
    index = min(range(len(sol.t)), key=lambda i: abs(sol.t[i]-j))

    ax1.set_xlim([0, L])
    ax1.set_ylim([0, 2])
    ax1.plot(xx, sol.y[:N, index])
    ax1.set_ylabel('H')
    ax1.set_xlabel('x')

    ax2.plot(xx, sol.y[N:, index])
    ax2.set_xlim([0, L])
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('E')
    ax2.set_xlabel('x')
    
    H = H00 + np.array([0.0 for i in range(N)])
    E = np.array([0.0 for i in range(N)])
    
    t = sol.t[index]
    for k in range(1,100):
        H = H + h.Hn(xx, t, k, L)
        E = E + h.En(xx, t, k, L)

    ax3.plot(xx, H)
    ax3.set_xlim([0, L])
    ax3.set_ylim([0, 2])
    ax3.set_ylabel('H')
    ax3.set_xlabel('x')
    
    ax4.plot(xx, E)
    ax4.set_xlim([0, L])
    ax4.set_ylim([0, 1])
    ax4.set_ylabel('E')
    ax4.set_xlabel('x')


    plt.tight_layout()
    
plt.tight_layout()
plt.show()
plt.clf()
