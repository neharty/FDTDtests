import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def gaussian(x):
        return np.exp(-((x-5)/2)**2)

L = 10
dx = 0.1
x = np.linspace(0, L, num=int(L/dx)+1)

Einit = gaussian(x)
#Einit = np.sin(np.pi*x/L)
Einit[0] = 0
Einit[-1] = 0

Hinit = gaussian(x)
#Hinit = np.sin(np.pi*x/L)
Hinit[0] = Hinit[1]
Hinit[-1] = Hinit[-2]

initial = np.concatenate((Hinit, Einit))

A = np.diag([1 for b in range(len(x)-1)], k=1) + np.diag([-1 for c in range(len(x)-1)], k=-1)

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
Em[0,1] = 4
Em[0,2] = -1

Em[1,:] = 0
Em[1, 2] = 1

Em[-2, :] = 0
Em[-2, -3] = -1

Em[-1, :] = 0
Em[-1, -2] = -4
Em[-1, -3] = 1
Em = 1/(2*dx)*Em

print(Hm)
print(Em)

coeffs = np.block([
    [np.zeros((len(x), len(x))), Em],
    [Hm, np.zeros((len(x), len(x)))]])

def odesys(t, y):
    y = np.transpose(y)
    dydt = np.dot(coeffs, y)
    #print(dydt[len(x)-2])
    #dydt[-1] = dydt[-2]
    return dydt

T=5
sol = solve_ivp(odesys, [0, T], initial, max_step=0.1, dense_output=True, method='LSODA')
#print(np.shape(sol.y))
#print(len(x))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

for j in range(len(sol.t)):
    index = min(range(len(sol.t)), key=lambda i: abs(sol.t[i]-j))
    #print(sol.t[index])
    plt.title('t = ' + str(sol.t[j]))
    
    #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    
    ax1.set_xlim([0, L])
    ax1.set_ylim([0, 2])
    ax1.plot(x, sol.y[:len(x), index])
    ax1.set_ylabel('H')
    ax1.set_xlabel('x')

    ax2.plot(x, sol.y[len(x):, index])
    ax2.set_xlim([0, L])
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('E')
    ax2.set_xlabel('x')
    
    

    plt.tight_layout()
    
plt.tight_layout()
plt.show()
plt.clf()
