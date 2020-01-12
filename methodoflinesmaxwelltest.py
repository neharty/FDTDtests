import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def gaussian(x, mu, sigma):
        return np.exp(-((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))

L = 10
dx = 0.05
x = np.linspace(0, L, num=int(L/dx)+1)

Einit = gaussian(x, L/2, 1)
Einit[0] = 0
Einit[-1] = 0

Hinit = gaussian(x, L/2, 1)
Hinit[0] = Hinit[1]
Hinit[-1] = Hinit[-2]

initial = np.concatenate((Hinit, Einit))

A = np.diag([1 for b in range(len(x)-1)], k=-1) + np.diag([-1 for c in range(len(x))])
A = (1/(dx))*A

Hm = np.copy(A)

Hm[0,:] = 0
Hm[1,:] = 0
#Hm[-2,-2] = -10
#Hm[-2, -1] = 0
#Hm[-2,:] = 0
Hm[-1,:] = 0


Em = np.copy(A)
Em[0,:] = 0
Em[-1,:] = 0

print(Hm)
print(Em)

coeffs = np.block([
    [np.zeros((len(x), len(x))), Hm],
    [Em, np.zeros((len(x), len(x)))]])

def odesys(t, y):
    y = np.transpose(y)
    dydt = np.dot(coeffs, y)
    #print(dydt[len(x)-2])
    dydt[-1] = dydt[-2]
    return dydt

T=7
sol = solve_ivp(odesys, [0, T], initial, max_step=0.1, dense_output=True, method='LSODA')
#print(np.shape(sol.y))
#print(len(x))

for j in range(T):
    index = min(range(len(sol.t)), key=lambda i: abs(sol.t[i]-j))
    print(sol.t[index])
    plt.subplot(1,2,1)
    plt.plot(x, sol.y[:len(x), index])
    
    plt.subplot(1,2,2)
    plt.plot(x, sol.y[len(x):, index])

'''
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center'
'''
plt.tight_layout()
plt.show()
plt.clf()
