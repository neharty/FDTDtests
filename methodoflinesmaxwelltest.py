import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def gaussian(x, mu, sigma):
        return np.exp(-((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))

L = 10
dx = 0.01
x = np.linspace(0, L, num=int(L/dx)+1)

Einit = gaussian(x, L/2, 1)
Einit[0] = 0
Einit[-1] = 0

Hinit = gaussian(x, L/2, 1)
Hinit[0] = 0
Hinit[-1] = 0

initial = np.concatenate((Hinit, Einit))

A = np.diag([1 for b in range(len(x)-1)], k=-1) + np.diag([-1 for c in range(len(x)-1)], k=1)
A = (1/(2*dx))*A

Hm = np.copy(A)
Em = np.copy(A)
Em[0,1] = 0
Em[-1,-2] = 0

print(Hm)
print(Em)

coeffs = np.block([
    [np.zeros((len(x), len(x))), Em],
    [Hm, np.zeros((len(x), len(x)))]])

def odesys(t, y):
    y = np.transpose(y)
    dydt = np.dot(coeffs, y)
    return dydt

T=10
sol = solve_ivp(odesys, [0, T], initial, max_step=0.1, dense_output=True, method='LSODA')
#print(np.shape(sol.y))
#print(len(x))

for j in range(T):
    index = min(range(len(sol.t)), key=lambda i: abs(sol.t[i]-j))
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
