import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def gaussian(x, mu, sigma):
        return np.exp(-((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))

L = 8
dx = 0.1
x = np.linspace(-L, L, num=int(L/dx)+1)

phi = gaussian(x, 0, 1)
#phi = np.zeros(len(x))
psi = np.zeros(len(x))
#a1 = np.where(x==-2)[0]
#a2 = np.where(x==2)[0]
#a1, a2 = a1[0], a2[0]
#psi = np.zeros(len(x))
#psi[a1:a2+1]=1
initial = np.concatenate((phi, psi))

unity = np.eye(len(x))
unity[0,0] = 0
unity[0,1] = 1
unity[-1,-1] = 0
unity[-1, -2] = 0

A = np.diag([-2 for a in range(len(x))]) + np.diag([1 for b in range(len(x)-1)], k=-1) + np.diag([1 for c in range(len(x)-1)], k=1)
A[0,0] = 0
A[0,1] = 0
A[-1,-1] = 0
A[-1, -2] = 0
A = (1/(dx**2))*A

print(A)
coeffs = np.block([
    [np.zeros((len(x), len(x))), unity],
    [A, np.zeros((len(x), len(x)))]])

def odesys(t, y):
    y = np.transpose(y)
    dydt = np.dot(coeffs, y)
    return dydt

sol = solve_ivp(odesys, [0, 8], initial, max_step=0.1, dense_output=True, method='LSODA')

for j in range(10):
    index = min(range(len(sol.t)), key=lambda i: abs(sol.t[i]-j))
    plt.plot(x, sol.y[:len(x), index])
plt.show()
plt.clf()
#plt.plot(sol.t, sol.y[10])
#plt.show()
