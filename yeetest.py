from matplotlib import pyplot as plt
import numpy as np
#from scipy.integrate import solve_ivp

def gaussian(x):
    return np.exp(-(x - 5)**2 / 2)/np.sqrt(2*np.pi)

dx = 0.01
L = 10
x = np.linspace(0, L, num = (L/dx) + 1)
N = len(x)

E = gaussian(x)
E[0] = 0
E[-1] = 0

B = gaussian(x)
B[0] = 0
B[-1] = 0

T = 10
dt = dx/2

s = dt/dx

m = int(round(T/dt))

for j in range(m):
    E[1:-1] = E[1:-1] + s*(B[2:] - B[:-2])
    B[1:-1] = B[1:-1] + s*(E[2:] - E[:-2])
    B[0] = B[1]
    B[-1] = B[-2]
    t = j*dt
    
    if(t==int(t)):
        plt.subplot(1,2,1)
        plt.plot(x, E)
        plt.ylabel('E')
                
        plt.subplot(1,2,2)
        plt.plot(x, B)
        plt.ylabel('B')
        
        plt.tight_layout()
        plt.show()
