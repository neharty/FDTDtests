import numpy as np
import matplotlib.pyplot as plt

def initial(x, mu):
    return np.exp(-((x - mu))**2) - np.exp(-(mu)**2)

def c(z):
    return 1/(1+z/L)

L = 50
numints = 5000
dx = L/numints
x = np.linspace(0, L, num = numints + 1)
N = len(x)

cmax = max(c(x))
dt = dx/(2*cmax)
s = (c(x[1:-1])*dt/dx)**2

u = initial(x, L/2)

deriv = np.zeros(len(x))

u1 = np.zeros(len(x))
u1[1:-1] = u[1:-1] + dt*deriv[1:-1]
T = 1500
m = int(round(T/dt))
for j in range(m):
    tmp = np.copy(u1)
    u1[1:-1] = s*(u1[2:] + u1[:-2]) + 2*(1-s)*u1[1:-1] - u[1:-1]
    u = np.copy(tmp)
    
    t = (j+2)*dt
    
    if(t%(3*L)==0):
        print(np.max(np.abs(u1-initial(x,L/2))))
        plt.plot(x, u1)
        #plt.plot(x, 0.5*(gaussian(x,L/2 - t,1) + gaussian(x, L/2+t, 1)))
        plt.ylabel('E')
        plt.xlabel('x')
        plt.title('t=' +str(t) + ', dx='+str(dx))
        
plt.tight_layout()   
#plt.show()
plt.savefig('./testresults/epsgaussian_waveqn.pdf')
