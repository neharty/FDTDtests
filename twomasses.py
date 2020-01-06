import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu, sig):
    return np.exp(-((x - mu)/sig)**2 /2)

c1 = 1
c2 = 4
dx = 0.01
L = 10
x = np.linspace(0, L, num = (L/dx) + 1)
N = len(x)

dt = np.min([dx/(2*c1), dx/(2*c2)])
su = (c1*dt/dx)**2
sw = (c2*dt/dx)**2

u = gaussian(x, 5, 1)
uderiv = np.zeros(len(x))
u1 = np.zeros(len(x))
u1[1:-1] = u[1:-1] + dt*uderiv[1:-1]
u[-1] = 0

w = np.zeros(len(x))
wderiv = np.zeros(len(x))
w1 = np.zeros(len(x))
w1[1:-1] = w1[1:-1] + dt*wderiv[1:-1]

w[0] = u[-1]
w1[0] = u1[-1]

T = 10
m = int(round(T/dt))

for j in range(m):
    u1[-1] = w1[0]
    u1[-1] = u1[-1]+(w1[1] - w1[0])
    tmp = np.copy(u1)
    u1[1:-1] = su*(u1[2:] + u1[:-2]) + 2*(1-su)*u1[1:-1] - u[1:-1]
    u = np.copy(tmp)
    
    #u1[] = u1[0]
    w1[0] = u1[-1]
    w1[0] = w1[1]+(u1[-2] - u1[-1])
    tmp = np.copy(w1)
    w1[1:-1] = sw*(w1[2:] + w1[:-2]) + 2*(1-sw)*w1[1:-1] - w[1:-1]
    w = np.copy(tmp)

    t = (j+1)*dt

    if(t == int(t)):
        plt.plot(np.linspace(0, 2*L, num = (2*L/dx) + 1), np.concatenate((u1,w1[1:])), label='t='+str(t))
        plt.axvline(x = 10, color='r', ls='--')
        plt.ylabel('u')
        plt.xlabel('x')
        plt.legend()
        plt.tight_layout()
        #plt.autoscale(enable=False)
        plt.ylim([-1, 1])
        plt.xlim([0, 20])
        plt.show()
        plt.clf()
                       
