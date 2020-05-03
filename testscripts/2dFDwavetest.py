from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import time

start = time.time()
#define heat eqn parameters
c = 1
dx = 0.05
L = 10
x = np.linspace(0, L, num = int(L/dx) + 1)
N = len(x)

xx, yy = np.meshgrid(x, x, sparse=True)

#heat eqn w/boundary conditions
u = np.sin(3*np.pi*xx/L)*np.sin(3*np.pi*yy/L)
#print(u)
'''
u[0:] = 0
u[:-1] = 0
u[-1:] = 0
u[:0] = 0
'''
#time in hrs and time step
T = 10
dt = dx/(8*c)
s = (c*dt/dx)**2
Dt = np.zeros((N, N))
u1 = np.zeros((N, N))
u1[1:-1, 1:-1] = u[1:-1, 1:-1] + dt*Dt[1:-1,1:-1]

#time step
m = int(round(T/dt))

for j in range(0, m):
    tmp = np.copy(u1)
    u1[1:-1, 1:-1] = s*(u1[2:, 1:-1] + u1[:-2, 1:-1] + u1[1:-1,2:] + u1[1:-1,:-2]) + 2*(1-2*s)*u1[1:-1, 1:-1] - u[1:-1,1:-1]
    u = np.copy(tmp)

    #time value
    t = (j+1)*dt

    if(t == int(t)):
        fig = plt.figure(figsize=[16, 9])
        ax = fig.add_subplot(1, 2, 1)
        cf = ax.contourf(x, x, u1)
        fig.colorbar(cf, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        X, Y = np.meshgrid(x, x)
        temp = ax.plot_surface(X, Y, u1, cmap='viridis')
        fig.colorbar(temp, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        plt.title('t=' + str(t))
        plt.tight_layout(pad=2.8)
        #plt.show()
        plt.savefig('testplots/2dwave' + str(int(t)) + '.pdf')
        plt.clf()
print('process took ' + str(time.time() - start) + ' seconds')
