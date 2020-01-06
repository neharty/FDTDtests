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
x = np.linspace(0, L, num = int(L*/dx) + 1)
N = len(x)

xx, yy = np.meshgrid(x, x)
#heat eqn w/boundary conditions
u = np.sin(np.pi*xx/L)*np.sin(np.pi*yy/L)
u[0:] = 0
u[:-1] = 0
u[-1:] = 0
u[:0] = 0

#time in hrs and time step
T = 10
dt = dx/(2*c)
s = (c*dt/dx)**2
Dt = np.zeros((N, N))
u1 = np.zeros((N, N))
u1[1:-1, 1:-1] = u[1:-1, 1:-1] + dt*Dt[1:-1,1:-1]

#time step
m = int(round(T/dt))

for j in range(0, m):
    tmp = np.copy(u1)
    u1[1:-1, 1:-1] = s*(u1[2:, 1:-1] + u1[:-2, 1:-1]) + 2*(1-s)*u1[1:-1, 1:-1] + s*(u1[1:-1,2:] + u1[1:-1,:-2]) + 2*(1-s)*u1[1:-1,1:-1] - 2*u[1:-1]
    u = np.copy(tmp)

    #time value
    t = (j+1)*dt

    if(t == int(t)):
        fig = plt.figure(figsize=[16, 9])
        ax = fig.add_subplot(1, 2, 1)
        cf = ax.contourf(x, x, C, cmap='coolwarm')
        fig.colorbar(cf, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        X, Y = np.meshgrid(x, x)
        temp = ax.plot_surface(X, Y, C, cmap='coolwarm')
        ax.set_zlim3d(273, 293)
        fig.colorbar(temp, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('T')

        fig.suptitle('t = ' + str(t) + string)
        plt.tight_layout(pad=2.8)
        plt.savefig('plots/heateqn/heateqn_RTB_' + str(int(t)) + 'hr.pdf')
        plt.clf()
        
        del fig
        del ax
        fig = plt.figure(figsize=[16, 9])
        ax = fig.add_subplot(1, 2, 1)
        cf = ax.contourf(x, x, np.log10(A))
        fig.colorbar(cf, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        X, Y = np.meshgrid(x, x)
        bactnum = ax.plot_surface(X, Y, np.log10(A), cmap='viridis')
        ax.set_zlim3d(2, 8)
        fig.colorbar(bactnum, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('$\log{N}$')
        
        fig.suptitle('t = ' + str(t) + string)
        plt.tight_layout(pad=2.8)
        plt.savefig('plots/bactnum/bactnum_RTB_' + str(int(t)) + 'hr.pdf')
        plt.clf()

print('process took ' + str(time.time() - start) + ' seconds')
