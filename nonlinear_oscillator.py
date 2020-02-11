import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def fn(z):
    return 0.5*np.exp(-z) + 1

def ode(t, y):
    return[y[1], -fn(y[0])*y[0]]

mmin = -1
mmax = 1

y1 = np.linspace(5*mmin, 5*mmax, 20)
y2 = np.linspace(mmin, mmax, 20)

Y1, Y2 = np.meshgrid(y1, y2)

t = 0

u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)

NI, NJ = Y1.shape

for i in range(NI):
    for j in range(NJ):
        x = Y1[i, j]
        y = Y2[i, j]
        yprime = ode(t, [x, y])
        u[i,j] = yprime[0]
        v[i,j] = yprime[1]

plt.quiver(Y1, Y2, u, v, color='r')

plt.xlabel('$Z$')
plt.ylabel('$dZ/dz$')
plt.xlim([5*mmin, 5*mmax])
plt.ylim([mmin, mmax])
plt.show()
plt.clf()

'''
sol = solve_ivp(ode, [0,20], [0,0.1], max_step=0.1)

fig, ax = plt.subplots(1,2)
fig.set_size_inches(16,9)
ax[0].plot(sol.t, sol.y[0])
#plt.plot(sol.t, np.sin(sol.t))
ax[1].plot(sol.y[0], sol.y[1])
plt.show()
'''
