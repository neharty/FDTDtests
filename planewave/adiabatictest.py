import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

n = 201
zlim = 20
x, y, z = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), np.linspace(0, zlim, n))

tenseps = np.zeros((3,3))
tenseps[0,0] = 4
tenseps[1,1] = 1
tenseps[2,2] = 1

phi = np.pi/4
theta = np.pi/2
s = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])

def getvpbasis(s, eps, mu, c):
    sx, sy, sz = s[0], s[1], s[2]
    ex, ey, ez = eps[0,0], eps[1,1], eps[2,2]
    vx, vy, vz = c/np.sqrt(mu*ex), c/np.sqrt(mu*ey), c/np.sqrt(mu*ez)
    
    M = -1*np.array([[(vx*sx)**2 - vx**2, sx*sy*vx**2, sx*sz*vx**2],
        [sx*sy*vy**2, (vy*sy)**2 - vy**2, sy*sz*vy**2],
        [sz*sx*vz**2, sz*sy*vz**2, (vz*sz)**2 - vz**2]])
    print(M)

    evals, evects = np.linalg.eig(M)
    evects = np.transpose(evects)
    #keep nonzero eigenthings
    evects = evects[evals>0]
    evals=evals[evals>0]
    return np.sqrt(evals), evects

tst = getvpbasis(s, tenseps, 1, 1)
print(tst[0])
vect1, vect2 = tst[1]
print(vect1)
print(vect2)
#print(vect3)
tmp1 = np.matmul(tenseps,np.transpose(vect1))
tmp2 = np.matmul(tenseps,np.transpose(vect2))

print(np.matmul(tenseps,np.transpose(vect1)))
print(np.dot(s, tmp1))
print(np.dot(s, tmp2))
print(np.dot(tmp1, tmp2))

def E(s,t,w,vp,phi):
    if(np.norm(s) != 1):
        s = s/np.norm(s)
    p=np.dot([x,y,z], s)
    return np.exp(1j*w(p/vp-t+phi))

#mask for z-axis
mx = x==0
my = y==0
mask = mx & my

epsend = 15/2

zmask = z >= epsend
#eps = np.array([4 for i in range(len(z[mask][:np.where(z[mask] == 5)[0][0]]))])
#eps = np.concatenate((eps, np.array([1 for i in range(len(z[mask][:np.where(z[mask] == 5)[0][0]]))])))

eps = 1/4*~zmask
eps += zmask
phaseshift = np.pi*(0.5-1)*zmask

#u = np.real(E(z, 0, np.pi/epsend, 1, 0))*mask
#v = np.real(E(z, 0, np.pi/epsend, eps, phaseshift))*mask
#w = y * 0

fig = plt.figure()
ax = fig.gca(projection='3d')

origin = [0], [0], [0]

vects = np.array([s, tmp1, tmp2])
print(vects)
ax.quiver(origin[0], origin[1], origin[2], vects[:,0], vects[:,1], vects[:,2], normalize=True)
ax.set_ylim(-1, 1)
ax.set_xlim(-1, 1)
ax.set_zlim(-1, 1)
#ax.axis('equal')
plt.show()
