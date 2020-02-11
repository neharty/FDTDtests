import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.optimize import root
import pandas as pd
from scipy.interpolate import interp1d

zinit = -900

#n = 201
#zlim = 20
#x, y, z = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), np.linspace(0, zlim, n))

#create permittivity matrix, measurements from doi:10.15784/601057
fin = pd.read_csv('../epsdata/evals_vs_depth.csv')

depth = -1*np.array(fin['Nominal Depth'])
e1, e2, e3 = np.array(fin['E1']), np.array(fin['E2']), np.array(fin['E3'])
def perm(z):
    #model from https://arxiv.org/pdf/1910.01471.pdf
    tenseps = np.zeros((3,3))
    de = 0.034
    ep = 3.157
    e1f = interp1d(depth, ep+e1*de)
    e2f = interp1d(depth, ep+e2*de)
    e3f = interp1d(depth, ep+e3*de)

    tenseps[0,0] = e1f(z)
    tenseps[1,1] = e2f(z)
    tenseps[2,2] = e3f(z)

    return tenseps

phi = np.pi/4
theta = np.pi/4
s = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])

def getvpbasis(s, z):
    c=1
    mu=1
    eps = perm(z)
    sx, sy, sz = s[0], s[1], s[2]
    ex, ey, ez = eps[0,0], eps[1,1], eps[2,2]
    vx, vy, vz = c/np.sqrt(mu*ex), c/np.sqrt(mu*ey), c/np.sqrt(mu*ez)

    M = -1*np.array([[(vx*sx)**2 - vx**2, sx*sy*vx**2, sx*sz*vx**2],
        [sx*sy*vy**2, (vy*sy)**2 - vy**2, sy*sz*vy**2],
        [sz*sx*vz**2, sz*sy*vz**2, (vz*sz)**2 - vz**2]])

    evals, evects = np.linalg.eig(M)
    evects = np.transpose(evects)

    #keep nonzero eigenthings
    evects = evects[evals>0]
    evals=evals[evals>0]
    return np.sqrt(evals), evects

def f(s, z, vpp, evect):
    c=1
    mu=1
    eps=perm(z)
    sx, sy, sz = s[0], s[1], s[2]
    ex, ey, ez = eps[0,0], eps[1,1], eps[2,2]
    vx, vy, vz = c/np.sqrt(mu*ex), c/np.sqrt(mu*ey), c/np.sqrt(mu*ez)
    
    vp, testvect = getvpbasis(s, z)
    print(vp)

    tst = np.abs(vpp-vp)
    if(tst[1] < tst[0]):
        vp = tst[1]
    else:
        vp = tst[0]

    M = np.array([[(vx*sx)**2 - vx**2 + vp**2, sx*sy*vx**2, sx*sz*vx**2],
        [sx*sy*vy**2, (vy*sy)**2 - vy**2 + vp**2, sy*sz*vy**2],
        [sz*sx*vz**2, sz*sy*vz**2, (vz*sz)**2 - vz**2 + vp**2]])
    F = np.dot(M, evect)
    return F

zz = np.linspace(-1500, -200, 1001)
vpp, evects = getvpbasis(s, zz[0])
r1, r2, r3 = np.zeros(len(zz)), np.zeros(len(zz)), np.zeros(len(zz))
r1[0], r2[0], r3[0] = root(f, [1,1,1]/np.sqrt(3), args=(zz[0], vpp, evects[0])).x
for i in range(1,len(zz)):
    vpp, evectsz = getvpbasis([r1[i-1], r2[i-1], r3[i-1]], zz[i-1])
    r1[i], r2[i], r3[i] = root(f, [r1[i-1], r2[i-1], r3[i-1]], args=(zz[i], vpp, evects[0])).x

fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)

ax1.plot(zz, r1, label='s1')
ax1.plot(zz, r2, label='s2')
ax1.plot(zz, r3, label='s3')
ax1.legend()

permvals = np.array([perm(zz[i]) for i in range(len(zz))])
ax2.plot(zz, permvals[:,0,0])
ax2.plot(zz, permvals[:,1,1])
ax2.plot(zz, permvals[:,2,2])

plt.show()

def E(s,t,w,vp,phi):
    if(np.norm(s) != 1):
        s = s/np.norm(s)
    p=np.dot([x,y,z], s)
    return np.exp(1j*w(p/vp-t+phi))
'''
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
'''

