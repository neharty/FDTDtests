import numpy as np
import matplotlib.pyplot as plt

numints = 1000
x = np.linspace(0,np.pi, num=numints+1)
dx = np.pi/numints

eps=1+x/np.pi

A = np.diag(-2*eps[1:-1]*np.ones(len(x)-2)) + np.diag(eps[1:-2]*np.ones(len(x)-3), k=1) + np.diag(eps[2:-1]*np.ones(len(x)-3), k=-1)
print(A)
A=A/(dx)**2

evals, evects = np.linalg.eig(A)

idx = evals.argsort()[::-1]
evals = evals[idx]
evects = evects[:,idx]

vec = np.zeros(len(x))
vec[1:-1] = evects[:,0]
print(np.sqrt(-evals[0]))

fig, ((ax11,ax12),(ax21,ax22)) = plt.subplots(2,2)

ax11.plot(x, -vec/np.max(np.abs(vec)))
ax12.plot(x, np.hstack([0, evects[:,1],0])/np.max(np.abs(np.hstack([0, evects[:,1],0]))))
ax21.plot(x, -np.hstack([0, evects[:,2],0])/np.max(np.abs(np.hstack([0, evects[:,2],0]))))
ax22.plot(x, -np.hstack([0, evects[:,3],0])/np.max(np.abs(np.hstack([0, evects[:,3],0]))))

ax11.plot(x, np.sin(x),'--')
ax12.plot(x, np.sin(2*x),'--')
ax21.plot(x, np.sin(3*x),'--')
ax22.plot(x, np.sin(4*x),'--')

plt.savefig('sltest.pdf')
