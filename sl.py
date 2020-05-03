import numpy as np
import matplotlib.pyplot as plt

numints = 100
x = np.linspace(0,np.pi, num=numints+1)
dx = np.pi/numints

A = np.diag(-2*np.ones(len(x)-2)) + np.diag(np.ones(len(x)-3), k=1) + np.diag(np.ones(len(x)-3), k=-1)

print(A)

evals, evects = np.linalg.eig(A)
#print(np.sqrt(-evals/dx))

vec = np.zeros(len(x))
vec[1:-1] = evects[:,0]

plt.plot(x, vec)
plt.show()
