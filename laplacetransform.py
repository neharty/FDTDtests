import numpy as np
from scipy.integrate import quadrature

xx = np.linspace(0, 2*np.pi, 10)

f = lambda s, x: 1/(s**2 +1)
g = np.zeros(len(xx))
for k in range(len(xx)):
    g[k], err = quadrature(lambda s: f(s, xx[k]), 2-1j*10000, 2+1j*10000)
print(g)
