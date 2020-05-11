import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

nrefrac = lambda x: 1+x
def fn(x,n):
    inte, err = quad(nrefrac, 0, np.pi)
    lam = n*np.pi/inte
    int2, err = np.sin(lam*quad(nrefrac, 0, x))
    return 1/np.sqrt(nrefrac(x)) * np.sin(lam*quad(nrefrac, 0, x))

g = lambda x: (nrefrac(x))**2*fn(x, 1)*fn(x, 2)
print(g(1))
print(quad(g, 0, np.pi))
