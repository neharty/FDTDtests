import numpy as np
from scipy.integrate import quad

def initial(x):
    return np.exp(-(x-5)**2) - np.exp(-25)

def a(n, l):
    f = lambda y: np.sin((n*np.pi/l)*y)*initial(y)
    sol, err = quad(f, 0, l)
    return 2*sol/l

def b(n, l):
    g = lambda x: np.cos((n*np.pi/l)*x)*initial(x)
    sol, err = quad(g, 0, l)
    return 2*sol/l

def En(x,t,n,l):
    w = n*np.pi/l
    return np.sin(w*x)*(-b(n, l)*np.sin(w*t) + a(n, l)*np.cos(w*t))

def Hn(x,t,n,l):
    w = n*np.pi/l
    return np.cos(w*x)*(a(n, l)*np.sin(w*t) + b(n, l)*np.cos(w*t))

def D0(x,l):
    sol, err = quad(initial, 0, l)
    return sol/l

