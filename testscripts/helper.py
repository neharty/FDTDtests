import numpy as np
from scipy.integrate import quad
from scipy.special import erf
#TODO: check and generalize formulas

def initial(x, s):
    return np.exp(-(x-s/2)**2) - np.exp(-s**2)
    #return np.cos(np.pi/10 *(x-5))

def a(n, l):
    f = lambda y: np.sin((n*np.pi/l)*y)*initial(y,l)
    sol, err = quad(f, 0, l, epsabs=1e-16, epsrel=1e-16, limit=1000)
    return 2*sol/l
    #sol = 2*((-1)**n - 1)/(n*np.exp(25)*np.pi) - 1/20.0*1j*np.exp(-(n*np.pi*(200*1j+np.pi*n))/400)*(-1+np.exp(n*np.pi*1j))*np.sqrt(np.pi)*(erf(5-(n*np.pi*1j/20))+erf(5+(n*np.pi*1j/20)))
    #return np.real(sol)

def b(n, l):
    g = lambda x: np.cos((n*np.pi/l)*x)*initial(x,l)
    sol, err = quad(g, 0, l, epsabs=1e-16, epsrel=1e-16, limit=1000)
    return 2*sol/l
    #sol =  1/20.0*np.exp(-(n*np.pi*(200*1j+np.pi*n))/400)*(1+np.exp(n*np.pi*1j))*np.sqrt(np.pi)*(erf(5-(n*np.pi*1j/20))+erf(5+(n*np.pi*1j/20)))
    #return np.real(sol)

def En(x,t,n,l):
    w = n*np.pi/l
    return np.sin(w*x)*(-b(n, l)*np.sin(w*t) + a(n, l)*np.cos(w*t))

def Hn(x,t,n,l):
    w = n*np.pi/l
    return np.cos(w*x)*(a(n, l)*np.sin(w*t) + b(n, l)*np.cos(w*t))

def D0(l):
    #sol = -10/np.exp(25)+np.sqrt(np.pi)*erf(5)
    sol, err = quad(lambda y: initial(y,l), 0, l, epsabs=1e-16, epsrel=1e-16, limit=1000)
    return sol/l
'''
def initial(x):
    return np.cos(np.pi/10 * (x-5))

def a(n,l):
    if(n==1):
        return 1
    else:
        return 0

def b(n,l):
    if(n==1):
        return 0
    else:
        return 2*((-1)**n +1)/(np.pi - n**2 * np.pi)
'''
