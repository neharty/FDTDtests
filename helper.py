import numpy as np
from scipy.integrate import quad
from scipy.special import erf
#TODO: check and generalize formulas

A = 1
alph = 400

def erfs(l,alpha,n):
    c1 = np.sqrt(alpha)*l/2 
    c2 = n*np.pi/(2*l*np.sqrt(alpha))
    erfs=erf(c1+1j*c2)+erf(c1-1j*c2)-2*erf(1j*c2)
    return A*(np.sqrt(np.pi/alpha)/l)*np.exp(-(c2**2))*erfs

def initial(x,l):
    return A*np.exp(-alph*(x-l/2)**2)

def a(n, l):
    sol = -np.cos(n*np.pi/2)*erfs(l,alph,n)
    return np.real(sol)

def b(n, l):
    sol=np.sin(n*np.pi/2)*erfs(l,alph,n)
    return np.real(sol)

def En(x,t,n,l):
    w = n*np.pi/l
    return np.sin(w*x)*(a(n, l)*np.sin(w*t) + b(n, l)*np.cos(w*t))

def Hn(x,t,n,l):
    w = n*np.pi/l
    return np.cos(w*x)*(b(n, l)*np.sin(w*t) - a(n, l)*np.cos(w*t))

def D0(x,l):
    sol = A*np.sqrt(np.pi/alph)*erf(np.sqrt(alph)*l/2)
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
