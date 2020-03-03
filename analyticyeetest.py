from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import quad

def initial(x):
    return np.exp(-(x-5)**2 / 2)

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
    g = lambda x: initial(x)
    sol, err = quad(g, 0, l)
    return sol/l

dx = 0.05
L = 10
xx = np.linspace(0, L, num = (L/dx) + 1)
N = len(xx)

T = 21

#print(a(3,L))

H00 = D0(xx, L)

for j in range(T):
    H = H00 + np.array([0.0 for i in range(N)])
    E = np.array([0.0 for i in range(N)])
    for k in range(1,100):
        H = H + Hn(xx, j, k, L)
        E = E + En(xx, j, k, L)
    plt.subplot(1,2,1)
    plt.plot(xx, E)
    plt.ylabel('E')
                
    plt.subplot(1,2,2)
    plt.plot(xx, H)
    plt.ylabel('H')
        
    plt.tight_layout()
'''
import commands
print commands.getoutput('convert -quality 100 ___t*.png giftest/yeetest.gif')
print commands.getoutput('rm ___t*.png') #remove temp files        
#plt.tight_layout()
#plt.savefig('testplots/yeetest.pdf')
#plt.show()
'''
plt.show()
