from matplotlib import pyplot as plt
import numpy as np
import helper as h

numints = 5000
L = 50.
dx = L/numints
x = np.linspace(0, L, num = numints + 1)
N = le
for j in range(m+1):
    Ha[1:-1] = Ha[1:-1] + s*(Ea[2:] - Ea[1:-1])
    
    Ha[0] = (4./3)*Ha[1] - (1./3)*Ha[2]
    Ha[-1] = (4./3)*Ha[-2] - (1./3)*Ha[-3]
    
    Ea[1:-1] = Ea[1:-1] + s*(Ha[1:-1] - Ha[:-2])/(n(x[1:-1]))**2
     
    Ea[0] = 0
    Ea[-1] = 0

    t = (j+1)*dt
    
