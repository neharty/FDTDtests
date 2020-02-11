import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

fin = pd.read_csv('../epsdata/evals_vs_depth.csv')

depth = np.array(fin['Nominal Depth'])
e1, e2, e3 = np.array(fin['E1']), np.array(fin['E2']), np.array(fin['E3'])

def line(x, a, b):
    return a*x+b

def expn(x, a, b, c):
    return a+b*np.exp(-x/c)

e1f = interp1d(depth, e1)
e2f = interp1d(depth, e2)
e3f = interp1d(depth, e3)

de = 0.034
ep = 3.157
e1, e2, e3 = ep+e1*de, ep+e2*de, ep+e3*de

for i in [e1, e2, e3]:
    plt.plot(depth, i, '.')

x = np.linspace(depth[0], depth[-1], 10000)
#plt.plot(x, e1f(x))
#plt.plot(x, e2f(x))
#plt.plot(x, e3f(x))
plt.xlabel('depth')
plt.ylabel('permittivity')
plt.legend(['$\epsilon_1$', '$\epsilon_2$', '$\epsilon_3$'])
plt.savefig('epsilonplot.png')
