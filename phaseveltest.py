import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, np.pi/2, 100)
d = (np.pi/2)/len(theta)+1

def phasevel(ang, delta):
    return (1+delta*(np.cos(ang)/np.sin(ang)))

plt.plot(theta, phasevel(theta, d))
plt.show()
