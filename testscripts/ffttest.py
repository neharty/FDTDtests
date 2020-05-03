import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

n = 1000.
xx = np.linspace(0, 50, int(n)+1)
x1 = np.exp(-(xx-25)**2)

y1 = fft(x1)

def ode(t, y):
    N = len(y)
    w = -2j*np.pi*fftfreq(N, d=50/n)
    return -w*y

T=10
sol = solve_ivp(ode, [0, T], y1, max_step=0.1, rtol = 1e-6, atol = 1e-12, method='RK45')

fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1)

for i in range(T+1):
    idx = (np.abs(sol.t - i)).argmin()
    x2 = ifft(sol.y[:,idx])
    ax1.plot(xx, np.real(x2), label=str(sol.t[idx]))
    #print(np.max(np.real(x2)))
    ax2.plot(xx, np.exp(-(xx+sol.t[idx] - 25)**2))
    #print(np.max(np.exp(-(xx+sol.t[idx] - 25)**2)))
    print(np.max(np.abs(np.real(x2) - np.exp(-(xx+sol.t[idx] - 25)**2))))

ax1.legend()
plt.show()
plt.clf()
