import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import diffcheb as d

m = 120
L=1
x,Dx = d.diffcheb(m,[0,1]);
trunc = lambda u: u[1:m]
extend = lambda v: np.hstack([0,v,0])

def dwdt(t,w):
    u=np.zeros(m+1)
    u[1:-1] = w[:m-1]
    z = w[m-1:]
    dudt = Dx@z
    dzdt = Dx@u
    return np.hstack([ trunc(dudt), dzdt ])

u_init = np.exp(-400*(x-0.5)**2)
z_init = -u_init
w_init = np.hstack([ trunc(u_init), z_init ])

T=np.arange(0, 4)

sol = solve_ivp(dwdt,T,w_init,dense_output=True,method="Radau")

N=len(x)
print(N)

#fig, (ax11,ax12) = plt.subplots(nrows = 1, ncols=2,figsize=(16,9))

for i in T:
    if(i%(2*L) == 0):
        '''
        y = sol.sol(i)

        Etmp=np.zeros(N)
        Etmp[1:-1]=y[:N-2]
        ax11.plot(x, Etmp)
        ax11.set_ylabel('E')

        Htmp = np.zeros(N)
        Htmp[:]=y[N-2:]
        ax12.plot(x, Htmp)
        ax12.set_ylabel('H')

        fig.suptitle('t='+str(i))
        plt.tight_layout()

        '''
        plt.plot(sol.y)

plt.show()
