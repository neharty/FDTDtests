from matplotlib import pyplot as plt
import numpy as np
import helper as h
#from scipy.integrate import solve_ivp

numints = 5000
L = 50.
dx = L/numints
x = np.linspace(0, L, num = numints + 1)
N = len(x)

def n(z):
    return 1 + z/L

#Ea = h.initial(x, L)
Ea = np.zeros(N)
#Ea[-1] = Ea[-2]
#Ea[0] = Ea[1]
#Ea[0] = Ea[-1]

Ea[0] = 0
Ea[-1] = 0

#Ha = h.initial(x, L)
Ha = np.zeros(N)
#Ha[-1] = Ha[-2] 
#Ha[0] = Ha[1]
#Ha[0] = Ha[-1]

Ha[0] = (18./11)*Ha[1] - (9./11)*Ha[2] + (2./11)*Ha[3]
Ha[-1] = (18./11)*Ha[-2] - (9./11)*Ha[-3] + (2./11)*Ha[-4]

T = 20

epsmax = max(n(x)**2)
cmax=max(1/n(x))
#dt = (epsmax*dx)**2
dt = dx/cmax
s = dt/(dx)

m = int(round(T/dt))

Herr = np.zeros(N)
Eerr = np.zeros(N)

fig, (ax11,ax12) = plt.subplots(nrows = 1, ncols=2,figsize=(16,9))

Htmp = np.zeros(len(Ha))
for j in range(m+1):
    Htmp[:] = Ha[:]
    Ha[1:-1] = Ha[1:-1] + s*(Ea[2:] - Ea[1:-1])
    #Ha[-1] = Ha[-2]
    #Ha[0] = Ha[1]
    
    #htmp = Ha[0]
    #Ha[0] = Ha[-1]
    #Ha[-1] = htmp

    #Ha[0] = (4./3)*Ha[1] - (1./3)*Ha[2]
    #Ha[-1] = (4./3)*Ha[-2] - (1./3)*Ha[-3]
    
    Ha[0] = (18./11)*Ha[1] - (9./11)*Ha[2] +(2./11)*Ha[3]
    Ha[-1] = (18./11)*Ha[-2] - (9./11)*Ha[-3] + (2./11)*Ha[-4]

    Ea[1:-1] = Ea[1:-1] + s*(Ha[1:-1] - Ha[:-2])
    #Ea[-1] = Ea[-2]
    #Ea[0] = Ea[1]
    
    #etmp = Ea[0]
    #Ea[0] = Ea[-1]
    #Ea[-1] = etmp
    
    Ea[0] = 0
    Ea[-1] = 0

    t = (j+1)*dt
    if(2*np.pi/T*t <= np.pi):
        Ea[int((N-1)/2)] = np.sin(2*np.pi/T*t)
    
    #if(t%(3*L)==0):
    if(t==int(t)):
        print(str(np.max(np.abs(Ea-h.initial(x,L)))) + '\t'+str(np.max(np.abs(Ha-h.initial(x,L)))))
        ax11.plot(x, Ea, label='t='+str(t))
        ax11.set_ylabel('E_z')
        #ax11.set_ylim([-1,1])
         
        ax12.plot(x, Ha)
        ax12.set_ylabel('H_y')
        #ax12.set_ylim([-1, 1])
        
        '''
        H = H00 + np.zeros(N)
        E = np.zeros(N)
        for k in range(1,151):
            H = H + h.Hn(x, t, k, L)
            E = E + h.En(x, t, k, L)
        Herr = np.abs(Ha - H)
        Eerr = np.abs(Ea - E)
        
        ax22.semilogy(x, Herr, '.')
        ax21.semilogy(x, Eerr, '.')
        '''
        ax11.legend()
        fig.suptitle('t='+str(t)+', dx = ' +str(dx))

plt.tight_layout()
#plt.show()

plt.savefig('../testresults/sinesource.pdf')

