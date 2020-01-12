import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu, sig):
    return np.exp(-((x - mu)/sig)**2 /2)

c = 1
dx = 0.01
L = 100
x = np.linspace(0, L, num = (L/dx) + 1)
N = len(x)

dt = dx/(2*c)
s = (c*dt/dx)**2

u = gaussian(x, L/2, 1)
#u=np.sin(5*np.pi*x/L)

#u[0] = 0
#u[-1] = u[-2]
#u[-1] = 0

deriv = np.zeros(len(x))
#deriv[len(x)/2 - 500 : len(x)/2 + 500] = 1
u1 = np.zeros(len(x))
#u1[1:-1] = np.copy((s/2)*(u[2:]+u[:-2])+(1-s)*u[1:-1]+dt*deriv[1:-1])
u1[1:-1] = u[1:-1] + dt*deriv[1:-1]
u1[-1]=u1[-2]
T = 40
m = int(round(T/dt))
diffmatrix = np.zeros((len(x), m))
for j in range(m):
    tmp = np.copy(u1)
    u1[1:-1] = s*(u1[2:] + u1[:-2]) + 2*(1-s)*u1[1:-1] - u[1:-1]
    #u1[-1] = u1[-2]
    u = np.copy(tmp)
    
    #diffmatrix[:,j] = np.abs(u1 -0.5*(gaussian(x,x - t,1) + gaussian(x, x+t, 1)))

    t = (j+2)*dt
    #print(np.max(u1 -0.5*(gaussian(x,L/2 - t,1) + gaussian(x, L/2+t, 1))))
    diffmatrix[:,j] = np.abs(u1 -0.5*(gaussian(x,L/2-t,1) + gaussian(x, L/2+t, 1)))

    if(t == int(t)):
        plt.plot(x, u1, label='t='+str(t))
        plt.plot(x, 0.5*(gaussian(x,L/2 - t,1) + gaussian(x, L/2+t, 1)))
        plt.ylabel('u')
        plt.xlabel('x')
        plt.legend()
        plt.tight_layout()
        #plt.show()
#plt.savefig('testplots/gaussianpulsering.pdf')
print(np.max(np.max(diffmatrix)))
