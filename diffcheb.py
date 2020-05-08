from scipy import *
from numpy import *
from matplotlib.pyplot import *
from scipy.linalg import *
from numpy.linalg import *
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp

def diffcheb(n,xspan):
    """
    diffcheb(n,xspan)
    Compute Chebyshev differentiation matrices on `n`+1 points in the
    interval `xspan`. Return a vector of nodes, and the matrices for the first
    and second derivatives.
    """
    x = -cos( arange(n+1)*pi/n )   # nodes in [-1,1]
    Dx = zeros([n+1,n+1])
    c = hstack([2.,ones(n-1),2.])    # endpoint factors

    # Off-diagonal entries
    Dx = zeros([n+1,n+1])
    for i in range(n+1):
            for j in range(n+1):
                    if i != j:
                            Dx[i,j] = (-1)**(i+j) * c[i] / (c[j]*(x[i]-x[j]))

    # Diagonal entries by the "negative sum trick"
    for i in range(n+1):
            Dx[i,i] = -sum( [Dx[i,j] for j in range(n+1) if j!=i] )

    # Transplant to [a,b]
    a,b = xspan
    x = a + (b-a)*(x+1)/2
    Dx = 2*Dx/(b-a)

    # Second derivative
    Dxx = Dx @ Dx

    return x, Dx
