import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Solve for finite difference coefficients.')
parser.add_argument('derivative',help='derivative order', type=int)
parser.add_argument('-n','--nodes', nargs='+', help='nodes array')

args = parser.parse_args()

nodes = np.array(args.nodes)
nodes = nodes.astype(np.int)
m = int(args.derivative)

n = len(nodes)

sig = np.zeros((n,n))

for i in range(n):
    sig[i,:] = nodes**i

b = np.zeros(n)
b[m] = np.math.factorial(m)

print(np.linalg.solve(sig,b))
