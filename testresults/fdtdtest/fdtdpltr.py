import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

fin = pd.read_csv('yeestability1k.out', sep=' ', header=None)
fin = np.array(fin)
plt.semilogy(fin[0], fin[1], '.', label='H error')
plt.semilogy(fin[0], fin[2], '.', label='E error')
#plt.loglog(fin[0], fin[-1], '--')
plt.xlabel('t')
plt.ylabel('error')
plt.title('Yee stability')
plt.legend()
plt.savefig('yeestability1k.pdf')


