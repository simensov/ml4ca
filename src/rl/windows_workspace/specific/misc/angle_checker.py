import numpy as np 
import matplotlib.pyplot as plt 
from tabulate import tabulate

angles = np.linspace(-np.pi, np.pi, 11)
sins = np.linspace(-1.0,1.0,11)
coss = np.linspace(-1.0,1.0,11)

r2d = 180 / np.pi

# sins = [np.sin(a) for a in angles]
# coss = [np.cos(a) for a in angles]

table = np.zeros((len(sins)+3,len(coss)+3))
table[3:,0] = np.array([s for s in sins]) 
table[3:,1] = np.array([np.arcsin(s) *r2d for s in sins]) 
table[0,3:] = np.array([c for c in coss]) 
table[1,3:] = np.array([np.arccos(c) *r2d for c in coss]) 

for i, s in enumerate(sins):
    for j, c in enumerate(coss):
        print('Sin and cos:')
        print('{:.2f}, {:.2f}'.format(s,c) )
        print('Angles coming from sin and cos')
        print('{:.2f}, {:.2f}'.format( np.arcsin(s) * r2d, np.arccos(c) *r2d))
        t = np.arctan2(s,c) * r2d
        print('Angle from arctan2:')
        print(t)
        table[i+3,j+3] = '{:.2f}'.format(t)

print(tabulate(table,numalign='center',stralign='center',tablefmt='presto'))