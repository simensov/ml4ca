from matplotlib.patches import Polygon
import matplotlib.pyplot as plt 
import numpy as np 


f, ax = plt.subplots(1)
ax.set_ylim(-5,5)
ax.set_xlim(-5,5)

L1 = 3
L2 = 2.2
W = 0.7
pos = np.array([[0,0],[0,L2],[W/2., L1],[W,L2],[W,0]])
poly = Polygon(xy=pos, closed=True, facecolor='grey')
ax.add_patch(poly)


plt.show()