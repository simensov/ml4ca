import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.markers import MarkerStyle


import sys
import os
import platform

current_dir = os.path.dirname(os.path.abspath(__file__))

if platform.system().lower() == 'linux':
    parent_dir = current_dir.rsplit('/',1)[0] # split current dir on last '/', which gives the parent dir in Ubuntu
elif platform.system().lower() == 'windows':
    parent_dir = current_dir.rsplit('\\',1)[0] # split current dir on last '/', which gives the parent dir in Windows

sys.path.append(parent_dir)

from common import methods, labels, colors, set_params, get_secondly_averages, absolute_error, IAE, plot_gray_areas, LARGE_SQUARE, RECTANGLE, SMALL_SQUARE

set_params() # sets global plot parameters

methods = ['NO'] # methods + ['RLintegral']
labels = ['NO'] # labels + ['RLI']
colors[3] = 'orange'

'''
Positional data
'''
path = 'bagfile__{}_observer_eta_ned.csv' # General path to eta

north, east, psi, time = [np.zeros((1,1))]*len(methods), [np.zeros((1,1))]*len(methods), [np.zeros((1,1))]*len(methods), [np.zeros((1,1))]*len(methods)
ALL_POS_DATA = []
for i in range(len(methods)):
    fpath = path.format(methods[i])
    posdata = np.genfromtxt(fpath,delimiter=',')
    # 0th elements is text

    north[i] = posdata[1:,1:2] - posdata[1:,1:2][0,0]
    east[i] = posdata[1:,2:3] - posdata[1:,2:3][0,0]
    psi[i] = posdata[1:,6:7]
    time[i] = posdata[1:,7:]
    ALL_POS_DATA.append([north[i], east[i], psi[i], time[i]] )

n_0, e_0, p_0 = north[0], east[0], psi[0]
# Points for the different box test square. These are only the coords and not the changes relative to eachother. Very first elements are nan
box_n = [n_0[1,0],  n_0[1,0] + 5, n_0[1,0] + 5.0,  n_0[1,0] + 5.0,  n_0[1,0],           n_0[1,0]]
box_e = [e_0[1,0],  e_0[1,0],     e_0[1,0] - 5.0,  e_0[1,0] - 5.0,  e_0[1,0] - 5.0,     e_0[1,0]]
box_p = [p_0[1,0],  p_0[1,0],     p_0[1,0],        p_0[1,0] - 45,   p_0[1,0] - 45,      p_0[1,0]]

setpoint_times = [0]
ref_north = box_n[0] * np.ones_like(north[0])
ref_east = box_e[0] * np.ones_like(east[0])
ref_yaw = box_p[0] * np.ones_like(psi[0])
ref_time = np.copy(time[0])
refdata = [ref_north, ref_east, ref_yaw]


'''
### NEDPOS
'''
f = plt.figure(figsize=SMALL_SQUARE)
ax = plt.gca()

ax.scatter(box_e[0], box_n[0],color = 'black',marker='8',s = 50,label='Initial position',zorder =20)
ax.set_ylim(-10,7) # 17 distanc
ax.set_xlim(-11,6)

for i in range(len(methods)):
    e, n = east[i], north[i]
    plt.plot(e,n,color = '#ac34c7', label='Vessel position')

nth = 600
for i, (e,n,p) in enumerate(zip(east[0], north[0], psi[0])):
    if i % nth == 0 or i == len(east[0]) - 1:
        m = MarkerStyle("^")
        m._transform.scale(5*0.6, 5*1)
        m._transform.rotate_deg(-(p - ref_yaw[0,0]))
        plt.scatter(e, n, s=225, marker = m, color = 'grey', linewidths = 1, edgecolors = 'black', alpha=0.5, zorder=0)

ax.plot([], [], color='grey', marker='^', linestyle='None', markersize=10, markeredgewidth=1,markeredgecolor = 'black', label='Vessel (to scale lengthwise)')
ax.set_xlabel('East [m]')
ax.set_ylabel('North [m]')
ax.legend(loc='best').set_draggable(True)

f.tight_layout()

'''
### North and East plots
'''
f0, axes = plt.subplots(3,1,figsize=SMALL_SQUARE,sharex = True)
plt.xlabel('Time [s]')
axes[0].set_ylabel('North [m]')
axes[1].set_ylabel('East [m]')
axes[2].set_ylabel('Yaw [deg]')

for axn,ax in enumerate(axes):
    for i in range(len(methods)):
        local_data = north[i], east[i], psi[i], time[i]
        t = local_data[3]
        ax.plot(t,local_data[axn],color='#ac34c7',label='Vessel pose')
    
    # Print reference lines
    ax.plot(ref_time, refdata[axn], '--',color='black', label = 'Initial pose' if axn == 0 else None)
    plot_gray_areas(ax, setpoint_times)

axes[0].legend(loc='best').set_draggable(True)
f0.tight_layout()

plt.show()
