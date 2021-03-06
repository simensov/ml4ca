import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import MaxNLocator

import sys
import os
import platform

current_dir = os.path.dirname(os.path.abspath(__file__))

if platform.system().lower() == 'linux':
    parent_dir = current_dir.rsplit('/',1)[0] # split current dir on last '/', which gives the parent dir in Ubuntu
elif platform.system().lower() == 'windows':
    parent_dir = current_dir.rsplit('\\',1)[0] # split current dir on last '/', which gives the parent dir in Windows

sys.path.append(parent_dir)

from common import methods, labels, colors, set_params, get_secondly_averages, absolute_error, IAE, plot_gray_areas, SMALL_SQUARE, RECTANGLE

set_params() # sets global plot parameters

methods = methods + ['RLintegral']
labels = labels + ['RLI']
colors[3] = 'orange'

'''
Positional data
'''
path = 'bagfile__{}_observer_eta_ned.csv' # General path to eta
path_ref = 'bagfile__RL_reference_filter_state_desired.csv'

ref_data = np.genfromtxt(path_ref,delimiter=',')
ref_north = (ref_data[1:,1:2] - ref_data[1:,1:2][0,0] ) 
ref_east = (ref_data[1:,2:3] - ref_data[1:,2:3][0,0])
ref_yaw = ref_data[1:,3:4]
ref_time = ref_data[1:,-1:]
refdata = [ref_north, ref_east, ref_yaw]
n_0, e_0, p_0 = ref_north, ref_east, ref_yaw

north, east, psi, time = [np.zeros((1,1))]*len(methods), [np.zeros((1,1))]*len(methods), [np.zeros((1,1))]*len(methods), [np.zeros((1,1))]*len(methods)
ALL_POS_DATA = []

for i in range(len(methods)):
    fpath = path.format(methods[i])
    posdata = np.genfromtxt(fpath,delimiter=',')
    # 0th elements are nan due to being text
    north[i] = ( posdata[1:,1:2] - ref_data[1:,1:2][0,0]) 
    east[i] = ( posdata[1:,2:3] - ref_data[1:,2:3][0,0]) 
    psi[i] = posdata[1:,6:7]
    time[i] = posdata[1:,7:]
    ALL_POS_DATA.append([north[i], east[i], psi[i], time[i]])

# Points for the different box test square. These are only the coords and not the changes relative to eachother. Very first elements are nan
box_n = [n_0[1,0],  n_0[1,0] + 12, n_0[1,0] + 12.0,  n_0[1,0] + 12.0,  n_0[1,0]]
box_e = [e_0[1,0],  e_0[1,0],     e_0[1,0]  + 12.0,  e_0[1,0] + 12.0,  e_0[1,0]]
box_p = [p_0[1,0],  p_0[1,0],     p_0[1,0],        p_0[1,0] + 90.0,   p_0[1,0] - 45.0]

setpointx  = [0] + (np.array([10,  10,  60,  60, 120, 120, 160, 160, 250,250,290]) + 2.0).tolist()
gray_areas = [el for i, el in enumerate(setpointx) if i%2 == 0] + [setpointx[-1]]

# currently unused
# setpoint_change_times                               = [0,10-2,60-2,120-2,140-2,190-2]
# setpointx                                           = [0,  10,  10,  60,  60, 120, 120, 160, 160, 240]
# box_coords_over_time = np.array([
#    (np.array([box_n[0]]*(len(setpointx))) + np.array([0,  0,  5,   5,   5,   5,   5,   5,   0,   0,   0,   0])).tolist(),
#    (np.array([box_e[0]]*(len(setpointx))) + np.array([0,  0,  0,   0,  -5,  -5,  -5,  -5,  -5,  -5,   0,   0])).tolist(),
#    (np.array([0]*(len(setpointx)))        + np.array([0,  0,  0,   0,   0,   0, -45, -45, -45, -45,   0,   0])).tolist()])

'''
### NEDPOS
'''
f = plt.figure(figsize=SMALL_SQUARE)
ax = plt.gca()
ax.scatter(box_e,box_n,color = 'black',marker='8',s=50,label='Set points')

for i in range(len(methods)):
    e, n = east[i], north[i]
    plt.plot(e,n,color = colors[i], label=labels[i])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


show_dir = True
if show_dir:
    x = box_e[0] - 1.1; y = (box_n[0] + box_n[1]) / 2; dx = 0; dy = (box_n[1] - box_n[0]) * 0.2
    ax.annotate("", xy=(x+dx,y+dy), xytext=(x, y), arrowprops=dict(arrowstyle="->"))
    x = (box_e[1] + box_e[2]) / 2 - (box_e[2] - box_e[1]) * 0.1; y = box_n[1] + 1; dx = (box_e[2] - box_e[1]) * 0.2; dy = 0
    ax.annotate("", xy=(x+dx,y+dy), xytext=(x, y), arrowprops=dict(arrowstyle="->"))
    x = (box_e[-1] + box_e[-2]) / 2 - (box_e[-2] - box_e[-1]) * 0.1 + 3; y = (box_n[-1] + box_n[-2]) / 2 + (box_n[-2] - box_n[-1]) * 0.1 - 1; dx = -(box_e[-2] - box_e[-1]) * 0.14; dy = -(box_n[-2] - box_n[-1]) * 0.14
    ax.annotate("", xy=(x+dx,y+dy), xytext=(x, y), arrowprops=dict(arrowstyle="->"))

ax.plot(ref_east, ref_north, '--', color='black',label='Reference')    
ax.set_xlabel('East [m]')
ax.set_ylabel('North [m]')
ax.legend(loc='best').set_draggable(True)
f.tight_layout()

'''
### North and East plots
'''
f0, axes = plt.subplots(3,1,figsize=RECTANGLE,sharex = True)
plt.xlabel('Time [s]')
axes[0].set_ylabel('North [m]')
axes[1].set_ylabel('East [m]')
axes[2].set_ylabel('Yaw [deg]')

for axn,ax in enumerate(axes):
    for i in range(len(methods)):
        local_data = north[i], east[i], psi[i], time[i]
        t = local_data[3]
        ax.plot(t,local_data[axn],color=colors[i],label=labels[i])
    
    # Print reference lines
    ax.plot(ref_time, refdata[axn], '--',color='black', label = 'Reference' if axn == 0 else None)
    plot_gray_areas(ax, [el for i, el in enumerate(setpointx) if i%2 == 0] + [setpointx[-1]])

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
   
axes[0].legend(loc='best').set_draggable(True)
f0.tight_layout()

'''
### Integral Absolute Error : int_0^t  sqrt ( error^2 ) dt
    - compare position to reference filter
    - gather data from ref_filter/state_desired and observer/eta/ned, which are very different, so take their averages per second.
'''

ref_data_averages = [] # list of tuples: (average times [whole seconds] (dim 1,), average data values (dim 3,xsteps))
for data in refdata:
    ref_data_averages.append(get_secondly_averages(ref_time, data))

pos_data_averages = [] # is a list of three lists containing tuples: (average times [whole seconds], average data values) 
for i in range(len(methods)):
    current_method_averages = []
    current_method_data = ALL_POS_DATA[i]
    t = current_method_data[-1]
    for j in range(len(current_method_data) - 1):
        dimensional_data = current_method_data[j].reshape(current_method_data[j].shape[0],).tolist()
        inn = get_secondly_averages(t, dimensional_data)
        current_method_averages.append(inn)
    
    pos_data_averages.append(current_method_averages)

# Here, three IAE plots will be shown ontop of eachother
etas = [] # list of columvectors
refs = [] # list of columvectors

for pos_data_method_i in pos_data_averages:
    local_ned = []
    for tup in pos_data_method_i:
        t, d = tup
        local_ned.append(d)
        
    etas.append(np.array(local_ned).T)

local_ned = []
for tup in ref_data_averages:
    t,d = tup
    local_ned.append(d)

refs = np.array(local_ned).T

f0, ax = plt.subplots(1,1,figsize=SMALL_SQUARE,sharex = True)
IAES = [] # cumulative errors over time
times = (np.array(ref_data_averages[0][0]) - 1.0).tolist()
for i in range(len(methods)):
    integrals, cumsums = IAE(etas[i] / np.array([5.,5.,25.]), refs / np.array([5.,5.,25.]), times)
    IAES.append(cumsums)
    ax.plot(times, IAES[i], color=colors[i], label=labels[i])

# Gray areas
plot_gray_areas(ax,areas = gray_areas)

ax.legend(loc='best').set_draggable(True)
ax.set_ylabel('IAE [-]')
ax.set_xlabel('Time [s]')

for i in range(len(methods)):
    val = IAES[i][-1] # extract IAE at last timestep
    x_coord = times[-1] + 0.25
    txt = '{:.2f}'.format(val)
    moveif = {'IPI': -0.01 * val, 'QP': 0.01 * val, 'RL': 0.01 * val, 'RLI': -0.01 * val}
    activation = 1.0
    ax.annotate(txt, (x_coord, 0.99 * val + (activation * moveif[labels[i]])),color=colors[i], weight='bold', size=9)

f0.tight_layout()
    
print('IAES')
for i in range(len(methods)): print(methods[i], ':', IAES[i][-1])

plt.show()
