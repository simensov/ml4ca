import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as patches

import sys
import os
import platform

current_dir = os.path.dirname(os.path.abspath(__file__))

if platform.system().lower() == 'linux':
    parent_dir = current_dir.rsplit('/',1)[0] # split current dir on last '/', which gives the parent dir in Ubuntu
elif platform.system().lower() == 'windows':
    parent_dir = current_dir.rsplit('\\',1)[0] # split current dir on last '/', which gives the parent dir in Windows

sys.path.append(parent_dir)

from common import methods, labels, colors, set_params, get_secondly_averages, absolute_error, IAE, plot_gray_areas, SMALL_SQUARE, RECTANGLE, SMALL_RECTANGLE

methods = methods + ['RLintegral']
labels = labels + ['RLI']
colors[3] = 'orange'

set_params() # sets global plot parameters

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
    ALL_POS_DATA.append([north[i], east[i], psi[i], time[i]] )


# Points for the different box test square. These are only the coords and not the changes relative to eachother. Very first elements are nan
box_n = [n_0[1,0],  n_0[1,0] + 5, n_0[1,0] + 5.0,  n_0[1,0] + 5.0,  n_0[1,0],           n_0[1,0]]
box_e = [e_0[1,0],  e_0[1,0],     e_0[1,0] - 5.0,  e_0[1,0] - 5.0,  e_0[1,0] - 5.0,     e_0[1,0]]
box_p = [p_0[1,0],  p_0[1,0],     p_0[1,0],        p_0[1,0] - 45,   p_0[1,0] - 45,      p_0[1,0]]

setpoint_times = [0,10, 80, 150, 190, 270, 350]

'''
### NEDPOS
'''
f = plt.figure(figsize=SMALL_SQUARE)
ax = plt.gca()
ax.scatter(box_e,box_n,color = 'black',marker='8',s=50,label='Set points')
show_current_angle = True
if show_current_angle:
    ang = np.deg2rad(135)
    r = 1
    x = box_e[1] - 3; y = box_n[1] + 2.2
    dx = r * np.sin(ang); dy = r * np.cos(ang)
    rect = patches.Rectangle( (x-0.9, y-1.15), 2.6,1.5,linewidth = 1,alpha = 0.5, facecolor='#6b4c9a') # 
    ax.add_patch(rect)

    #ax.arrow(x,y,dx,dy)
    ax.annotate("$\\nu_c = 0.2 m/s, \\beta_c = 135^\\circ$ ",xy=(x-0.7,y+0.1), size=9)
    ax.annotate("", xy=(x+dx,       y+dy),      xytext=(x, y), arrowprops=dict(arrowstyle="->"))
    ax.annotate("",  xy=(x+dx-0.2,   y+dy-0.2),  xytext=(x-0.2, y-0.2), arrowprops=dict(arrowstyle="->"))
    ax.annotate("",  xy=(x+dx-0.4,     y+dy-0.4),    xytext=(x-0.4, y-0.4), arrowprops=dict(arrowstyle="->"))

show_dir = True
if show_dir:
    # North
    x = box_e[0] - 0.3; y = (box_n[0] + box_n[1]) / 2 - 0.5; dx = 0; dy = (box_n[1] - box_n[0]) * 0.2
    ax.annotate("", xy=(x+dx,y+dy), xytext=(x, y), arrowprops=dict(arrowstyle="->"))
    # West
    x = (box_e[1] + box_e[2]) / 2 - (box_e[2] - box_e[1]) * 0.1 + 0.5; y = box_n[1] + 0.3; dx = (box_e[2] - box_e[1]) * 0.2; dy = 0
    ax.annotate("", xy=(x+dx,y+dy), xytext=(x, y), arrowprops=dict(arrowstyle="->"))
    # South
    x = box_e[2] - 0.3; y = (box_n[-3] + box_n[-2]) / 2 + 0.5; dx = 0; dy = -(box_n[-3] - box_n[-2]) * 0.2
    ax.annotate("", xy=(x+dx,y+dy), xytext=(x, y), arrowprops=dict(arrowstyle="->"))
    # East
    x = (box_e[-1] + box_e[-2]) / 2 - (box_e[-1] - box_e[-2]) * 0.05; y = box_n[-1] - 0.3; dx = (box_e[-1] - box_e[-2]) * 0.2; dy = 0
    ax.annotate("", xy=(x+dx,y+dy), xytext=(x, y), arrowprops=dict(arrowstyle="->"))

for i in range(len(methods)):
    e, n = east[i], north[i]
    plt.plot(e,n,color = colors[i], label=labels[i])

ax.plot(ref_east, ref_north, '--', color='black',label='Reference')
ax.set_xlabel('East [m]')
ax.set_ylabel('North [m]')
ax.legend(loc='best').set_draggable(True)
f.tight_layout()

'''
### North and East plots
'''
f0, axes = plt.subplots(3,1,figsize=SMALL_RECTANGLE,sharex = True)
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
    plot_gray_areas(ax, setpoint_times)

   
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

if False: # This is used to verify that averages is true to the real data
    f0, axes = plt.subplots(3,1,figsize=RECTANGLE,sharex = True)
    for axn,ax in enumerate(axes):
        ax.plot(ref_data_averages[axn][0], ref_data_averages[axn][1], '--', color='black')
        for i in range(len(methods)):
            local_time = pos_data_averages[i][axn][0]
            local_data = pos_data_averages[i][axn][1]
            ax.plot(local_time,local_data,color = colors[i])
    f0.tight_layout()
    plt.show()
    sys.exit()

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
plot_gray_areas(ax,areas = setpoint_times)

ax.legend(loc='best').set_draggable(True)
ax.set_ylabel('IAE [-]')
ax.set_xlabel('Time [s]')

for i in range(len(methods)):
    val = IAES[i][-1] # extract IAE at last timestep
    x_coord = setpoint_times[-1] + 0.5
    txt = '{:.2f}'.format(val)
    moveif = {'IPI':-0.02*val, 'QP': 0.0* val, 'RL': 0.02 * val, 'RLI': 0.0 * val}
    activation = 1.0
    ax.annotate(txt, (x_coord, 0.99 * val + (activation * moveif[labels[i]])),color=colors[i], weight='bold', size=9)

f0.tight_layout()
    
print('IAES')
for i in range(len(methods)): print(methods[i], ':', IAES[i][-1])

plt.show()
