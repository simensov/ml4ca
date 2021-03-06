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

from common import methods, labels, colors, set_params, get_secondly_averages, absolute_error, IAE, plot_gray_areas, LARGE_SQUARE, RECTANGLE, SMALL_SQUARE

set_params() # sets global plot parameters

headings = [-158,-135,-113,-90,-68,-45,-23,0,23,45,68,90,113,135,158,180]
methods = ['pseudo{}deg'.format(val) for val in headings]
colors[3] = colors[0]
methods = ['RLintegral{}deg'.format(val) for val in headings]
colors[3] = 'orange'

labels = ['${}^\\circ$'.format(val) for val in headings]

'''
Positional data
'''
path = 'bagfile__{}_observer_eta_ned.csv' # General path to eta
path_ref = 'bagfile__RLintegral-135deg_reference_filter_state_desired.csv'

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
    # 0th elements is text
    north[i] = ( posdata[1:,1:2] - ref_data[1:,1:2][0,0]) 
    east[i] = ( posdata[1:,2:3] - ref_data[1:,2:3][0,0]) 
    psi[i] = posdata[1:,6:7]
    time[i] = posdata[1:,7:]
    ALL_POS_DATA.append([north[i], east[i], psi[i], time[i]] )


# Points for the different box test square. These are only the coords and not the changes relative to eachother. Very first elements are nan
box_n = [n_0[1,0],  n_0[1,0] + 5, n_0[1,0] + 5.0,  n_0[1,0] + 5.0,  n_0[1,0],           n_0[1,0]]
box_e = [e_0[1,0],  e_0[1,0],     e_0[1,0] - 5.0,  e_0[1,0] - 5.0,  e_0[1,0] - 5.0,     e_0[1,0]]
box_p = [p_0[1,0],  p_0[1,0],     p_0[1,0],        p_0[1,0] - 45,   p_0[1,0] - 45,      p_0[1,0]]

setpoint_times = [0]

'''
### NEDPOS
'''
if False:
    f = plt.figure(figsize=SMALL_SQUARE)
    ax = plt.gca()
    # ax.scatter(box_e,box_n,color = 'black',marker='8',s=50,label='Set points')
    ax.scatter(box_e[0],box_n[0],color = 'black',marker='8',s = 50,label='Set point')

    for i in range(len(methods)):
        e, n = east[i], north[i]
        plt.plot(e,n,label=labels[i])#,color = colors[i])

    show_dir = False
    if show_dir:
        # North
        x = box_e[0] + 0.6; y = (box_n[0] + box_n[1]) / 2 - 0.5; dx = 0; dy = (box_n[1] - box_n[0]) * 0.2
        ax.annotate("", xy=(x+dx,y+dy), xytext=(x, y), arrowprops=dict(arrowstyle="->"))
        # West
        x = (box_e[1] + box_e[2]) / 2 - (box_e[2] - box_e[1]) * 0.1; y = box_n[1] + 1.2; dx = (box_e[2] - box_e[1]) * 0.2; dy = 0
        ax.annotate("", xy=(x+dx,y+dy), xytext=(x, y), arrowprops=dict(arrowstyle="->"))
        # South
        x = box_e[2] - 0.3; y = (box_n[-3] + box_n[-2]) / 2 + 0.5; dx = 0; dy = -(box_n[-3] - box_n[-2]) * 0.2
        ax.annotate("", xy=(x+dx,y+dy), xytext=(x, y), arrowprops=dict(arrowstyle="->"))
        # East
        x = (box_e[-1] + box_e[-2]) / 2 - (box_e[-1] - box_e[-2]) * 0.05; y = box_n[-1] - 0.3; dx = (box_e[-1] - box_e[-2]) * 0.2; dy = 0
        ax.annotate("", xy=(x+dx,y+dy), xytext=(x, y), arrowprops=dict(arrowstyle="->"))

    ax.plot(ref_east, ref_north, '--', color='black',label='Reference')
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
        ax.plot(t,local_data[axn],label=labels[i])#,color=colors[i])
    
    # Print reference lines
    ax.plot(ref_time, refdata[axn], '--',color='black', label = 'Reference' if axn == 0 else None)
    plot_gray_areas(ax, setpoint_times)
    if axn <= 1:
        ax.plot(ref_time, refdata[axn] + 1.0,'--', color='red', label = 'Boundary' if axn == 0 else None)
        ax.plot(ref_time, refdata[axn] - 1.0,'--', color='red')
        ax.set_ylim(-1.5,1.5)
    else:
        ax.plot(ref_time, refdata[axn] + 5.0,'--', color='red')
        ax.plot(ref_time, refdata[axn] - 5.0,'--', color='red')
        ax.set_ylim(-18,18)

f0.tight_layout()

'''
### Integral Absolute Error : int_0^t  sqrt ( error^2 ) dt
    - compare position to reference filter
    - gather data from ref_filter/state_desired and observer/eta/ned, which are very different, so take their averages per second.
'''

ref_data_averages = [] # list of tuples: (average times [whole seconds] (dim 1,), average data values (dim 3,xsteps))
for data in refdata:
    ref_data_averages.append(get_secondly_averages(ref_time, data))

pos_data_averages = [] # is a list of num methods lists containing tuples: (average times [whole seconds], average data values) 
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
    f0, axes = plt.subplots(3,1,figsize=(12,9),sharex = True)
    for axn,ax in enumerate(axes):
        ax.plot(ref_data_averages[axn][0], ref_data_averages[axn][1], '--', color='black')
        for i in range(len(methods)):
            local_time = pos_data_averages[i][axn][0]
            local_data = pos_data_averages[i][axn][1]
            ax.plot(local_time,local_data)#,color = colors[i])
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
    ax.plot(times, IAES[i], label=labels[i]) # , color=colors[i])

# Gray areas
plot_gray_areas(ax,areas = setpoint_times)

# ax.legend(loc='best').set_draggable(True)
ax.set_ylabel('IAE [-]')
ax.set_xlabel('Time [s]')

vals = []
for i in range(len(methods)):
    vals.append(IAES[i][-1])
print('Average IAES:', np.mean(vals))

for i in range(len(methods)):
    val = IAES[i][-1] # extract IAE at last timestep
    x_coord = t[-1] + 0.25
    txt = '{:.2f}'.format(val)
    moveif = {'IPI':0, 'QP': 0.00*val, 'RL': -0.00*val, 'RLI':0, 'NO':0, 'RLintegral-135deg':0}
    activation = 1.0 if axn <= 1 else 0.0
    # ax.annotate(txt, (x_coord, 0.99*val + (activation * moveif[labels[i]])),color=colors[i], weight='bold')
    # ax.annotate(txt, (x_coord, 0.99*val + (activation * 0))) # color=colors[i], )

ax.annotate('Average IAE: {:.2f}'.format(np.mean(vals)), (t[0] + 1, np.mean(vals)), weight='bold')

f0.tight_layout()
    


plt.show()
