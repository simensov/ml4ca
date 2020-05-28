import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys
import os
import platform

current_dir = os.path.dirname(os.path.abspath(__file__))

if platform.system().lower() == 'linux':
    parent_dir = current_dir.rsplit('/',1)[0] # split current dir on last '/', which gives the parent dir in Ubuntu
elif platform.system().lower() == 'windows':
    parent_dir = current_dir.rsplit('\\',1)[0] # split current dir on last '/', which gives the parent dir in Windows

sys.path.append(parent_dir)

from common import methods, labels, colors, set_params, plot_gray_areas, LARGE_SQUARE, SMALL_SQUARE, RECTANGLE, NARROW_RECTANGLE, wrap_angle
import math

methods = methods + ['RLintegral']
labels = labels + ['RLI']
colors[3] = 'orange'


save = False
set_params()

def plot_zeros(ax,data):
    ax.plot(t,[0]*len(data),'-', linewidth=0.5, color='grey', alpha=0.5,label=None)

PLOT_SETPOINT_AREAS = True

'''
Positional data
'''
stern_setpoints_path = 'bagfile__{}_thrusterAllocation_stern_thruster_setpoints.csv'
stern_angles_path = 'bagfile__{}_thrusterAllocation_pod_angle_input.csv'
bow_params_path = 'bagfile__{}_bow_control.csv'

nport, nstar, nbow, aport, astar, abow, time_nstern, time_astern, time_bow = \
    [np.zeros((1,1))]*len(methods), [np.zeros((1,1))]*len(methods),[np.zeros((1,1))]*len(methods), [np.zeros((1,1))]*len(methods),[np.zeros((1,1))]*len(methods),\
        [np.zeros((1,1))]*len(methods),[np.zeros((1,1))]*len(methods),[np.zeros((1,1))]*len(methods),[np.zeros((1,1))]*len(methods) 

for i in range(len(methods)):
    s_t_path = stern_setpoints_path.format(methods[i])
    s_a_path = stern_angles_path.format(methods[i])
    bow_path = bow_params_path.format(methods[i])

    s_t_data = np.genfromtxt(s_t_path,delimiter=',')
    s_a_data = np.genfromtxt(s_a_path,delimiter=',')
    bow_data = np.genfromtxt(bow_path,delimiter=',')

    nstar[i] = s_t_data[1:,1:2]
    nport[i] = s_t_data[1:,2:3]
    aport[i] = s_a_data[1:,1:2]
    astar[i] = s_a_data[1:,2:3]
    nbow[i] = bow_data[1:,1:2]
    abow[i] = bow_data[1:,2:3]
    time_nstern[i] = s_t_data[1:,3:]
    time_astern[i] = s_a_data[1:,3:]
    time_bow[i] = bow_data[1:,4:] # lin_act_bow is in pos 3

setpointx = [10, 60, 110, 140, 190]
setpnt_areas = [0] + setpointx + [240]
setpnt_areas =[0,10,80,150,190,270,340]

'''
### ANGLES
'''
f0, axes = plt.subplots(2,1,figsize=SMALL_SQUARE,sharex=True)
plt.xlabel('Time [s]')
# axes[0].set_ylabel('Bow thruster angle [deg]')
axes[0].set_ylabel('Port angle [deg]')
axes[1].set_ylabel('Starboard angle [deg]')

for axn,ax in enumerate(axes):
    for i in range(len(methods)):
        t = time_astern[i]
        if axn == 0:
            relevant_data = aport[i]
        else:
            relevant_data = astar[i]

        ax.plot(t,relevant_data, color=colors[i],label=labels[i],alpha=0.9)
        plot_zeros(ax,relevant_data)
    
    if not PLOT_SETPOINT_AREAS:
        lab = 'Set point changes' if axn == 0 and i == 0 else '' # TODO get labels to work
        for c in setpointx:
            ax.axvline(c, linestyle='--', color='black', alpha = 0.8, label=lab)
    else:
        plot_gray_areas(ax, areas=setpnt_areas)
    
# Print reference lines
axes[0].legend(loc='best').set_draggable(True)
f0.tight_layout(pad=0.4)


'''
### THRUST
'''
f, axes = plt.subplots(3,1,figsize=SMALL_SQUARE,sharex = True)
plt.xlabel('Time [s]')
axes[0].set_ylabel('$n_{bow}$ [%]')
axes[1].set_ylabel('$n_{port}$ [%]')
axes[2].set_ylabel('$n_{star}$ [%]')

for axn,ax in enumerate(axes):
    for i in range(len(methods)):

        if axn == 0:
            t = time_bow[i]
            relevant_data = nbow[i]
        else:
            t = time_nstern[i]
            if axn == 1:
                relevant_data = nport[i]
            else:
                relevant_data = nstar[i]

        ax.plot(t,relevant_data, color=colors[i],label=labels[i],alpha=0.9) # use lower alpha to view all better ontop of eachother
        plot_zeros(ax,relevant_data)
    
    if not PLOT_SETPOINT_AREAS:
        lab = 'Set point changes' if axn == 0 and i == 0 else '' # TODO get labels to work
        for c in setpointx:
            ax.axvline(c, linestyle='--', color='black', alpha = 0.8, label=lab)
    else:
        plot_gray_areas(ax,areas=setpnt_areas)

# Print reference lines
axes[0].legend(loc='best', facecolor='#FAD7A0', framealpha=0.3).set_draggable(True)
f.tight_layout()

'''
### POWER Equivalents
'''
    
rps_max     = {'bow': 33.0, 'stern': 11.0}
diameters   = {'bow': 0.06, 'stern': 0.15}
KQ_0        = {'bow': 0.035 * 0.001518 / 0.0027, 'stern': 0.028}
rho         = 1025.0

def power(n,which):
	''' n comes as -100% to 100%, and must be divided on 100 for multiplication with rps to give fraction'''
	return np.sign(n) * KQ_0[which] * 2 * np.pi * rho * diameters[which]**5 * (n /100.0 * rps_max[which])**3

powers = [[[]]*3] * len(methods) # Contains power timeseries on [method0: [bow,port,star], method1: ...] for each method
powers = [[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]]]

for i in range(len(methods)):
    tbow = time_bow[i]
    nb = nbow[i]
    pb = 0

    for j in range(len(tbow) - 1): # trapezoidal integration
        dt = tbow[j+1] - tbow[j]
        p_avg = (power(nb[j+1] , 'bow') + power(nb[j] , 'bow')) / 2 # average |n^3| for current time interval
        powers[i][0].append(p_avg * dt)

    taft = time_nstern[i]
    nprt = nport[i]
    nstr = nstar[i]

    for j in range(len(taft) - 1): # trapezoidal integration
        dt = taft[j+1] - taft[j]
        p_avg_port = (power(nprt[j+1] , 'stern') + power(nprt[j] , 'stern')) / 2
        p_avg_star = (power(nstr[j+1] , 'stern') + power(nstr[j] , 'stern')) / 2
        powers[i][1].append(p_avg_port * dt)
        powers[i][2].append(p_avg_star * dt)

f, axes = plt.subplots(3,1,figsize=SMALL_SQUARE,sharex = True)
plt.xlabel('Time [s]')
axes[0].set_ylabel('$P^*_{bow}$ [W]')
axes[1].set_ylabel('$P^*_{port}$ [W]')
axes[2].set_ylabel('$P^*_{star}$ [W]')

for axn,ax in enumerate(axes):
    for i in range(len(methods)):
        if axn == 0:
            t = time_bow[i][:-1]
            relevant_data = powers[i][axn]
        else:
            t = time_nstern[i][:-1]
            relevant_data = powers[i][axn]

        ax.plot(t,relevant_data, color=colors[i],label=labels[i],alpha=0.9)
        plot_zeros(ax,relevant_data)
    
    if not PLOT_SETPOINT_AREAS:
        lab = 'Set point changes' if axn == 0 and i == 0 else '' # TODO get labels to work
        for c in setpointx:
            ax.axvline(c, linestyle='--', color='black', alpha = 0.8, label=lab)
    else:
        plot_gray_areas(ax,areas=setpnt_areas)

axes[0].legend(loc='best').set_draggable(True)
f.tight_layout()

cumsums = [[[]]*3]*len(methods)
cumsums = [[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]]]

for i in range(len(methods)):
    for j in range(3):
        cumsums[i][j] = np.cumsum(powers[i][j])

f, axes = plt.subplots(4,1,figsize=SMALL_SQUARE,sharex = True)
plt.xlabel('Time [s]')
axes[0].set_ylabel('$W^*_{bow}$ [J]')
axes[1].set_ylabel('$W^*_{port}$ [J]')
axes[2].set_ylabel('$W^*_{star}$ [J]')
axes[3].set_ylabel('$W^*_{total}$ [J]')

for axn,ax in enumerate(axes):
    for i in range(len(methods)):
        if axn == 0:
            t = time_bow[i][:-1]
            relevant_data = cumsums[i][axn]
        elif axn in [1,2]:
            t = time_nstern[i][:-1]
            relevant_data = cumsums[i][axn]
        else:
        	vals = [cumsums[i][axn_loc] for axn_loc in range(3)] # list of size 3 - cumssums for the current method for all three thrusters
        	shortest = min([a for val in vals for a in val.shape ]) # choose the shortest value in order to make lists equally long - bow and stern messages does not necessilty share same dimensions
        	new_vals = [cumsums[i][axn_loc][0:shortest] for axn_loc in range(3)] # adjust for this possible size difference
        	relevant_data = sum(new_vals) # take the sum over all of the three thrusters to get the total
        	t = time_nstern[i][:shortest] # adjust the time list for same reason as above

        ax.plot(t,relevant_data, color=colors[i],label=labels[i],alpha=0.9)

        # annotate the ending value
        val = relevant_data[-1] # extract final value
        x_coord = t[-1] + 1
        txt = '{:.2f}'.format(val)
        moveif = {'IPI' : -0.05 * val, 'QP': 0.1 * val, 'RL': -0.1 * val, 'RLI': 0.1 * val}
        activation = 1.0
        if ax == 0:
            moveif['RLI'] = val

        ax.annotate(txt, (x_coord, 0.95 * val + (activation * moveif[labels[i]])),color=colors[i], weight='bold',size=9)
    
    if not PLOT_SETPOINT_AREAS:
        lab = 'Set point changes' if axn == 0 and i == 0 else '' # TODO get labels to work
        for c in setpointx:
            ax.axvline(c, linestyle='--', color='black', alpha = 0.8, label=lab)
    else:
        plot_gray_areas(ax,setpnt_areas)

axes[0].legend(loc='best').set_draggable(True)
f.tight_layout()

# END RESULTS
final_powers = [[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]] ]


for i,p_method in enumerate(powers):
    # all p_method comes as power timeseries [p_bow, p_port, p_star]
    for j,p_thruster in enumerate(p_method):
        # p_thruster is now power time series for a given thruster, for the given method i
        final_powers[i][j] = sum(p_thruster)

for i in range(len(methods)):
    print('Method {}'.format(methods[i]))
    print(sum(final_powers[i]))


if False:
    f = plt.figure(figsize=LARGE_SQUARE)

    bars = []
    for i in range(3): # all three thrusters
        for j in range(len(methods)):
            bars.append(final_powers[j][i][0]) # collects [p_bow_dnvgl, p_bow_qp, p_bow_rl, p_port_dnvgl, ..., p_star_rl]

    for i in range(len(methods)):
        bars.append(sum(final_powers[i])[0])   

    bar_positions = []
    shift = 0
    for j in range(4): # there will always be four column sections
    	for i in range(len(methods)):
    		bar_positions.append(i + shift)
    	shift += len(methods) + 1

    tick_pos = [1,5,10,15]
    txts = ['Bow thruster', 'Port thruster', 'Starboard thruster', 'Total']
    bar_colors = colors * 4
    elements = plt.bar(bar_positions, bars, color = bar_colors)
    plt.xticks(tick_pos, txts)

    ax = plt.gca()
    ax.legend(elements[:], labels,loc='best').set_draggable(True)
    ax.set_ylabel('$W^*$ [J]')
    f.tight_layout()


'''
### Integral Absolute Derivative Commanded
'''

derivs = [[[],[],[],[],[],[]], [[],[],[],[],[],[]], [[],[],[],[],[],[]], [[],[],[],[],[],[]]]
average_data = []

for i in range(len(methods)):
    tbow = time_bow[i]
    nb = nbow[i]
    ab = abow[i]
    pb = 0

    for j in range(len(tbow) - 1): 
        dt = tbow[j+1] - tbow[j]

        # assume thrust derivs independent of rps_max for now
        nb_deriv = (np.abs( (nb[j+1] - nb[j]) / dt) / 100.0)[0]
        ab_deriv = 0 # There is no action from the angle of the bow thruster (np.abs( wrap_angle(ab[j+1] - ab[j], deg=True) / dt) / 180.0)[0]

        derivs[i][0].append(nb_deriv)
        derivs[i][1].append(ab_deriv)

    taft = time_nstern[i]
    nprt = nport[i]
    aprt = aport[i]
    nstr = nstar[i]
    astr = astar[i]

    for j in range(len(taft) - 1): 
        dt = taft[j+1] - taft[j]
        dt = max(0.20, dt)
        
        nprt_deriv = (np.abs( (nprt[j+1] - nprt[j]) / dt) / 100.0)[0]
        aprt_deriv = (np.abs( wrap_angle(aprt[j+1] - aprt[j], deg=True) / dt) / 180.0)[0]
        nstr_deriv = (np.abs( (nstr[j+1] - nstr[j]) / dt) / 100.0)[0]
        astr_deriv = (np.abs( wrap_angle(astr[j+1] - astr[j], deg=True) / dt) / 180.0)[0]

        derivs[i][2].append(nprt_deriv)
        derivs[i][3].append(aprt_deriv)
        derivs[i][4].append(nstr_deriv)
        derivs[i][5].append(astr_deriv)


f0, ax = plt.subplots(1,1,figsize=SMALL_SQUARE,sharex = True)

IADC = [[0],[0],[0],[0]]
IADC_cumsums = []

for i in range(len(methods)):
    cumsums = [0]
    time_data = time_bow[i] if len(time_bow[i]) < len(time_nstern[i]) else time_nstern[i]

    for j in range(len(time_data)- 1):

        val = float(sum( [derivs[i][k][j] for k in range(6)]))
        val = np.clip(val,0,400) # * 0.2
        
        IADC[i].append(val)
        # print(val, i, time_data[j]) if val > 100.0 else None 
        cumsums.append(cumsums[j] + val)

    IADC_cumsums.append(cumsums[:-1])

for i in range(len(methods)):
    time_data = time_bow[i][:-1] if len(time_bow[i]) < len(time_nstern[i]) else time_nstern[i][:-1]
    ax.plot(time_data, IADC_cumsums[i], color=colors[i], label=labels[i])

# Gray areas
plot_gray_areas(ax,areas = setpnt_areas)

ax.legend(loc='best').set_draggable(True)
ax.set_ylabel('IADC [-]')
ax.set_xlabel('Time [s]')

for i in range(len(methods)):
    val = IADC_cumsums[i][-1] # extract at last timestep
    x_coord = time_data[-1] + 0.25
    txt = '{:.2f}'.format(val)
    moveif = {'IPI':0, 'QP': 0.00*val, 'RL': -val, 'RLI':val}
    activation = 1.0
    ax.annotate(txt, (x_coord, 0.99*val + (activation * moveif[labels[i]])),color=colors[i], weight='bold')

f0.tight_layout()
    
print('IADC')
for i in range(len(methods)): print(methods[i], ':', IADC_cumsums[i][-1])

plt.show()