import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
from common import methods, labels, colors, set_params
import math

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

'''
### ANGLES
'''
f0, axes = plt.subplots(2,1,figsize=(12,9),sharex=True)
plt.xlabel('Time [s]')
# axes[0].set_ylabel('Bow thruster angle [deg]')
axes[0].set_ylabel('Port thruster angle [deg]')
axes[1].set_ylabel('Starboard thruster angle [deg]')

for axn,ax in enumerate(axes):
    for i in range(len(methods)):
        t = time_astern[i]
        if axn == 0:
            relevant_data = aport[i]
        else:
            relevant_data = astar[i]

        ax.plot(t,relevant_data, color=colors[i],label=labels[i],alpha=0.9)
        plot_zeros(ax,relevant_data)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.yaxis.grid(color='grey', linestyle='--', alpha=0.5)
    
    if not PLOT_SETPOINT_AREAS:
        lab = 'Set point changes' if axn == 0 and i == 0 else '' # TODO get labels to work
        for c in setpointx:
            ax.axvline(c, linestyle='--', color='black', alpha = 0.8, label=lab)
    else:
        clrs = ['grey','white']
        clrctr = 0
        for i in range(len(setpnt_areas) - 1):
            ax.axvspan(setpnt_areas[i],setpnt_areas[i+1], facecolor=clrs[clrctr], alpha=0.1)
            clrctr = int(1 - clrctr)
    
    # Print reference lines
# ax.plot(setpointx,targets,'--',color=colors[3], label = 'Reference' if axn == 0 else None)
axes[0].legend(loc='best', facecolor='#FAD7A0', framealpha=0.3).set_draggable(True)
f0.tight_layout(pad=0.4)


'''
### THRUST
'''
f, axes = plt.subplots(3,1,figsize=(12,9),sharex = True)
plt.xlabel('Time [s]')
axes[0].set_ylabel('Bow thrust [% of max]')
axes[1].set_ylabel('Port thrust [% of max]')
axes[2].set_ylabel('Starboard thrust [% of max]')

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
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.yaxis.grid(color='grey', linestyle='--', alpha=0.5)
    
    if not PLOT_SETPOINT_AREAS:
        lab = 'Set point changes' if axn == 0 and i == 0 else '' # TODO get labels to work
        for c in setpointx:
            ax.axvline(c, linestyle='--', color='black', alpha = 0.8, label=lab)
    else:
        clrs = ['grey','white']
        clrctr = 0
        for i in range(len(setpnt_areas) - 1):
            ax.axvspan(setpnt_areas[i],setpnt_areas[i+1], facecolor=clrs[clrctr], alpha=0.1)
            clrctr = int(1 - clrctr)

    # Print reference lines
# ax.plot(setpointx,targets,'--',color=colors[3], label = 'Reference' if axn == 0 else None)
axes[0].legend(loc='best', facecolor='#FAD7A0', framealpha=0.3).set_draggable(True)
f.tight_layout()

'''
### POWER Equivalents
'''

# bow doesnt work under 3%
#for i in range(len(methods)):
#	nbow[i][np.abs(nbow[i]) < 3.0] = 0.0
    
rps_max = {'bow': 33.0, 'stern': 11.0}
diameters = {'bow': 0.06, 'stern': 0.15}
KQ_0 = {'bow': 0.035 * 0.001518 / 0.0027, 'stern':0.028}
rho = 1025.0

def power(n,which):
	''' n comes as -100% to 100%, and must be divided on 100 for multiplication with rps to give fraction'''
	return np.sign(n) * KQ_0[which] * 2 * np.pi * rho * diameters[which]**5 * (n /100.0 * rps_max[which])**3

powers = [[[],[],[]], [[],[],[]], [[],[],[]]] # Contains power timeseries on [method0: [bow,port,star], method1: ...] for each method
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

f, axes = plt.subplots(3,1,figsize=(12,9),sharex = True)
plt.xlabel('Time [s]')
axes[0].set_ylabel('$P_{bow}$ equivalent')
axes[1].set_ylabel('$P_{port}$ equivalent')
axes[2].set_ylabel('$P_{starboard}$ equivalent')

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
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.yaxis.grid(color='grey', linestyle='--', alpha=0.5)
    
    if not PLOT_SETPOINT_AREAS:
        lab = 'Set point changes' if axn == 0 and i == 0 else '' # TODO get labels to work
        for c in setpointx:
            ax.axvline(c, linestyle='--', color='black', alpha = 0.8, label=lab)
    else:
        clrs = ['grey','white']
        clrctr = 0
        for i in range(len(setpnt_areas) - 1):
            ax.axvspan(setpnt_areas[i],setpnt_areas[i+1], facecolor=clrs[clrctr], alpha=0.1)
            clrctr = int(1 - clrctr)

axes[0].legend(loc='best', facecolor='#FAD7A0', framealpha=0.3).set_draggable(True)
f.tight_layout()

cumsums = [[[],[],[]], [[],[],[]], [[],[],[]]]
for i in range(len(methods)):
    for j in range(3):
        cumsums[i][j] = np.cumsum(powers[i][j])

f, axes = plt.subplots(4,1,figsize=(12,9),sharex = True)
plt.xlabel('Time [s]')
axes[0].set_ylabel('$P_{bow}$ cumulative')
axes[1].set_ylabel('$P_{port}$ cumulative')
axes[2].set_ylabel('$P_{starboard}$ cumulative')
axes[3].set_ylabel('$P_{total}$ cumulative')

for axn,ax in enumerate(axes):
    for i in range(len(methods)):
        if axn == 0:
            t = time_bow[i][:-1]
            relevant_data = cumsums[i][axn]
        elif axn in [1,2]:
            t = time_nstern[i][:-1]
            relevant_data = cumsums[i][axn]
        else:
        	vals = [cumsums[i][axn_loc] for axn_loc in range(3)]
        	shortest = min([a for val in vals for a in val.shape ])
        	new_vals = [cumsums[i][axn_loc][0:shortest] for axn_loc in range(3)]
        	s = sum(new_vals)
        	relevant_data = s

        	t = time_nstern[i][:shortest]

        ax.plot(t,relevant_data, color=colors[i],label=labels[i],alpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.yaxis.grid(color='grey', linestyle='--', alpha=0.5)
    
    if not PLOT_SETPOINT_AREAS:
        lab = 'Set point changes' if axn == 0 and i == 0 else '' # TODO get labels to work
        for c in setpointx:
            ax.axvline(c, linestyle='--', color='black', alpha = 0.8, label=lab)
    else:
        clrs = ['grey','white']
        clrctr = 0
        for i in range(len(setpnt_areas) - 1):
            ax.axvspan(setpnt_areas[i],setpnt_areas[i+1], facecolor=clrs[clrctr], alpha=0.1)
            clrctr = int(1 - clrctr)

axes[0].legend(loc='best', facecolor='#FAD7A0', framealpha=0.3).set_draggable(True)
f.tight_layout()

final_powers = [[[],[],[]], [[],[],[]], [[],[],[]]]

for i,p_method in enumerate(powers):
    # all p_method comes as power timeseries [p_bow, p_port, p_star]
    for j,p_thruster in enumerate(p_method):
        # p_thruster is now power time series for a given thruster, for the given method i
        final_powers[i][j] = sum(p_thruster)

for i in range(len(methods)):
    print('Method {}'.format(methods[i]))
    print(sum(final_powers[i]))

f = plt.figure()

bars = []
for i in range(3): # all three thrusters
    for j in range(len(methods)):
        bars.append(final_powers[j][i][0]) # collects [p_bow_dnvgl, p_bow_qp, p_bow_rl, p_port_dnvgl, ..., p_star_rl]

for i in range(len(methods)):
    bars.append(sum(final_powers[i])[0])   

bar_positions = [0,1,2,4,5,6,9,10,11,14,15,16]
tick_pos = [1,5,10,15]
txts = ['Bow thruster', 'Port thruster', 'Starboard thruster', 'Total']
bar_colors = [colors[0], colors[1], colors[2]]*4
elements = plt.bar(bar_positions, bars, color = bar_colors)
plt.xticks(tick_pos, txts)

ax = plt.gca()
ax.legend(elements[0:3], ['DNVGL\'s pseudoinverse', 'Quadratic Programming', 'Reinforcement Learning'],loc='best', facecolor='#FAD7A0', framealpha=0.3).set_draggable(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

f.tight_layout()
plt.show()
