import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
from common import methods, labels, colors, set_params
#plt.gca().spines['top'].set_visible(False)
# gridspec.GridSpec(3,3)
# plt.subplot2grid((2,3),(0,1)); plt.subplot2grid((2,3),(1,0),colspan=3)

save = False
set_params()

'''
Positional data
'''
path = 'bagfile__{}_observer_eta_ned.csv' # General path to eta
path_ref = 'bagfile__reference_filter_state_desired_new.csv'

north, east, psi, time = [np.zeros((1,1))]*len(methods), [np.zeros((1,1))]*len(methods), [np.zeros((1,1))]*len(methods), [np.zeros((1,1))]*len(methods)
data = []
for i in range(len(methods)):
    fpath = path.format(methods[i])
    posdata = np.genfromtxt(fpath,delimiter=',')
    # 0th elements are nan for some reason

    north[i] = posdata[1:,1:2]
    if False and methods[i] == 'RL':
        north[i] = posdata[1:,1:2] + 0.1 * np.ones_like(posdata[1:,1:2])

    east[i] = posdata[1:,2:3]
    if False and methods[i] == 'QP':
        east[i] = posdata[1:,2:3] - 0.8 * np.ones_like(posdata[1:,2:3])
    elif False and methods[i] == 'RL':
        east[i] = posdata[1:,2:3] + 0.45 * np.ones_like(posdata[1:,2:3])

    psi[i] = posdata[1:,6:7]
    time[i] = posdata[1:,7:]
    data.append([north[i], east[i], psi[i], time[i]] )


refdata = np.genfromtxt(path_ref,delimiter=',')
ref_north = refdata[1:,1:2]
ref_east = refdata[1:,2:3]
ref_yaw = refdata[1:,3:4]
ref_time = refdata[1:,-1:]
if False:
    ref_time = ref_time - 2 *np.ones_like(ref_time)
    ref_time[ref_time < 0] = 0.0
    ref_time = ref_time[1:]
    ref_time = np.vstack( (ref_time,np.array([240])))
refdata = [ref_north, ref_east, ref_yaw]

n_0, e_0, p_0 = north[0], east[0], psi[0]
# Points for the different box test square. These are only the coords and not the changes relative to eachother. Very first elements are nan
box_n = [n_0[1,0],  n_0[1,0] + 5, n_0[1,0] + 5.0,  n_0[1,0] + 5.0,  n_0[1,0],           n_0[1,0]]
box_e = [e_0[1,0],  e_0[1,0],     e_0[1,0] - 5.0,  e_0[1,0] - 5.0,  e_0[1,0] - 5.0,     e_0[1,0]]
box_p = [p_0[1,0],  p_0[1,0],     p_0[1,0],        p_0[1,0] - 45,   p_0[1,0] - 45,      p_0[1,0]]

setpoint_change_times                               = [0,10-2,60-2,120-2,140-2,190-2]
setpointx                                           = [0,  8,  8,  58,  58, 108, 108, 138, 138, 188, 188, 238]
box_coords_over_time = np.array([
    (np.array([box_n[0]]*(len(setpointx))) + np.array([0,  0,  5,   5,   5,   5,   5,   5,   0,   0,   0,   0])).tolist(),
    (np.array([box_e[0]]*(len(setpointx))) + np.array([0,  0,  0,   0,  -5,  -5,  -5,  -5,  -5,  -5,   0,   0])).tolist(),
    (np.array([0]*(len(setpointx)))        + np.array([0,  0,  0,   0,   0,   0, -45, -45, -45, -45,   0,   0])).tolist()
])
'''
### NEDPOS
'''
f = plt.figure(figsize=(12,9))
ax = plt.gca()
ax.scatter(box_e,box_n,color = colors[3],marker='8',s=50,label='Set points')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(color='grey', linestyle='--', alpha=0.5)

for i in range(len(methods)):
    e, n = east[i], north[i]
    plt.plot(e,n,color = colors[i], label=labels[i])

ax.set_xlabel('East [m from NED frame origin]')
ax.set_ylabel('North [m from NED frame origin]')
ax.legend(loc='best', facecolor='#FAD7A0', framealpha=0.3).set_draggable(True)
f.tight_layout()

'''
### North and East plots
'''
f0, axes = plt.subplots(3,1,figsize=(12,9),sharex = True)
plt.xlabel('Time [s]')
axes[0].set_ylabel('North [m]')
axes[1].set_ylabel('East [m]')
axes[2].set_ylabel('Yaw [deg]')

for axn,ax in enumerate(axes):

    for i in range(len(methods)):
        local_data = north[i], east[i], psi[i], time[i]
        t = local_data[3]
        relevant_data = local_data[axn]
        ax.plot(t,relevant_data,color=colors[i],label=labels[i])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(color='grey', linestyle='--', alpha=0.5)
    
    # Print reference lines
    targets = box_coords_over_time[axn]
    # ax.plot(setpointx,targets,'--',color=colors[3], label = 'Reference' if axn == 0 else None)
    ax.plot(ref_time, refdata[axn], '--',color=colors[3], label = 'Reference' if axn == 0 else None)

axes[0].legend(loc='best', facecolor='#FAD7A0', framealpha=0.3).set_draggable(True)
f0.tight_layout()


plt.show()
