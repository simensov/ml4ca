import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.markers import MarkerStyle
import matplotlib.patheffects as PathEffects
from matplotlib.patches import Polygon

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

headings = [-158,-135,-113,-90,-68,-45,-23,0,23,45,68,90,113,135,158,180]
labels = ['${}^\\circ$'.format(val) for val in headings]
methods = ['pseudo{}deg'.format(val) for val in headings] # + ['RLintegral{}deg'.format(val) for val in headings]
colors[3] = colors[0]
# methods = ['RLintegral{}deg'.format(val) for val in headings]
# colors[3] = 'orange'



set_params()

def plot_zeros(ax,data):
    ax.plot(t,[0]*len(data),'-', linewidth=0.5, color='grey', alpha=0.5,label=None)

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

setpnt_areas =[0]

'''
### POWER AND THRUST Equivalents
'''
    
rps_max     = {'bow': 33.0, 'stern': 11.0}
diameters   = {'bow': 0.06, 'stern': 0.15}
KQ_0        = {'bow': 0.02, 'stern': 0.036}
KT_0        = {'bow': 0.0009, 'stern': 0.00205}
rho         = 1025.0

'''
### THRUST
'''

def abs_thrust(n,which):
    ''' n comes as -100% to 100%. This calculation is done with the empirical formula T = K* n^2 using n as percentage,
     NOT T = rho * D^4 * K_T * n^2 using n as RPS'''
    return KT_0[which] * (n)**2

thrusts = [[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]]
max_thrusts = [abs_thrust(100,'bow'), abs_thrust(100,'stern'), abs_thrust(100,'stern')]

for i in range(len(methods)):
    tbow = time_bow[i]
    nb = nbow[i]
    pb = 0

    for j in range(len(tbow) - 1):
        thrusts[i][0].append(abs_thrust(nb[j] , 'bow'))

    taft = time_nstern[i]
    nprt = nport[i]
    nstr = nstar[i]

    for j in range(len(taft) - 1): # trapezoidal integration
        thrusts[i][1].append(abs_thrust(nprt[j] , 'stern'))
        thrusts[i][2].append(abs_thrust(nstr[j] , 'stern'))

'''
### THRUST CAPABILITY
'''

capabilities = [] # Will hold a number between 0-100 per method

for i in range(len(methods)):
    means = []
    for j in range(3): # three thrusters
        means.append(np.mean(thrusts[i][j]) / max_thrusts[j] )
    
    capabilities.append(np.mean(means))

thetas = wrap_angle( np.deg2rad(np.array(headings + [headings[0]]) + 180), deg=False) # skew 180 degrees because e.g. 0 deg means environment going straight north, but it should show as coming 180 deg on the vessel (from behind)
r = np.array(capabilities + [capabilities[0]]) * 100

f = plt.figure(figsize=SMALL_SQUARE)
ax = f.add_subplot(111, projection='polar')
ax.set_xticks(np.pi / 180. * np.linspace(180,  -180, len(headings)/2, endpoint=False))
ax.set_thetalim(-np.pi, np.pi)
ax.plot(thetas, r,color = colors[3] )

rlabels = ax.get_ymajorticklabels()
for label in rlabels:
    label.set_color('black')
    label.set_weight('bold')
    label.set_zorder(30)
    # label.set_path_effects([PathEffects.withStroke(linewidth=2, foreground=colors[3])])

# Draw vessel
m = MarkerStyle("^")
m._transform.scale(7.5*0.6, 7.5*1)
plt.scatter(0, 0, s=225, marker = m, color = 'grey', linewidths = 1, edgecolors = 'black', alpha=0.5, zorder=0)

ax.set_rmax(max(r)*1.1)
ax.set_rlabel_position(60)  # get radial labels away from plotted line
ax.grid(True)
label_position=ax.get_rlabel_position()
ax.text(np.radians(label_position), ax.get_rmax()*1.2 ,'% thrust',
        rotation=30,ha='center',va='center',color='black',weight='bold')

test =ax.text(np.pi/45, ax.get_rmax()*1.085 ,'External loads\' incoming angles',
        rotation=0,ha='left',va='center')


ax.set_theta_zero_location("N")  # theta = 0 pos (could be N, S, W, E etc.)
ax.set_theta_direction(-1)  # theta increasing clockwise
f = plt.gcf()
f.tight_layout()


'''
### POWER
'''

def power(n,which):
	''' n comes as -100% to 100%, and must be divided on 100 for multiplication with rps to give fraction of max'''
	return np.sign(n) * KQ_0[which] * 2 * np.pi * rho * diameters[which]**5 * (n / 100.0 * rps_max[which])**3

powers = [[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]]
max_powers = [power(100,'bow'), power(100,'stern'), power(100,'stern')]

for i in range(len(methods)):
    tbow = time_bow[i]
    nb = nbow[i]
    pb = 0

    for j in range(len(tbow) - 1):
        powers[i][0].append(power(nb[j] , 'bow'))

    taft = time_nstern[i]
    nprt = nport[i]
    nstr = nstar[i]

    for j in range(len(taft) - 1): # trapezoidal integration
        powers[i][1].append(power(nprt[j] , 'stern'))
        powers[i][2].append(power(nstr[j] , 'stern'))

# PLOT POWER OVER TIME
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

        ax.plot(t,relevant_data, label=labels[i],alpha=0.9),# color=colors[i],)
        plot_zeros(ax,relevant_data)
    
    plot_gray_areas(ax,areas=setpnt_areas)

# axes[0].legend(loc='best').set_draggable(True)
f.tight_layout()

'''
### POWER CAPABILITY
'''
if False:
    capabilities = [] # Will hold a number between 0-100 per method

    for i in range(len(methods)):
        means = []
        for j in range(3): # three thrusters
            means.append(np.mean(powers[i][j]) / max_powers[j] )
        
        capabilities.append(np.mean(means))

    thetas = wrap_angle( np.deg2rad(np.array(headings + [headings[0]]) + 180), deg=False) # skew 180 degrees because e.g. 0 deg means environment going straight north, but it should show as coming 180 deg on the vessel (from behind)
    r = np.array(capabilities + [capabilities[0]]) * 100

    ax = plt.subplot(111, projection='polar')
    ax.plot(thetas, r,color = colors[3] )
    ax.set_rmax(max(r)*1.1)
    ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)

    ax.set_theta_zero_location("N")  # theta = 0 at the top
    ax.set_theta_direction(-1)  # theta increasing clockwise


'''
### ENERGY
'''

# THESE ELEMENTS DOES NOT CONTAINT "REAL" POWER: THEY CONTAIN THE ELEMENTS/AREAS THAT ARE TO BE SUMMED UP FOR DISCRETE INTEGRATION IN CUMCUMS
work_elements = [[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]]
for i in range(len(methods)):
    tbow = time_bow[i]
    nb = nbow[i]
    pb = 0

    for j in range(len(tbow) - 1): # trapezoidal integration
        dt = tbow[j+1] - tbow[j]
        p_avg = (power(nb[j+1] , 'bow') + power(nb[j] , 'bow')) / 2 # average |n^3| for current time interval
        work_elements[i][0].append(p_avg * dt)

    taft = time_nstern[i]
    nprt = nport[i]
    nstr = nstar[i]

    for j in range(len(taft) - 1): # trapezoidal integration
        dt = taft[j+1] - taft[j]
        p_avg_port = (power(nprt[j+1] , 'stern') + power(nprt[j] , 'stern')) / 2
        p_avg_star = (power(nstr[j+1] , 'stern') + power(nstr[j] , 'stern')) / 2
        work_elements[i][1].append(p_avg_port * dt)
        work_elements[i][2].append(p_avg_star * dt)



cumsums = [[[]]*3]*len(methods)
cumsums = [[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]]

for i in range(len(methods)):
    for j in range(3):
        cumsums[i][j] = np.cumsum(work_elements[i][j])

f, axes = plt.subplots(4,1,figsize=SMALL_SQUARE,sharex = True)
plt.xlabel('Time [s]')
axes[0].set_ylabel('$W^*_{bow}$ [J]')
axes[1].set_ylabel('$W^*_{port}$ [J]')
axes[2].set_ylabel('$W^*_{star}$ [J]')
axes[3].set_ylabel('$W^*_{total}$ [J]')

total_energies = []

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

        ax.plot(t,relevant_data,label=labels[i],alpha=0.9) #color=colors[i],)

        # annotate the ending value
        val = relevant_data[-1] # extract final value
        x_coord = t[-1] + 1
        txt = '{:.2f}'.format(val)
        moveif = {'IPI' : -0.05 * val, 'QP': 0.1 * val, 'RL': -0.1 * val, 'RLI': 0.1 * val}
        activation = 1.0
        if ax == 0:
            moveif['RLI'] = val

        # ax.annotate(txt, (x_coord, 0.95 * val + (activation * 0)),size=9) # ,color=colors[i], 

        if axn == 3:
        	total_energies.append(val)
    
    plot_gray_areas(ax,setpnt_areas)

axes[3].annotate('Average total energy: {:.2f}'.format(np.mean(total_energies)), (t[0] + 1, np.mean(total_energies)), weight='bold')
# axes[0].legend(loc='best').set_draggable(True)
f.tight_layout()

# END RESULTS
final_powers = [[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]]

for i,p_method in enumerate(work_elements):
    # all p_method comes as power timeseries [p_bow, p_port, p_star]
    for j,p_thruster in enumerate(p_method):
        # p_thruster is now power time series for a given thruster, for the given method i
        final_powers[i][j] = sum(p_thruster)

for i in range(len(methods)):
    print('Method {}'.format(methods[i]))
    print(sum(final_powers[i]))


plt.show()