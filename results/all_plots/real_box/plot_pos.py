import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as patches
from matplotlib.markers import MarkerStyle
import matplotlib.lines as mlines
from matplotlib.patches import Polygon
import copy

import sys
import os
import platform

current_dir = os.path.dirname(os.path.abspath(__file__))

if platform.system().lower() == 'linux':
    parent_dir = current_dir.rsplit('/',1)[0] # split current dir on last '/', which gives the parent dir in Ubuntu
elif platform.system().lower() == 'windows':
    parent_dir = current_dir.rsplit('\\',1)[0] # split current dir on last '/', which gives the parent dir in Windows

sys.path.append(parent_dir)

from common import methods, labels, colors, set_params, get_secondly_averages, absolute_error, IAE, plot_gray_areas, SMALL_SQUARE, RECTANGLE, runningMean, SMALL_RECTANGLE


# This method was crap - probably due to the integral being way to fast for heading... Should have been set to 0
# methods = ['RLI']
# labels = ['RLI']
# colors = ['orange']
# path_ref = 'bagfile__RLI_reference_filter_state_desired.csv'

methods = ['pseudo','RL']
labels = ['IPI','RL']
colors = [colors[0],colors[2]]
path_ref = 'bagfile__RL_reference_filter_state_desired.csv'

# methods = ['pseudo']
# labels = ['IPI']
# colors = [colors[0]]
# path_ref = 'bagfile__pseudo_reference_filter_state_desired.csv'

LAMBDA = 1.0 # / 20.0 # set to one if using model sized data
TIMESCALE = (1 / (LAMBDA**0.5))
REFERENCE_ERROR = 1.00

set_params() # sets global plot parameters

'''
Positional data
'''
path = 'bagfile__{}_observer_eta_ned.csv' # General path to eta

ref_data = np.genfromtxt(path_ref,delimiter=',')
ref_north = (ref_data[1:,1:2] - ref_data[1:,1:2][0,0] )  * (1/LAMBDA)
ref_east = (ref_data[1:,2:3] - ref_data[1:,2:3][0,0]) * (1/LAMBDA) * REFERENCE_ERROR
ref_yaw = ref_data[1:,3:4] - ref_data[1:,3:4][0,0]
ref_time = ref_data[1:,-1:] * TIMESCALE
refdata = [ref_north, ref_east, ref_yaw]
n_0, e_0, p_0 = ref_north, ref_east, ref_yaw

north, east, psi, time = [np.zeros((1,1))]*len(methods), [np.zeros((1,1))]*len(methods), [np.zeros((1,1))]*len(methods), [np.zeros((1,1))]*len(methods)
roll = [np.zeros((1,1))]*len(methods)
ALL_POS_DATA = []
for i in range(len(methods)):
    fpath = path.format(methods[i])
    posdata = np.genfromtxt(fpath,delimiter=',')
    # 0th elements are NaN due to column text
    north[i] = ( posdata[1:,1:2] - ref_data[1:,1:2][0,0]) * (1/LAMBDA)
    east[i] = ( posdata[1:,2:3] - ref_data[1:,2:3][0,0]) * (1/LAMBDA)
    psi[i] = posdata[1:,6:7] - ref_data[1:,3:4][0,0]
    time[i] = posdata[1:,7:] * TIMESCALE
    ALL_POS_DATA.append([north[i], east[i], psi[i], time[i]] )

    roll[i] = posdata[1:,4:5]

N = 50 # 300 pnts is ish 15 seconds of observer messages, but gives too early reactions. Neeed to filter some of the noise from the roll motions!
for i in range(len(methods)):
    north[i] = runningMean(north[i],N).reshape(north[i].shape)
    east[i] = runningMean(east[i],N).reshape(east[i].shape)
    psi[i] = runningMean(psi[i],N).reshape(psi[i].shape)
    ALL_POS_DATA[i] = [north[i], east[i], psi[i], time[i]]

incr = 5.0 * (1/LAMBDA)
# Points for the different box test square. These are only the coords and not the changes relative to eachother. Very first elements are nan
box_e = [e_0[1,0],  e_0[1,0],           e_0[1,0] - incr * REFERENCE_ERROR,  e_0[1,0] - incr* REFERENCE_ERROR,  e_0[1,0] - incr*REFERENCE_ERROR,     e_0[1,0]]
box_n = [n_0[1,0],  n_0[1,0] + incr,    n_0[1,0] + incr,  n_0[1,0] + incr,  n_0[1,0],           n_0[1,0]]
box_p = [p_0[1,0],  p_0[1,0],           p_0[1,0],        p_0[1,0] - 45,     p_0[1,0] - 45,      p_0[1,0]]

# setpoint_times = np.hstack( ([0], np.array([10, 80, 150, 190, 270, 350])+9) ) # from before modifications to csv-files
setpoint_times = (np.array([0, 10, 80, 150, 190, 270, 350]) * TIMESCALE).tolist()

if False:
    f, ax = plt.subplots(1,1,figsize=RECTANGLE,sharex = True)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Roll [deg]')
    for i in range(len(methods)):
        plt.plot(time[i],roll[i], color = colors[i], label=labels[i], zorder=10)
    f.tight_layout()

'''
### NEDPOS
'''
f = plt.figure(figsize=SMALL_SQUARE,dpi=100)
ax = plt.gca()
ax.scatter(box_e,box_n,color = 'black',marker='8',s = 50,label='Set points')

ax.set_xlim((east[0][0,0] - 8.5)*1/LAMBDA, (east[0][0,0] + 2.5)*1/LAMBDA)
ax.set_ylim((north[0][0,0] - 2.5)*1/LAMBDA,  (north[0][0,0] + 8.5)*1/LAMBDA)

for i in range(len(methods)):
    e, n = east[i], north[i]
    plt.plot(e,n,color = colors[i], label=labels[i], zorder=10)

marker_style = 2 # 0 if small triangle, 1 is full sized model. Everything else drops plotting the vessel

if marker_style == 0:
    if len(methods) == 1:
        nth = 200
        corners = {(box_e[0], box_n[0], box_p[0]) : 0, (box_e[1],box_n[1], box_p[1]):0, (box_e[2],box_n[2],box_p[2]):0,(box_e[3],box_n[3],box_p[3]):0, (box_e[4],box_n[4],box_p[4]):0}
        for i, (e,n,p) in enumerate(zip(east[0], north[0], psi[0])):
            skip = False
            if i % nth == 0 or i == 0:
                for j, tup in enumerate(corners):
                    if np.sqrt((e-tup[0])**2 + (n-tup[1])**2 ) < (0.5 * (1/LAMBDA)) and np.abs(p - tup[2]) < 10.0:
                        if j == 2 or j == 3:
                            if corners[tup] < 1:
                                corners[tup] +=1
                            else:
                                skip = True
                        else:
                            if corners[tup] < 1:
                                corners[tup] +=1
                            else:
                                skip = True

                if not skip:
                    m = MarkerStyle("^")
                    m._transform.scale(0.6, 1)
                    m._transform.rotate_deg(-(p - ref_yaw[0,0]))
                    plt.scatter(e, n, s=225, marker = m, color = 'grey', linewidths = 1, edgecolors = 'black', alpha=0.8, zorder=20)

elif marker_style == 1:
    if len(methods) == 1:
        from matplotlib import transforms

        #            [start,    onway northeast,    northeast,  onway northwest,    northwest,  nw with rot,    onway southwest, sw,  onway back]
        draw_times = (np.array([6,        21,                 73.75,         105,                140,        162,            210,             252, 293])*TIMESCALE).tolist() # used with running mean
        # draw_times = [7.5,      23,                 75,         105,                140,        163,            210,             252, 294] # used with no running mean
        time_dict = { i : 0 for i in draw_times}
        data_time = time[i]
        e, n, p = east[i], north[i], psi[i]
        ts = ax.transData

        for t in range(len(data_time)):
            for t_ref in draw_times:
                if data_time[t] > t_ref and time_dict[t_ref] == 0:
                    time_dict[t_ref] += 1
                    e_cg, n_cg, psi_cg = float(e[t]), float(n[t]), float(p[t]) # represents position of CG and rotation around it in tthe NED frame
                    L1 = 3.0*(1/LAMBDA); L2 = 2.2*(1/LAMBDA); W = 0.7*(1/LAMBDA); L_cg = 1.65*(1/LAMBDA) # full length, lengt from stern to bow-curvature, width
                    e_skew = -W / 2; n_skew = -L_cg # scew iof polygon vertices away from cg

                    vertices = np.array([[e_cg + e_skew, n_cg + n_skew],
                                         [e_cg + e_skew, n_cg + n_skew + L2],
                                         [e_cg + e_skew + W/2, n_cg + n_skew + L1],
                                         [e_cg + e_skew + W, n_cg + n_skew + L2],
                                         [e_cg + e_skew + W, n_cg + n_skew]])

                    # transform polygon to rotate around CG
                    coords = ts.transform([e_cg, n_cg])
                    rotation = transforms.Affine2D().rotate_deg_around(coords[0], coords[1], -(psi_cg - ref_yaw[0,0]))
                    trans = ts + rotation
                    poly = Polygon(xy=vertices, closed=True, facecolor='grey', edgecolor = 'black',alpha= 0.6, transform = trans,zorder=0)
                    ax.add_patch(poly)
                    plt.draw()
                    ts = ax.transData
                    
if marker_style in [0,1]:
    ax.plot([], [], color='grey', marker='^', linestyle='None', markersize=10, markeredgewidth=1,markeredgecolor = 'black', alpha=0.6,label='Vessel (to scale)')

ax.plot(ref_east, ref_north, '--', color='black',label='Reference')
ax.set_xlabel('East [m]')
ax.set_ylabel('North [m]')
ax.legend(loc='best').set_draggable(True)
f.tight_layout()

# plt.savefig('realBox_pos_ned.pdf')

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
    
    # Plot reference lines
    ax.plot(ref_time, refdata[axn], '--',color='black', label = 'Reference' if axn == 0 else None)
    plot_gray_areas(ax, setpoint_times)

axes[0].legend(loc='best').set_draggable(True)
f0.tight_layout()

'''
### Integral Absolute Error : int_0^t  sqrt ( error^2 ) dt
    - compare position to reference filter
    - gather data from ref_filter/state_desired and observer/eta/ned, which are very different, so take their averages per second.
    - TODO a mistake done during the thesis was that the NED-frame coordinates were used, while eta should have been transformed to body-frame errors first tbh
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
IAES_N = [] 
IAES_E = [] 
IAES_Y = [] 
times = (np.array(ref_data_averages[0][0]) - 1.0).tolist()
for i in range(len(methods)):
    integrals, cumsums = IAE(etas[i] / np.array([5.*1/LAMBDA,5.*1/LAMBDA,25.]), refs / np.array([5.*1/LAMBDA,5.*1/LAMBDA,25.]), times)
    IAES.append(cumsums)
    n_vals = etas[i][:,0]
    r_vals_n = refs[:,0]
    e_vals = etas[i][:,1]
    r_vals_e = refs[:,1]
    y_vals = etas[i][:,2]
    r_vals_y = refs[:,2]
    
    _, cumsums_N = IAE( n_vals.reshape(n_vals.shape[0],1) / np.array([5.*1/LAMBDA]), r_vals_n.reshape(r_vals_n.shape[0],1) / np.array([5.*1/LAMBDA]), times)
    IAES_N.append(cumsums_N)
    _, cumsums_E = IAE( e_vals.reshape(e_vals.shape[0],1) / np.array([5.*1/LAMBDA]), r_vals_e.reshape(r_vals_e.shape[0],1) / np.array([5.*1/LAMBDA]), times)
    IAES_E.append(cumsums_E)
    _, cumsums_Y = IAE( y_vals.reshape(y_vals.shape[0],1) / np.array([25]), r_vals_y.reshape(r_vals_y.shape[0],1) / np.array([25]), times)
    IAES_Y.append(cumsums_Y)
    ax.plot(times, IAES[i], color=colors[i], label=labels[i])
    ax.plot(times, IAES_N[i],'--', color=colors[i], label=str( labels[i] + ' surge contribution' ), alpha=1.0) # TODO this is wrong, as eta from NED-frame was actually used. But since ReVolt heads mainly towards North, it is approximately the same
    ax.plot(times, IAES_E[i],':', color=colors[i], label=str( labels[i] + ' sway contribution' ), alpha=1.0)
    ax.plot(times, IAES_Y[i],'-.', color=colors[i], label=str( labels[i] + ' yaw contribution' ), alpha=1.0)

# Gray areas
plot_gray_areas(ax,areas = setpoint_times)

ax.legend(loc='best').set_draggable(True)
ax.set_ylabel('IAE [-]')
ax.set_xlabel('Time [s]')

for i in range(len(methods)):
    val = IAES[i][-1] # extract IAE at last timestep
    x_coord = setpoint_times[-1] + 0.5
    txt = '{:.2f}'.format(val)
    moveif = {'IPI':-0.00*val, 'QP': 0.0* val, 'RL': 0.00 * val, 'RLI': 0.0 * val}
    activation = 1.0
    ax.annotate(txt, (x_coord, 0.99 * val + (activation * moveif[labels[i]])),color=colors[i], weight='bold', size=9)

f0.tight_layout()
    
print('IAES')
for i in range(len(methods)): print(methods[i], ':', IAES[i][-1])

plt.show()
