import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys

methods = ['pseudo','QP', 'RL']
labels  = ['DNVGL', 'QP', 'RL']
colors  = ['#3e9651', '#3969b1', '#cc2529', '#000000']

# print(plt.rcParams.keys()) # for all alternatives

def set_params():
    # plt.rcParams['axes.labelweight'] = 'bold'
    params = {
    'font.serif':           'Computer Modern Roman',
    'axes.labelsize':       12,
    'axes.labelweight':     'normal',
    'axes.spines.right':    False,
    'axes.spines.top':      False,
    'legend.fontsize':      10,
    'legend.framealpha':    0.5,
    'legend.facecolor':     '#FAD7A0',
    'xtick.labelsize':      10,
    'ytick.labelsize':      10,
    'text.usetex':          False,
    'figure.figsize':       [12, 9]
    }

    plt.rcParams.update(params)

def get_secondly_averages(time_data, data):
    current_second = 0
    ref, avgs, tavg = [], [], []
    for t, second in enumerate(time_data):
        if int(second) < (current_second + 1):
            ref.append(data[t])
        else:
            avgs.append( np.mean(ref) )
            ref = []
            tavg.append(int(second) + 0.5)
            current_second += 1

    return tavg, avgs

def absolute_error(vec1, vec2):
    return np.sqrt( (vec1 - vec2).T.dot(vec1 - vec2) )

def IAE(data1, data2, time):
    ''' Integrating data point differences over time using trapezoidal integration. Returns list of errors over time and list cummulative vals  
    data1, data2 are np arrays with etas along the rows, e.g data1 = [[n0,e0,y0], [n1,e1,y1], ... ]
    '''
    integrals = [0]
    cumsum = [0]

    for i, t in enumerate(time):
        if i < len(time) - 1:
            dt = time[i+1] - t
            int_val = (absolute_error(data1[i],data2[i]) + absolute_error(data1[i+1],data2[i+1])) / 2 * dt
            integrals.append(int_val)
            cumsum.append(cumsum[-1] + int_val) # adding from previous sum

    return integrals, cumsum # these should have the properties that sum(integrals) == cumsum[-1]

def plot_gray_areas(ax,areas = [0] + [11, 61, 111, 141, 191] + [240]):
    clrs = ['grey','white']
    clrctr = 0
    for i in range(len(areas) - 1):
        ax.axvspan(areas[i],areas[i+1], facecolor=clrs[clrctr], alpha=0.1)
        clrctr = int(1 - clrctr)
