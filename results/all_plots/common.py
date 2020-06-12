import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys

methods = ['pseudo','QP', 'RL']
labels  = ['IPI', 'QP', 'RL']
colors  = ['#3e9651', '#3969b1', '#cc2529', '#000000']
LARGE_SQUARE = (9,9)
SMALL_SQUARE = (6,6)
RECTANGLE = (12,9)
NARROW_RECTANGLE =(12,5)

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

def get_secondly_averages(time_data, data, absolute=False):
    current_second = 0
    ref, avgs, tavg = [], [], []
    for t, second in enumerate(time_data):
        if int(second) < (current_second + 1):
            if absolute:
                ref.append(np.abs(data[t]))
            else:
                ref.append(data[t])
        else:
            if ref: # non empty
                avgs.append( np.mean(ref) )
            else:
                avgs.append( avgs[-1])
                
            tavg.append(int(second) + 0.5)
            ref = []
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


def wrap_angle(angle, deg = False):
    ''' Wrap angle between -180 and 180 deg. deg == True means degrees, == False means radians
        Handles if angle is a vector or list of angles. In both cases, a (x,) shaped numpy array is returned '''
    ref = 180.0 if deg else np.pi

    if isinstance(angle,np.ndarray): # handle arrays
        ref = np.ones_like(angle) * ref
    elif isinstance(angle,list):
        angle = np.array(angle)
        ref = np.ones_like(angle) * ref

    return np.mod(angle + ref, 2*ref) - ref

def runningMean(x, N):
    ''' Returns a numpy array which contains as many datapoints as there is in x, 
    only each now representing the mean over the this -> N next datapoints.
    Could be quite slow for large arrays'''
    
    y = np.zeros((len(x),))
    for ctr in range(len(x)):
         y[ctr] = np.mean(x[ctr:(ctr+N)])
    return y