import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys

# methods = ['pseudo','QP','QPold']
# labels  = ['DNVGL pseudoinverse', 'Quadratic Programming', 'QP Old']

methods = ['pseudo','QP','QPold']
labels  = ['DNVGL pseudoinverse', 'Quadratic Programming', 'Reinforcement Learning']
colors  = [            '#009e73',               '#0072b2',             '#CD5C5C',  '#000000']

# print(plt.rcParams.keys())

def set_params():
    # plt.rcParams['axes.labelweight'] = 'bold'
    params = {
    'font.serif': 'Computer Modern Roman',
    'axes.labelsize': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'text.usetex': False,
    'figure.figsize': [12, 9]
    }

    plt.rcParams.update(params)

# plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# plt.rc{'font', 'cm'}
# plt.rc('text', usetex=True)

        