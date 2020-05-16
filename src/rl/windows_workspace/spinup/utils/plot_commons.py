
import matplotlib.pyplot as plt 

colors  = ['#3e9651', '#3969b1', '#cc2529', '#000000']

def set_params():
    # plt.rcParams['axes.labelweight'] = 'bold'
    params = {
    'font.serif':        'Computer Modern Roman',
    'axes.labelsize':    13,
    'axes.labelweight':  'normal',
    'axes.spines.right':  False,
    'axes.spines.top':  False,
    'legend.fontsize':   10,
    'legend.framealpha': 0.5,
    'legend.facecolor':  '#FAD7A0',
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'text.usetex':       False,
    'figure.figsize':    [12, 9]
    }

    plt.rcParams.update(params)