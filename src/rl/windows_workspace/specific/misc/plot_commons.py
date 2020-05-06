
import matplotlib.pyplot as plt 

def set_params():
    # plt.rcParams['axes.labelweight'] = 'bold'
    params = {
    'font.serif': 'Computer Modern Roman',
    'axes.labelsize': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'text.usetex': False
    # 'figure.figsize': [12, 9]
    }

    plt.rcParams.update(params)