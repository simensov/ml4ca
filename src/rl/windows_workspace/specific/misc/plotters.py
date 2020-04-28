
import numpy as np
import matplotlib.pyplot as plt

'''
### FROM common in ROS
'''
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

    
def plot_policytest_data(args,data):
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    plt.xlabel('Time [s]')
    ax1.set_ylabel('$\sqrt{ {\~{x}}^2 + {\~{y}}^2 }$ [m]')
    ax2.set_ylabel('$\~{\psi}$ [deg]')
    ax3.set_ylabel('Reward, $R_t$')

    fig, ax4 = plt.subplots()
    ax4.set_xlabel('Error sway')
    ax4.set_ylabel('Error surge')

    axes = [ax1, ax2, ax3, ax4]
    
    step_len = 0.001 if args.realtime else 0.1
    ep_no, ep_len = 0, 0
    for episode in data:
        ep_len = len(episode)
        eucl_dists, headings, rewards, steps = [], [], [], [i*step_len for i in range(ep_len)]
        pos = {'sway': [], 'surge' : []}

        for step in episode:
            state, reward = step # state is [errorframe] + [velocities]
            eucl_dists.append( np.sqrt(state[0]**2 + state[1]**2) )
            pos['surge'].append(state[0])
            pos['sway'].append(state[1])
            headings.append(state[2] * 180 / np.pi)
            rewards.append(reward)
        
        print('Plotting ep', ep_no, 'in errorframe')
        ax1.plot(steps,         eucl_dists,     label='Run {}'.format(ep_no+1))
        ax2.plot(steps,         headings,       label='Run {}'.format(ep_no+1))
        ax3.plot(steps,         rewards,        label='Run {}'.format(ep_no+1))
        ax4.plot(pos['sway'],  pos['surge'],    label='Run {}'.format(ep_no+1))
        ep_no += 1

    for ax in axes:
        ax.grid(False)
        ax.yaxis.grid(color='grey', linestyle='--', alpha=0.5)
        # ax.legend(loc='best').set_draggable(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.xaxis.set_tick_params(which='both', labelbottom=True)

    circle = plt.Circle((0, 0), radius=5, color='grey', fill=False)
    ax4.add_artist(circle)
    ax4.set_xlim((-5, 5))
    ax4.set_ylim((-5, 5))

    axes[0].legend(loc='best', facecolor='#FAD7A0', framealpha=0.3).set_draggable(True)
    fig.tight_layout()


def plot_NED_data(args,data):
    '''
    :args:
        - data (list): a list of lists, giving all NED_pos, NED_ref lists of all time steps of all episodes
                        e.g. [ [([1,2,3], [1,1,1]), ([1,1,2], [1,1,1])] , [([1,2,3], [0,0,0]), ([1,0,0], [0,0,0])]]
    '''
    f1, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    plt.xlabel('Time [s]')
    ax1.set_ylabel('North pos [m]')
    ax2.set_ylabel('East pos [m]')
    ax3.set_ylabel('Heading [deg]')

    f2, ax4 = plt.subplots()
    ax4.set_xlabel('NED East')
    ax4.set_ylabel('NED North')

    axes = [ax1, ax2, ax3, ax4]
    
    step_len = 0.001 if args.realtime else 0.1
    ep_no, ep_len = 0, 0
    for episode in data:
        ep_len = len(episode)
        steps = [i*step_len for i in range(ep_len)]
        pos = {'north': [], 'east' : [], 'heading' : []}
        ref = {'north': [], 'east' : [], 'heading' : []}
        for step in episode:
            state, reference = step
            pos['north'].append(state[0]);      ref['north'].append(reference[0])
            pos['east'].append(state[1]);       ref['east'].append(reference[1])
            pos['heading'].append(state[2]*180/np.pi );    ref['heading'].append(reference[2] * 180 / np.pi)
        
        print('Plotting ep', ep_no, 'in ned_data')
        ax1.plot(steps,         pos['north'],   label='Run {}'.format(ep_no+1))
        ax2.plot(steps,         pos['east'],    label='Run {}'.format(ep_no+1))
        ax3.plot(steps,         pos['heading'], label='Run {}'.format(ep_no+1))
        ax4.plot(pos['east'],   pos['north'],   label='Run {}'.format(ep_no+1))
        ep_no += 1

    for ax in axes:
        ax.grid(False)
        ax.yaxis.grid(color='grey', linestyle='--', alpha=0.5)
        # ax.legend(loc='best').set_draggable(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.xaxis.set_tick_params(which='both', labelbottom=True)

    circle = plt.Circle((0,0), radius = 0.2, color='red', fill=True)
    ax4.add_artist(circle)
    ax4.set_xlim(-5,5)
    ax4.set_ylim(-5,5)

    axes[0].legend(loc='best', facecolor='#FAD7A0', framealpha=0.3).set_draggable(True)
    f2.tight_layout()


def plot_action_data(args,data,env):

    fig, axes = plt.subplots(nrows=3,ncols=2,sharex=True)
    axes[2,0].set_xlabel('Time [s]')
    axes[2,1].set_xlabel('Time [s]')
    idx = 0
    for r,row in enumerate(axes):
        for i, ax in enumerate(row):
            if i == 0:
                ax.set_ylabel('n{}'.format(idx+1))
            if i == 1:
                ax.set_ylabel('Angle {}'.format(idx+1))
        idx += 1

    step_len = 0.001 if args.realtime else 0.1
    ep_no, ep_len = 0, 0

    for episode in data:
        ep_len = len(episode)
        steps = [i*step_len for i in range(ep_len)]
        actions = None # Each action_vec will be stacked ontop of each other, giving [[act1], [act2], [act3]] etc.
        for action_vec in episode:
            if actions is None:
                actions = np.copy(action_vec)
            else:
                actions = np.vstack( (actions, action_vec) )
                
        print('Plotting ep', ep_no, 'in action_data')
        actions[:,-3:] = actions[:,-3:] * 180/np.pi
        idx = 0
        for row in axes:
            for i, ax in enumerate(row):
                if i == 0:
                    ax.plot(steps,actions[:,idx], label='Run {}'.format(ep_no+1))
                elif i == 1:
                    ax.plot(steps,actions[:,idx+3], label='Run {}'.format(ep_no+1))
            idx += 1

        ep_no += 1

    for row in axes:
        for ax in row:
            ax.grid(False)
            ax.yaxis.grid(color='grey', linestyle='--', alpha=0.5)
            # ax.legend(loc='best').set_draggable(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # ax.xaxis.set_tick_params(which='both', labelbottom=True)

    axes[0,0].legend(loc='best', facecolor='#FAD7A0', framealpha=0.3).set_draggable(True)
    # fig.tight_layout()


if __name__ == '__main__':
    pass