
import numpy as np
import matplotlib.pyplot as plt
from specific.misc.plot_commons import set_params,colors

set_params()
    
def plot_policytest_data(args,data,env):
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(6,6), sharex=True)
    plt.xlabel('Time [s]')
    ax1.set_ylabel('$d$ [m]')
    ax2.set_ylabel('$\~{\psi}$ [deg]')
    ax3.set_ylabel('Reward, $r_t$')

    axes = [ax1, ax2, ax3]
    
    step_len = env.dt
    ep_no, ep_len = 0, 0
    first = True
    for episode in data:
        ep_len = len(episode)
        eucl_dists, headings, rewards, steps = [], [], [], [i*step_len for i in range(ep_len)]

        for step in episode:
            state, reward = step # state is [errorframe] + [velocities]
            eucl_dists.append( np.sqrt(state[0]**2 + state[1]**2) )
            headings.append(state[2] * 180 / np.pi)
            rewards.append(reward)
        
        print('Plotting ep', ep_no, 'in errorframe')
        ax1.plot(steps,         eucl_dists,     label='Run {}'.format(ep_no+1))
        ax2.plot(steps,         headings,       label='Run {}'.format(ep_no+1))
        ax3.plot(steps,         rewards,        label='Run {}'.format(ep_no+1))
        if first:
            rew_plot = ax3.plot(steps, [3.5]*len(steps), '--',color='black', label='Max reward',alpha=0.8,zorder=0)
            first=False
            
        ep_no += 1

    axes[0].legend(loc='best').set_draggable(True)
    ax3.legend(handles=rew_plot, loc='best').set_draggable(True)
    fig.tight_layout()

def plot_NED_data(args,data,env):
    '''
    :args:
        - data (list): a list of lists, giving all NED_pos, NED_ref lists of all time steps of all episodes
                        e.g. [ [([1,2,3], [1,1,1]), ([1,1,2], [1,1,1])] , [([1,2,3], [0,0,0]), ([1,0,0], [0,0,0])]]
    '''
    f1, (ax1, ax2, ax3) = plt.subplots(3, figsize=(6,6), sharex=True)
    plt.xlabel('Time [s]')
    ax1.set_ylabel('North pos [m]')
    ax2.set_ylabel('East pos [m]')
    ax3.set_ylabel('Yaw [deg]')

    f2, ax4 = plt.subplots(figsize=(6,6))
    ax4.set_xlabel('East position from setpoint in NED frame [m]')
    ax4.set_ylabel('North position from setpoint in NED frame [m]')

    axes = [ax1, ax2, ax3, ax4]

    step_len = env.dt
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

    circle = plt.Circle((0,0), radius = 0.2, color='red', alpha=0.5, fill=True,label='Set point')
    ax4.add_artist(circle)
    ax4.set_xlim(-5,5)
    ax4.set_ylim(-5,5)

    axes[0].legend(loc='best').set_draggable(True)
    ax4.legend(loc='best').set_draggable(True)
    f2.tight_layout()

def plot_action_data(args,data,env):

    fig, axes = plt.subplots(nrows=3,ncols=2,figsize=(6,6),sharex=True)
    axes[2,0].set_xlabel('Time [s]')
    axes[2,1].set_xlabel('Time [s]')
    idx = 0
    for r,row in enumerate(axes):
        for i, ax in enumerate(row):
            if i == 0:
                ax.set_ylabel('$n_{}$ [%]'.format(idx+1))
            if i == 1:
                ax.set_ylabel('$\\alpha_{}$ [deg]'.format(idx+1))
        idx += 1

    step_len = env.dt
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
        actions[:,-3:] *= 180 / np.pi # Scale to degrees
        idx = 0
        for row in axes:
            for i, ax in enumerate(row):
                if i == 0:
                    ax.plot(steps,actions[:,idx], label='Run {}'.format(ep_no+1))
                elif i == 1:
                    ax.plot(steps,actions[:,idx+3], label='Run {}'.format(ep_no+1))
            idx += 1

        ep_no += 1

    axes[0,1].legend(loc='best').set_draggable(True)
    fig.tight_layout()

if __name__ == '__main__':
    pass