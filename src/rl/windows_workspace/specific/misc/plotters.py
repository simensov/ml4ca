
import numpy as np
import matplotlib.pyplot as plt

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs
    
def plot_policytest_data(args,data):
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    plt.xlabel('Time [s]')
    ax1.set_ylabel('$\sqrt{ {\~{x}}^2 + {\~{y}}^2 }$ [m]')
    ax2.set_ylabel('$\~{\psi}$ [deg]')
    ax3.set_ylabel('Reward, $R_t$')

    fig, ax4 = plt.subplots()
    ax4.set_xlabel('Error sway')
    ax4.set_ylabel('Error surge')
    
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

    ax1.grid(True)
    ax1.legend(loc='best').set_draggable(True)
    ax2.grid(True)
    ax2.legend(loc='best').set_draggable(True)
    ax3.grid(True)
    ax3.legend(loc='best').set_draggable(True)
    ax4.grid(True)
    ax4.legend(loc='best').set_draggable(True)
    circle = plt.Circle((0, 0), radius=5, color='grey', fill=False)
    ax4.add_artist(circle)
    ax4.set_xlim((-5, 5))
    ax4.set_ylim((-5, 5))
    fig.tight_layout()


def plot_NED_data(args,data):
    '''
    :args:
        - data (list): a list of lists, giving all NED_pos, NED_ref lists of all time steps of all episodes
                        e.g. [ [([1,2,3], [1,1,1]), ([1,1,2], [1,1,1])] , [([1,2,3], [0,0,0]), ([1,0,0], [0,0,0])]]
    '''
    f1, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    plt.xlabel('Time [s]')
    ax1.set_ylabel('North pos')
    ax2.set_ylabel('East pos')
    ax3.set_ylabel('Heading [rad]')

    f2, ax4 = plt.subplots()
    ax4.set_xlabel('NED East')
    ax4.set_ylabel('NED North')
    
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
            pos['heading'].append(state[1]);    ref['heading'].append(reference[2])
        
        print('Plotting ep', ep_no, 'in ned_data')
        ax1.plot(steps,         pos['north'],   steps, ref['north'],    label='Run {}'.format(ep_no+1))
        ax2.plot(steps,         pos['east'],    steps, ref['east'],     label='Run {}'.format(ep_no+1))
        ax3.plot(steps,         pos['heading'], steps, ref['heading'],  label='Run {}'.format(ep_no+1))
        ax4.plot(pos['east'],   pos['north'],   label='Run {}'.format(ep_no+1))
        ep_no += 1

    ax1.grid(True)
    ax1.legend(loc='best').set_draggable(True)
    ax2.grid(True)
    ax2.legend(loc='best').set_draggable(True)
    ax3.grid(True)
    ax3.legend(loc='best').set_draggable(True)
    ax4.grid(True)
    ax4.legend(loc='best').set_draggable(True)
    circle = plt.Circle((0,0), radius = 0.2, color='red', fill=True)
    ax4.add_artist(circle)
    ax4.set_xlim(-5,5)
    ax4.set_ylim(-5,5)
    f2.tight_layout()


def plot_action_data(args,data,env):

    fig, axes = plt.subplots(nrows=3,ncols=2,sharex=True)
    plt.xlabel('Time [s]')
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
            ax.grid(True)
            ax.legend(loc='best').set_draggable(True)
    
    fig.tight_layout()


if __name__ == '__main__':
    # grid()
    # summed_gaussian()
    # plot_gaussian(0,5.7**2) 
    # contour()
    # rews = [-1,5,6,3,8,-6,3,4,-2,-9]
    # print(np.array(rews))
    # print(reward_to_go(rews))
    # print(investigate_gaussian())
    # view_distribution()

    pass