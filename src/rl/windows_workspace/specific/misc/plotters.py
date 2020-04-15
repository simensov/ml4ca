
import numpy as np
import matplotlib.pyplot as plt

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def investigate_gaussian():
    from mathematics import gaussian_like

    vals =  np.array([-1,2,4])
    mean = np.zeros_like(vals)
    variance = np.square(np.array([1,1,1]))
    return gaussian_like(vals,mean,variance)

def view_distribution():
    import random 
    x,y = [], []
    for _ in range(2000):
        x.append((random.uniform(-1,1)))
        y.append((random.uniform(-1,1)))

    for i in range(len(x)):
        x[i] = (x[i] - min(x)) / (max(x) - min(x))

    # plt.scatter(x,y)
    plt.hist(x,len(x))
    plt.show()

def plot_gaussian(mean,var):
    from mathematics import gaussian, gaussian_like
    x = np.linspace(-50,50,201)
    y = [gaussian([i], mean=[mean], var=[var]) for i in x]
    # y = [gaussian_like(np.array([i]), mean=np.array([mean]), var=np.array([var])) for i in x]
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(x,y)
    plt.show()

    
def plot_policytest_data(args,data):
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    plt.xlabel('Time [s]')
    ax1.set_ylabel('$\sqrt{ {\~{x}}^2 + {\~{y}}^2 }$ [m]')
    ax2.set_ylabel('$\~{\psi}$ [deg]')
    ax3.set_ylabel('Immediate reward, $R_t$')

    fig, ax4 = plt.subplots()
    ax4.set_ylabel('Error surge')
    ax4.set_xlabel('Error sway')

    step_len = 0.001 if args.realtime else 0.1
    ep_no, ep_len = 0, 0
    for episode in data:
        ep_len = len(episode)
        eucl_dists, headings, rewards, steps = [], [], [], [i*step_len for i in range(ep_len)]
        pos = {'sway': [], 'surge' : []}

        for step in episode:
            state, reward = step
            eucl_dists.append( np.sqrt(state[0]**2 + state[1]**2) )
            pos['sway'].append(state[0])
            pos['surge'].append(state[1])
            headings.append(state[2] * 180 / np.pi)
            rewards.append(reward)
        
        print('Plotting ep', ep_no)
        ax1.plot(steps,         eucl_dists,     label='Run {}'.format(ep_no+1))
        ax2.plot(steps,         headings,       label='Run {}'.format(ep_no+1))
        ax3.plot(steps,         rewards,        label='Run {}'.format(ep_no+1))
        ax4.plot(pos['surge'],  pos['sway'],    label='Run {}'.format(ep_no+1))
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

    plt.show()

def plot_NED_data(args,data):
    '''
    :args:
        - data (list): a list of lists, giving all NED_pos, NED_ref lists of all time steps of all episodes
                        e.g. [ [([1,2,3], [1,1,1]), ([1,1,2], [1,1,1])] , [([1,2,3], [0,0,0]), ([1,0,0], [0,0,0])]]
    '''
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    plt.xlabel('Time [s]')
    ax1.set_ylabel('$\sqrt{ {\~{x}}^2 + {\~{y}}^2 }$ [m]')
    ax2.set_ylabel('$\~{\psi}$ [deg]')

    fig, ax4 = plt.subplots()
    ax4.set_ylabel('NED North')
    ax4.set_xlabel('NED East')

    step_len = 0.001 if args.realtime else 0.1
    ep_no, ep_len = 0, 0
    for episode in data:
        ep_len = len(episode)
        eucl_dists, headings, references, steps = [], [], [], [i*step_len for i in range(ep_len)]
        pos = {'sway': [], 'surge' : []}
        for step in episode:
            state, reference = step
            eucl_dists.append( np.sqrt(state[0]**2 + state[1]**2) )
            pos['sway'].append(state[0])
            pos['surge'].append(state[1])
            headings.append(state[2] * 180 / np.pi)
            references.append(reference)
        
        ax1.plot(steps, eucl_dists, label='Run {}'.format(ep_no+1))
        ax1.plot(steps, [0 for _ in range(ep_len)], 'g--',label='Goal')
        ax2.plot(steps, headings, label='Run {}'.format(ep_no+1))
        ax2.plot(steps, [0 for _ in range(ep_len)], 'g--')

        ax4.plot(pos['surge'],pos['sway'],label='Run {}'.format(ep_no+1))
        ep_no += 1

    ax1.grid(True)
    ax1.legend(loc='best').set_draggable(True)
    ax2.grid(True)
    ax2.legend(loc='best').set_draggable(True)
    ax4.grid(True)
    ax4.legend(loc='best').set_draggable(True)
    circle = plt.Circle((0, 0), radius=5, color='grey', fill=False)
    ax4.add_artist(circle)
    ax4.set_xlim((-5, 5))
    ax4.set_ylim((-5, 5))
    fig.tight_layout()

    plt.show()

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