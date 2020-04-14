
import numpy as np
import matplotlib.pyplot as plt

def grid():
    from scipy.stats import multivariate_normal
    from mpl_toolkits.mplot3d import Axes3D

    #Parameters to set
    mu_x, mu_y = 0, 0
    var_x, var_y = np.square(1), np.square(5)

    #Create grid and multivariate normal
    x = np.linspace(-10,10,500)
    y = np.linspace(-30,30,500)
    X, Y = np.meshgrid(x,y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    rv = multivariate_normal([mu_x, mu_y], [[var_x, 0], [0, var_y]])

    #Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Z = 100* rv.pdf(pos)
    Z = Z - 0.6 # shift by constant
    test = ax.plot_surface(X, Y, Z, cmap='viridis',linewidth=0)
    ax.set_xlabel('$\~{x}$',size=12)
    ax.set_ylabel('$\~{y}$',size=12)
    ax.set_zlabel('$r_{err}$',size=12)
    bar = plt.colorbar(test)
    bar.set_label('Reward ', rotation = 90, size = 12)
    plt.show()

def summed_gaussian():
    from mathematics import gaussian
    from mpl_toolkits.mplot3d import Axes3D

    x = np.linspace(-8,8,101)
    y = np.linspace(-8,8,101)
    yaw = np.linspace(-20, 20, 101) * np.pi/180
    X, Y = np.meshgrid(x,y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    Z1 = np.zeros((x.shape[0], y.shape[0]))
    Z2 = np.zeros((x.shape[0], y.shape[0]))

    x_vals = [gaussian([i]) for i in x]
    y_vals = [gaussian([i]) for i in y]
    for i, x_val in enumerate(x_vals):
        for j, y_val in enumerate(y_vals):
                Z1[i,j] = x_val + y_val  

    for i in range(len(x)):
        for j in range(len(y)):
            r = np.sqrt( x[i]**2 + y[j]**2 )
            val = gaussian( [r] )
            Z2[i,j] = val

    fig=plt.figure()
    ax1=fig.add_subplot(1,2,1,projection='3d')
    ax2=fig.add_subplot(1,2,2,projection='3d')
    plt1 = ax1.plot_surface(X, Y, Z1, cmap='viridis',linewidth=0)
    plt2 = ax2.plot_surface(X, Y, Z2, cmap='viridis',linewidth=0)

    for ax, plot in [(ax1,plt1) ,(ax2,plt2)]:
        ax.set_xlabel('$\~{x}$', size=12)
        ax.set_ylabel('$\~{y}$', size=12)
        ax.set_zlabel('$r_{err}$', size=12)
        # bar = plt.colorbar(plot, orientation='horizontal',pad=0.2)
        # bar.set_label('Reward ',  size = 12)    

    max_r = np.sqrt(2 * 8**2)
    rads = np.linspace(-max_r, max_r, 101)
    yaws = np.linspace(-20, 20, 101)
    X, Y = np.meshgrid(rads,yaws)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    Z = np.zeros((x.shape[0], y.shape[0]))

    for rew in range(len(rads)):
        for y in range(len(yaws)):
            Z[rew,y] = gaussian([rads[rew]]) + gaussian([yaws[y]], var = [(180/np.pi*0.1)**2])

    f1 = plt.figure()
    ax = f1.gca(projection='3d')
    total = ax.plot_surface(X, Y, Z, cmap='viridis',linewidth=0)
    ax.set_xlabel('$\~{r}$',size=12)
    ax.set_ylabel('$\~{\psi}$',size=12)
    ax.set_zlabel('$Reward$',size=12)
    bar = plt.colorbar(total)
    bar.set_label('Reward ', rotation = 90, size = 12)

    plt.show()
    




def contour():
    import matplotlib.pyplot as plt
    from matplotlib import style
    style.use('fivethirtyeight')
    from scipy.stats import multivariate_normal

    #Parameters to set
    mu_x = 0
    var_x = 1
    mu_y = 0
    var_y = 1
    x = np.linspace(-10,10,500)
    y = np.linspace(-10,10,500)
    X,Y = np.meshgrid(x,y)
    pos = np.array([X.flatten(),Y.flatten()]).T
    rv = multivariate_normal([mu_x, mu_y], [[var_x, 0], [0, var_y]])
    fig = plt.figure(figsize=(10,10))
    ax0 = fig.add_subplot(111)
    ax0.contour(rv.pdf(pos).reshape(500,500))
    plt.show()

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

def unitary_multivar_normal(x,mu,var):
    ''' Take x, my, var lists - output scalar value '''

    size = len(x)
    x = np.array(x).reshape((len(x),1))
    mu = np.array(mu).reshape((len(mu),1))
    var = np.array(var).reshape((len(var),1))

    covar = np.zeros((size,size))
    for i in range(size):
        covar[i,i] = var[i]
    
    covar_inv = np.linalg.inv(covar)
    
    diff = x-mu

    return np.exp(-0.5 * (diff.T).dot(covar_inv).dot(diff)) # * 30 / np.sqrt( np.sqrt( (2*np.pi)**(size) * np.linalg.det(covar) ))

def plot_unitary():
    from mpl_toolkits.mplot3d import Axes3D

    x = np.linspace(-8,8,201)
    y = np.linspace(-20,20,201)
    X, Y = np.meshgrid(x,y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    Z = np.zeros((x.shape[0], y.shape[0]))

    for i in range(len(x)):
        for j in range(len(y)):
            Z[i,j] = 10.0 * unitary_multivar_normal(x = [x[i],y[j]], mu=[0,0], var=[1**2, 5**2])

    f1 = plt.figure()
    ax = f1.gca(projection='3d')
    total = ax.plot_surface(X, Y, Z, cmap='viridis',linewidth=0)
    ax.set_xlabel('$\~{r}$',size=12)
    ax.set_ylabel('$\~{\psi}$',size=12)
    ax.set_zlabel('$Reward$',size=12)
    bar = plt.colorbar(total)
    bar.set_label('Reward ', rotation = 90, size = 12)
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
    plot_unitary()

    pass