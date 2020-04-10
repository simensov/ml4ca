
import numpy as np
import matplotlib.pyplot as plt

def grid():
    from scipy.stats import multivariate_normal
    from mpl_toolkits.mplot3d import Axes3D

    #Parameters to set
    mu_x, mu_y = 0, 0
    var_x, var_y = np.square(5), np.square(5)

    #Create grid and multivariate normal
    x = np.linspace(-10,10,500)
    y = np.linspace(-10,10,500)
    X, Y = np.meshgrid(x,y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    rv = multivariate_normal([mu_x, mu_y], [[var_x, 0], [0, var_y]])

    #Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Z = 1000 * rv.pdf(pos)
    Z[0,:] = -np.ones_like(Z[0,:])
    Z[-1,:] = -np.ones_like(Z[-1,:])
    Z[:,0] = -np.ones_like(Z[:,0])
    Z[:,-1] = -np.ones_like(Z[:,-1])
    print(Z, type(Z))
    test = ax.plot_surface(X, Y, Z, cmap='viridis',linewidth=0)
    ax.set_xlabel('$\~{x}$',size=12)
    ax.set_ylabel('$\~{y}$',size=12)
    ax.set_zlabel('$r_{err}$',size=12)
    bar = plt.colorbar(test)
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

    


if __name__ == '__main__':

    # grid()
    # contour()
    # rews = [-1,5,6,3,8,-6,3,4,-2,-9]
    # print(np.array(rews))
    # print(reward_to_go(rews))
    # print(investigate_gaussian())

    view_distribution()