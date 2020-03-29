
import numpy as np

def grid():
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal
    from mpl_toolkits.mplot3d import Axes3D

    #Parameters to set
    mu_x, mu_y = 0, 0
    var_x, var_y = 1, 1

    #Create grid and multivariate normal
    x = np.linspace(-10,10,500)
    y = np.linspace(-10,10,500)
    X, Y = np.meshgrid(x,y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    rv = multivariate_normal([mu_x, mu_y], [[var_x, 0], [0, var_y]])
    # TODO figure out how to scale this - multiply with a constant

    #Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
    ax.set_xlabel('$\~{x}$')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
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

if __name__ == '__main__':

    # grid()
    # contour()
    rews = [-1,5,6,3,8,-6,3,4,-2,-9]
    print(np.array(rews))
    print(reward_to_go(rews))