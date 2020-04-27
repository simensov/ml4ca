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

def summed_gaussian_vs_2D(pltno = 0):
    from mathematics import gaussian, gaussian_like
    from mpl_toolkits.mplot3d import Axes3D

    if pltno == 0:
        x = np.linspace(-8,8,101)
        y = np.linspace(-8,8,101)
        yaw = np.linspace(-20, 20, 101) * np.pi/180
        X, Y = np.meshgrid(x,y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        Z1 = np.zeros((x.shape[0], y.shape[0]))
        Z2 = np.zeros((x.shape[0], y.shape[0]))

        x_vals = [gaussian_like([i]) for i in x]
        y_vals = [gaussian_like([i]) for i in y]
        for i, x_val in enumerate(x_vals):
            for j, y_val in enumerate(y_vals):
                    Z1[i,j] = x_val + y_val  


        if False: # was used initially for finding out how to avoid sparsity
        # for i in range(len(x)):
        #     for j in range(len(y)):
        #         r = np.sqrt( x[i]**2 + y[j]**2 )
        #         val1 = gaussian_like( [r], var=[(2)**2]) 
        #         val2 = (1-0.1*r) # + x_vals[i] + y_vals[j]
        #         Z2[i,j] = val1 + val2

        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1,projection='3d')
        ax2 = fig.add_subplot(2,2,2,projection='3d')
        ax3 = fig.add_subplot(2,2,3)
        ax4 = fig.add_subplot(2,2,4)
        plt1 = ax1.plot_surface(X, Y, Z1, cmap='viridis',linewidth=0)
        plt2 = ax2.plot_surface(X, Y, Z2, cmap='viridis',linewidth=0)
        plt3 = ax3.contour(X,Y,Z1)
        plt4 = ax4.contour(X,Y,Z2)

        for ax, plot in [(ax1,plt1) ,(ax2,plt2)]:
            ax.set_xlabel('$\~{x}$', size=12)
            ax.set_ylabel('$\~{y}$', size=12)
            ax.set_zlabel('$Reward$', size=12)
            # bar = plt.colorbar(plot, orientation='horizontal',pad=0.2)
            # bar.set_label('Reward ',  size = 12)    

    elif pltno == 1:
        max_r = np.sqrt(2 * 8**2)
        rads = np.linspace(-max_r, max_r, 101)
        yaws = np.linspace(-45, 45, 101)
        X, Y = np.meshgrid(rads,yaws)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        Z1 = np.zeros((rads.shape[0], yaws.shape[0]))
        Z2 = np.zeros((rads.shape[0], yaws.shape[0]))
        Z3 = np.zeros((rads.shape[0], yaws.shape[0]))
        Z4 = np.zeros((rads.shape[0], yaws.shape[0]))
        for i in range(len(rads)):
            for j in range(len(yaws)):
                measure = np.sqrt((rads[i])**2 + (yaws[j]/4)**2)
                yaw_pen = 0.5 # -np.abs(yaws[j])/45.0
                anti_sparity = max(-1.0, ( 1 - 0.1 * measure ))
                multivar = 2 * unitary_multivar_normal( [rads[i], yaws[j]], mu = [0,0], var=[1.0**2, 5.0**2])
                Z1[i,j] = multivar + anti_sparity + yaw_pen
                Z2[i,j] = multivar
                Z3[i,j] = anti_sparity
                Z4[i,j] = yaw_pen + 10 * yaws[i] # TODO WTFWTF IS UP WITH THE AXIS?!?! ENSURE LARGER YAW PEN

        f1 = plt.figure()
        ax1 = f1.add_subplot(2,2,1,projection='3d')
        ax2 = f1.add_subplot(2,2,2,projection='3d')
        ax3 = f1.add_subplot(2,2,3,projection='3d')
        ax4 = f1.add_subplot(2,2,4,projection='3d')
        total1 = ax1.plot_surface(X, Y, Z1, cmap='viridis',linewidth=0)
        total2 = ax2.plot_surface(X, Y, Z2, cmap='viridis',linewidth=0)
        
        if False:
            plt3 = ax3.contour(X,Y,Z1)
            plt4 = ax4.contour(X,Y,Z2)
        else:
            ax3.plot_surface(X, Y, Z3, cmap='viridis',linewidth=0)
            ax4.plot_surface(X, Y, Z4, cmap='viridis',linewidth=0)

        ax1.set_xlabel('$\~{r}$',size=12)
        ax1.set_ylabel('$\~{\psi}$',size=12)
        # ax1.set_zlabel('$Reward$',size=12)
        ax2.set_xlabel('$\~{r}$',size=12)
        ax2.set_ylabel('$\~{\psi}$',size=12)
        # ax2.set_zlabel('$Reward$',size=12)
        ax3.set_xlabel('$\~{r}$',size=12)
        ax3.set_ylabel('$\~{\psi}$',size=12)
        ax4.set_xlabel('$\~{r}$',size=12)
        ax4.set_ylabel('$\~{\psi}$',size=12)


    elif pltno == 2:
        x = np.linspace(-8,8,101)
        y = np.linspace(-8,8,101)
        yaw = np.linspace(-20, 20, 101) * np.pi/180
        X, Y = np.meshgrid(x,yaw)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        Z1 = np.zeros((x.shape[0], y.shape[0]))
        Z2 = np.zeros((x.shape[0], y.shape[0]))

        x_vals = [gaussian_like([i]) for i in x]
        y_vals = [gaussian_like([i]) for i in y]
        for i, x_val in enumerate(x_vals):
            for j, y_val in enumerate(y_vals):
                    Z1[i,j] = x_val + y_val  

        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1,projection='3d')
        ax2 = fig.add_subplot(2,2,2,projection='3d')
        ax3 = fig.add_subplot(2,2,3)
        ax4 = fig.add_subplot(2,2,4)
        plt1 = ax1.plot_surface(X, Y, Z1, cmap='viridis',linewidth=0)
        plt2 = ax2.plot_surface(X, Y, Z2, cmap='viridis',linewidth=0)
        plt3 = ax3.contour(X,Y,Z1)
        plt4 = ax4.contour(X,Y,Z2)

        for ax, plot in [(ax1,plt1) ,(ax2,plt2)]:
            ax.set_xlabel('$\~{x}$', size=12)
            ax.set_ylabel('$\~{y}$', size=12)
            ax.set_zlabel('$Reward$', size=12)
            # bar = plt.colorbar(plot, orientation='horizontal',pad=0.2)
            # bar.set_label('Reward ',  size = 12)    


    else:
        print('No valid pltno given')


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
            measure = np.sqrt( x[i]**2 + (y[j]/4)**2 )
            Z[i,j] = 2 * unitary_multivar_normal(x = [x[i],y[j]], mu=[0,0], var=[1**2, 5.7**2]) + max(0.0 , (1-0.1*measure)) + 0.5

    f1 = plt.figure()
    ax = f1.gca(projection='3d')
    total = ax.plot_surface(X, Y, Z, cmap='viridis',linewidth=0)
    ax.set_xlabel('$\~{r}$',size=12)
    ax.set_ylabel('$\~{\psi}$',size=12)
    ax.set_zlabel('$Reward$',size=12)
    bar = plt.colorbar(total)
    bar.set_label('Reward ', rotation = 90, size = 12)
    # plt.show()

if __name__ == '__main__':
    # plot_unitary()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plotno', '-n', type=int, default=0)
    args = parser.parse_args()


    summed_gaussian_vs_2D(plotno = args.plotno)
    plt.show()
    pass