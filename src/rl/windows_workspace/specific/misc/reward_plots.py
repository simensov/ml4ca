import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from plot_commons import set_params,colors
set_params()

FS = (6,6)

def summed_gaussian_vs_2D(pltno = 0):
    from mathematics import gaussian, gaussian_like
    from mpl_toolkits.mplot3d import Axes3D

    if pltno == 0:
        # Plots the summed gaussian on the left, 
        max_r  = np.sqrt(2 * 8**2)
        rads   = np.linspace(-max_r, max_r, 251)
        yaws   = np.linspace(-20, 20, 251)
        (X, Y) = np.meshgrid(rads,yaws)
        pos    = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        Z1 = np.zeros((rads.shape[0], yaws.shape[0]))
        Z2 = np.zeros((rads.shape[0], yaws.shape[0]))

        r_vals   = [gaussian_like([i], mean=[0], var=[1.0**2]) for i in rads]
        yaw_vals = [gaussian_like([i], mean=[0], var=[5.0**2]) for i in yaws]

        for i, r_val in enumerate(r_vals):
            for j, yaw_val in enumerate(yaw_vals):
                    Z1[i,j] = r_val + yaw_val
                    Z2[i,j] = unitary_multivar_normal( [rads[i], yaws[j]], mu = [0, 0], var=[1.0**2, 5.0**2])

        fig1 = plt.figure(figsize=FS)
        ax1 = fig1.add_subplot(1,1,1,projection='3d')
        plt1 = ax1.plot_surface(X, Y, Z1, cmap = cm.get_cmap(), linewidth=0)
        
        fig2 = plt.figure(figsize=FS)
        ax2 = fig2.add_subplot(1,1,1,projection='3d')
        plt2 = ax2.plot_surface(X, Y, Z2, cmap = cm.get_cmap(), linewidth=0)

        fig3 = plt.figure(figsize=FS)
        ax3 = fig3.add_subplot(1,1,1)
        plt3 = ax3.contourf(X,Y,Z1, levels=8,cmap = cm.get_cmap())
        bar = plt.colorbar(plt3)
        bar.set_label('Reward ', rotation = 90, size = 12)

        fig4 = plt.figure(figsize=FS)
        ax4 = fig4.add_subplot(1,1,1)
        plt4 = ax4.contourf(X,Y,Z2, levels=8, cmap = cm.get_cmap())
        bar = plt.colorbar(plt4)
        bar.set_label('Reward ', rotation = 90, size = 12)
        
        for axn, (ax, plot) in enumerate([(ax1,plt1) ,(ax2,plt2),(ax3,plt3),(ax4,plt4)]):
            ax.set_xlabel('$\~{x}\ [m]$')
            ax.set_ylabel('$\~{\psi}\ [deg]$')

            for el in ['x','y','z']:
                ax.locator_params(axis=el, nbins=5)
            if axn in [0,1]:
                ax.set_zlabel('$Reward$')    
                ax.view_init(30, -45)

        fig1.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()
        fig4.tight_layout()

    elif pltno == 1:
        # FINAL REWARD FUNCTION SHAPE
        max_r  = np.sqrt(2 * 8**2)
        rads   = np.linspace(-max_r, max_r, 251)
        yaws   = np.linspace(-45, 45, 251)
        (X, Y) = np.meshgrid(rads,yaws)
        pos    = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        Z1 = np.zeros((rads.shape[0], yaws.shape[0]))
        Z2 = np.zeros((rads.shape[0], yaws.shape[0]))
        Z3 = np.zeros((rads.shape[0], yaws.shape[0]))
        Z4 = np.zeros((rads.shape[0], yaws.shape[0]))
        for i in range(len(rads)):
            for j in range(len(yaws)):
                measure = np.sqrt((rads[i])**2 + (yaws[j]/4)**2)
                const = 0.5
                anti_sparity = max(-1.0, ( 1 - 0.1 * measure ))
                multivar = 2 * unitary_multivar_normal( [rads[i], yaws[j]], mu = [0,0], var=[1.0**2, 5.0**2])
                Z1[i,j] = multivar + anti_sparity + const
                Z2[i,j] = multivar
                Z3[i,j] = anti_sparity
                Z4[i,j] = const

        # FIRST, PLOT ALL COMPONENTS OF THE FINAL REWARD FUNCTION
        
        f1   = plt.figure(figsize=FS)
        ax1  = f1.add_subplot(1,1,1,projection='3d')
        plt1 = ax1.plot_surface(X, Y, Z1,cmap = cm.get_cmap(), linewidth=0)
        
        f2 = plt.figure(figsize=(6,4.5))
        ax2  = f2.add_subplot(1,1,1,projection='3d')
        plt2 = ax2.plot_surface(X, Y, Z2,cmap = cm.get_cmap(), linewidth=0)

        f3 = plt.figure(figsize=FS)
        ax3  = f3.add_subplot(1,1,1,projection='3d')
        plt3 = ax3.plot_surface(X, Y, Z3, cmap = cm.get_cmap(), linewidth=0)

        f4 = plt.figure(figsize=FS)
        ax4  = f4.add_subplot(1,1,1,projection='3d')
        plt4 = ax4.plot_surface(X, Y, Z4, cmap = cm.get_cmap(), linewidth=0)

        for axn, (ax, plot) in enumerate([(ax1,plt1) ,(ax2,plt2),(ax3,plt3),(ax4,plt4)]):
            ax.set_xlabel('$\~{x}\ [m]$')
            ax.set_ylabel('$\~{\psi}\ [deg]$')
            for el in ['x','y','z']:
                ax.locator_params(axis=el, nbins=5)
            
            ax.set_zlabel('$Reward$',linespacing=0.2)
            ax.view_init(20, -45)
        
        f1.tight_layout()
        f2.tight_layout()
        f3.tight_layout()
        f4.tight_layout()

        # THEN, PLOT THE FINAL REWARD FUNCTION IN 3D AND IN A CONTOUR MAP
        fig = plt.figure(figsize=(6,6))
        ax1 = fig.add_subplot(1,1,1,projection='3d')
        plt1 = ax1.plot_surface(X, Y, Z1, cmap = cm.get_cmap(), linewidth=0)
        # bar = plt.colorbar(plt1)
        # bar.set_label('Reward ', rotation = 90, size = 12)
        for axn, (ax, plot) in enumerate([(ax1,plt1)]):
            ax.set_xlabel('$\~{x}\ [m]$')
            ax.set_ylabel('$\~{\psi}\ [deg]$')
            ax.set_zlabel('$Reward$')
            ax.view_init(20, -45)

            for el in ['x','y','z']:
                ax.locator_params(axis=el, nbins=5)

        fig.tight_layout()

        fig = plt.figure(figsize=(6,6))
        ax1 = fig.add_subplot(1,1,1)
        plt1 = ax1.contourf(X, Y, Z1, levels=8, cmap = cm.get_cmap())
        bar = plt.colorbar(plt1)
        bar.set_label('Reward ', rotation = 90, size = 12)

        for axn, (ax, plot) in enumerate([(ax1,plt1)]):
            ax.set_xlabel('$\~{x}\ [m]$')
            ax.set_ylabel('$\~{\psi}\ [deg]$')
            for el in ['x','y','z']:
                ax.locator_params(axis=el, nbins=5)

        fig.tight_layout()

    else:
        print('No valid pltno given')

def unitary_multivar_normal(x,mu,var):
    ''' Take x, my, var lists - output scalar value '''
    size  = len(x)
    x     = np.array(x).reshape((len(x),1))
    mu    = np.array(mu).reshape((len(mu),1))
    var   = np.array(var).reshape((len(var),1))
    covar = np.zeros((size,size))
    for i in range(size):
        covar[i,i] = var[i]
    
    covar_inv = np.linalg.inv(covar)
    diff      = x-mu
    return np.exp(-0.5 * (diff.T).dot(covar_inv).dot(diff))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pltno', '-n', type=int, default=0)
    args = parser.parse_args()

    summed_gaussian_vs_2D(pltno = args.pltno)
    plt.show()
    pass