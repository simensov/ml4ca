# Generate data for supervised learning
# TRY WITH BOTH FIXED AZIMUTHS AND ROTATING TO TEST PERFORMANCE!
# Attempt 1: fixed bow thruster and same angles for stern thrusters. Remember to add a shortest path test for

import numpy as np
from math import sin, cos
from math import radians as deg2rad
import pandas as pd 
import time


# Order of thrusters:
# Alfeim and Muggerud used port, starboard, bow representation

#         Thruster setup
#    _________________________________
#    |                                 \
#    |   X-  (a1)                        \
#    |                       X-  (a3)     )
#    |   X-  (a2)                        /
#    |________________________________ /


# Distance from CG, defined by bow,starboard,down oriented body frame

class SupervisedTau():

    def __init__( self, a = np.array([[0,0,0]]).T, u = np.array([[0,0,0]]).T,data=[]):
        self.a = a
        self.u = u
        self.ly = [-0.15, 0.15, 0]
        self.lx = [-1.12, -1.12, 1.08]
        self.data = data
        self.filename = 'dataset_train.npy' #.format(time.strftime("%Y%m%d_%H%M%S")) # alternatively strftime("%d-%m-%Y_%I-%M-%S_%p")

    def B(self,a):
        '''
        Returns the effectiveness matrix, full shape numpy array (e.g. (3,3))

        :params:
            a - An (3,1) np.array of azimuth angles
        '''
        return np.array([[cos(a[0]), 									cos(a[1]), 										cos(a[2])],
                         [sin(a[0]), 									sin(a[1]), 										sin(a[2])],
                         [self.lx[0]*sin(a[0]) - self.ly[0]*cos(a[0]), 	self.lx[1]*sin(a[1]) - self.ly[1]*cos(a[1]), 	self.lx[2]*sin(a[2]) - self.ly[2]*cos(a[2])]
                         ])


    def F(self,u):
        '''
        Calculate real force vector from applied control inputs. 
        The relation used was found from Alfheim and Muggerud, 2016, ch. 8.3.
        K1pm = K2pm = 2.7e-3 for stern thrusters, both ways
        K3m = 1.518e−3 for positive u, or 6.172e−4 for negative u 
        u is given in percentages

        => Fi = Ki|ui|ui

        :params:
            u - A 3x1 array of applied thruster forces, ranging from 0 - 100%
        '''
        return np.array([[float(0.0027 * abs(u[0]) * u[0])],
                         [float(0.0027 * abs(u[1]) * u[1])],
                         [float(0.001518 * abs(u[2]) * u[2]) if u[2] >= 0 else float(0.0006172 * abs(u[2]) * u[2])]
                        ])

    def tau(self,a,u):
        '''
        Calculates tau = B(alpha) * F(u)

        :params:
            a	- 3x1 np.array of the angles of the thrusters, in radians (-pi,pi]
            u	- 3x1 np.array of the thruster inputs, in percentages (-100 to 100)
        '''

        return np.dot(self.B(a), self.F(u))	


    def generateData(self):
        '''
        Generates dataset for the supervised learning task of 
        '''
        # Create dataset. Constrain bow thruster to +-270 degrees
        # Use radians. Alfheim and Muggerud somehow allows both -180 and 180 deg by their implementation
        # I am trying the same, intending to use (-180,180] later

        a0s = np.linspace(-np.pi,np.pi,361) # including zero with odd number of spacings. Using same angles for stern thrusters for now TODO
        a2s = [np.pi/2]
        # a3 = np.linspace(deg2rad(-270),deg2rad(270),) # including zero with odd number of spacings TODO what is the problem with +- 270??

        # IDEA: use same values for u0 and u1 to always move them in the same direction?
        u0s = np.linspace(-100,100,401)
        u1s = np.linspace(-100,100,401)
        u2s = np.linspace(-100,100,401)

        # Each column of the data will contain one datapoint: [tau_x, tau_y, tau_psi, u1, u2, u3, a1, a2, a3].T

        ##### TESTVALS
        randnum = 21
        a0s = np.linspace(-np.pi,np.pi,randnum) # including zero with odd number of spacings # Using same angles for stern thrusters
        a2s = [np.pi/2]
        u0s = np.linspace(-100,100,randnum)
        u1s = np.linspace(-100,100,randnum)
        u2s = np.linspace(-100,100,randnum)
        ### TESTVALS

        # IDEA: store some of the instances as test data

        for a0 in a0s:
            for a2 in a2s:
                for u0 in u0s:
                    for u1 in u1s:
                        for u2 in u2s:
                            u = np.array([[u0,u1,u2]]).T
                            a = np.array([[a0,a0,a2]]).T
                            tau = self.tau(a,u)
                            ua = np.vstack((u,a))
                            datapoint = np.vstack((tau,ua)).reshape(9,)
                            self.data.append(datapoint) # Each row is one datapoint

        self.data = np.array(self.data)

    def displayData(self):
        '''
        Use Pandas to visualize the dataset
        '''
        # index = [str(i) for i in range(data.shape[0])]
        cols = ['Tx','Ty','Tp','u1','u2','u3','a1','a2','a3',]

        if self.data == []:
            print('Data is an empty list')
            return

        df = pd.DataFrame(	data=self.data[0:,0:],
                            index=[i for i in range(self.data.shape[0])],
                            columns=cols) 
                                
        print(df)

    def saveData(self):
        '''
        Store data for later
        '''
        np.save(self.filename,self.data)
    
    def loadData(self,filename):
        self.data = np.load(filename)

### END CLASS
        
# if __name__ == "__main__":

#     obj = SupervisedTau()
#     obj.generateData()
#     obj.displayData()