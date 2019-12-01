#!/usr/bin/env python3

# Generate data for supervised learning
# TRY WITH BOTH FIXED AZIMUTHS AND ROTATING TO TEST PERFORMANCE!
# Attempt 1: fixed bow thruster and same angles for stern thrusters. Remember to add a shortest path test for rotational thrusters.

import numpy as np
from math import sin, cos, floor
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

    def __init__( self, a = np.array([[0,0,0]]).T, u = np.array([[0,0,0]]).T,data=[], df = 0):
        self.a = a
        self.u = u
        self.ly = [-0.15, 0.15, 0]
        self.lx = [-1.12, -1.12, 1.08]
        self.tau_max = np.array([[69,30,80]]).T # Saturation according to dp_controller/DP_PID.py
        self.data = data
        self.df = 0
        self.filename = 'dataset_train.npy' #.format(time.strftime("%Y%m%d_%H%M%S")) # alternatively strftime("%d-%m-%Y_%I-%M-%S_%p")

    def B(self,a):
        '''
        Returns the effectiveness matrix B in tau = B*u, full shape numpy array (e.g. (3,3))

        :params:
            a - A (3,1) np.array of azimuth angles in radians
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
        K3 = 1.518e−3 for positive u, or 6.172e−4 for negative u 
            - pm means plus/minus
        u is given in percentages

        => Fi = Ki|ui|ui

        :params:
            u - A (3,1) array of thruster rotational velocities, ranging from -100% - 100% of maximum thrust - NOT RPM or rad/s
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


    def generateData(self,azimuth_discretization=37,thrust_discretization = 21):
        '''
        Generates dataset for the supervised learning task of thrust allocation.
        Each row of the data will contain one datapoint: [lx1,ly1,lx2,ly2,lx3,ly3,tau_x,tau_y,tau_psi, u1, u2, u3, a1, a2, a3].T
        The data is added to the member variable self.data

        :params:
            azimuth_discretization  - An integer of number of discretization points for the azimuth angles
            thrust_discretization   - An integer of number of discretization points for the thrusters' rotational velocities

        '''
        # TODO Alfheim and Muggerud somehow allows both -180 and 180 deg by their implementation. I am trying the same, intending to use (-180,180] later

        stern_angles = np.linspace(-np.pi,np.pi,azimuth_discretization) # Including zero with odd number of spacings - Using same angles for stern thrusters
        stern_angles = [3*np.pi/4]
        bow_angles = [np.pi/2] # TODO kept constant during training due to the weird definition of +- 270??
        throttle = np.linspace(-100,100,thrust_discretization)


        # Make the vector of distances from CG
        l = []
        for lxi, lyi in zip(self.lx,self.ly):
            l.append(lxi); l.append(lyi)

        l = np.array([l]).T

        for a0 in stern_angles:
            for a2 in bow_angles:
                for u0 in throttle:
                    for u1 in throttle:
                        for u2 in throttle:
                            u = np.array([[u0,u1,u2]]).T
                            a = np.array([[-a0,a0,a2]]).T # TODO note that there might be a minus for the port side thruster, used when they are set fixed
                            tau = self.tau(a,u)


                            # Do not add very small elements
                            if True:
                                if np.any(np.abs(tau) < 1):
                                    continue

                            if True:
                                # Do not add elements that are larger than the PID saturation, set in the previous implementations Revolt Source Code. This is however not quite realistic when the bow thruster has a fixed angle.
                                if np.any( np.abs(tau) > self.tau_max):
                                    continue
                            
                            # Scale dataset labels to -1,1
                            u = u / 100.0
                            a = a / np.pi

                            if True:
                                tauscale = np.array([[1/54,1/69.2,1/76.9]]).T # calculated maximum taux, tauy, taup
                                tau = np.multiply(tau,tauscale) # elementwise multiplication , NOT dot product!
                            

                            # Add the positions of the thrusters to the dataset to help the NN understanding relationships of force and moment.
                            datapoint = np.vstack((l,tau,u,a)).reshape(15,)
                            self.data.append(datapoint)

        # Convert list of np.arrays to big np.array
        self.data = np.array(self.data) # Each row is one datapoint. First nine columns are input vals, six next are labels


    def generateDataRandom(self,azimuth_discretization=37,thrust_discretization = 21):
        '''
        Generates dataset for the supervised learning task in the same way that Skulstad did: considering each thruster on its own
        '''
        stern_angles = np.linspace(-np.pi,np.pi,azimuth_discretization) # including zero with odd number of spacings # Using same angles for stern thrusters
        # a1s = np.linspace(-np.pi,np.pi,azimuth_discretization) # not do this
        # stern_angles = [3*np.pi/4]
        bow_angles = [np.pi/2] # TODO what is the problem with +- 270??
        throttle = np.linspace(-100,100,thrust_discretization) # TODO bow thruster does not generate rotations on low thruster inputs (see thruster_allocation). But Alfeim et al hasn't restricted it lol 

        throttle = throttle[np.abs(throttle) > 20]
        # throttle = 100*(throttle/100)**2 * np.sign(throttle) # bias larger thruster inputs
        
        stern_angles = np.pi * (stern_angles/np.pi)**2 * np.sign(stern_angles)


        # Use the same angles for thrusters 1 and 2.
        tauxs = set()
        tauys = set()
        taups = set()
        # u0set = set()
        # u1set = set()
        # u2set = set()
        # stern_angleset = set()

        for _ in range(20000):
            u0 = np.random.choice(throttle)
            u1 = np.random.choice(throttle)
            u2 = np.random.choice(throttle)
            a0 = np.random.choice(stern_angles)

            # if u0 in u0set and u1 in u1set and u2 in u2set and a0 in stern_angleset:
            #     continue
            # else:
            #     u0set.add(u0); u1set.add(u1); u2set.add(u2); stern_angleset.add(a0)

            
            a2 = bow_angles[0]

            u = np.array([[u0,u1,u2]]).T
            a = np.array([[a0,a0,a2]]).T # TODO note that there might be a minus for the port side thruster, used when they are set fixed
            
            tau = self.tau(a,u)

            # Do not add very small elements
            if np.any(np.abs(tau) < 1):
                continue
            
            tx, ty, tp = tau[:,0]

            if float(tx) in tauxs or float(ty) in tauys or float(tp) in taups:
                continue
            else:
                tauxs.add(float(tx)); tauys.add(float(ty)); taups.add(float(tp))

            # Scale dataset labels to -1,1
            u = u / 100.0
            a = a / np.pi
            
            # TODO scale tau here as well?
            #tauscale = np.array([[1/54,1/69.2,1/76.9]]).T # calculated maximum taux, tauy, taup
            #tau = np.multiply(tau,tauscale) # elementwise multiplication , NOT dot product!

            # Add the positions of the thrusters to the dataset to help the NN understanding relationships of force and moment.

            l = []
            for lxi, lyi in zip(self.lx,self.ly):
                l.append(lxi); l.append(lyi)

            l = np.array([l]).T

            datapoint = np.vstack((l,tau,u,a)).reshape(15,)
            
            self.data.append(datapoint) 

        # Convert list of np.arrays to big np.array
        self.data = np.array(self.data) # Each row is one datapoint. First three columns are input vals, six next are labels


    def generateDataFrame(self):
        '''
        Use Pandas to visualize the dataset
        '''
        # index = [str(i) for i in range(data.shape[0])]
        cols = ['x1','y1','x2','y2','x3','y3','Tx','Ty','Tp','u1','u2','u3','a1','a2','a3']

        if self.data == []:
            print('Data is an empty list')
            return

        df = pd.DataFrame(	data=self.data[0:,0:],
                            index=[i for i in range(self.data.shape[0])],
                            columns=cols) 
                                
        self.df = df

    def saveData(self,filename='dataset_train.npy'):
        '''
        Store data for later
        '''
        self.filename = filename
        np.save(self.filename,self.data)
    
    def loadData(self,filename):
        self.data = np.load(filename)
        self.generateDataFrame()

    def maxMSE(self):
        '''
        Calculate the maximum squared error of the training data
        '''
        X = self.data[:,0:3]
        largest_squares = [ (np.max(X[:,col])-np.min(X[:,col]))**2 for col in range(X.shape[1])]
        return sum(largest_squares)

    def getNormalizedData(self):

        normalized = []
        
        for col in range(self.data.shape[1]):
            arr = self.data[:,col]
            normalized.append( (arr - np.mean(arr)) / (np.max(arr) - np.min(arr)))

        return np.array(normalized).T

    def getStandardizedData(self):

        return


### END CLASS
        
# if __name__ == "__main__":

#     obj = SupervisedTau()
#     obj.generateData()
#     obj.displayData()