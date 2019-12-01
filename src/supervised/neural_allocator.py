#!/usr/bin/python

from __future__ import division
import rospy
from std_msgs.msg import Float64
from custom_msgs.msg import podAngle, SternThrusterSetpoints, bowControl
from geometry_msgs.msg import Wrench
import numpy as np 
import sys # for sys.exit()
import time
from keras.models import load_model

'''
ROS Node for using a neural network for solving the thrust allocation problem on the ReVolt model ship.
The neural network was trained with supervised learning, where a dataset was generated with all possible thruster inputs and azimuth angles, 
with the corresponding forces and moments generated. The neural network therefore approximates the pseudoinverse.

It was trained on the dataset where the azimuth angles were rotating from -pi to pi, but wrapped in sin(x/2) to keep the values within -1 and 1,
while still having each values representing a unique angle. Thruster inputs were scaled to -1,1 by dividing the thruster input on 100.0, as the inputs were given in percentages.
The corresponding forces as a function of angles and thruster input percentages was derived according to Alfheim and Muggerud, 2016.

See SupervisedTau.py for more details on the dataset generation. There it is also understandable how the predictions of the network has to be rescaled.
Also, the input forces needs to be rescaled due to the scaling of them as inputs during training. See train.py for more details on the dataset augmentation.

@author: Simen Sem Oevereng, simensem@gmail.com. November 2019.
'''
class NNTA(object):
    '''
    Neural Network Thrust Allocation
    '''

    def __init__(self):
        ''' 
        Initialization of Thrust allocation object, member variables, and subs and publishers.
        '''

        # Init ROS Node
        rospy.init_node('neuralAllocator', anonymous = True)

        # Init Neural Network
        
        try:
            # This compiles 
            self.model = load_model('/home/revolt/revolt_ws/src/neural_allocator/src/model.h5')
            rospy.logerr("Loading of neural network model for thrust allocation successful.")
            # self.model.summary()
        except:
            rospy.logerr("Loading of neural network model was not successful. Shutting down")
            sys.exit()


        # Scaling factor for the forces and moments used when training the neural network
        self.tau_scale = np.array([[1/54,1/69.2,1/76.9]]).T
        self.u_scale = np.array([[100,100,100,np.pi,np.pi,np.pi]]).T

        self.max_forces_forward = np.array([[25,25,14]]).T # In Newton
        self.max_forces_backward = np.array([[25,25,6.1]]).T # In Newton - bow thruster is asymmetrical, thus lower force backwards

        self.forwards_K  = np.array([[0.0027, 0.0027, 0.001518]]).T
        self.backwards_K = np.array([[0.0027, 0.0027, 0.0006172]]).T

        # Init variable for storing the previous state each time, so that it is possible to send the thrusters the right way using shortest path calculations
        self.previous_thruster_state = [0,0,0,0,0,0]
        self.prev_time_step = rospy.get_time()

      # Init variable that contains the positions of the thrusters: [lx1 ly1 lx2 ly2 lx3 ly3]
        self.lx = [-1.12, -1.12, 1.08]
        self.ly = [-0.15, 0.15, 0.0]
        temp = []
        for lxi, lyi in zip(self.lx,self.ly):
            temp.append(lxi); temp.append(lyi)
        self.l = np.array([temp]).T

        # Init Publishers TODO use neuralAllocator
        self.pub_stern_angles             = rospy.Publisher('NN/thrusterAllocation/pod_angle_input', podAngle, queue_size=1)
        self.pub_stern_thruster_setpoints = rospy.Publisher("NN/thrusterAllocation/stern_thruster_setpoints", SternThrusterSetpoints, queue_size=1)
        self.pub_bow_control              = rospy.Publisher("NN/bow_control", bowControl, queue_size=1)
        self.pub_tau_diff                 = rospy.Publisher("NN/tau_diff", Wrench, queue_size=1)
        self.pub_thruster_forces          = rospy.Publisher("NN/F", Wrench, queue_size=1)

        # Init subscriber for control forces from either DP controller or RC remote (manual T.A.)
        rospy.Subscriber("tau_controller", Wrench, self.tau_controller_callback)


    def shortestPath(self,a_prev,a):
        '''
        Calculates the shortest angluar path between two angles.

        :params:
            a_prev  - a float of the previous TODO could just use self.previous_thruster_state[3:]
            a       - a float of the desired angle

        :returns:
            a float of the shortest angular distance (in radians) between current azimuth angle and the desired one (could be positive or negative)

        '''
        pass

    def format_tau(self,tau_d):
        '''
        Formats the incoming desired tau to an input that the neural network has been trained on in order for it to be able to make a prediction of the thruster states.
        The model was trained on tau-values scaled with the calculated maximum thrust, so it has to be scaled when precicting 

        :params:
            tau_d   - A Wrench message

        :returns:
            A numpy array of shape (9,1) with the positions of the thrusters, and the desired forces/moments
        '''
        taux = float(tau_d.force.x)
        tauy = float(tau_d.force.y)
        taup = float(tau_d.torque.z)
        tau = np.array([[taux,tauy,taup]]).T

        # Return a 9x1 column vector of [lx1 ly1 lx2 ly2 lx3 ly3 taux tauy taup]
        # NB the tau-values were scaled when training. Therefore, they has to be scaled again
        tau = np.multiply(tau,self.tau_scale)

        return np.vstack((self.l,tau))

    def predictThrusterStates(self,data):
        '''
        Returns the thruster inputs and angles from 
        Since the values were scaled during training, they have to be rescaled after predicting.

        :params:
            data    - a (9x1) numpy array of [lx1 ly1 lx2 ly2 lx3 ly3 taux tauy taup]

        :returns:
            A (6x1) numpy array of [u1 u2 u3 a1 a2 a3] in percentages and radians.

        '''
        
        # predict believes that it receives a batch of input values to make predictions on
        # Therefore, the data must be added to a "batch"

        prediction_batch = np.array([data.reshape(data.shape[0],)])
        predictions = self.model.predict( prediction_batch )

        # If more predictions than one are done; choose the one with the lowest norm
        lowest_norm = np.inf
        u = predictions[0:1,:]
        for p in predictions:
            if np.linalg.norm(p[0:3]) < lowest_norm:
                u = p
                lowest_norm = np.linalg.norm(p[0:3])

        # ELEMENTWISE multiplication to scale the thruster inputs AND angles according to how the model was trained!
        u = u.reshape(6,1)
        scaled_u = np.multiply(u, self.u_scale) 

        return scaled_u


    def mapAngle(self, val_input, val_min, val_max, out_min, out_max):
        '''
        Maps val_input from the range given by val_min,max to range given by out_min,max

        :params:
            val_input   - input value 
            val_min     - float of min in original range
            val_max     - float of max in original range
            out_min     - float of min in new range
            out_max     - float of max in new range

        :returns:
            An integer of the mapped value
        '''
        return int((val_input - val_min) * (out_max - out_min) / (val_max - val_min) + out_min)

    def saturateThrustPercentage(self, u):
        ''' Checks if throttle is in valid area.

        :params:
            u   - (3x1) vector of thrusterinputs (u1, u2, u3)
        
        :returns:
            A (3x1) with the thruster inputs constrained between -100% and 100%
        '''

        # Set the values of exeeding saturations to the saturation levels
        u[np.where(u > 100.0)] = 100.0
        u[np.where(u < -100.0)] = -100.0

        return u


    def tau_controller_callback(self, tau_d):
        '''
        Callback for DP and Manual Thrust allocation mode. Performs thrust allocation
        Publishes stern_angles, stern_thruster_setpoints, bow_control

        :params:
            tau_d   - a Wrench message consisting of members .force and .torque

        Thruster setup
           _________________________________
           |                                 \
           |   X-  (a1)                        \
           |                       X-  (a3)     )
           |   X-  (a2)                        /
           |________________________________ /
        '''

        # A column vector for the neural network model
        data_to_NN = self.format_tau(tau_d) 

        # Create a prediction of the data, giving a (6,1) shaped numpy array with [u1 u2 u3 a1 a2 a3]
        u = self.predictThrusterStates(data_to_NN)

        # Collect the thruster inputs (while saturating) and angles
        thruster_percentages = self.saturateThrustPercentage(u[0:3,:])

        thruster_angles = u[3:,:] # TODO these are not entirely constant - setting them to the trained value
        thruster_angles = np.array([[-3*np.pi/4, 3*np.pi/4, np.pi / 2]]).T

        # Set messages as angles in degrees, and thruster inputs in RPM TODO I believe this is in percentages
        pod_angle = podAngle()        
        pod_angle.port = float(np.rad2deg(thruster_angles[0,0]))
        pod_angle.star = float(np.rad2deg(thruster_angles[1,0]))

        stern_thruster_setpoints = SternThrusterSetpoints()
        stern_thruster_setpoints.port_effort = float(thruster_percentages[0,0])
        stern_thruster_setpoints.star_effort = float(thruster_percentages[1,0])
        
        bow_control = bowControl()
        bow_control.throttle_bow = float(thruster_percentages[2,0]) # Bow throttle doesn't work at low inputs. (DC Motor)
        bow_control.position_bow = np.rad2deg(thruster_angles[2,0])
        bow_control.lin_act_bow = 2

        # Publish the stern pod angles and thruster rpms, as well as the bow message
        self.pub_stern_angles.publish(pod_angle)
        self.pub_stern_thruster_setpoints.publish(stern_thruster_setpoints)
        self.pub_bow_control.publish(bow_control)

        # TODO currently not used for anything. Just nice to have when implementing shorest angle blabla
        self.previous_thruster_state = np.vstack((thruster_percentages,thruster_angles))

        # TESTINGTESTING TODO

        # Constant K values F = K*|n|*n (see Alheim and Muggerud, 2016, for this empirical formula). The bow thruster is unsymmetrical, and this has lower coefficient for negative directioned thrust.
        if u[2] >= 0:
            K = self.forwards_K
        else:
            K = self.backwards_K

        F = K * np.abs(thruster_percentages) * thruster_percentages # ELEMENTWISE multiplication

        out_forces = Wrench() # Contains three forces from port, starboard and bow thruster
        out_forces.force.x = float(F[0,0])
        out_forces.force.y = float(F[1,0])
        out_forces.force.z = float(F[2,0])
        self.pub_thruster_forces.publish(out_forces)

        tau_res = Wrench()
        alpha = thruster_angles
        tau_res.force.x  = float(np.cos(alpha[0,0]) * F[0,0] + np.cos(alpha[1,0]) * F[1,0] + np.cos(alpha[2,0]) * F[2,0])
        tau_res.force.y  = float(np.sin(alpha[0,0]) * F[0,0] + np.sin(alpha[1,0]) * F[1,0] + np.sin(alpha[2,0]) * F[2,0])
        tau_res.torque.z = float( (self.lx[0]*np.sin(alpha[0,0]) - self.ly[0]*np.cos(alpha[0,0])) * F[0,0] + (self.lx[1]*np.sin(alpha[1,0]) - self.ly[1]*np.cos(alpha[1,0])) * F[2,0] + (self.lx[2]*np.sin(alpha[2,0]) - self.ly[2]*np.cos(alpha[2,0])) * F[2,0])

        tau_diff = Wrench()
        tau_diff.force.x = tau_res.force.x - float(tau_d.force.x)
        tau_diff.force.y = tau_res.force.y - float(tau_d.force.z)
        tau_diff.torque.z = tau_res.torque.z - float(tau_d.torque.z)
        self.pub_tau_diff.publish(tau_diff)

if __name__ == '__main__':
    
    try:
        nn = NNTA()
        rospy.spin()
    except rospy.ROSInterruptException:
        # TODO implement
        pass
