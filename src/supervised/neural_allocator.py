#!/usr/bin/env python3

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

from __future__ import division
import rospy
from std_msgs.msg import Float64
from custom_msgs.msg import NorthEastHeading, podAngle, SternThrusterSetpoints, bowControl, diffThrottleStern
from geometry_msgs.msg import Wrench
import numpy as np 
import sys # for sys.exit()
import time
from keras.models import load_model

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
            self.model = load_model('model.h5')
        except:
            rospy.logerr("Loading of neural network model was not successful. Shutting down")
            sys.exit()

        # Scaling factor for the forces and moments used when training the neural network
        self.tau_scale = np.array([[54,69.2,76.9]]).T

        self.u_scale = np.array([[100,100,100,np.pi,np.pi,np.pi]]).T

        self.max_forces_forward = np.array([[25,25,14]]).T # In Newton
        self.max_forces_backward = np.array([[25,25,6.1]]).T # In Newton - bow thruster is asymmetrical, thus lower force backwards

        self.forwards_K  = np.array([[0.0027, 0.0027, 0.001518]]).T
        self.backwards_K = np.array([[0.0027, 0.0027, 0.0006172]]).T

        # Init variable for storing the previous state each time, so that it is possible to send the thrusters the right way using shortest path calculations
        self.previous_thruster_state = [0,0,0,0,0,0]

        # Init variable that contains the positions of the thrusters: [lx1 ly1 lx2 ly2 lx3 ly3]
        temp = []
        for lxi, lyi in zip([-1.12, -1.12, 1.08],[-0.15, 0.15, 0]):
            temp.append(lxi); temp.append(lyi)
        self.l = np.array([temp]).T

        # Init Publishers
        self.pub_stern_angles             = rospy.Publisher('neuralAllocator/pod_angle_input', podAngle, queue_size=1)
        self.pub_stern_thruster_setpoints = rospy.Publisher("neuralAllocator/stern_thruster_setpoints", SternThrusterSetpoints, queue_size=1)
        self.pub_bow_control              = rospy.Publisher("neuralAllocator/bow_control", bowControl, queue_size=1)

        # Init Subscribers
        # Subscriber for control forces from either DP controller or RC remote (manual T.A.)
        rospy.Subscriber("tau_controller", Wrench, self.tau_controller_callback)

        # TODO: These methods needs a differently trained model. Not suitable for trained neural networks for DP mode
        # Subscriber for differential throttle used in manual mode
        #rospy.Subscriber("CME/diff_throttle_stern", diffThrottleStern, self.diff_throttle_stern_callback)
        #rospy.Subscriber("headingController/output", Float64, self.heading_control_effort_callback)
        # Subscriber for control effort from speed controller
        #rospy.Subscriber("speedController/output", Float64, self.speed_controller_effort_callback)

    def shortestPath(self,a_prev,a):
        '''
        Calculates the shortest angluar path between two angles.

        :params:
            a_prev  - a float of the previous TODO could just use self.previous_thruster_state[3:]
            a       - a float of the desired angle

        :returns:
            a float of the shortest angular distance (in radians) between current azimuth angle and the desired one (could be positive or negative)

        '''

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
        u = self.model.predict(data)

        # ELEMENTWISE multiplication to scale the thruster inputs and angles
        return np.multiply(u, self.u_scale)


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
            u   - float of thruster input as percentage
        
        :returns:
            A float with the thruster input, constrainted between -100% and 100%
        '''
        if u < -100:
            return -100.0
        elif u > 100:
            return 100.0
        else:
            return u


    def tau_controller_callback(self, tau_d):
        '''
        Callback for DP and Manual Thrust allocation mode. Performs thrust allocation

        :params:
            tau_d   - a Wrench message consisting of members .force and .torque

        Publishes stern_angles, stern_thruster_setpoints, bow_control

        Thruster setup
           _________________________________
           |                                 \
           |   X-  (a1)                        \
           |                       X-  (a3)     )
           |   X-  (a2)                        /
           |________________________________ /

        Parameters
        ----------
        tau_d : float
            Desired throttle

        '''

        # A column vector for the neural network model
        data_to_NN = self.format_tau(tau_d) 

        # Create a prediction of the data, giving a (6,1) shaped numpy array with [u1 u2 u3 a1 a2 a3]
        u = self.predictThrusterStates(data_to_NN)

        thruster_percentages = u[:,0:2]
        thruster_angles = u[:,3:] 

        # Constant K values F = K*n*n (see Alheim and Muggerud, 2016, for this empirical formula). The bow thruster is unsymmetrical, and this has lower coefficient for negative directioned thrust.
        if u[2] >= 0:
            K = self.forwards_K
            forces_produced = np.multiply(self.max_forces_forward, thruster_percentages / 100) # [N] elementwise multiplication

        else:
            K = np.array([0.0027, 0.0027, 0.0006172])
            forces_produced = np.multiply(self.max_forces_backward, thruster_percentages / 100) # [N] elementwise multiplication

        # TODO does the original allocation calculate the thrust outputs as forces????
        # TODO make the check above to a function - it is messy
        # TODO do I have to do this? I have already used the thruster input percentages as training data - the predictions are according to the physics anyway
        # TODO that might be right, but I do not have the maximum rpms available anywhere (at least not underwater)
        # TODO therefore, the recalculation might has to be done unless a transformation of u(%) -> n(RPM) is found. BUT maybe the rpm signals are just given as percentages??
        # TODO FIRST TEST: use the percentages from the neural network directly!
        
        # Calculate n [rpm] : f = K*n*abs(n) from thrust input given as percentages

        # Translate the thruster forces into thruster RPMS
        Tf = forces_produced
        n = np.sign(Tf / K) * np.sqrt(np.abs(Tf / K)) # a (3,1) array of the thruster RPMS

        rospy.loginfo('n.Port: %s , n.Star: %s, n.Bow: %s  ', n[0], n[1], n[2])

        # Initialize messages for publishing
        pod_angle = podAngle()
        stern_thruster_setpoints = SternThrusterSetpoints()
        bow_control = bowControl()

        # Set messages as angles in degrees, and thruster inputs in RPM
        pod_angle.port = float(u[3] * 180 / np.pi)
        pod_angle.star = float(u[4] * 180 / np.pi)
        stern_thruster_setpoints.star_effort = float(n[1])
        stern_thruster_setpoints.port_effort = float(n[0])
        
        bow_spd = float(n[2])
        # Bow throttle doesn't work at low inputs. (DC Motor)
        if bow_spd > 0:
            bow_spd = float(bow_spd)
        if bow_spd < 3 and bow_spd > -3:
            bow_spd = float(bow_spd)
        else:
            bow_spd = float(bow_spd)

        # Map the bow thruster angle into -100 and 100 percent
        servo_out = self.mapAngle(u[5] * 180 / np.pi, -270, 270, -100, 100)

        # Linear Actuator For Bow Pod. 2 == Go down
        act_out = 2

        # Fill bow control custom message
        bow_control.throttle_bow = bow_spd
        bow_control.position_bow = servo_out
        bow_control.lin_act_bow = act_out

        self.previous_thruster_state = np.array([[]]).T

        # Publish the stern pod angles and thruster rpms, as well as the bow message
        self.pub_stern_angles.publish(pod_angle)
        self.pub_stern_thruster_setpoints.publish(stern_thruster_setpoints)
        self.pub_bow_control.publish(bow_control)

if __name__ == '__main__':
    try:
        NNTA()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
