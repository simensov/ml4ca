#!/usr/bin/python

from __future__ import division
import rospy
from std_msgs.msg import Float64
from custom_msgs.msg import NorthEastHeading, podAngle, SternThrusterSetpoints, bowControl, diffThrottleStern
from geometry_msgs.msg import Wrench
import numpy as np 
import sys # for sys.exit()
import time
from keras.models import load_model

'''
ROS Node for using Quadratic Programming for solving the thrust allocation problem on the ReVolt model ship.
The QP is formulated with nonlinear constraints due to the nature of how the forces and moments are calculated.
The QP is solved each time step, using the previous thruster states as initial values, finding the STERN azimuth angles and all thruster forces.
The forces are manually translated into percentage thrust using formulas found in Alfheim and Muggerud (2016).

@author: Simen Sem Oevereng, simensem@gmail.com. November 2019.
'''

class QPTA(object):
    '''
    Quadratic Programming Thrust Allocation
    '''

    def __init__(self):
        ''' 
        Initialization of Thrust allocation object, member variables, and subs and publishers.
        '''

        # Init ROS Node
        rospy.init_node('neuralAllocator', anonymous = True)

        # Scaling factor for the forces and moments used when training the neural network
        self.max_forces_forward = np.array([[25,25,14]]).T # In Newton
        self.max_forces_backward = np.array([[25,25,6.1]]).T # In Newton - bow thruster is asymmetrical, thus lower force backwards
        self.forwards_K  = np.array([[0.0027, 0.0027, 0.001518]]).T
        self.backwards_K = np.array([[0.0027, 0.0027, 0.0006172]]).T

        # Init variable for storing the previous state each time, so that it is possible to send the thrusters the right way using shortest path calculations
        self.previous_thruster_state = [0,0,np.pi/2,0,0,0]

        # Init variable that contains the positions of the thrusters: [lx1 ly1 lx2 ly2 lx3 ly3]
        # TODO this don't need a for loop
        temp = []
        for lxi, lyi in zip([-1.12, -1.12, 1.08],[-0.15, 0.15, 0]):
            temp.append(lxi); temp.append(lyi)
        self.l = np.array([temp]).T

        # Init Publishers TODO use neuralAllocator
        self.pub_stern_angles             = rospy.Publisher('QPAllocation/pod_angle_input', podAngle, queue_size=1)
        self.pub_stern_thruster_setpoints = rospy.Publisher("QPAllocation/stern_thruster_setpoints", SternThrusterSetpoints, queue_size=1)
        self.pub_bow_control              = rospy.Publisher("QPAllocation/bow_control", bowControl, queue_size=1)

        # Init subscriber for control forces from either DP controller or RC remote (manual T.A.)
        rospy.Subscriber("tau_controller", Wrench, self.tau_controller_callback)

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

        #print(u.shape,'from prediction')

        # Collect the thruster inputs (while saturating) and angles
        thruster_percentages = self.saturateThrustPercentage(u[0:3,:])
        thruster_angles = u[3:,:] 

        #print(thruster_percentages, 'after sat')
        
        #print(np.rad2deg(thruster_angles),'angles')

        # Not currently used since checking if percentages work directly
        if False:
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

            #rospy.loginfo('n.Port: %s , n.Star: %s, n.Bow: %s  ', n[0], n[1], n[2])

        # Initialize messages for publishing
        pod_angle = podAngle()
        stern_thruster_setpoints = SternThrusterSetpoints()
        bow_control = bowControl()

        # Set messages as angles in degrees, and thruster inputs in RPM TODO I believe this is in percentages
        pod_angle.port = float(np.rad2deg(thruster_angles[0,0]))
        pod_angle.star = float(np.rad2deg(thruster_angles[1,0]))

        stern_thruster_setpoints.port_effort = float(thruster_percentages[0,0])
        stern_thruster_setpoints.star_effort = float(thruster_percentages[1,0])
        
        bow_spd = float(thruster_percentages[2,0])
        # Bow throttle doesn't work at low inputs. (DC Motor) TODO this isn't doing anything
        if bow_spd > 0:
            bow_spd = float(bow_spd)
        if bow_spd < 3 and bow_spd > -3:
            bow_spd = float(bow_spd)
        else:
            bow_spd = float(bow_spd)

        # Map the bow thruster angle into -100 and 100 percent
        # TODO this makes no sense so far. They have assumed that the range was between -270, 270, but nothing has explicitly stated that yet

        # servo_out = self.mapAngle(np.rad2deg(thruster_angles[2,0]), -270, 270, -100, 100)
        servo_out = np.rad2deg(thruster_angles[2,0])

        # Linear Actuator For Bow Pod. 2 == Go down
        act_out = 2

        # Fill bow control custom message
        bow_control.throttle_bow = bow_spd
        bow_control.position_bow = servo_out # note that this is 
        bow_control.lin_act_bow = act_out


        # Publish the stern pod angles and thruster rpms, as well as the bow message
        self.pub_stern_angles.publish(pod_angle)
        self.pub_stern_thruster_setpoints.publish(stern_thruster_setpoints)
        self.pub_bow_control.publish(bow_control)

        # TODO currently not used for anything. Just nice to have when implementing shorest angle blabla
        self.previous_thruster_state = np.vstack((thruster_percentages,thruster_angles))

if __name__ == '__main__':
    
    try:
        QPTA()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
