#!/usr/bin/python

from __future__ import division
import rospy
from custom_msgs.msg import podAngle, SternThrusterSetpoints, bowControl
from geometry_msgs.msg import Wrench
import numpy as np 
import sys # for sys.exit()
import time
from scipy.optimize import minimize

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
        rospy.init_node('QPAllocator', anonymous = True)
        self.rate = rospy.Rate(5) # time step of 0.2 s^1 = 5 Hz

        # Scaling factor for the forces and moments used when training the neural network
        self.max_forces_forward = np.array([[25.0,25.0,14.0]]).T # In Newton
        self.max_forces_backward = np.array([[25.0,25.0,6.1]]).T # In Newton - bow thruster is asymmetrical, thus lower force backwards
        self.max_force_rate = [10, 10, 4] # 5, 5, 2 was nice
        self.max_rotational_rate = [np.pi/6, np.pi/6, np.pi/6] # pi/12 on all was nice

        self.forwards_K  = np.array([[0.0027, 0.0027, 0.001518]]).T
        self.backwards_K = np.array([[0.0027, 0.0027, 0.0006172]]).T

        # Init variable for storing the previous state each time, so that it is possible to send the thrusters the right way using shortest path calculations
        self.bow_angle_fixed = np.pi/2
        self.previous_thruster_state = [0,0,self.bow_angle_fixed,0,0,0] # NB: these states are expressed in [N, N, N, rad, rad, rad]

        # Init variable that contains the positions of the thrusters: [lx1 ly1 lx2 ly2 lx3 ly3]
        self.lx = [-1.12, -1.12, 1.08]
        self.ly = [-0.15, 0.15, 0.0]

        temp = []
        for lxi, lyi in zip(self.lx,self.ly):
            temp.append(lxi); temp.append(lyi)
        self.l = np.array([temp]).T

        # Init Publishers TODO use neuralAllocator
        self.pub_stern_angles             = rospy.Publisher('QPthrusterAllocation/pod_angle_input', podAngle, queue_size=1)
        self.pub_stern_thruster_setpoints = rospy.Publisher("QPthrusterAllocation/stern_thruster_setpoints", SternThrusterSetpoints, queue_size=1)
        self.pub_bow_control              = rospy.Publisher("QPbow_control", bowControl, queue_size=1)
        self.pub_resulting_tau            = rospy.Publisher("QPresulting_tau", Wrench, queue_size=1)

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

    def mapToPi(self,angles):
        '''
        Maps the elements in a vector to [-pi,pi)

        :params:
            angles  - A numpy column vector of size (m,1), m > 0
        '''

        return np.mod( angles + np.pi, 2 * np.pi) - np.pi

    def solve_QP(self,tau_d): # TODO use previous states
        '''
        Solves a quadratic program (optimization based) for the thruster FORCES [N] and stern azimuth ANGLES [rad]
        '''
        # TODO optimize using numpy / math functions for their relevant usage
        
        s_t = self.previous_thruster_state # states at time t
        
        def objective(x):
            # Objective to minimize (is quadratic) - does not constrain the angles, but it is possible to constraint the size depending on the previous angle in s_t
            # Decision variables for the optimization: x = [f1 f2 f3 a1 a2 s1 s2 s3]
            FandS = np.hstack( (x[0:3], x[5:]) )
            return (FandS.T).dot(FandS)
            #return (x.T).dot(x) # This will also minimize the angles themselves, but it is the ANGULAR RATES that should be minimized over time for wear and tear

        ### PHYSICAL CONSTRAINTS DEPENDENT ON THE THRUSTER SETUP: B(alpha)*F = tau_d
        # Since these are equality constrains, a slack variable is added and the goal is to minimize s^2 - added in the objective functions
        # The constraints are written as ||| B(alpha)*F - tau_d - s = 0 |||, so that if s is minimized, the produced forces are as close as possible to tau_d
        # scipy.optimize.minimize only takes scalar functions, so the each row has to be written seperately
        def c1(x):  return np.cos(x[3]) * x[0] + np.cos(x[4]) * x[1] + np.cos(np.pi/2) * x[2] - x[5] - float(tau_d[0,0]) # top row
        def c2(x):  return np.sin(x[3]) * x[0] + np.sin(x[4]) * x[1] + np.sin(np.pi/2) * x[2] - x[6] - float(tau_d[1,0]) # mid row
        def c3(x):  return (self.lx[0]*np.sin(x[3]) - self.ly[0]*np.cos(x[3])) * x[0] + (self.lx[1]*np.sin(x[4]) - self.ly[1]*np.cos(x[4])) * x[1] + (self.lx[2]*np.sin(np.pi/2) - self.ly[2]*np.cos(np.pi/2))*x[2] - x[7] - float(tau_d[2,0]) # bottom row

        ### FORCE RATE CONSTRAINTS
        # Rate constraint on force increase/decrease: 
        # dfMin < dF < dFMax  becomes ||| dFMax - dF >= 0||| AND ||| -dFMin + dF >= 0 ||| where -dFmin is -(-dFMax) 
        # Note that e-3 added makes it possible for the rate to be zero! It just overlaps the other constraint, so it is all good - a good ol' optimization trick
        def c4(x): return self.max_force_rate[0] - (x[0] - s_t[0])
        def c5(x): return self.max_force_rate[0] + (x[0] - s_t[0])
        def c6(x): return self.max_force_rate[1] - (x[1] - s_t[1])
        def c7(x): return self.max_force_rate[1] + (x[1] - s_t[1])
        def c8(x): return self.max_force_rate[2] - (x[2] - s_t[2])
        def c9(x): return self.max_force_rate[2] + (x[2] - s_t[2])
        
        ### ANGULAR RATE CONSTRAINTS
        def c10(x): return self.max_rotational_rate[0] + (x[3] - s_t[3])
        def c11(x): return self.max_rotational_rate[0] - (x[3] - s_t[3])
        def c12(x): return self.max_rotational_rate[1] + (x[4] - s_t[4])
        def c13(x): return self.max_rotational_rate[1] - (x[4] - s_t[4])

        # Gather all constrains
        con1  = {'type': 'eq', 'fun': c1}
        con2  = {'type': 'eq', 'fun': c2}
        con3  = {'type': 'eq', 'fun': c3}
        con4  = {'type': 'ineq', 'fun': c4}
        con5  = {'type': 'ineq', 'fun': c5}
        con6  = {'type': 'ineq', 'fun': c6}
        con7  = {'type': 'ineq', 'fun': c7}
        con8  = {'type': 'ineq', 'fun': c8}
        con9  = {'type': 'ineq', 'fun': c9}
        con10 = {'type': 'ineq', 'fun': c10}
        con11 = {'type': 'ineq', 'fun': c11}
        con12 = {'type': 'ineq', 'fun': c12}
        con13 = {'type': 'ineq', 'fun': c13}
            
        cons = ([con1, con2, con3, con4, con5, con6, con7, con8, con9, con10, con11, con12, con13])

        # Bound decision variables for maximum force, angular range (allowed to -360,360 deg to allow rotation between -pi/+pi border TODO might allow too large range for angles; contrain to pi + pi/6 elns) 
        # np.inf is used to disable constrains, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.Bounds.html#scipy.optimize.Bounds
        s_bnd = 1.0 # TODO these has to be reset since the desired forces may be too large compared to the previous thruster states
        
        bnds = ((-25,25),(-25,25),(-6.1,14),\
                (-2*np.pi,2*np.pi),(-2*np.pi,2*np.pi),\
                (-s_bnd,s_bnd),(-s_bnd,s_bnd),(-s_bnd,s_bnd)) 

        # Set initial condition according to previous state, and set slack variables to zero
        # TODO maybe this is not the proper way to go?

        x0 = np.array([s_t[0], s_t[1], s_t[2], s_t[3], s_t[4], 0.0, 0.0, 0.0]) # Initial value - changes with each time instance

        # SOLVE!
        solution = minimize(objective, x0, method='SLSQP', bounds = bnds, constraints = cons) # Using Sequential Least Squares Quadratic Programming

        # solution has attributes such as x, success, message
        x = solution.x
        print(solution.message)

        # Clean solution for very small values to avoid flickering TODO see if this causes flickering, evt. put boundary higher
        x[np.where(np.abs(x) < 0.01)] = 0.0

        # Ensure that the node isn't run faster than 5 times per second when QP has been called since the rotational rates has been calculated with 0.2s as time step
        #self.rate.sleep() # TODO
        
        return x, solution.success # (8,)-shaped numpy array

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

        tau_desired     = np.array([[float(tau_d.force.x), float(tau_d.force.y), float(tau_d.torque.z)]]).T
        solution, succsess = self.solve_QP(tau_desired) # returns a vector of [F1,F2,F3,alpha1,alpha2,slack1,slack3,slack3]

        if not succsess:
            rospy.logerr('QP found no solution - setting thruster states == previous states')
            solution = self.previous_thruster_state

        # Extract thruster forces F and angles alpha
        F = np.array([[ solution[0], solution[1], solution[2] ]]).T # Has been constrained between max and min force in QP solver
        alpha = np.array([[solution[3],solution[4], self.bow_angle_fixed]]).T
        alpha = self.mapToPi(alpha) # the Qp solver has been allowed to change angles witin -2pi and 2pi to know that rotation from -175 to 175 deg is possible without rotating all the way around

        # Constant K values F = K*n*|n| (see Alheim and Muggerud, 2016, for this empirical formula). 
        # The bow thruster is unsymmetrical, and this has lower coefficient for negative directioned thrust.
        if F[2] >= 0:
            K = self.forwards_K
        else:
            K = self.backwards_K

        K = self.forwards_K if F[0] >= 0.0 else self.backwards_K

        # Calculate n [% thrust] : f = K*n*abs(n). Note that these operations are performed elementwise
        fk = np.divide(F,K) # F/ K
        n = np.multiply(np.sign(fk), np.sqrt(np.abs(fk))) # sgn(F/K) * sqrt(|F/K|) a (3,1) array of the propeller rotation percentages

        print('Thruster inputs')

        print(n)
        # Initialize messages for publishing
        pod_angle = podAngle()
        stern_thruster_setpoints = SternThrusterSetpoints()
        bow_control = bowControl()

        # Set messages as angles in degrees, and thruster inputs in percentages
        pod_angle.port = float(np.rad2deg(alpha[0,0]))
        pod_angle.star = float(np.rad2deg(alpha[1,0]))

        stern_thruster_setpoints.port_effort = float(n[0,0])
        stern_thruster_setpoints.star_effort = float(n[1,0])
        bow_spd                              = float(n[2,0]) # TODO Bow throttle doesn't work at low inputs (DC Motor) 

        # Map the bow thruster angle into -100 and 100 percent TODO this makes no sense so far. They have assumed that the range was between -270, 270, but nothing has explicitly stated that yet

        # servo_out = self.mapAngle(np.rad2deg(alpha[2,0]), -270, 270, -100, 100)
        servo_out = np.rad2deg(alpha[2,0])

        # Linear Actuator For Bow Pod. 2 == Go down
        act_out = 2

        # Fill bow control custom message
        bow_control.throttle_bow = bow_spd
        bow_control.position_bow = servo_out
        bow_control.lin_act_bow = act_out

        # Publish the stern pod angles and thruster rpms, as well as the bow message
        self.pub_stern_angles.publish(pod_angle)
        self.pub_stern_thruster_setpoints.publish(stern_thruster_setpoints)
        self.pub_bow_control.publish(bow_control)

        # TODO currently not used for anything. Just nice to have when implementing shorest angle blabla
        self.previous_thruster_state = [float(F[0,0]),float(F[1,0]),\
                                        float(F[2,0]),float(alpha[0,0]),\
                                        float(alpha[1,0]),float(alpha[2,0])] # regular list of three forces and angles

        # TODO TESTING PERFORMANCE
        tau_out = Wrench()
        tau_out.force.x = float(np.cos(alpha[0,0]) * F[0,0] + np.cos(alpha[1,0]) * F[1,0] + np.cos(alpha[2,0]) * F[2,0])
        tau_out.force.y = float(np.sin(alpha[0,0]) * F[0,0] + np.sin(alpha[1,0]) * F[1,0] + np.sin(alpha[2,0]) * F[2,0])
        tau_out.torque.z = float( (self.lx[0]*np.sin(alpha[0,0]) - self.ly[0]*np.cos(alpha[0,0])) * F[0,0] + (self.lx[1]*np.sin(alpha[1,0]) - self.ly[1]*np.cos(alpha[1,0])) * F[2,0] + (self.lx[2]*np.sin(alpha[2,0]) - self.ly[2]*np.cos(alpha[2,0])) * F[2,0])
        self.pub_resulting_tau.publish(tau_out)

if __name__ == '__main__':
    
    try:
        QPTA()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
