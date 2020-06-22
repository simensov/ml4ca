#!/usr/bin/python

'''
NOTE: This node ONLY acts in the case of controlmode being DP. If changing away to other controlmode, e.g. sysID, there
      In order for this node to ensure that the system works as intended, only a few functions needs to be added.
      See dp_controller/nodesThruster_allocation.py for how callbacks from sysID etc. is handled.
      
ROS Node for using Quadratic Programming for solving the thrust allocation problem on the ReVolt model ship.
The QP is formulated with nonlinear constraints due to the nature of how the forces and moments are calculated.
The QP is solved each time step, using the previous thruster states as initial values, finding the STERN azimuth angles and all thruster forces.
The forces are manually translated into percentage thrust using formulas found in Alfheim and Muggerud (2016).

@author: Simen Sem Oevereng, simensem@gmail.com. November 2019.
'''

from __future__ import division
import rospy
from custom_msgs.msg import podAngle, SternThrusterSetpoints, bowControl, diffThrottleStern
from geometry_msgs.msg import Wrench
import numpy as np 
import sys # for sys.exit()
import time
from scipy.optimize import minimize

DEBUGGING = False
SKEWED_BOW_THRUSTER = False
SIMULATION = True

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
        self.dt = 0.20
        self.rate = rospy.Rate(5) # time step of 0.2 s^1 = 5 Hz

        # Scaling factor for forces 
        if SKEWED_BOW_THRUSTER:
            self.max_forces_forward  = np.array([[20.5,20.5,9.0]]).T # In Newton
            self.max_forces_backward = np.array([[20.5,20.5,3.7]]).T # In Newton - bow thruster is asymmetrical, thus lower force backwards
            self.forwards_K          = np.array([[0.00205, 0.00205, 0.0009]]).T
            self.backwards_K         = np.array([[0.00205, 0.00205, 0.00037]]).T
        else:
            # This is how the propeller forces are set currently in the simulator - Alfheim and Muggerudss values are OLD AND GIVES BAD RESULTS
            self.max_forces_forward  = np.array([[20.5,20.5,9.0]]).T # In Newton
            self.max_forces_backward = np.copy(self.max_forces_forward) # In Newton - bow thruster is ASSUMED SYMMETRICAL IN SIMULATOR
            self.forwards_K          = np.array([[0.00205, 0.00205, 0.0009]]).T
            self.backwards_K         = np.copy(self.forwards_K)

        self.max_force_rate = [10.0/2.0, 10.0/2.0, 4.0/2.0] # 5, 5, 2 was also nice
        self.max_rotational_rate = [np.pi/6.0 / 2.0, np.pi/6.0 / 2.0, np.pi/32.0 / 2.0] # pi/12 on all was also nice

        # self.max_force_rate = [10.0, 10.0, 4.0] # 5, 5, 2 was also nice
        # self.max_rotational_rate = [np.pi/6.0, np.pi/6.0, np.pi/32.0] # pi/12 on all was also nice
        # note that all rates has been divided by 2 due to the changed rate of the node in master's compared to in project thesis

        # Init variable for storing the previous state each time, so that it is possible to send the thrusters the right way using shortest path calculations
        self.bow_angle_fixed = np.pi/2
        self.previous_thruster_state = [0,0,0,0,0,self.bow_angle_fixed] # Thruster states are expressed in [N, N, N, rad, rad, rad]

        # Init variable that contains the positions of the thrusters: [lx1 ly1 lx2 ly2 lx3 ly3]
        self.lx = [-1.12, -1.12, 1.08]
        self.ly = [-0.15, 0.15, 0.0]

        temp = []
        for lxi, lyi in zip(self.lx,self.ly):
            temp.append(lxi); temp.append(lyi)
        self.l = np.array([temp]).T

        # Init Publishers TODO use neuralAllocator
        self.pub_stern_angles             = rospy.Publisher('thrusterAllocation/pod_angle_input', podAngle, queue_size=1)
        self.pub_stern_thruster_setpoints = rospy.Publisher("thrusterAllocation/stern_thruster_setpoints", SternThrusterSetpoints, queue_size=1)
        self.pub_bow_control              = rospy.Publisher("bow_control", bowControl, queue_size=1)

        # Init subscriber for control forces from either DP controller or RC remote (manual T.A.)
        rospy.Subscriber("tau_controller", Wrench, self.tau_controller_callback, queue_size=1) # critical with size 1 when using rospy.rate.sleep()
        self.callback_time = rospy.get_time()

        rospy.Subscriber("CME/diff_throttle_stern", diffThrottleStern, self.diff_throttle_stern_callback)
        rospy.loginfo('QP allocator started, running at {} Hz'.format(1.0 / self.dt))

    def saturateThrustPercentage(self, u):
        ''' Controls that throttle is in valid area.
        :params:
            u   - (3x1) vector of thrusterinputs (u1, u2, u3)
        
        :returns:
            A (3x1) with the thruster inputs constrained between -100% and 100%
        '''
        u[np.where(u > 100.0)] = 100.0
        u[np.where(u < -100.0)] = -100.0
        return u

    def mapToPi(self,angles):
        ''' Maps the elements in an array to [-pi,pi)
        :params:
            angles  - A numpy column vector of size (m,1), m > 0
        '''
        return np.mod( angles + np.pi, 2 * np.pi) - np.pi

    def solve_QP(self,tau_d, weight_matrix = None, reduce_fuel=True, reduce_flickering = True, reduce_angular = True):
        '''Solves a quadratic program (optimization based) for the thruster FORCES [N] and stern azimuth ANGLES [rad]
        :params:
            - tau_d (Wrench): message containing the 3 DOF desired force/moment vector coming from the PID controller
        '''
        
        s_t = self.previous_thruster_state # states at time t, being [Fport, Fstar, Fbow, aport, astar, abow]
        
        def objective(x, Q = weight_matrix, fuel=reduce_fuel, flick=reduce_flickering, ang=reduce_angular):
            ''' Create objective to minimize (is quadratic in decision variables)
                Decision variables for the optimization: x = [f1 f2 f3 a1 a2 s1 s2 s3]
                The objective function is constructed depending on which factors are being considered:
                    - the objective function ALWAYS consists, as a minimum, of f1,f2,f3,s1,s2,s3 - being minimized in a regular, quadratic fashion
                    - fuel: penalizes forces to the power of 3/2
                    - flick: penalizes changes of force linearly
                    - ang: penalizes changes of angles linearly
            '''
            obj = x[5:] # minimize slack variables

            if fuel:
                obj = np.hstack((obj,np.abs(x[0:3])**1.5)) # minimize power, which is proportional to thrust^(3/2)
            else:
                obj = np.hstack((obj,x[0:3])) # minimize the magnitude of the thrust

            if ang:
                obj = np.hstack((obj, np.abs((x[3] - s_t[3])))) # minimize size between previous (commanded) angle the the current
                obj = np.hstack((obj, np.abs((x[4] - s_t[4]))))

            if flick:
                obj = np.hstack((obj, np.abs(x[0:3] - np.array(s_t[0:3]))))

            if Q is None:
                Q = np.zeros((len(obj), len(obj))) # set up weighting matrix
                np.fill_diagonal(Q,1.0) 
                if ang: # these are always at pos 6 and 7
                    Q[6,6] = 0.25
                    Q[7,7] = 0.25 # don't prioritize lowering these derivatives as much
                if flick:
                    idx = 6 if not ang else 8 # these have positions depending on if angles are being reduced
                    for i in range(idx,idx+3):
                        Q[i,i] = 0.25

            return 0.5 * (obj.T).dot(Q).dot(obj) # 0.5 x^t Q x - 0.5 is not neccessary as the minimum of x^2 and 0.5 x^2 is the same, but it gives a prettier derivative

        ### PHYSICAL CONSTRAINTS DEPENDENT ON THE THRUSTER SETUP: B(alpha)*F = tau_d
        # Since these are equality constrains, a slack variable is added and the goal is to minimize s^2 - added in the objective functions
        # The constraints are written as ||| B(alpha)*F - tau_d - s = 0 |||, so that if s is minimized, the produced forces are as close as possible to tau_d
        # scipy.optimize.minimize only takes scalar functions, so the each row has to be written seperately
        def c1(x):  return np.cos(x[3]) * x[0] + np.cos(x[4]) * x[1] + np.cos(np.pi/2) * x[2] - x[5] - float(tau_d[0,0]) # top row
        def c2(x):  return np.sin(x[3]) * x[0] + np.sin(x[4]) * x[1] + np.sin(np.pi/2) * x[2] - x[6] - float(tau_d[1,0]) # mid row
        def c3(x):  return (self.lx[0]*np.sin(x[3]) - self.ly[0]*np.cos(x[3])) * x[0] + (self.lx[1]*np.sin(x[4]) - self.ly[1]*np.cos(x[4])) * x[1] + (self.lx[2]*np.sin(np.pi/2) - self.ly[2]*np.cos(np.pi/2))*x[2] - x[7] - float(tau_d[2,0]) # bottom row

        ### FORCE RATE CONSTRAINTS
        # Rate constraint on force increase/decrease: 
        # dfMin < dF < dFMax  becomes ||| dFMax - dF >= 0||| AND ||| -dFMin + dF >= 0 ||| where -dFmin == -(-dFMax) 
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

        # Bound decision variables for maximum force, angular range (allowed to -360,360 deg to allow rotation between -pi/+pi border - requires shortest assigned angle later on)
        # np.inf is used to disable constrains, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.Bounds.html#scipy.optimize.Bounds
        s_bnd = 1.0 # these has to be reset since the desired forces may be too large compared to the previous thruster states
        
        bnds = ((-self.max_forces_backward[0,0],self.max_forces_forward[0,0]),(-self.max_forces_backward[1,0],self.max_forces_forward[1,0]),(-self.max_forces_backward[2,0],self.max_forces_forward[2,0]),\
                (-2*np.pi,2*np.pi),(-2*np.pi,2*np.pi),\
                (-s_bnd,s_bnd),(-s_bnd,s_bnd),(-s_bnd,s_bnd)) 

        # Set initial condition according to previous state, and set slack variables to zero
        x0 = np.array([s_t[0], s_t[1], s_t[2], s_t[3], s_t[4], 0.0, 0.0, 0.0]) # Initial value - changes with each time instance

        # SOLVE!
        solution = minimize(objective, x0, method='SLSQP', bounds = bnds, constraints = cons) # Using Sequential Least Squares Quadratic Programming

        # TODO here the logic can be altered to gradually increase the s_bnds if not solution was found
        while (solution.success is False) and ( (rospy.get_time() - self.callback_time) > (self.dt / 4.0) ): # look for new solution within time limit, including safety
            rospy.loginfo('No sol found - attempting to iteratively find new sol') if DEBUGGING else None

            s_bnd += 1.0 # TODO set this to whatever suitable

            bnds = ((-self.max_forces_backward[0,0],self.max_forces_forward[0,0]),(-self.max_forces_backward[1,0],self.max_forces_forward[1,0]),(-self.max_forces_backward[2,0],self.max_forces_forward[2,0]),\
                    (-2*np.pi,2*np.pi),(-2*np.pi,2*np.pi),\
                    (-s_bnd,s_bnd),(-s_bnd,s_bnd),(-s_bnd,s_bnd)) 
            
            # Continue the optimization using the slack variables from where they ended the last time
            x = solution.x
            s1, s2, s3 = x[-3:]

            x0 = np.array([s_t[0], s_t[1], s_t[2], s_t[3], s_t[4], s1, s2, s3])
            solution = minimize(objective, x0, method='SLSQP', bounds = bnds, constraints = cons)

            if solution.success and DEBUGGING:
                rospy.loginfo('Found sol within time constraint!')
        
        # solution has attributes such as x, success, message
        x = solution.x
        
        # Clean solution for very small values to avoid flickering
        x[np.where(np.abs(x) < 0.01)] = 0.0
        
        return x, solution.success # (8,)-shaped numpy array, and a boolean

    def tau_controller_callback(self,tau_d):
        # This acts more stable than self.rate.sleep() and queue_size = 1 since it actually discards messages coming in at more than self.dt seconds,
        # while self.rate.sleep() seems to not do the trick
        
        if self.callback_time is None or ( (rospy.get_time() - self.callback_time) >= self.dt * 0.50): 
            if self.callback_time is None:
                self.callback_time = rospy.get_time() 

            self.tau_controller_callback_func(tau_d)
            self.callback_time = rospy.get_time()

    def tau_controller_callback_func(self, tau_d):
        '''
        Callback for DP and Manual Thrust allocation mode. Performs thrust allocation
        Publishes stern_angles, stern_thruster_setpoints, bow_control

        :params:
            tau_d (Wrench): a message consisting of members the 6 DOF force/moment vector coming from the PID controller

        Thruster setup
           _________________________________
           |                                 \
           |   X-  (a1)                        \
           |                       X-  (a3)     )
           |   X-  (a2)                        /
           |________________________________ /
        '''
      
        tau_desired     = np.array([[float(tau_d.force.x), float(tau_d.force.y), float(tau_d.torque.z)]]).T
        solution, success = self.solve_QP(tau_desired) # returns a vector of [F1,F2,F3,alpha1,alpha2,slack1,slack3,slack3]

        if not success:
            rospy.logwarn('QP found no solution - setting thruster states == previous states')
            solution = self.previous_thruster_state
        else:
            pass
            # rospy.loginfo('QP allocation: Solution found')

        # Extract thruster forces F and angles alpha
        F = np.array([[ solution[0], solution[1], solution[2] ]]).T # Has been constrained between max and min force in QP solver
        alpha = np.array([[solution[3],solution[4], self.bow_angle_fixed]]).T
        alpha = self.mapToPi(alpha) # the Qp solver has been allowed to change angles witin -2pi and 2pi to know that rotation from -179 to 179 deg is possible without rotating all the way around

        # Constant K values F = K*n*|n| (see Alheim and Muggerud, 2016, for this empirical formula). 
        # The bow thruster is unsymmetrical, and this has lower coefficient for negative directioned thrust.
        if SKEWED_BOW_THRUSTER:
            K = self.forwards_K if F[2] >= 0 else self.backwards_K
        else:
            K = self.forwards_K # NOTE new simulator values assumes same force profile for bow thruster

        # Calculate n [% thrust] : f = K*n*abs(n). Note that these operations are performed elementwise
        fk = np.divide(F,K) # F/ K
        n = np.multiply(np.sign(fk), np.sqrt(np.abs(fk))) # sgn(F/K) * sqrt(|F/K|) a (3,1) array of the propeller rotation percentages

        # Set messages as angles in degrees, and thruster inputs in percentages
        pod_angle = podAngle()
        pod_angle.port = float(np.rad2deg(alpha[0,0]))
        pod_angle.star = float(np.rad2deg(alpha[1,0]))

        stern_thruster_setpoints = SternThrusterSetpoints()
        stern_thruster_setpoints.port_effort = float(n[0,0])
        stern_thruster_setpoints.star_effort = float(n[1,0])

        # Fill bow control custom message
        bow_control = bowControl()
        
        if SIMULATION:
            bow_control.throttle_bow = float(n[2,0])
            bow_control.position_bow = np.rad2deg(alpha[2,0])
            
        else:
            val = np.clip(float(n[2,0]) * 2.5, -100.0, 100.0) # Bow throttle doesn't work at low inputs (DC Motor) ca. ???% Seems like 50%. 2.5 is empirically found OK value. Clipped here to avoid too large values
            bow_control.throttle_bow = val
            bow_control.position_bow = int(45) # This value was found empirically in real testing, putting the thruster at ca. 90 degrees

        bow_control.lin_act_bow = 2 # down == 2

        # Publish the stern pod angles and thruster rpms, as well as the bow message
        self.pub_stern_angles.publish(pod_angle)
        self.pub_stern_thruster_setpoints.publish(stern_thruster_setpoints)
        self.pub_bow_control.publish(bow_control)

        # Update previous thruster states, used as state if QP finds no solution
        self.previous_thruster_state = [float(F[0,0]),float(F[1,0]),\
                                        float(F[2,0]),float(alpha[0,0]),\
                                        float(alpha[1,0]),float(alpha[2,0])] # regular list of three forces and angles


    def diff_throttle_stern_callback(self, diff_throttle_input):
        '''Callback for system identification control mode. Publishes stern thruster setpoints.
        :params:
            - diff_throttle_input (SternThrusterSetpoints message): 
                a message which controls the vessel by using a virtual rudder angle to control the vessel by
                lowering and increasing thrust on port/starboard side oppositely of each other
        '''

        stern_thruster_setpoints = SternThrusterSetpoints()
        throttle_in = self.clip_scalar(diff_throttle_input.throttle) # Not self
        rudder_in = self.clip_scalar(diff_throttle_input.rudder)
        # Mixing of throttle and rudder inputs
        stern_thruster_setpoints.port_effort = float(throttle_in) + float(rudder_in)
        stern_thruster_setpoints.star_effort = float(throttle_in) - float(rudder_in)
        # Note: Port propeller is mirrored physically and needs to rotate the other way
        self.pub_stern_thruster_setpoints.publish(stern_thruster_setpoints)

    def clip_scalar(self, input_check, bound=(-100.0,100.0)):
        ''' Checks if throttle is in valid area.
        :params:
            - input_check (float): input to clip
            - bound (tuple of floats): lower and upper bound of clipping region
        :returns:
            - a float being the clipped scalar value
        '''
        if not (isinstance(input_check,float) or isinstance(input_check, int)): raise Exception('Clipping function received an object which is not a float or an integer!')
        return np.clip(input_check,bound[0],bound[1])

if __name__ == '__main__':
    
    try:
        QPTA()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
