#!/usr/bin/python3

"""
NOTE that this node is using Python 3!!!

ROS Node for using a neural network for solving the thrust allocation problem on the ReVolt model ship.
The neural network was trained using the policy gradient method PPO using an actor critic structure. It was trained during 48 hours of simulation on a Huawei Matebook Pro X computer running on Windows OS.
The model was not given any information about the vessel or the environment, and is therefore based 100% on "self play".
This node was created as a part of an MSc thesis, named TODO "Solving thrust allocation for surface vessels using deep reinforcement learning". 

The output of the policy network (the part that decides which action to take) is in the area (-1,1), and therefore has to be scaled up to (-100%, 100%).
This varies with if we are looking at %RPM or angles, in which the %RPM is in scaled with a factor of 100, and the angles are scaled according to the environment, 
usually pi/2 since the network picks angles between -pi/2 and pi/ for the stern angles. Note that the bow thruster is always fixed to 90 degrees, so the agent only chooses the %RPM for it.

Requirements:
    - Tensorflow 1.x - NOTE that the system pip (for python2.7) uses tensorflow 2. It is wanted to keep tf.__version__ >= 2.0.0 as basis as that is by far the best documented module for now.
                       neural allocator uses Keras, which is based on using tf2 as backend. Therefore, this node (and the other supportive files in this package) runs with the system's Python 3,
                       in order to access tensorflow 1

TODO by user:
    - Decide the type of environment to setup
        The "environment" describes the type of thruster setups used. Various were trained to evaluate progress, and the differences are explained in __init__()


@author: Simen Sem Oevereng, simensem@gmail.com. June 2020.
"""

from __future__ import division

SIMULATION = False
INTEGRATOR = False
NU_INPUTS = False # the model was not trained on anything else than the goals of velocities being zero. Setting this to True gives overshoots - dont use it, but keep code

'''
rosbag record /observer/eta/ned /bow_control /thrusterAllocation/pod_angle_input /thrusterAllocation/stern_thruster_setpoints /reference_filter/state_desired --duration=400 -O test.bag
'''
import rospy
from std_msgs.msg import Float64
from custom_msgs.msg import podAngle, SternThrusterSetpoints, bowControl, NorthEastHeading, diffThrottleStern
from geometry_msgs.msg import Wrench, Twist, Pose2D
import numpy as np 
import sys # for sys.exit()
import time
from errorFrame import ErrorFrame, wrap_angle
from utils import load_policy, create_publishable_messages, shutdown_handler

class RLTA(object):
    '''
    Reinforcement Learning Control and Thrust Allocation
    '''
    def __init__(self):
        ''' Initialization of Thrust allocation object, member variables, and subs and publishers.

        A state to the action network is [bodyframe_error_surge, 
                                          bodyframe_error_sway, 
                                          bodyframe_error_yaw, 
                                          bodyframe_velocity_surge,
                                          bodyframe_velocity_sway,
                                          bodyframe_velocity_yaw,
                                          previous_thrust_bow [as a fraction of 100% in (-1,1)],
                                          previous_thrust_port [as a fraction of 100% in (-1,1)],
                                          previous_thrust_star [as a fraction of 100% in (-1,1)]
                                          ]
                                          or in brief
                                          [x_tilde, y_tilde, psi_tilde, u, v, r, n_bow_prev, n_port_prev, n_star_prev]

        The output depends on the environment. Below is an overview of what each environment controls, and what is set to be constant.
        Choices are shown as ranges (-1,1), and the constants are given as fixed numbers.

        ENV:    \  action       | Thrust bow    | Thrust port   | Thrust star   | Angle bow | Angle port                        | Angle star
        simple  (3 actions)     | (-1,1)        | (-1,1)        | (-1,1)        | pi/2      | -3*pi/4                           | 3*pi/4
        limited (5 actions)     | (-1,1)        | (-1,1)        | (-1,1)        | pi/2      | (-1,1) scales to -pi/2 to pi/2    | (-1,1) scales to -pi/2 to pi/2
        full    (6 actions)     | (-1,1)        | (-1,1)        | (-1,1)        | pi/2      | (-1,1) scales to -pi to pi        | (-1,1) scales to -pi to pi

        NOTE that the order in which the thrust is added to the state is not the same as the convention in the other allocator nodes.
        This is due to that the enumeration of the thrusters are not the same in the ROS code as it is in the Revolt Simulator!
        
        Thruster setup in ROS                       Thruster setup in Cybersea where the actor was trained
        _________________________________            _________________________________
        |                                 \          |                                 \ 
        |   X-  (a1)                        \        |   X-  (a2)                        \
        |                       X-  (a3)     )       |                       X-  (a1)     )
        |   X-  (a2)                        /        |   X-  (a3)                        /
        |________________________________ /          |________________________________ /
        '''

        # Init ROS Node
        rospy.init_node('rlAllocator', anonymous = True)

        '''
        Parameters for the different possible network setups:
        Output of simple has actions  [0: n_bow,  1: n_port, 2: n_star]
        Output of limited has actions [0: n_bow,  1: n_port, 2: n_star, 3: a_port, 4: a_star]
        Output of full has actions    [0: n_bow,  1: n_port, 2: n_star, 3: a_bow,  4: a_port, 5: a_star]
        Output of full with continous angle representation has actions    [0: n_bow,  1: n_port, 2: n_star, 3: a_bow,  4: sin(a_port), 5: cos(a_port), 6: sin(a_star), 7: cos(a_star)]
        Setup in ROS has actions      [0: n_port, 1: n_star, 2: n_bow,  3: a_port, 4: a_star, 5: a_bow]
        '''
        act_bnd = {'simple' : [100.0]*3,
                  'limited' : [100.0]*3 + [np.pi/2]*2, 
                    'final' : [100.0]*3 + [np.pi]*2,
                     'full' : [100.0]*3 + [np.pi]*3} # Real life action boundaries - This must not be removed as it acts as the basis for all the other environments in terms of setting default actions etc

        act_map = {'simple' : {0:2, 1:0, 2:1}, 
                  'limited' : {0:2, 1:0, 2:1, 3:3, 4:4}, 
                    'final' : {0:2, 1:0, 2:1, 3:3, 4:4}, 
                     'full' : {0:2, 1:0, 2:1, 3:5, 4:3, 5:4}} # Mapping from action output of network to corresponding message

        act_def = {'simple' : [0, 0, 0, np.pi/2, -3*np.pi/4, 3*np.pi/4], 
                  'limited' : [0, 0, 0, np.pi/2, 0, 0],
                    'final' : [0, 0, 0, np.pi/2, 0, 0], 
                     'full' : [0]*6 } # Action boundaries used

        self.params = {'simple': {'act_bnd': act_bnd['simple'],  'act_map' : act_map['simple'],  'default_actions': act_def['simple']},
                      'limited': {'act_bnd': act_bnd['limited'], 'act_map' : act_map['limited'], 'default_actions': act_def['limited']},
                        'final': {'act_bnd': act_bnd['final'],   'act_map' : act_map['final'],   'default_actions': act_def['final']},
                         'full': {'act_bnd': act_bnd['full'],    'act_map' : act_map['full'],    'default_actions': act_def['full'] }  
                      }

        ### TODO Set environment type (see ml4ta/src/rl/windows_workspace/specific/customEnv for definitions)
        ### THIS HAS TO BE SET MANUALLY - NOT VERY FLEXIBLE YET
        self.env = 'final'
        self.number_of_hidden_layers = 3
        self.cont_ang = True
        ###
        ### ABOVE PARAMS MUST BE SET

        if self.env == 'simple': raise Exception('No simple enviroments has been trained using previous thrust in the state vector / extended state space vector - sorry')

        path = '/home/revolt/revolt_ws/src/rl_allocator/src/models/{}'.format(self.env)
        rospy.loginfo('Launching RLTA - it takes a few seconds to load...')
        try:
            self.actor = load_policy(fpath = path, num_hidden_layers=self.number_of_hidden_layers) # a FUNCTION which takes in a state, and outputs an action
            rospy.loginfo('... loading RL policy network using {} thruster setup for thrust allocation successful.'.format(self.env))
        except:
            rospy.logerr('... loading RL policy network for thrust allocation was not successful. Shutting down node')
            sys.exit()

        self.rate = rospy.Rate(5) # time step of 0.1 s^1 = 10 Hz or 0.2 s^1 = 5 Hz

        # Init variable for storing the previous state each time, so that it is possible to send the thrusters the right way using shortest path calculations
        self.state                    = np.zeros((9,)) # State vector contains [error_surge, error_sway, error_yaw, u, v, r, thr_bow/100, thr_port/100, thr_star/100]
        self.prev_thrust_state        = np.zeros((6,))
        self.integrator               = np.zeros((3,))
        self.velocities = np.zeros((3,))
        self.use_bodyframe_integrator = INTEGRATOR
        self.time_arrival             = time.time()
        self.EF                       = ErrorFrame(use_integral_effect=False)
        self.time_prev                = rospy.get_time()
        self.error_time_prev          = rospy.get_time()
        self.h                        = 0.0
        rospy.loginfo('RL allocation loaded {} integral action'.format( 'with' if self.use_bodyframe_integrator else 'without' ))

        # Init Publishers
        self.pub_stern_angles             = rospy.Publisher('thrusterAllocation/pod_angle_input', podAngle, queue_size=1)
        self.pub_stern_thruster_setpoints = rospy.Publisher("thrusterAllocation/stern_thruster_setpoints", SternThrusterSetpoints, queue_size=1)
        self.pub_bow_control              = rospy.Publisher("bow_control", bowControl, queue_size=1)
        self.pub_state                    = rospy.Publisher("RLallocation/state", NorthEastHeading, queue_size=1)

        # Init Subscribers for reference, states and sysID
        rospy.Subscriber("reference_filter/state_desired", NorthEastHeading, self.state_desired_callback,queue_size=1) # TODO old reference filter
        rospy.Subscriber("observer/eta/ned", Twist, self.eta_obs_callback)
        rospy.Subscriber('observer/nu/body', Twist, self.nu_obs_callback)
        rospy.Subscriber("CME/diff_throttle_stern", diffThrottleStern, self.diff_throttle_stern_callback)

    def eta_obs_callback(self, eta):
        ''' Callback function for pose. Updates self.state.
        :params:
            - eta (float): A Twist message containing the 6-dimensional NED position vector. Angles are in degrees.
        ''' 
        eta = np.array([eta.linear.x, eta.linear.y, np.deg2rad(eta.angular.z)])
        eta[2] = wrap_angle(eta[2], deg=False) # wrap yaw in (-pi,pi)

        # Update errorFrame with new position, and update state vector
        self.EF.update(pos=eta.tolist())
        self.state[0:3] = self.get_error_states() # (3,) shaped ndarray

    def nu_obs_callback(self, nu):
        ''' Callback function for velocity observed in body frame, nu_obs. Updates self.state.
        :params:
            - nu (float): Twist message containing observed velocities. Angles are in rad/s.
        '''
        self.velocities = np.array([nu.linear.x,nu.linear.y,nu.angular.z])
        if not NU_INPUTS:
            self.state[3:6] = np.array([nu.linear.x,nu.linear.y,nu.angular.z])

    def state_desired_callback(self, eta_des):
        ''' Performs thrust allocation. Updates self.state and publishes stern_angles, stern_thruster_setpoints, bow_control
        :params:
            - eta_des (NorthEastHeading): Message containing desired state in 9 dimensions: 3 x pos, 3 x vel and 3 x acc.
        '''

        # Compute timestep h for finding thruster derivatives
        self.h = (rospy.get_time() - self.time_prev)
        self.time_prev = rospy.get_time()

        # Extract eta_desired, update errorFrame with new reference, and update state vector
        self.EF.update(ref = [eta_des.pos_north, eta_des.pos_east, np.deg2rad(eta_des.pos_heading)])
        self.state[0:3] = self.get_error_states(step = self.h)
        
        # If using reference filter's desired velocities, update states accordingly. Else, just use the velocity of the vessel
        if NU_INPUTS:
            desired_velocities = np.array([eta_des.vel_north, eta_des.vel_east, eta_des.vel_heading])
            self.state[3:6] = self.velocities - desired_velocities
        else:
            self.state[3:6] = self.velocities

        # Select action
        u = self.get_action() # [n1,n2,n3,a1,a2,a3] (6,) shaped array according to ROS format

        # Publish action
        pod_angle, stern_thruster_setpoints, bow_control = create_publishable_messages(u, simulation=SIMULATION)
        self.pub_stern_angles.publish(pod_angle)
        self.pub_stern_thruster_setpoints.publish(stern_thruster_setpoints)
        self.pub_bow_control.publish(bow_control)

        # Update state vector with previous thrust
        self.prev_thrust_state = np.copy(u)
        self.state[-3:] = np.array([u[2], u[0], u[1]]) / 100.0 # Updating state using neural net input

        self.rate.sleep() # Ensure that the published messages doesn't exceed given rate

    def scale_and_clip(self,action):
        bnds = np.array(self.params[self.env]['act_bnd'])
        action = np.multiply(action,bnds) # scale
        action = np.clip(action,-bnds,bnds) # clip
        return action

    def get_action(self):
        # Get action selection from the policy network and scale according to environment
        action = self.actor(self.state) # (x,) shaped array as output from the neural network

        if self.env == 'final' and self.cont_ang: # TODO there has been added a change to the case where cont_ang is False - wrapping before clipping
            action = self.handle_continuous_angles(action)

        action = self.scale_and_clip(action)

        curr_act_map = self.params[self.env]['act_map'] # The action map from chosen action to the ROS format
        full_act_map = self.params['full']['act_map']   # The action map from default actions to the ROS format
        arr = np.zeros((6,))

        # Translate the output from the neural network to outputs understood by the ROS format:
        # (1) set all default actions in the ROS format using the current format's default values with the full format action map
        for i, default in enumerate(self.params[self.env]['default_actions']): # [nbow,nport,nstar,abow,aport,astar]
            arr[full_act_map[i]] = default

        # (2) fill the chosen action values into the ROS format using the current environment's action map
        for i,act in enumerate(action):
            arr[curr_act_map[i]] = act

        return arr # [nport,nstar,nbow,aport,astar,abow]

    def get_error_states(self, step = 0.1):
        current_state = np.array(self.EF.get_pose())
        print(['{:.3f}'.format(val) if i <= 1 else '{:.3f}'.format(np.rad2deg(val)) for i,val in enumerate(current_state) ])

        if self.use_bodyframe_integrator:
            surge, sway, yaw = current_state

            # TODO this effect might be unneccessary in real testing when SIMULATION is false
            if (np.abs(surge) > 5.0 or np.abs(sway) > 5.0 or np.abs(yaw) > np.deg2rad(140)): # stop integral effect between setpoints
                self.integrator = np.zeros(3)
                self.time_arrival = time.time()
            else:
                if (time.time() - self.time_arrival) > 5.0: # If the vessel has been within the setpoint over x seconds: initiate integral effect
                    unbounded = self.integrator + step * np.multiply(np.array([0.05, 0.05, 0.05]), current_state)
                    bnds = np.array([0.5, 1, np.pi/32])
                    self.integrator = np.clip(unbounded, -bnds, bnds)
            
            print(['{:.3f}'.format(val) if i <= 1 else '{:.3f}'.format(np.rad2deg(val)) for i,val in enumerate(self.integrator) ])
            return current_state + self.integrator
        else:
            self.integrator = np.zeros((3,)) # reset in case it has been turned on before
            return current_state

    def handle_continuous_angles(self,action):
        assert self.env.lower() == 'final', 'Using continuous angles is only made to work with the final environment using fully rotating stern thrusters'
        sin_port, cos_port = action[3], action[4]
        sin_star, cos_star = action[5], action[6]
        a_port = np.arctan2(sin_port, cos_port) / self.params[self.env]['act_bnd'][-1] # Since the scale and clip-function assumes an action between -1 and 1, the angle is scaled according to maximum
        a_star = np.arctan2(sin_star, cos_star) / self.params[self.env]['act_bnd'][-1]
        new_action = np.hstack( (action[0:3], np.array([a_port, a_star])) )
        action = new_action.copy()
        return action


    def shortestPath(self,a_prev,a,deg = False):
        ''' Calculates the shortest angluar path BETWEEN two angles. This does NOT return the new angle, but should be used for adding.
        :params:
            a_prev  - a float of the previous TODO could just use self.prev_thrust_state [3:]
            a       - a float of the desired angle
        :returns:
            a float of the shortest angular distance (in radians) between current azimuth angle and the desired one (could be positive or negative)
        '''
        # TODO this function needs to be used with the scale_and_clip function in order to get the shortest angle BEFORE clipping
        ref = 180.0 if deg else np.pi
        temp = np.mod((a-a_prev), 2 * ref) # get REMAINDER from division with 2pi in case the difference is larger than 2pi
        shortest = np.mod( (temp + 2 * ref), 2 * ref) # map the remainder into being positive in region [0, 2pi)
        addition = shortest - 2 * ref if shortest > ref else shortest # if it is larger than pi, return the equivalent on the other side of 0 instead
        return addition

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
        # Simulate rudder with varying thrust
        stern_thruster_setpoints.port_effort = float(throttle_in) + float(rudder_in) # Note: Port propeller is mirrored physically and needs to rotate the other way
        stern_thruster_setpoints.star_effort = float(throttle_in) - float(rudder_in)
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

    def shutdown_handler(self):
        # shutdown_handler() # this does not do the job as the node is still publishing other messages
        pass


if __name__ == '__main__':

    try:
        node = RLTA()
        rospy.on_shutdown(node.shutdown_handler)
        rospy.spin()
    except rospy.ROSInterruptException:
        # TODO implement
        pass
