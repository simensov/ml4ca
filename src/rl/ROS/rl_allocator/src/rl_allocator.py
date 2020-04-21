#!/usr/bin/python3

"""
NOTE that this node is using Python 3!!!

ROS Node for using a neural network for solving the thrust allocation problem on the ReVolt model ship. #!/home/revolt/revolt_ws/src/rl_venv/bin/python
The neural network was trained using the actor critic, policy gradient method PPO. It was trained during 24 hours of simulation on a Windows computer.
The model was not given any information about the vessel or the environment, and is therefore based 100% on "self play".
The node was created as a part of an MSc thesis, named "Solving thrust allocation for surface vessels using deep reinforcement learning". 

The output of the policy network (the part that decides which action to take) is in the area (-1,1), and therefore has to be scaled up to (-100%, 100%).
This varies with if we are looking at %RPM or angles, in which the %RPM is in scaled with a factor of 100, and the angles are scaled according to the environment, 
usually pi/2 since the network picks angles between -pi/2 and pi/2.

Requirements:
    - Tensorflow 1.x - NOTE that the system pip (for python2.7) uses tensorflow 2. It is wanted to keep tf.__version__ >= 2.0.0 as basis as that is by far the best documented modules for now.
                       neural allocator uses Keras, which is based on using tf2 as backend. Therefore, this node (and the other supportive files in this package) runs with the system's Python 3.

@author: Simen Sem Oevereng, simensem@gmail.com. June 2020.
"""

from __future__ import division
import rospy
from std_msgs.msg import Float64
from custom_msgs.msg import podAngle, SternThrusterSetpoints, bowControl, NorthEastHeading
from geometry_msgs.msg import Wrench, Twist
import numpy as np 
import sys # for sys.exit()
import time
from errorFrame import ErrorFrame, wrap_angle
from tf_utils import load_policy

def createPublishableMessages(u):
    '''
    :params:
        - u (ndarray): a (6,) shaped array containing [n1,n2,n3,a1,a2,a3] in percentages and degrees
    '''
    thruster_percentages = u[:3]
    thruster_angles = u[3:]

    # Set messages as angles in degrees, and thruster inputs in RPM percentages
    pod_angle = podAngle()        
    pod_angle.port = float(np.rad2deg(thruster_angles[0]))
    pod_angle.star = float(np.rad2deg(thruster_angles[1]))

    stern_thruster_setpoints = SternThrusterSetpoints()
    stern_thruster_setpoints.port_effort = float(thruster_percentages[0])
    stern_thruster_setpoints.star_effort = float(thruster_percentages[1])
    
    bow_control = bowControl()
    bow_control.throttle_bow = float(thruster_percentages[2]) # Bow throttle doesn't work at low inputs. (DC Motor)
    bow_control.position_bow = np.rad2deg(thruster_angles[2])
    bow_control.lin_act_bow = 2

    return pod_angle, stern_thruster_setpoints, bow_control


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
                                          previous_trust_bow [as a fraction of 100% in (-1,1)],
                                          previous_trust_port [as a fraction of 100% in (-1,1)],
                                          previous_trust_star [as a fraction of 100% in (-1,1)]
                                          ]
                                          or
                                          [x_tilde, y_tilde, psi_tilde, u, v, r, n_bow_prev, n_port_prev, n_star_prev]

        The output depends on the environment. Below is an overview of what each environment controls, and what is set to be constant.
        Choices are shown as ranges (-1,1), and the constants are given as fixed numbers.

        ENV:    \  action       | Thrust bow    | Thrust port   | Thrust star   | Angle bow | Angle port                        | Angle star
        simple  (3 actions)     | (-1,1)        | (-1,1)        | (-1,1)        | pi/2      | -3*pi/4                           | 3*pi/4
        limited (5 actions)     | (-1,1)        | (-1,1)        | (-1,1)        | pi/2      | (-1,1) scales to -pi/2 to pi/2    | (-1,1) scales to -pi/2 to pi/2
        full    (6 actions)     | (-1,1)        | (-1,1)        | (-1,1)        | pi/2      | (-1,1) scales to -pi to pi        | (-1,1) scales to -pi to pi

        Not that the order in which the thrust is added to the state is not the same as the convention in the other allocator nodes.
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
        print('Launching RLTA')

        rospy.init_node('rlAllocator', anonymous = True)

        '''
        Parameters for the different possible network setups:
        Output of simple has actions  [0: n_bow,  1: n_port, 2: n_star]
        Output of limited has actions [0: n_bow,  1: n_port, 2: n_star, 3: a_port, 4: a_star]
        Output of full has actions    [0: n_bow,  1: n_port, 2: n_star, 3: a_bow,  4: a_port, 5: a_star]
        Setup in ROS has actions      [0: n_port, 1: n_star, 2: n_bow,  3: a_port, 4: a_star, 5: a_bow]
        '''
        act_bnd = {'simple' : [100.0]*3,
                  'limited' : [100.0]*3 + [np.pi/2]*2, 
                     'full' : [100.0]*3 + [np.pi]*3} # Real life action boundaries

        act_map = {'simple' : {0:2, 1:0, 2:1}, 
                  'limited' : {0:2, 1:0, 2:1, 3:3, 4:4}, 
                     'full' : {0:2, 1:0, 2:1, 3:5, 4:3, 5:4}} # Mapping from action output of network to corresponding message

        act_def = {'simple' : [0, 0, 0, np.pi/2, -3*np.pi/4, 3*np.pi/4], 
                  'limited' : [0, 0, 0, np.pi/2, 0, 0], 
                     'full' : [0]*6 } # Action boundaries used

        self.params = {'simple': {'act_bnd': act_bnd['simple'],  'act_map' : act_map['simple'],  'default_actions': act_def['simple']},
                      'limited': {'act_bnd': act_bnd['limited'], 'act_map' : act_map['limited'], 'default_actions': act_def['limited']},
                         'full': {'act_bnd': act_bnd['full'],    'act_map' : act_map['full'],    'default_actions': act_def['full'] }  
                      }

        self.env = 'limited'
        # path = '/home/revolt/revolt_ws/src/rl_allocator/src/simactpen' # TODO get a new simple env with ext state
        path = '/home/revolt/revolt_ws/src/rl_allocator/src/{}'.format(self.env)
        
        try:
            self.actor = load_policy(fpath=path) # a FUNCTION which takes in a state, and outputs an action
            rospy.loginfo("Loading of RL policy network for thrust allocation successful.")
        except:
            rospy.logerr("Loading of RL policy network fir thrust allocation was not successful. Shutting down")
            sys.exit()

        self.rate = rospy.Rate(10) # time step of 0.1 s^1 = 10 Hz

        # Scaling factor for the forces and moments used when training the neural network
        self.max_forces_forward  = np.array([[25,       25,     14]]).T         # In Newton
        self.max_forces_backward = np.array([[25,       25,     6.1]]).T        # In Newton - bow thruster is asymmetrical, thus lower force backwards
        self.forwards_K          = np.array([[0.0027,   0.0027, 0.001518]]).T
        self.backwards_K         = np.array([[0.0027,   0.0027, 0.0006172]]).T

        # Init variable for storing the previous state each time, so that it is possible to send the thrusters the right way using shortest path calculations
        self.state = np.zeros((9,))
        self.EF = ErrorFrame()
        self.previous_thruster_state = np.zeros((6,))
        self.time_prev = rospy.get_time()
        self.h = 0.0

        # Init Publishers TODO use neuralAllocator
        self.pub_stern_angles             = rospy.Publisher('RLNN/thrusterAllocation/pod_angle_input', podAngle, queue_size=1)
        self.pub_stern_thruster_setpoints = rospy.Publisher("RLNN/thrusterAllocation/stern_thruster_setpoints", SternThrusterSetpoints, queue_size=1)
        self.pub_bow_control              = rospy.Publisher("RLNN/bow_control", bowControl, queue_size=1)
        self.pub_tau_diff                 = rospy.Publisher("RLNN/tau_diff", Wrench, queue_size=1)
        self.pub_thruster_forces          = rospy.Publisher("RLNN/F", Wrench, queue_size=1)
        self.pub_thruster_forces          = rospy.Publisher("RLNN/F", Wrench, queue_size=1)

        # Init subscriber for control forces from either DP controller or RC remote (manual T.A.)
        rospy.Subscriber("reference_filter/state_desired", NorthEastHeading, self.state_desired_callback)
        rospy.Subscriber("observer/eta/ned", Twist, self.eta_obs_callback)
        rospy.Subscriber('observer/nu/body', Twist, self.nu_obs_callback)

    def eta_obs_callback(self, eta):
        ''' Callback function for pose. Updates self.state.
        :params:
            - eta (float): A Twist message containing the 6-dimensional NED position vector. Angles are in degrees.
            Twist
        ''' 
        eta = np.array([eta.linear.x, eta.linear.y, eta.angular.z * np.pi / 180])
        eta = wrap_angle(eta, deg=False)

        # Update errorFrame with new position, and update state vector
        self.EF.update(pos=eta.tolist())
        self.state[0:3] = np.array(self.EF.get_pose()) # (3,) shaped ndarray

    def nu_obs_callback(self, nu):
        ''' Callback function for velocity observed in body frame, nu_obs. Updates self.state.
        :params:
            - nu (float): Twist message containing observed velocities. Angles are in rad/s.
        '''
        self.state[3:6] = np.array([nu.linear.x,nu.linear.y,nu.angular.z])

    def shortestPath(self,a_prev,a):
        ''' Calculates the shortest angluar path between two angles.
        :params:
            a_prev  - a float of the previous TODO could just use self.previous_thruster_state[3:]
            a       - a float of the desired angle
        :returns:
            a float of the shortest angular distance (in radians) between current azimuth angle and the desired one (could be positive or negative)
        '''
        pass

    def scale_and_clip(self,action):
        bnds = np.array(self.params[self.env]['act_bnd'])
        action = np.multiply(action,bnds) # scale
        return np.clip(action,-bnds,bnds) # clip

    def get_action(self):
        # Get action selection from the policy network and scale according to environment
        action = self.actor(self.state) # (x,) shaped array
        action = self.scale_and_clip(action)

        curr_act_map = self.params[self.env]['act_map'] # The action map from chosen action to the ROS format
        full_act_map = self.params['full']['act_map']   # The action map from default actions to the ROS format
        arr = np.zeros((6,))

        # First, set all default actions in the ROS format using the current format's default values with the full format action map
        for i, default in enumerate(self.params[self.env]['default_actions']): # [nbow,nport,nstar,abow,aport,astar]
            arr[full_act_map[i]] = default

        # Then fill the chosen action values into the ROS format using the current environment's action map
        for i,act in enumerate(action):
            arr[curr_act_map[i]] = act

        return arr # [nport,nstar,nbow,aport,astar,abow]

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
        self.state[0:3] = np.array(self.EF.get_pose())

        # Select action
        u = self.get_action() # [n1,n2,n3,a1,a2,a3] in (6,) shaped array in ROS format
        print(u)

        # Publish action
        pod_angle, stern_thruster_setpoints, bow_control = createPublishableMessages(u)
        self.pub_stern_angles.publish(pod_angle)
        self.pub_stern_thruster_setpoints.publish(stern_thruster_setpoints)
        self.pub_bow_control.publish(bow_control)

        # Update state vector with previous thrust
        self.previous_thruster_state = np.copy(u)
        self.state[-3:] = np.array([u[1], u[2], u[0]]) # Updating the state must be done according to the mapping between windows setup and ros setup

        '''
        Messages used only for plotting purposes
        '''
        # Constant K values F = K*|n|*n (see Alheim and Muggerud, 2016, for this empirical formula). The bow thruster is unsymmetrical, and this has lower coefficient for negative directioned thrust.
        K = self.forwards_K if u[2] >0 else self.backwards_K
        F = K * np.abs(u[:3]) * u[:3] # ELEMENTWISE multiplication, using thruster percentages


if __name__ == '__main__':
    
    try:
        node = RLTA()
        rospy.spin()
    except rospy.ROSInterruptException:
        # TODO implement
        pass
