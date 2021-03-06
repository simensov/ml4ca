import gym 
from gym import spaces
import numpy as np
from specific.misc.simtools import get_pose_3DOF, get_vel_3DOF, get_pose_on_state_space, get_random_pose_on_radius, get_vel_on_state_space, get_fixed_pose_on_radius
from specific.misc.mathematics import gaussian, gaussian_like, wrap_angle
from specific.errorFrame import ErrorFrame
import time

DEBUGGING = False

class Revolt(gym.Env):
    """ Custom Environment that follows OpenAI's gym API.
        Max velocities measured with no thrust losses activated. "Full" means rotating stern azimuths only - bow thruster remains fixed at 90 degrees.
            Full:   surge, sway, yaw = (+2.20, -1.60) m/s, +-0.35 m/s, +-0.60 rad/s
            Simple: surge, sway, yaw = (+1.75, -1.40) m/s, +-0.30 m/s, +-0.51 rad/s
        With thrust losses:
            Full:   surge, sway, yaw = (+1.4, -1.1) m/s, +-0.30 m/s, +-0.52 rad/s
            Simple: surge, sway, yaw = (+1.1, -1.2) m/s, +-0.26 m/s, +-0.43 rad/s LOL The thrusters should never have been set to -135 and 135, but rather 45 and -45 degrees (then speed became +1.3, -1.0)
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 digitwin       = None,
                 num_actions    = 6,
                 num_states     = 6,
                 real_ss_bounds = [8.0, 8.0, np.pi/2, 1.4, 0.30, 0.52], # By mistake, these velocities (three last elements) was not set lower. Of course, the bounds must LIMIT the agent; these are its REAL limits...
                 testing        = False,
                 realtime       = False,
                 max_ep_len     = 800,
                 extended_state = False,
                 reset_acts     = False,
                 cont_ang       = False):

        super(Revolt, self).__init__()
        assert digitwin is not None, 'No digitwin was passed to Revolt environment'
        self.dTwin = digitwin
        self.name = 'full'
        
        ''' +++++++++++++++++++++++++++++++ '''
        '''     STATE AND ACTION SPACE      '''
        ''' +++++++++++++++++++++++++++++++ '''
        self.extended_state = extended_state
        self.num_actions = num_actions
        self.num_states  = num_states if not self.extended_state else num_states + 3

        # Set the name of actions in Cybersea
        self.actions = [
            {'idx': 0, 'module': 'THR1', 'feature': 'ThrustOrTorqueCmdMtc'}, # bow
            {'idx': 1, 'module': 'THR2', 'feature': 'ThrustOrTorqueCmdMtc'}, # stern, portside
            {'idx': 2, 'module': 'THR3', 'feature': 'ThrustOrTorqueCmdMtc'}, # stern, starboard
            {'idx': 3, 'module': 'THR1', 'feature': 'AzmCmdMtc'}, 
            {'idx': 4, 'module': 'THR2', 'feature': 'AzmCmdMtc'}, 
            {'idx': 5, 'module': 'THR3', 'feature': 'AzmCmdMtc'} ]

        bnds = {'action':{'low': -1*np.ones((self.num_actions,)), 'high': np.ones((self.num_actions,)) },
                'spaces':{'low': -1*np.ones((self.num_states,)),  'high': np.ones((self.num_states,))} }

        self.default_actions      = {0:0,1:0,2:0,3:0,4:0,5:0} # 
        self.act_2_act_map        = {0:0,1:1,2:2,3:3,4:4,5:5} # A map between action number and the environment specific action number (look on the other subclasses for examples of non-"one to one" mappings)
        self.act_2_act_map_inv    = self.act_2_act_map
        self.valid_action_indices = list(range(6))[0:self.num_actions] # NOTE  Only works this way for full env and simple: a list of all idx in self.actions that is allowed for this environment.
        self.action_space         = spaces.Box(low=bnds['action']['low'], high=bnds['action']['high'], dtype=np.float64) # action space bound in environment
        self.real_action_bounds   = [100] * 3 + [np.pi] * 3 # action space IRL
        self.observation_space    = spaces.Box(low=bnds['spaces']['low'], high=bnds['spaces']['high'], dtype=np.float64) # state space bound in environment # TODO not used
        self.real_ss_bounds       = real_ss_bounds # state space bound IRL
        self.EF                   = ErrorFrame()

        # Parameters used for the extended state vector
        self.prev_thrust = [0, 0, 0]
        self.prev_angles = [0, 0, 0]
        self.current_angles = [0, 0, 0]
        self.state_ext = np.zeros((9,))
        self.reset_actions = reset_acts

        ''' +++++++++++++++++++++++++++++++ '''
        '''     REWARD AND TEST PARAMS      '''
        ''' +++++++++++++++++++++++++++++++ '''
        self.vel_rew_coeffs = [0.5,0.5,1.0] # weighting between surge, sway and heading deviations used in reward function. Punish one rad/s twice as much as one m/s
        timesteps = 20
        self.n_steps    = 1 if (testing and realtime) else timesteps # I dont want to step at 100 Hz ever, really
        self.dt         = 0.01 * self.n_steps
        self.testing    = testing # stores if the environment is being used while testing policy, or is being used for training
        self.max_ep_len = int(max_ep_len * 10.0/self.n_steps) # 800 was specialized for 10 Hz - 5Hz needs less

        ''' Unitary multivariate gaussian reward parameters '''
        self.covar = np.array([ [1**2,      0   ],  # meters
                                [0,         5.0**2]])  # degrees
        self.covar_inv = np.linalg.inv(self.covar)

        self.cont_ang = cont_ang # predict sin_val  ~= sin(theta) and cos_val ~= cos(theta) instead of theta directly for circular continuity, finding theta = atan2(sin_val / cos_val)

    def step(self, action, new_ref=None):
        ''' Step a fixed number of steps in the Cybersea simulator 
        :args:
            action (numpy array): a (x,) shaped action provided by the agent
        :returns:
            state (numpy array): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """'''
        self.prev_angles = self.current_angles[:] # TODO wrap this so that there is no occurence of losing control over number of rotations

        if self.name == 'revoltfinal':
            if self.cont_ang: # transform sin(theta) and cos(theta) predictions into the action vector being passed on to Cybersea
                action = self.handle_continuous_angles(action)
            else:
                action = self.wrap_stern_angles(action)

        action = self.scale_and_clip(action)

        if DEBUGGING:
            outstr = ''
            for a in action: outstr += '{:.2f}\t'.format(a)
            print(outstr)

        for a in self.actions:
            if a['idx'] in self.valid_action_indices:
                idx = self.act_2_act_map[a['idx']]
                self.dTwin.val(a['module'], a['feature'], action[idx])
                if a['idx'] in [3,4,5]: # it is an angle, and in the valid action indices meaning that it has been changed
                    self.current_angles[a['idx'] - 3] = action[idx]

        self.dTwin.step(self.n_steps) # ReVolt is operating at 10 Hz. Input to step() is number of steps at 100 Hz
        s = self.state() if not self.extended_state else self.state_extended() # this uses previous time step thrust, so that the reward function can penalize it!
        self.prev_thrust = [action[0], action[1], action[2]] # These three will always be the first elements of the action vector...

        r = self.reward()
        d = self.is_terminal()

        if new_ref is not None: self.EF.update(ref=new_ref)

        return s,r,d, {'None': 0}

    def reset(self, new_ref = None, fraction = 0.8, fixed_point = None, **init):
        """ Resets the state of the environment and returns an initial observation.
        :returns:
            observation (object): the initial observation.
        """
        # Decide which initial values shall be set
        if not init:
            N, E, Y, u, v, r = 0, 0, 0, 0, 0, 0
            if not self.testing:
                    N, E, Y = get_pose_on_state_space(self.real_ss_bounds[0:3], fraction = fraction)
                    u, v, r = get_vel_on_state_space(self.real_ss_bounds[3:], fraction = 0.30 * fraction) # velocities are going to be low during DP, so dont sample too much of that
            else: 
                if fixed_point is None:
                    N, E, Y = get_random_pose_on_radius()
                else:
                    N, E, Y = get_fixed_pose_on_radius(n = fixed_point)

            init = {'Hull.PosNED':[N,E],'Hull.PosAttitude':[0,0,Y], 'Hull.VelocityNu':[u,v,0,0,0,r]}

        # Update error frame in case reference point has changed
        if self.testing and new_ref is not None:
            self.EF.update(ref=new_ref)

        # Update features like initial hull position and velocity
        for modfeat in init:
            module, feature = modfeat.split('.')
            self.dTwin.val(module, feature, init[modfeat])
            
        #reset critical models to clear states from last episode
        self.dTwin.val('Hull', 'StateResetOn', 1)
        self.dTwin.val('THR1', 'LinActuator', 2.0) # Make bow thruster come down from the hull
        self.dTwin.step(50)
        self.dTwin.val('Hull', 'StateResetOn', 0)

        for i in range(3):
            self.dTwin.val('THR'+str(i+1), 'MtcOn', 1) # turn on the motor control for all three thrusters

        # Notify simulator of all default thruster states
        for a in self.actions:
            default = self.default_actions[a['idx']]
            self.dTwin.val(a['module'], a['feature'], default) # set all default thruster states
            if a['idx'] in [3,4,5]: # It is an angle
                self.prev_angles[a['idx'] - 3] = default # TODO the simulator does not have time to get the angles back to zero - only the command of zero is being sent
        
        if self.reset_actions:
            action = np.zeros((len(self.valid_action_indices),))
            action[0:3] = np.random.normal(loc = 0.0, scale = 0.1, size=3)
            action = self.scale_and_clip(action)
            for a in self.actions:        
                if a['idx'] in self.valid_action_indices and a['idx'] in [0,1,2]: # This only affects thrust at the moment
                    idx = self.act_2_act_map[a['idx']]
                    self.dTwin.val(a['module'], a['feature'], action[idx])

            self.prev_thrust = action[0:3].copy()
        else:
            self.prev_thrust = [0,0,0]

        self.current_angles = self.prev_angles.copy()
        s = self.state() if not self.extended_state else self.state_extended()
        return s

    def state(self):
        ''' Returns the standard state vector of body frame errors + body frame velocities'''
        self.EF.update(get_pose_3DOF(self.dTwin))
        return np.array( self.EF.get_pose() + get_vel_3DOF(self.dTwin) ) # (x,) numpy array

    def state_extended(self):
        ''' Updates and returns the extended state formulation. 
        NB: since this function uses prev_thrust, the state_ext must be used when calculating the thrust derivatives to avoid getting 0 derivatives at each time step'''
        self.state_ext = np.hstack((self.state(),np.array(self.prev_thrust.copy()) / 100.0)) # (x,) numpy array
        return self.state_ext

    def is_terminal(self):
        ''' Returns true if the vessel has travelled too far from the set point.'''
        for s, bound in zip(self.state(),self.real_ss_bounds):
            if np.abs(s) > bound:
                return True

        return False

    def scale_and_clip(self,action):
        ''' Action from actor close to being a vector with vals between ish -1 and 1. Scale 100%, and clip.
        :args:
            - action (numpy array): an action provided by the agent
        :returns:
            A list of the scaled and clipped actions NB not a numpy array
         '''
        bnds = np.array(self.real_action_bounds) # select bounds according to environment specifications
        action = np.multiply(action,bnds) # The action comes as choices between -1 and 1...
        action = np.clip(action,-bnds,bnds) # ... but the std_dev in the stochastic policy means that we have to clip
        return action.tolist()

    def handle_continuous_angles(self,action):
        assert self.name.lower() == 'revoltfinal' and self.cont_ang is True, 'Using continuous angles is only made to work with the final environment fully rotating stern thrusters'
        sin_port, cos_port = action[3], action[4]
        sin_star, cos_star = action[5], action[6]
        a_port = np.arctan2(sin_port, cos_port) / self.real_action_bounds[3] # Since the scale and clip-function assumes an action between -1 and 1, the angle is scaled according to maximum
        a_star = np.arctan2(sin_star, cos_star) / self.real_action_bounds[3]
        new_action = np.hstack( (action[0:3], np.array([a_port, a_star])) )
        action = new_action.copy()
        return action

    def wrap_stern_angles(self,action):
        assert self.name.lower() == 'revoltfinal' and self.cont_ang is False, 'Using continuous angles is only made to work with the final environment fully rotating stern thrusters'
        # Actor output comes in -1,1 -> upscale before wrapping, then downscale again
        actor_port_angle = wrap_angle(action[3] * self.real_action_bounds[3],deg=False) / self.real_action_bounds[3]
        actor_star_angle = wrap_angle(action[4] * self.real_action_bounds[3],deg=False) / self.real_action_bounds[3]
        new_action = np.hstack( (action[0:3], np.array([actor_port_angle, actor_star_angle])) )
        action = new_action.copy()
        return action

    def render(self):
        pass # The environment will always be rendered in Cybersea
    
    ''' +++++++++++++++++++++++++++++++ '''
    '''           REWARD SHAPING        '''
    ''' +++++++++++++++++++++++++++++++ '''

    def reward(self):
        ''' Gives calculates reward in each state. 
        :returns:
            - A float representing the scalar reward of the agent being in the current state
        '''
        ### This is the reward function used on the last limited env (not mentioned in the thesis)
        # rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.1,0.1,0.1]) + self.action_derivative_penalty([0.05,0.075,0.075], angular = False) # actderros # changed derivatives according to what seems nice in ROS. high used 0.1, 0.1, 0.1

        ### Final Env
        # This reward function is the one used in the master's thesis.
        rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.20,0.30,0.30]) + self.action_derivative_penalty(thrust=True, pen_coeff=[0.05,0.05,0.05], angular=True, ang_coeff=[0.00,0.01,0.01])

        return rew  

    def vel_reward(self, coeffs = None):
        ''' Penalizes high velocities. Maximum penalty with real_bounds and all coeffs = 1 : -2.37. All coeffs = 0.5 : -1.675'''
        if coeffs is None:
            coeffs = self.vel_rew_coeffs
        assert len(coeffs) == 3 and isinstance(coeffs,list), 'Vel coeffs must be a list of length 3'

        return -np.sqrt(sum( [e**2 * c for e,c in zip(get_vel_3DOF(self.dTwin), coeffs)] ))

    def multivariate_gaussian(self,yaw_penalty=True):
        ''' Using a multivariate gaussian distribution without normalizing area to 1, with a diagonal covariance matrix and a linear "sparsity-regularizer" '''
        surge, sway, yaw = self.EF.get_pose()

        # multivariate gaussian reward function in r-yaw space
        r = np.sqrt(surge**2 + sway**2) # meters
        yaw = yaw * 180 / np.pi # Use degrees since that was the standard when creating the reward function - easier visualized than radians
        x = np.array([[r, yaw]]).T
        multivar = 2 * np.exp(-0.5 * (x.T).dot(self.covar_inv).dot(x))

        # Avoid sparse reward function
        low = -1.0 if yaw_penalty else 0.0
        special_measurement = np.sqrt(r**2 + (yaw * 0.25)**2) 
        anti_sparity = 1 * max(low, (1-0.1*special_measurement)) # this is steeper in the region of yaw, so the more negative the lower bound is, the more yaw_dist is penalized
        const = 0.5
        return  multivar + anti_sparity + const # can be viewed in reward_plots.py
        
    def thrust_penalty(self, pen_coeff = [0.1, 0.1, 0.1], torque_based = False):
        # assert np.all(np.array(pen_coeff) >= 0.0) and np.all(np.array(pen_coeff) <= 0.33), 'Action penalty coefficients must be in range 0.0 - 0.33'
        pen = 0
        if torque_based: # The penalty is based on torque, meaning that 
            for n,c in zip(self.prev_thrust, pen_coeff):
                pen -= np.abs(n)**3 / 10**5 * c # actually penalize like torque!
        else:
            for n,c in zip(self.prev_thrust, pen_coeff):
                pen -= np.abs(n) / 100.0 * c

        return pen

    def action_derivative_penalty(self,pen_coeff=[0.1,0.1,0.1], thrust = True, angular = False, ang_coeff = [0.03, 0.03, 0.03]):
        if not self.extended_state:
            # Dont use derivative penalties if the previous thrust is not in the state vector representation
            return 0

        pen = 0
        if thrust:
            derr = ( np.array(self.prev_thrust)-self.state_ext[-3:] * 100.0) / self.dt # prev_thrust stores the current thrust in (-100,100), while the last three elements of the extended state stores the prev thrust in (-1,1)
            for dT,c in zip(derr, pen_coeff):
                pen -= np.abs(dT / 100.0) * c # 200 is the maximum change from one second to another

        if angular:
            angpen = 0
            derr = (np.array(self.current_angles) - np.array(self.prev_angles)) / self.dt
            for dA, c in zip(derr,ang_coeff):
                bnd = self.real_action_bounds[4] # 4 is always an angle, being the last action
                angpen -= np.abs(dA / bnd) * c # 2 * bnd is the maximum change

            angpen = max(-1.0, angpen)
            pen += angpen        

        return pen

class RevoltSimple(Revolt):
    ''' +++++++++++++++++++++++++++++++ '''
    '''      FIXED THRUSTER SETUP       '''
    ''' +++++++++++++++++++++++++++++++ '''
    def __init__(self,digitwin, testing = False, realtime = False, max_ep_len = 800, extended_state=False, reset_acts = False,cont_ang=False):
        super().__init__(digitwin = digitwin, num_actions = 3, num_states = 6, testing = testing, 
                         realtime = realtime, max_ep_len = max_ep_len, extended_state = extended_state, reset_acts=reset_acts, cont_ang = False)

        self.name = 'revoltsimple'
        # Overwrite environment bounds according to measured max velocity for this specific setup
        self.real_ss_bounds = [8.0, 8.0, np.pi/2, 1.75, 0.30, 0.51]
       
        self.real_action_bounds   = [100] * 3
        # Overwrite default actions
        self.default_actions = {0: 0,
                                1: 0,
                                2: 0,
                                3: np.pi / 2,
                                4: -3 * np.pi / 4, # +- 135 degrees
                                5: 3 * np.pi / 4}

        self.act_2_act_map = {0: 0, 1: 1, 2: 2}
        self.act_2_act_map_inv = self.act_2_act_map

class RevoltLimited(Revolt):
    ''' +++++++++++++++++++++++++++++++ '''
    '''   LIMITED STERN AZIMUTH ANGLES  '''
    ''' +++++++++++++++++++++++++++++++ '''
    def __init__(self, digitwin, testing = False, realtime = False, max_ep_len = 800, extended_state = False, reset_acts = False, cont_ang=False):
        super().__init__(digitwin = digitwin, num_actions = 5, num_states = 6, testing = testing,
                         realtime = realtime, max_ep_len = max_ep_len, extended_state = extended_state, reset_acts = reset_acts, cont_ang=cont_ang)

        # Not choosing the bow angle means (1) one less action bound, (2) remove one valid index, (3) set bow angle default to pi/2
        self.name = 'revoltlimited'
        self.real_ss_bounds[2]    = 45 * np.pi / 180
        self.real_action_bounds   = [100] * 3 + [np.pi / 2 ] * 2
        self.valid_action_indices = [0,    1,      2,      4,      5]
        self.act_2_act_map        = {0:0,  1:1,    2:2,    4:3,    5:4} # {index in self.actions : index in action vector outputed by this actor for this env}
        self.act_2_act_map_inv    = {0:0,  1:1,    2:2,    3:4,    4:5} # {index in action vector outputed by this actor for this env : index self.actions}
        self.default_actions      = {0: 0, 
                                     1: 0, 
                                     2: 0, 
                                     3: np.pi / 2, 
                                     4: 0, 
                                     5: 0}

class RevoltFinal(Revolt):
    ''' +++++++++++++++++++++++++++++++ '''
    '''    FULL STERN AZIMUTH ANGLES    '''
    ''' +++++++++++++++++++++++++++++++ '''
    def __init__(self, digitwin, testing = False, realtime = False, max_ep_len = 800, extended_state = False, reset_acts = False, cont_ang = False):

        n_actions = 7 if cont_ang else 5

        super().__init__(digitwin = digitwin, num_actions = n_actions, num_states = 6, testing = testing,
                         realtime = realtime, max_ep_len = max_ep_len, extended_state = extended_state, reset_acts = reset_acts, cont_ang = cont_ang)

        # Not choosing the bow angle means (1) one less action bound, (2) remove one valid index, (3) set bow angle default to pi/2
        self.name = 'revoltfinal'
        self.real_ss_bounds[2]    = 45 * np.pi / 180
        
        # TODO using continous angles must be compatible with the valid indices etc. The easiest should be to have an extra transformation right after actor output

        self.real_action_bounds   = [100] * 3 + [np.pi] * 2
        self.valid_action_indices = [0,    1,      2,      4,      5]
        self.act_2_act_map        = {0:0,  1:1,    2:2,    4:3,    5:4} # {index in self.actions : index in action vector outputed by this actor for this env}
        self.act_2_act_map_inv    = {0:0,  1:1,    2:2,    3:4,    4:5} # {index in action vector outputed by this actor for this env : index self.actions}
        self.default_actions      = {0: 0, 
                                     1: 0, 
                                     2: 0, 
                                     3: np.pi / 2, 
                                     4: 0, 
                                     5: 0}


'''
REWARD FUNCTION STORAGE - all of these reward functions was used during reward shaping. Left here for legacy

# act_der_low - suggest 0.075 instead! 0.10 overfits
rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.1,0.1,0.1]) + self.action_derivative_penalty([0.05,0.05,0.05], angular = False)                 

# act_torque_high - gets better at using less thrust early
rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.1,0.3,0.3], torque_based=True)                                                                  

# actderangle - managed to get rid of angle flucts without having angles in the state vector, but stopped at 0 and 90 degs, which is actually OK as it does not lock in singular configuration
rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.1,0.1,0.1]) + self.action_derivative_penalty([0.05,0.05,0.05], thrust = False, angular = True)  

# Test strict heading by adding a penalty on heading, yawstdpen (this worked fine, and the minimum on the antisparity has been set to -1 per standard)
rew = self.vel_reward() + self.multivariate_gaussian(yaw_penalty=True) + self.thrust_penalty([0.1,0.1,0.1])                                                                     

# realtorquelow
rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.03,0.03,0.03], torque_based=True)                                                               

# acrderanglelow
rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.1,0.1,0.1]) + self.action_derivative_penalty(thrust=False, angular=True,ang_coeff=[0.03,0.03,0.03]) 

# actderallsmall with all worked fine!
rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.1,0.1,0.1]) + self.action_derivative_penalty(pen_coeff=[0.01,0.01,0.01], thrust=True, angular=True, ang_coeff=[0.02,0.02,0.02]) 

# antisparitized gaussian trained for longer
rew = self.vel_reward() + self.summed_gaussian_like() + self.thrust_penalty([0.1,0.1,0.1])                                                                                      

# Old gaussian summed with multivar for best of both worlds
rew = self.vel_reward() + self.summed_gaussian_with_multivariate() + self.thrust_penalty([0.1,0.1,0.1])                                                                        

# actderallsmallest to see if it can become stable
rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.1,0.1,0.1]) + self.action_derivative_penalty(pen_coeff=[0.00,0.01,0.01], thrust=True, angular=True, ang_coeff=[0.00,0.01,0.01])  

# rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.03,0.03,0.03], torque_based=True) + self.action_derivative_penalty(thrust=False,                            angular=True, ang_coeff=[0.02,0.02,0.02]) # finconttorque
# rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.1,0.1,0.1])                       + self.action_derivative_penalty(thrust=True, pen_coeff=[0.00,0.01,0.01], angular=True, ang_coeff=[0.00,0.01,0.01]) # fincontactderall
# rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.15,0.15,0.15])                    + self.action_derivative_penalty(thrust=True, pen_coeff=[0.00,0.05,0.05], angular=True, ang_coeff=[0.00,0.01,0.01]) # finconttotal
# rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.20,0.30,0.30])                    + self.action_derivative_penalty(thrust=True, pen_coeff=[0.00,0.05,0.05], angular=True, ang_coeff=[0.00,0.01,0.01]) # finconttotalhigh
# rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.10,0.10,0.10])                    + self.action_derivative_penalty(thrust=True, pen_coeff=[0.05,0.10,0.10], angular=True, ang_coeff=[0.00,0.01,0.01]) # finconttotalhderhigh
# rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.05,0.05,0.05], torque_based=True) + self.action_derivative_penalty(thrust=True, pen_coeff=[0.02,0.02,0.02], angular=True, ang_coeff=[0.02,0.02,0.02]) # finconttotaltorque
# rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.075,0.075,0.075], torque_based=True) + self.action_derivative_penalty(thrust=True, pen_coeff=[0.05,0.075,0.075], angular=True, ang_coeff=[0.02,0.02,0.02]) # finconttorqueactderrosangle
# rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.10,0.10,0.10])                    + self.action_derivative_penalty(thrust=True, pen_coeff=[0.02,0.02,0.02], angular=True, ang_coeff=[0.00,0.02,0.02]) # finactderallangleup
# rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.15,0.15,0.15]) + self.action_derivative_penalty(thrust=True, pen_coeff=[0.00,0.05,0.05], angular=True, ang_coeff=[0.00,0.01,0.01]) # finconttotal - good for iterative approach
# rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.15,0.15,0.15]) + self.action_derivative_penalty(thrust=True, pen_coeff=[0.02,0.02,0.02], angular=True, ang_coeff=[0.00,0.04,0.04]) # finactderallangleup / finimprovement
# rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.15,0.15,0.15]) + self.action_derivative_penalty(thrust=True, pen_coeff=[0.05,0.05,0.05], angular=True, ang_coeff=[0.00,0.01,0.01]) # finconttotbowder
# rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.15,0.20,0.20]) + self.action_derivative_penalty(thrust=True, pen_coeff=[0.05,0.05,0.05], angular=True, ang_coeff=[0.00,0.01,0.01]) # finconttotallup
# rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.15,0.30,0.30]) + self.action_derivative_penalty(thrust=True, pen_coeff=[0.05,0.05,0.05], angular=True, ang_coeff=[0.00,0.02,0.02]) # finconttothighall
# rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.20,0.30,0.30]) + self.action_derivative_penalty(thrust=True, pen_coeff=[0.05,0.05,0.05], angular=True, ang_coeff=[0.00,0.01,0.01]) # finconttothighbowder
'''
