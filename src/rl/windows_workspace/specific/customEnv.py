import gym 
from gym import spaces
import numpy as np
from specific.misc.simtools import get_pose_3DOF, get_vel_3DOF, get_pose_on_state_space, get_random_pose_on_radius, get_vel_on_state_space, get_fixed_pose_on_radius
from specific.misc.mathematics import gaussian, gaussian_like
from specific.errorFrame import ErrorFrame
import time

class Revolt(gym.Env):
    """ Custom Environment that follows OpenAI's gym API.
        Max velocities measured with no thrust losses activated. Full means rotating stern azimuths only.
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
                 real_ss_bounds = [8.0, 8.0, np.pi/2, 1.4, 0.30, 0.52],
                 testing        = False,
                 realtime       = False,
                 max_ep_len     = 800,
                 extended_state = False,
                 reset_acts = False):

        super(Revolt, self).__init__()
        assert digitwin is not None, 'No digitwin was passed to Revolt environment'
        self.dTwin = digitwin
        self.name = 'revolt'
        
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
        self.n_steps    = 1 if (testing and realtime) else 10 # I dont want to step at 100 Hz ever, really
        self.dt         = 0.01 * self.n_steps
        self.testing    = testing # stores if the environment is being used while testing policy, or is being used for training
        self.max_ep_len = max_ep_len * int(10/self.n_steps)

        ''' Unitary multivariate gaussian reward parameters '''
        self.covar = np.array([ [1**2,      0   ],  # meters
                                [0,         5.0**2]])  # degrees
        self.covar_inv = np.linalg.inv(self.covar)

    def step(self, action, new_ref=None):
        ''' Step a fixed number of steps in the Cybersea simulator 
        :args:
            action (numpy array): an action provided by the agent
        :returns:
            observation (numpy array): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """'''
        self.prev_angles = self.current_angles[:]

        action = self.scale_and_clip(action)
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
        # TODO if the previous actions are to be put into the state-vector, reset() must set random previous actions, or all previous actions must be set to zero

        # Decide which initial values shall be set
        if not init:
            N, E, Y, u, v, r = 0, 0, 0, 0, 0, 0
            if not self.testing:
                    N, E, Y = get_pose_on_state_space(self.real_ss_bounds[0:3], fraction = fraction)
                    u, v, r = get_vel_on_state_space(self.real_ss_bounds[3:], fraction = 0.25 * fraction)
            else: 
                if fixed_point is None:
                    N, E, Y = get_random_pose_on_radius()
                else:
                    N, E, Y = get_fixed_pose_on_radius(n = fixed_point)

            init = {'Hull.PosNED':[N,E],'Hull.PosAttitude':[0,0,Y], 'Hull.VelocityNu':[u,v,0,0,0,r]}

        if self.testing and new_ref is not None:
            self.EF.update(ref=new_ref)

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
        # TODO this more pretty, change between using default and random vals based on self.reset_actions!
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
                    # TODO reset angles if adding them to state vector also

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
        ''' Action from actor if close to being -1 and 1. Scale 100%, and clip.
        :args:
            - action (numpy array): an action provided by the agent
        :returns:
            A list of the scaled and clipped actions NB not a numpy array
         '''
        bnds = np.array(self.real_action_bounds[0:self.num_actions]) # select bounds according to environment specifications
        action = np.multiply(action,bnds) # The action comes as choices between -1 and 1...
        action = np.clip(action,-bnds,bnds) # ... but the std_dev in the stochastic policy means that we have to clip
        return action.tolist()

    def render(self):
        pass # The environment will always be rendered in Cybersea
    
    ''' +++++++++++++++++++++++++++++++ '''
    '''           REWARD SHAPING        '''
    ''' +++++++++++++++++++++++++++++++ '''

    def reward(self):
        '''
        :returns:
            - A float representing the scalar reward of the agent being in the current state
        '''
        # rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.1,0.1,0.1]) # BEST, but probably only since the penalty avoids thrusters being on MAX, but doesnt necessary minimize the usage

        # experiences using action penalties:
        # rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.1,0.1,0.1]) + self.action_derivative_penalty([0.05,0.05,0.05], angular = False) # act_der_low - suggest 0.075 instead! 0.10 overfits
        # rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.1,0.3,0.3], torque_based=True) # 1 # act_torque_high - gets better at using less thrust early
        # rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.1,0.1,0.1]) + self.action_derivative_penalty([0.05,0.05,0.05], thrust = False, angular = True) # actderangle - managed to get rid of angle flucts without having angles in the state vector, but stopped at 0 and 90 degs, which is actually OK as it does not lock in singular configuration

        # rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.1,0.1,0.1]) # best with long training
        # rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.1,0.03,0.03], torque_based=True) # realtorquelow
        # rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.1,0.1,0.1]) + self.action_derivative_penalty(thrust=False, angular=True,ang_coeff=[0.03,0.03,0.03]) 
        # rew = self.vel_reward() + self.multivariate_gaussian(yaw_penalty=True) + self.thrust_penalty([0.1,0.1,0.1]) # Test strict heading by adding a penalty on heading
        # rew = self.vel_reward() + self.smaller_yaw_dist() + self.thrust_penalty([0.1,0.1,0.1]) # Test old gaussian to see if it actually works going for the yaw first
        rew = self.vel_reward() + self.multivariate_gaussian() + self.thrust_penalty([0.1,0.1,0.1]) + self.action_derivative_penalty(pen_coeff=[0.01,0.01,0.01], thrust=True, angular=True, ang_coeff=[0.02,0.02,0.02]) # actderallsmall

        return rew  

    def vel_reward(self, coeffs = None):
        ''' Penalizes high velocities. Maximum penalty with real_bounds and all coeffs = 1 : -2.37. All coeffs = 0.5 : -1.675'''
        if coeffs is None:
            coeffs = self.vel_rew_coeffs
        assert len(coeffs) == 3 and isinstance(coeffs,list), 'Vel coeffs must be a list of length 3'

        return -np.sqrt(sum( [e**2 * c for e,c in zip(get_vel_3DOF(self.dTwin), coeffs)] ))

    def vel_reward_2(self, coeffs = None):
        ''' Penalizes high velocities. Maximum penalty with real_bounds and all coeffs = 1 : -2.37. All coeffs = 0.5 : -1.675'''
        if coeffs is None:
            coeffs = self.vel_rew_coeffs
        assert len(coeffs) == 3 and isinstance(coeffs,list), 'Vel coeffs must be a list of length 3'

        pen = 0
        bnds = self.real_ss_bounds[-3:]
        vels = get_vel_3DOF(self.dTwin)
        for vel, bnd, cof in zip(vels,bnds,coeffs):
            pen -= np.abs(vel) / bnd * cof

        return pen

    def smaller_yaw_dist(self):
        ''' First reward function that fixed the steady state error in yaw by sharpening the yaw gaussian '''
        surge, sway, yaw = self.EF.get_pose()
        rews = gaussian_like([surge,sway]) # mean 0 and var == 1
        yawrew = gaussian_like([yaw], var=[0.1**2]) # Before, using var = 1, there wasnt any real difference between surge and sway and yaw
        r = np.sqrt(surge**2 + sway**2) # meters
        special_measurement = np.sqrt(r**2 + (yaw * 0.25)**2) 
        anti_sparity = max(0.0, (1-0.1*special_measurement))
        return sum(rews) / 2 + 2 * yawrew + anti_sparity

    def multivariate_gaussian(self,yaw_penalty=False):
        ''' Using a multivariate gaussian distribution without normalizing area to 1, with a diagonal covariance matrix and a linear "sparsity-regularizer" '''
        surge, sway, yaw = self.EF.get_pose()
        r = np.sqrt(surge**2 + sway**2) # meters
        yaw = yaw * 180 / np.pi # Use degrees since that was the standard when creating the reward function - easier visualized than radians
        special_measurement = np.sqrt(r**2 + (yaw * 0.25)**2) 
        x = np.array([[r, yaw]]).T

        yaw_pen = 0 
        # if yaw_penalty: # TODO this seems to penalize radius more...
        #     yaw_pen = max(-1,0 - np.abs(yaw)/45.0)
        if yaw_penalty:
            low = -1.0
        else:
            low = 0.0

        return 2 * np.exp(-0.5 * (x.T).dot(self.covar_inv).dot(x)) + 1 * max(low, (1-0.1*special_measurement)) + 0.5 + yaw_pen # can be viewed in reward_plots.py
        
    def multivar(self):
        surge, sway, yaw = self.EF.get_pose()
        r = np.sqrt(surge**2 + sway**2)
        yaw_vs_r_factor = 0.25 # how much one degree is weighted vs a meter
        r3d = np.sqrt(r**2 + (yaw * 180 / np.pi * 0.25)**2)
        return gaussian_like(val = [r3d], mean = [0], var = [2.0**2]) + max(0.0, (1-0.05*r3d))

    def thrust_penalty(self, pen_coeff = [0.1, 0.1, 0.1], torque_based = False):
        # assert np.all(np.array(pen_coeff) >= 0.0) and np.all(np.array(pen_coeff) <= 0.33), 'Action penalty coefficients must be in range 0.0 - 0.33'
        pen = 0
        if torque_based: # The penalty is based on torque, meaning that 
            for n,c in zip(self.prev_thrust, pen_coeff):
                # pen -= np.abs(n / 100.0)**1.5 * c # NBNB the power is lower than regular penalty between 0.0 and 1.0!!!
                # pen -= np.abs(n)**1.5 / 400.0 * c # This allows for the max action penalty to become 2.5 - as large as max vel penalty
                pen -= np.abs(n)**3 / 10**5 * c # actually penalize like torque!
        else:
            for n,c in zip(self.prev_thrust, pen_coeff):
                pen -= np.abs(n) / 100.0 * c

        return pen # maximum penalty is 1 per time step if coeffs are <= 0.33

    def action_derivative_penalty(self,pen_coeff=[0.2,0.2,0.2], thrust = True, angular = False, ang_coeff = [0.03, 0.03, 0.03]):
        if not self.extended_state:
            return 0

        # assert np.all(np.array(pen_coeff) >= 0.0) and np.all(np.array(pen_coeff) <= 0.33), 'Action penalty coefficients must be in range 0.0 - 0.33'
        pen = 0

        if thrust:
            derr = ( np.array(self.prev_thrust)-self.state_ext[-3:] * 100.0) / self.dt # prev_thrust stores the current thrust in (-100,100), while the last three elements of the extended state stores the prev thrust in (-1,1)

            for dT,c in zip(derr, pen_coeff):
                pen -= np.abs(dT / 100.0) * c # 200 is the maximum change from one second to another

            pen = max(-1.0, pen)

        if angular:
            angpen = 0
            derr = (np.array(self.current_angles) - np.array(self.prev_angles)) / self.dt
            for dA, c in zip(derr,ang_coeff):
                bnd = self.real_action_bounds[4] # 4 is always an angle, being the last action
                angpen -= np.abs(dA / bnd) * c # 2 * bnd is the maximum change

            angpen = max(-1.0, angpen)
            pen += angpen        

        return pen # maximum penalty is 1 per time step if coeffs are <= 0.33 and division is done by 200.0

class RevoltSimple(Revolt):
    ''' +++++++++++++++++++++++++++++++ '''
    '''      FIXED THRUSTER SETUP       '''
    ''' +++++++++++++++++++++++++++++++ '''
    def __init__(self,digitwin, testing = False, realtime = False, max_ep_len = 800, extended_state=False, reset_acts = False):
        super().__init__(digitwin = digitwin, num_actions = 3, num_states = 6, testing = testing, 
                         realtime = realtime, max_ep_len = max_ep_len, extended_state = extended_state, reset_acts=reset_acts)

        self.name = 'revoltsimple'
        # Overwrite environment bounds according to measured max velocity for this specific setup
        self.real_ss_bounds = [8.0, 8.0, np.pi/2, 1.75, 0.30, 0.51] # TODO vel could be set to much smaller values: (scale 10,10,100 times to get them in the same range as the positional arguments)

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
    '''      LIMITED AZIMUTH ANGLES     '''
    ''' +++++++++++++++++++++++++++++++ '''
    def __init__(self, digitwin, testing = False, realtime = False, max_ep_len = 800, extended_state = False, reset_acts = False):
        super().__init__(digitwin = digitwin, num_actions = 5, num_states = 6, testing = testing,
                         realtime = realtime, max_ep_len = max_ep_len, extended_state = extended_state, reset_acts = reset_acts)

        # Not choosing the bow angle means (1) one less action bound, (2) remove one valid index, (3) set bow angle default to pi/2
        self.name = 'revoltlimited'
        self.real_ss_bounds[2]    = 45 * np.pi / 180 # TODO increased
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
