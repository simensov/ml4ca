import gym 
from gym import spaces
import numpy as np
import math
from specific.misc.simtools import get_pose_3DOF, get_vel_3DOF, get_pose_on_state_space, get_pose_on_radius, get_vel_on_state_space
from specific.misc.mathematics import gaussian, gaussian_like
from specific.errorFrame import ErrorFrame

class Revolt(gym.Env):
    """ Custom Environment that follows OpenAI's gym API.
        Max velocities measured with no thrust losses activated. Full means rotating stern azimuths only.
            Full:   surge, sway, yaw = (+2.20, -1.60) m/s, +-0.35 m/s, +-0.60 rad/s
            Simple: surge, sway, yaw = (+1.75, -1.40) m/s, +-0.30 m/s, +-0.51 rad/s
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 digitwin       = None,
                 num_actions    = 6,
                 num_states     = 6,
                 real_ss_bounds = [8.0, 8.0, np.pi/2, 2.2, 0.35, 0.60],
                 testing        = False,
                 realtime       = False,
                 max_ep_len     = 800,
                 extended_state = False):

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
        self.real_action_bounds   = [100] * 3 + [math.pi] * 3 # action space IRL
        self.observation_space    = spaces.Box(low=bnds['spaces']['low'], high=bnds['spaces']['high'], dtype=np.float64) # state space bound in environment
        self.real_ss_bounds       = real_ss_bounds # state space bound IRL
        self.EF                   = ErrorFrame()

        ''' +++++++++++++++++++++++++++++++ '''
        '''     REWARD AND TEST PARAMS      '''
        ''' +++++++++++++++++++++++++++++++ '''
        self.vel_rew_coeffs = [0.5,0.5,0.5] # weighting between surge, sway and heading deviations used in reward function
        self.n_steps    = 1 if (testing and realtime) else 10 # I dont want to step at 100 Hz ever, really
        self.dt         = 0.01 * self.n_steps
        self.testing    = testing # stores if the environment is being used while testing policy, or is being used for training
        self.max_ep_len = max_ep_len * int(10/self.n_steps)

        ''' Unitary gaussian reward parameters '''
        self.covar = np.array([ [1**2,     0   ],  # meters
                                [0,     (5)**2]])  # degrees
        self.covar_inv = np.linalg.inv(self.covar)
        self.covar_det = np.linalg.det(self.covar)

        ''' Storage of previous thrust '''
        self.prev_thrust = [0, 0, 0]

    def __str__(self):
        return str(self.dTwin.name)

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

        action = self.scale_and_clip(action)
        for a in self.actions:
            if a['idx'] in self.valid_action_indices:
                # self.dTwin.val(a['module'], a['feature'], action[a['idx']]) # original, working with revoltsimple, but once the action vector couldnt be followed chronologically as for limited, it broke
                idx = self.act_2_act_map[a['idx']]
                self.dTwin.val(a['module'], a['feature'], action[idx])

        self.dTwin.step(self.n_steps) # ReVolt is operating at 10 Hz. Input to step() is number of steps at 100 Hz
        s = self.state() if not self.extended_state else self.state_extended() # this uses previous time step thrust
        self.prev_thrust = [action[0], action[1], action[2]]
        r = self.reward()
        d = self.is_terminal()
        

        if new_ref is not None:
            self.EF.update(ref=new_ref)

        return s,r,d, {'None': 0}

    def scale_and_clip(self,action, scale=True, clip=True):
        ''' Action from actor if close to being -1 and 1. Scale 100%, and clip.
        :args:
            - action (numpy array): an action provided by the agent
        :returns:
            A list of the scaled and clipped actions
         '''
        bnds = np.array(self.real_action_bounds[0:self.num_actions]) # select bounds according to environment specifications
        if scale:
            action = np.multiply(action,bnds) # The action comes as choices between -1 and 1...
        if clip:
            action = np.clip(action,-bnds,bnds) # ... but the std_dev in the stochastic policy means that we have to clip
        return action.tolist()

    def reset(self, new_ref = None, fraction = 0.8, **init):
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
                N, E, Y = get_pose_on_radius()

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

        for a in self.actions:
            self.dTwin.val(a['module'], a['feature'], self.default_actions[a['idx']]) # set all default thruster states

        self.prev_thrust = [0,0,0]
        s = self.state() if not self.extended_state else self.state_extended()
        return s

    def state(self):
        self.EF.update(get_pose_3DOF(self.dTwin))
        return np.array( self.EF.get_pose() + get_vel_3DOF(self.dTwin) ) # (x,) numpy array

    def state_extended(self):
        return np.hstack((self.state(),np.array(self.prev_thrust) / 100.0)) # (x,) numpy array

    def is_terminal(self):
        ''' Returns true if the vessel has travelled too far from the set point.'''
        for s, bound in zip(self.state(),self.real_ss_bounds):
            if np.abs(s) > bound:
                return True
        return False

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
        # rew = self.vel_reward() + self.pose_reward()
        # rew = self.vel_reward() + self.smaller_yaw_dist()
        # rew = self.vel_reward() + self.unitary_multivariate_reward_2D() # multivar
        rew = self.vel_reward() + self.smaller_yaw_dist() + self.action_penalty(pen_coeff = [0.2, 0.1, 0.1]) # simactpen, limactpen has 0.5 on bow
        # rew = self.vel_reward() + self.new_multivar() + self.action_penalty(pen_coeff = [0.1, 0.1, 0.1]) # limcomplicated WORKED WELL! 
        # rew = self.vel_reward() + self.new_multivar() + self.action_penalty([0.2,0.1,0.1])
        # rew = self.vel_reward() + self.unitary_multivariate_reward() #  + self.action_penalty([0.1,0.1,0.1]) # newcomplicated and newlimcomplicated
        # rew = self.vel_reward() + SOME_MULTIVAR + self.action_penalty([0.1,0.1,0.1]) # newcomplicated and newlimcomplicated
        return rew  

    def pose_reward(self):
        rews = gaussian(self.EF.get_pose())
        return sum(rews)

    def vel_reward(self):
        ''' Penalizes high velocities. Maximum penalty with real_bounds and all coeffs = 1 : -2.37. All coeffs = 0.5 : -1.675'''
        return -math.sqrt(sum( [e**2 * c for e,c in zip(get_vel_3DOF(self.dTwin), self.vel_rew_coeffs)] ))

    def smaller_yaw_dist(self):
        ''' First reward function that fixed the steady state error in yaw by sharpening the yaw gaussian '''
        surge, sway, yaw = self.EF.get_pose()
        rews = gaussian([surge,sway]) # mean 0 and var == 1
        yawrew = gaussian([yaw], var=[0.1**2]) # Before, using var = 1, there wasnt any real difference between surge and sway and yaw
        return sum(rews) + yawrew

    def unitary_multivariate_reward(self):
        surge, sway, yaw = self.EF.get_pose()
        r = math.sqrt(surge**2 + sway**2) # meters
        yaw = yaw * 180 / math.pi # Use degrees since that was the standard when creating the reward function - easier visualized than radians
        special_measurement = math.sqrt(r**2 + (yaw * 0.25)**2) 
        x = np.array([[r, yaw]]).T
        return 2 * np.exp(-0.5 * (x.T).dot(self.covar_inv).dot(x)) + max(0.0, (1-0.1*special_measurement)) + 0.5 # can be viewed in reward_plots.py
        
    def new_multivar(self):
        surge, sway, yaw = self.EF.get_pose()
        r = math.sqrt(surge**2 + sway**2)
        yaw_vs_r_factor = 0.25 # how much one degree is weighted vs a meter
        r3d = math.sqrt(r**2 + (yaw * 180 / np.pi * 0.25)**2)
        return gaussian_like(val = [r3d], mean = [0], var = [2.0**2]) + max(0.0, (1-0.05*r3d))

    def action_penalty(self, pen_coeff = [0.1, 0.1, 0.1], torque_based = False):
        assert np.all(np.array(pen_coeff) >= 0.0) and np.all(np.array(pen_coeff) <= 0.33), 'Action penalty coefficients must be in range 0.0 - 0.33'
        pen = 0

        if torque_based: # The penalty is based on torque, meaning that 
            for n,c in zip(self.prev_thrust, pen_coeff):
                pen -= np.abs(n**1.5) / 1000.0 * c
        else:
            for n,c in zip(self.prev_thrust, pen_coeff):
                pen -= np.abs(n) / 100.0 * c

        
        return pen # maximum penalty is 1 per time step if coeffs are <= 0.33

    def action_der_pen(self,pen_coeff=[0.2,0.2,0.2]):
        if not self.extended_state:
            return 0

        assert np.all(np.array(pen_coeff) >= 0.0) and np.all(np.array(pen_coeff) <= 0.33), 'Action penalty coefficients must be in range 0.0 - 0.33'
        pen = 0
        # prev_thrust stores the current thrust in (-100,100), while the last three elements of the extended state stores the prev thrust in (-1,1)
        derr = (np.array(self.prev_thrust) - self.state_extended()[-3:]*100 )/ self.dt 
        for dT,c in zip(derr, pen_coeff):
            pen -= np.abs(dT) / 200.0 * c # 200 is the maximum change from one second to another
        return pen # maximum penalty is 1 per time step if coeffs are <= 0.33

class RevoltSimple(Revolt):
    ''' +++++++++++++++++++++++++++++++ '''
    '''      FIXED THRUSTER SETUP       '''
    ''' +++++++++++++++++++++++++++++++ '''
    def __init__(self,digitwin, testing = False, realtime = False, max_ep_len = 800, extended_state=False):
        super().__init__(digitwin = digitwin, num_actions = 3, num_states = 6,
                         testing = testing, realtime = realtime, max_ep_len = max_ep_len, extended_state = extended_state)

        self.name = 'revoltsimple'
        # Overwrite environment bounds according to measured max velocity for this specific setup
        self.real_ss_bounds = [8.0, 8.0, np.pi/2, 1.75, 0.30, 0.51] # TODO vel could be set to much smaller values: (scale 10,10,100 times to get them in the same range as the positional arguments)

        # Overwrite default actions
        self.default_actions = {0: 0,
                                1: 0,
                                2: 0,
                                3: math.pi / 2,
                                4: -3 * math.pi / 4, # +- 135 degrees
                                5: 3 * math.pi / 4}

        self.act_2_act_map = {0: 0, 1: 1, 2: 2}
        self.act_2_act_map_inv = self.act_2_act_map

class RevoltLimited(Revolt):
    ''' +++++++++++++++++++++++++++++++ '''
    '''      LIMITED AZIMUTH ANGLES     '''
    ''' +++++++++++++++++++++++++++++++ '''
    def __init__(self,digitwin,testing=False,realtime=False, extended_state = False):
        super().__init__(digitwin,num_actions=5,num_states=6,testing=testing,realtime=realtime,extended_state=extended_state)

        # Not choosing the bow angle means (1) one less action bound, (2) remove one valid index, (3) set bow angle default to pi/2
        self.real_ss_bounds       = [8.0, 8.0, 30 * np.pi / 180, 2.2, 0.35, 0.60] 
        self.real_action_bounds   = [100] * 3 + [math.pi / 2 ] * 2
        self.valid_action_indices = [ 0,    1,      2,      4,      5]
        self.act_2_act_map        = { 0:0,  1:1,    2:2,    4:3,    5:4} # {index in self.actions : index in action vector outputed by this actor for this env}
        self.act_2_act_map_inv    = { 0:0,  1:1,    2:2,    3:4,    4:5} # {index in action vector outputed by this actor for this env : index self.actions}
        self.default_actions      = {0:0, 1:0, 2:0, 3: math.pi / 2, 4:0, 5:0}
        self.name = 'revoltlimited'

        
        
