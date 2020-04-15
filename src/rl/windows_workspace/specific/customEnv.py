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
                 digitwin,
                 num_actions = 6, 
                 num_states = 6,
                 real_ss_bounds=[8.0, 8.0, np.pi/2, 2.2, 0.35, 0.60], 
                 norm_env = False,
                 testing = False, 
                 realtime = False, 
                 max_ep_len = 800, 
                 curriculum = False):

        super(Revolt, self).__init__()
        self.dTwin = digitwin
        
        ''' +++++++++++++++++++++++++++++++ '''
        '''     STATE AND ACTION SPACE      '''
        ''' +++++++++++++++++++++++++++++++ '''
        self.num_actions          = num_actions
        self.num_states           = num_states

        # Set the name of actions in Cybersea
        self.actions = [
            {'idx': 0, 'module': 'THR1', 'feature': 'ThrustOrTorqueCmdMtc'}, # bow
            {'idx': 1, 'module': 'THR2', 'feature': 'ThrustOrTorqueCmdMtc'}, # stern, portside
            {'idx': 2, 'module': 'THR3', 'feature': 'ThrustOrTorqueCmdMtc'}, # stern, starboard
            {'idx': 3, 'module': 'THR1', 'feature': 'AzmCmdMtc'}, 
            {'idx': 4, 'module': 'THR2', 'feature': 'AzmCmdMtc'}, 
            {'idx': 5, 'module': 'THR3', 'feature': 'AzmCmdMtc'} ]

        bnds = {'action':{'low': -1*np.ones((num_actions,)), 'high': np.ones((num_actions,)) },
                'spaces':{'low': -1*np.ones((num_states,)),  'high': np.ones((num_states,))} }

        self.default_actions      = {0:0,1:0,2:0,3:0,4:0,5:0}
        self.act_2_act_map        = {0:0,1:1,2:2,3:3,4:4,5:5}
        self.valid_action_indices = list(range(6))[0:num_actions] # NOTE  Only works this way for full env and simple: a list of all idx in self.actions that is allowed for this environment.
        self.action_space         = spaces.Box(low=bnds['action']['low'], high=bnds['action']['high'], dtype=np.float64) # action space bound in environment
        self.real_action_bounds   = [100] * 3 + [math.pi] * 3 # action space IRL
        self.observation_space    = spaces.Box(low=bnds['spaces']['low'], high=bnds['spaces']['high'], dtype=np.float64) # state space bound in environment
        self.real_ss_bounds       = real_ss_bounds # state space bound IRL
        self.norm_env             = norm_env
        self.EF                   = ErrorFrame()

        ''' +++++++++++++++++++++++++++++++ '''
        '''     REWARD AND TEST PARAMS      '''
        ''' +++++++++++++++++++++++++++++++ '''
        self.vel_rew_coeffs = [1,1,1] # weighting between surge, sway and heading deviations used in reward function
        self.n_steps    = 1 if (testing and realtime) else 10 # I dont want to step at 100 Hz ever, really
        self.testing    = testing # stores if the environment is being used while testing policy, or is being used for training
        self.max_ep_len = max_ep_len * int(10/self.n_steps)
        self.curriculum = curriculum # parameter for gradually increasing the sampled state space during training


        ''' 2D reward parameters '''
        self.covar = np.array([[1,0],[0,0.1**2]]) # meters and radians
        self.covar_inv = np.linalg.inv(self.covar)
        self.covar_det = np.linalg.det(self.covar)

        ''' 3D Reward parameters '''
        self.Sigma = np.array([[1.0,0.0,0.0],
                               [0.0,1.0,0.0],
                               [0.0,0.0,0.1**2]]) # meters and radians
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        self.Sigma_det = np.linalg.det(self.Sigma)

        ''' Storage of previous thrust '''
        self.thrust_taken = [0, 0, 0]

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
        
        self.thrust_taken = [action[0], action[1], action[2]]

        for a in self.actions:
            if a['idx'] in self.valid_action_indices:
                # self.dTwin.val(a['module'], a['feature'], action[a['idx']]) # original, working with revoltsimple, but once the action vector couldnt be followed chronologically as for limited, it broke
                idx = self.act_2_act_map[a['idx']]
                self.dTwin.val(a['module'], a['feature'], action[idx])

        self.dTwin.step(self.n_steps) # ReVolt is operating at 10 Hz. Input to step() is number of steps at 100 Hz
        s = self.state()
        r = self.reward()
        d = self.is_terminal(s)

        if self.norm_env: s = self.normalize_state(s)

        if new_ref is not None:
            self.EF.update(ref=new_ref)

        return s,r,d, {'None': 0}

    def scale_and_clip(self,action):
        ''' Action from actor if close to being -1 and 1. Scale 100%, and clip.
        :args:
            - action (numpy array): an action provided by the agent
        :returns:
            A list of the scaled and clipped action
         '''
        bnds = np.array(self.real_action_bounds[0:self.num_actions]) # select bounds according to environment specifications
        action = np.multiply(action,bnds) # The action comes as choices between -1 and 1...
        action = np.clip(action,-bnds,bnds) # ... but the std_dev in the stochastic policy means that we have to clip
        return action.tolist()

    def reset(self, new_ref = None, fraction = 1.0, **init):
        """ Resets the state of the environment and returns an initial observation.
        :returns:
            observation (object): the initial observation.
        """
        # TODO if the previous actions are to be put into the state-vector, reset() must set random previous actions, or all previous actions must be set to zero

        # Decide which initial values shall be set
        if not init:
            N, E, Y, u, v, r = 0, 0, 0, 0, 0, 0
            if not self.testing:
                if not self.curriculum: # TODO this shouldnt be a part of env
                    N, E, Y = get_pose_on_state_space(self.real_ss_bounds[0:3], fraction = fraction)
                else:
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

        s = self.state()
        if self.norm_env: s = self.normalize_state(s)

        self.thrust_taken = [0,0,0]
        return s

    def state(self):
        self.EF.update(get_pose_3DOF(self.dTwin))
        return np.array( self.EF.get_pose() + get_vel_3DOF(self.dTwin) ) # (6,) numpy array

    def normalize_state(self,state):
        ''' Based on normalizing a symmetric state distribution: x = (x - x_min) / (x_max - x_min) -> https://www.statisticshowto.com/normalized/
        :args:
            - state: (x,) numpy array, representing the full state
        :returns:
            - (x,) shaped numpy array, normalized according to 
        '''
        assert len(state) == len(self.real_ss_bounds), 'The state and bounds are not of same length!'
        for i, b in enumerate(self.real_ss_bounds):
            state[i] = (state[i] - (-b)) / (b - (-b))

        return state


    def is_terminal(self,state):
        ''' Returns true if the vessel has travelled too far from the set point.'''

        for s, bound in zip(state,self.real_ss_bounds):
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
        TODO: expand
        NOTE: The unitary gaussian is weird since the meters and radians are not comparable. Should be normalized wrt. bounds or something
        NOTE: The velocity is being penalized too much it seems, as the agent looks "satisfied" with a certain error_pose, as long as the velocity is kept low
              Especially, the yaw rate seems to be the reason to why the agent stagnates with a 
        '''
        # rew = self.vel_reward() + self.pose_reward()
        # rew = self.vel_reward() + self.smaller_yaw_dist()
        # rew = self.vel_reward() + self.unitary_multivariate_reward_2D() # multivar
        rew = self.vel_reward() + self.unitary_multivariate_reward_3D() # multivar3D MUST BE MUCH WIDER - REWARD IS TOO SPARSE!
        # rew = self.vel_reward() + self.smaller_yaw_dist() + self.action_penalty(pen_coeff = [0.5, 1.0, 1.0]) # simactpen, limactpen has 0.5 on bow
        # rew = self.vel_reward() + 10 * self.unitary_multivariate_reward_3D() + self.action_penalty() # simeqactlargemulti has 1.0 on bowpen
        rew = self.vel_reward() + self.new_multivar()

        return rew  

    def pose_reward(self):
        rews = gaussian(self.EF.get_pose())
        return sum(rews)

    def vel_reward(self):
        return -math.sqrt(sum( [e**2 * c for e,c in zip(get_vel_3DOF(self.dTwin), self.vel_rew_coeffs)] ))

    def eucledian_gaussian(self):
        surge, sway, yaw = self.EF.get_pose()
        dist = np.sqrt(surge**2 + sway**2)
        rews = gaussian([dist, yaw])
        return sum(rews)

    def smaller_yaw_dist(self):
        surge, sway, yaw = self.EF.get_pose()
        rews = gaussian([surge,sway]) # mean 0 and var == 1
        yawrew = gaussian([yaw], var=[0.1**2]) # Before, using var = 1, there wasnt any real difference between surge and sway and yaw
        return sum(rews) + yawrew

    def unitary_multivariate_reward_2D(self):
        surge, sway, yaw = self.EF.get_pose()
        r = math.sqrt(surge**2 + sway**2)
        x = np.array([[r,yaw]]).T
        
        return 4 * np.exp(-0.5 * (x.T).dot(self.covar_inv).dot(x))

    def new_gaussian(self):
        ''' This still encourages reducing r before yaw or vice versa '''
        surge, sway, yaw = self.EF.get_pose()
        r = math.sqrt(surge**2 + sway**2)

        g1 = gaussian_like(val = [r], mean = [0], var = [4])[0] # a wide, but sparse multivariate gaussian
        g2 = max(1 - 0.1 * r, 0.0) # reduce sparsity, but don't give negative reward far away
        g3 = gaussian_like(val = [yaw], mean = [0], var = [0.1**2])[0] # a narrow gaussian (std_dev = 5.7 deg)
        g4 = max(1 - 0.10 * np.abs(yaw) * 180 / np.pi, 0.0) # reduce sparsity around 10 degrees using 0.1 as constant
        return (g1 + g2 + g3 + g4) # maximum score of 4

    def new_multivar(self):
        surge, sway, yaw = self.EF.get_pose()
        r = math.sqrt(surge**2 + sway**2)

        yaw_vs_r_factor = 0.25 # how much one degree is weighted vs a meter
        r3d = math.sqrt(r**2 + (yaw * 180 / np.pi * 0.25)**2)
        return gaussian_like(val = [r3d], mean = [0], var = [2.0**2]) + max(0.0, (1-0.05*r3d))

    def action_penalty(self, pen_coeff = [1., 1., 1.]):
        pen = 0
        # must not be greater than one
        for T,c in zip(self.thrust_taken, pen_coeff):
            pen -= np.abs(T) / 100.0 * c

        return pen # maximum penalty is 1 per time step

    def unitary_multivariate_reward_3D(self):
        surge, sway, yaw = self.EF.get_pose()
        x = np.array([[surge,sway,yaw]]).T
        return np.exp(-0.5 * (x.T).dot(self.Sigma_inv).dot(x))

class RevoltSimple(Revolt):
    ''' +++++++++++++++++++++++++++++++ '''
    '''      FIXED THRUSTER SETUP       '''
    ''' +++++++++++++++++++++++++++++++ '''
    def __init__(self,digitwin, testing = False, realtime = False, norm_env = False, max_ep_len = 800, curriculum = False):
        super().__init__(digitwin = digitwin, num_actions = 3, num_states = 6,
                         testing = testing, realtime = realtime, norm_env = norm_env, max_ep_len = max_ep_len, curriculum=curriculum)

        # Overwrite environment bounds according to measured max velocity for this specific setup
        self.real_ss_bounds = [8.0, 8.0, np.pi/2, 1.75, 0.30, 0.51] # TODO could be set to much smaller values: (scale 10,10,100 times to get them in the same range as the positional arguments)

        # Overwrite default actions
        self.default_actions = {0: 0,
                                1: 0,
                                2: 0,
                                3: math.pi / 2,
                                4: -3 * math.pi / 4,
                                5: 3 * math.pi / 4}

        self.act_2_act_map = {0: 0, 1: 1, 2: 2}

class RevoltLimited(Revolt):
    ''' +++++++++++++++++++++++++++++++ '''
    '''      LIMITED AZIMUTH ANGLES     '''
    ''' +++++++++++++++++++++++++++++++ '''
    def __init__(self,digitwin,testing=False,realtime=False, norm_env=False,curriculum=False):
        super().__init__(digitwin,num_actions=5,num_states=6,testing=testing,realtime=realtime,norm_env=norm_env,curriculum=curriculum)

        # Not choosing the bow angle means (1) one less action bound, (2) remove one valid index, (3) set bow angle default to pi/2
        self.real_ss_bounds       = [8.0, 8.0, 30 * np.pi / 180, 2.2, 0.35, 0.30] 
        self.real_action_bounds   = [100] * 3 + [math.pi / 2 ] * 2
        self.valid_action_indices = [ 0,    1,      2,      4,      5]
        self.act_2_act_map        = { 0:0,  1:1,    2:2,    4:3,    5:4} # {index in self.actions : index in action vector}
        self.default_actions      = {0:0, 1:0, 2:0, 3: math.pi / 2, 4:0, 5:0}

        
        
