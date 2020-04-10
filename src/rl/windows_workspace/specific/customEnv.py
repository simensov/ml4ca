import gym 
from gym import spaces
import numpy as np
import math
from specific.misc.simtools import get_pose_3DOF, get_vel_3DOF, get_pose_on_state_space, get_pose_on_radius
from specific.misc.mathematics import gaussian
from specific.errorFrame import ErrorFrame

class Revolt(gym.Env):
    """ Custom Environment that follows OpenAI's gym API.
        Max velocities measured with no thrust losses activated. Full means rotating stern azimuths only.
            Full:   surge, sway, yaw = (+2.20, -1.60) m/s, +-0.35 m/s, +-0.60 rad/s
            Simple: surge, sway, yaw = (+1.75, -1.40) m/s, +-0.30 m/s, +-0.51 rad/s
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, digitwin, num_actions = 6, num_states = 6,
                 real_bounds=[8.0, 8.0, np.pi/2, 2.2, 0.35, 0.60], norm_env = False,
                 testing = False, realtime = False, max_ep_len = 1000):

        super(Revolt, self).__init__()

        self.dTwin = digitwin
        self.EF = ErrorFrame()

        # Set the name of actions in Cybersea
        self.actions = [
            {'idx': 0, 'module': 'THR1', 'feature': 'ThrustOrTorqueCmdMtc'}, # bow
            {'idx': 1, 'module': 'THR2', 'feature': 'ThrustOrTorqueCmdMtc'}, # stern, portside
            {'idx': 2, 'module': 'THR3', 'feature': 'ThrustOrTorqueCmdMtc'}, # stern, starboard
            {'idx': 3, 'module': 'THR1', 'feature': 'AzmCmdMtc'}, 
            {'idx': 4, 'module': 'THR2', 'feature': 'AzmCmdMtc'}, 
            {'idx': 5, 'module': 'THR3', 'feature': 'AzmCmdMtc'} 
        ]

        self.act_bnd = [100] * 3 + [math.pi] * 3 # TIP FROM RØRVIK: USE PI/2 INITIALLY
        self.num_actions = num_actions
        self.num_states = num_states
        self.valid_indices = list(range(6))[0:num_actions] # allows for num_actions == 3 for simplified thruster setup

        self.default_actions = {0:0,1:0,2:0,3:0,4:0,5:0}

        bnds = {'action':{'low': -1*np.ones((num_actions,)), 'high': np.ones((num_actions,)) },
                'spaces':{'low': -1*np.ones((num_states,)),  'high': np.ones((num_states,))} }

        self.action_space      = spaces.Box(low=bnds['action']['low'], high=bnds['action']['high'], dtype=np.float64)
        self.observation_space = spaces.Box(low=bnds['spaces']['low'], high=bnds['spaces']['high'], dtype=np.float64)

        self.real_bounds = real_bounds
        self.testing = testing # stores if the environment is being used while testing policy, or is being used for training
        self.n_steps = 1 if (testing and realtime) else 10 
        self.max_ep_len = max_ep_len * int(10/self.n_steps) # 1000 is a good length while training
        # TODO add more states - store previous actions etc

        self._norm_state = norm_env

    def __str__(self):
        return str(self.dTwin.name)

    def step(self, action):
        ''' Step a fixed number of steps in the Cybersea simulator 
        :args:
            action (object): an action provided by the agent
        :returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """'''

        action = self.scale_and_clip(action)

        for a in self.actions:
            if a['idx'] in self.valid_indices:
                self.dTwin.val(a['module'], a['feature'], action[a['idx']])

        self.dTwin.step(self.n_steps) # ReVolt is operating at 10 Hz. Input to step() is number of steps at 100 Hz
        s = self.state()
        r = self.reward()
        d = self.is_terminal(s)

        # if d: r += -1000 # TODO enforce large negative reward if terminal (there is no goal state, only death states)
        if self._norm_state:
            s = self.normalize_state(s) # TODO control

        return s,r,d, {'None': 0}

    def scale_and_clip(self,action):
        ''' Action from actor if close to being -1 and 1. Scale 100%, and clip. '''
        # TODO adjust bow thruster - 100% and -100% is not the same! Limit the strongest side, and scale it up to 100% from fractional upper hand
        bnds = np.array(self.act_bnd[0:self.num_actions]) # select bounds according to environment specifications
        action = np.multiply(action,bnds) # The action comes as choices between -1 and 1...
        action = np.clip(action,-bnds,bnds) # ... but the std_dev in the stochastic policy means that we have to clip
        return action.tolist()

    def reset(self,**init):
        """ Resets the state of the environment and returns an initial observation.
        :returns:
            observation (object): the initial observation.
        """
        # TODO if the previous actions are to be put into the state-vector, reset() must set random previous actions, or all previous actions must be set to zero

        if not init:
            if not self.testing:
                N, E, Y = get_pose_on_state_space(self.real_bounds[0:3]) 
            else: 
                N, E, Y = get_pose_on_radius(r=3)

            init = {'Hull.PosNED':[N,E],'Hull.PosAttitude':[0,0,Y]}

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

        state = self.state()
        if self._norm_state:
            state = self.normalize_state(state) # TODO control 

        return state

    def state(self):
        self.EF.update(get_pose_3DOF(self.dTwin))
        return np.array( self.EF.get_pose() + get_vel_3DOF(self.dTwin) ) # (6,) numpy array

    def normalize_state(self,state):
        ''' Based on normalizing a symmetric state distribution. 
            Since reset samples uniformly (and the sampled state space distribution is known to not be normally distributed),
            the states are normalized instead of standardized with means and stds 
        
        :args:
            - state: (x,) numpy array, representing the full state
        
        :returns:
            - (x,) shaped numpy array, normalized according to 
        '''
        assert len(state) == len(self.real_bounds), 'The state and bounds are not of same length!'

        # OLD WAY OF DOING IT
        # for i,b in enumerate(self.real_bounds):
        #     state[i] /= (1.0 * b) 

        for i, b in enumerate(self.real_bounds):
            state[i] = (state[i] - (-b)) / (b - (-b)) # https://www.statisticshowto.com/normalized/

        return state

    def reward(self):
        '''
        TODO: expand
        NOTE: The unitary gaussian is weird since the meters and radians are not comparable. Should be normalized wrt. bounds or something
        NOTE: The velocity is being penalized too much it seems, as the agent looks "satisfied" with a certain error_pose, as long as the velocity is kept low
              Especially, the yaw rate seems to be the reason to why the agent stagnates with a 
        '''
        return self.initial_reward_function()

    def initial_reward_function(self):
        vel = -math.sqrt(sum( [e**2 * c for e,c in zip(get_vel_3DOF(self.dTwin), [1,1,1])] ))
        gaus_rews = gaussian(self.EF.get_pose())
        gaus_rews[2] = 0 if abs(self.EF.get_pose()[2]) > np.pi/2 else gaus_rews[2] # set reward of yaw angle higher, or to zero
        return vel + sum(gaus_rews)

    def pose_reward(self):
        rews = gaussian(self.EF.get_pose())
        rews[2] = 0 if abs(self.EF.get_pose()[2]) > np.pi/2 else rews[2]
        return sum(rews)

    def is_terminal(self,state):
        ''' Returns true if the vessel has travelled too far from the set point.
        TODO: sync this with the bounds used to normalize the state space input '''

        for s, bound in zip(state,self.real_bounds):
            if np.abs(s) > bound:
                return True

        return False

    def render(self):
        pass # The environment will always be rendered in Cybersea


class RevoltSimple(Revolt):
    def __init__(self,digitwin, testing = False, realtime = False, norm_env = False):
        super().__init__(digitwin = digitwin, num_actions = 3, num_states = 6,
                         testing = testing, realtime = realtime, norm_env = norm_env)

        # Overwrite environment bounds according to measured max velocity for this specific setup
        self.real_bounds = [8.0, 8.0, np.pi/2, 1.75, 0.30, 0.51] # TODO could be set to much smaller values: (scale 10,10,100 times to get them in the same range as the positional arguments)

        # Overwrite default actions
        self.default_actions = {0: 0,
                                1: 0,
                                2: 0,
                                3: math.pi / 2,
                                4: -3 * math.pi / 4,
                                5: 3 * math.pi / 4}

class RevoltLimited(Revolt):
    ''' Limiting stern angles '''
    def __init__(self,digitwin,testing=False,realtime=False):
        super().__init__(digitwin,num_actions=5,num_states=6,testing=testing,realtime=realtime)

        # Not choosing the bow angle means (1) one less action bound, (2) remove one valid index, (3) set bow angle default to pi/2
        self.act_bnd = [100] * 3 + [math.pi / 2 ] * 2
        self.valid_indices = [0,1,2,4,5] 
        self.default_actions = {0:0, 1:0, 2:0, 3: math.pi / 2, 4:0, 5:0}
        
