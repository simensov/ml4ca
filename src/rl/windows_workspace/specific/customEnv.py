import gym 
from gym import spaces
import numpy as np
import math
from specific.misc.simtools import get_pose_3DOF, get_vel_3DOF, get_pose_on_state_space, get_pose_on_radius
from specific.misc.mathematics import gaussian
from specific.errorFrame import ErrorFrame

class Revolt(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self,digitwin,num_actions=6,num_states=6,real_bounds=[20.0,20.0,np.pi,2.0,2.0,1.0],testing=False):
        ''' The states and actions must be standardized! '''
        super(Revolt, self).__init__()
        # Define action and observation space as gym.spaces objects

        self.dTwin = digitwin
        self.EF = ErrorFrame()

        # Set the name of actions in Cybersea
        self.actions = [
            {'idx': 3, 'module': 'THR1', 'feature': 'AzmCmdMtc'}, # bow
            {'idx': 4, 'module': 'THR2', 'feature': 'AzmCmdMtc'}, # stern, portside
            {'idx': 5, 'module': 'THR3', 'feature': 'AzmCmdMtc'}, # stern, starboard
            {'idx': 0, 'module': 'THR1', 'feature': 'ThrustOrTorqueCmdMtc'},
            {'idx': 1, 'module': 'THR2', 'feature': 'ThrustOrTorqueCmdMtc'},
            {'idx': 2, 'module': 'THR3', 'feature': 'ThrustOrTorqueCmdMtc'}
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
        # TODO add more states - store previous actions etc

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

        self.dTwin.step(10) # ReVolt is operating at 10 Hz. Input to step() is number of steps at 100 Hz
        s = self.state() # These three lines could be put into return, but this order ensures err.update() after step()
        r = self.reward()
        d = self.is_terminal(s)

        return s,r,d, {'None': 0}

    def scale_and_clip(self,action):
        ''' Action from actor might be out of bounds TODO but shouldnt be. scale thus is upper boundary'''
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
                N, E, Y = get_pose_on_state_space(n=self.real_bounds[0],e=self.real_bounds[1],y=self.real_bounds[2]) 
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

        return self.state()

    def state(self):
        self.EF.update(get_pose_3DOF(self.dTwin))
        return np.array( self.EF.get_pose() + get_vel_3DOF(self.dTwin) ) # (6,) numpy array

    def reward(self):
        # TODO expand
        vel = -math.sqrt(sum( [e**2 * c for e,c in zip(get_vel_3DOF(self.dTwin), [1,1,5])] ))
        gaus_rews = gaussian(self.EF.get_pose())
        gaus_rews[2] = 0 if abs(self.EF.get_pose()[2]) > np.pi/2 else 5*gaus_rews[2] # set reward of yaw angle higher, or to zero
        return vel + sum(gaus_rews)

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
    def __init__(self,dt,testing=False):
        super().__init__(dt,num_actions=3,num_states=6,testing=testing)
        # Overwrite default actions
        self.default_actions = {0: 0,
                                1: 0,
                                2: 0,
                                3: math.pi / 2,
                                4: -3 * math.pi / 4,
                                5: 3 * math.pi / 4}