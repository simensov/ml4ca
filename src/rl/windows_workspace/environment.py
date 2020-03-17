from abc import abstractmethod, ABC
import random
import math
import numpy as np
from utils.simtools import get_pose_3DOF, get_vel_3DOF
from errorFrame import ErrorFrame

class Environment(ABC):
    ''' Abstract class for the different simulator setups '''
    
    def __init__(self,sim,observation_bounds):
        super().__init__()
        self._sim = sim
        self.report_reset = False
        self.bounds = observation_bounds
        self.err = ErrorFrame()

    def reset(self,**init):
        # TODO if the previous actions are to be put into the state-vector, reset() must set random actions

        for modfeat in init:
            module, feature = modfeat.split('.')
            self.sim.val(module, feature, init[modfeat])
            
        #reset critical models to clear states from last episode
        self.sim.val('Hull', 'StateResetOn', 1, self.report_reset)
        self.sim.step(50) #min 50 steps should do it
        self.sim.val('Hull', 'StateResetOn', 0, self.report_reset)
        self.sim.val('THR2', 'MtcOn', 1, self.report_reset)
        self.sim.val('THR2', 'ThrustOrTorqueCmdMtc', 0, self.report_reset)
        self.sim.val('THR2', 'AzmCmdMtc', 0*math.pi, self.report_reset)
        self.sim.val('THR3', 'MtcOn', 1, self.report_reset)
        self.sim.val('THR3', 'ThrustOrTorqueCmdMtc', 0, self.report_reset)
        self.sim.val('THR3', 'AzmCmdMtc', 0*math.pi, self.report_reset)

        state = self.create_state()
        return state

    def step(self,action):
            # TODO these actions might has to be extracted from "inverse standardization"?

            self.perform_action(action) # tell simulator to make a change
            self.sim.step() # perform that change
            s_ = self.create_state() # observe change
            r = self.calculate_reward() # calculate reward of change
            done = self.check_if_terminal() # check if change was terminal
            info = {'None': 0} # TODO add some information about the change
            return s_, r, done, info

    @abstractmethod
    def create_state(self):
        return NotImplementedError

    @abstractmethod
    def perform_action(self):
        return NotImplementedError

    @abstractmethod
    def calculate_reward(self):
        return NotImplementedError

    @abstractmethod
    def is_terminal(self):
        return NotImplementedError

class FixedThrusters(Environment):

    def __init__(self,sim):
        super().__init__(sim)
        self.reward_coeffs = [1,1,2]

    def create_state(self):
        pose_ned = get_pose_3DOF(self.sim)
        self.err.update(pose_ned)
        error_pose_body = self.err.get_pose()
        vel_body = get_vel_3DOF(self.sim)
        return np.array(error_pose_body + vel_body) # np array of shape (6,) after concatination lists

    def perform_action(self,action):
        ''' Actions should be a vector of [f1,f2,f3]'''

        azi_bow,azi_port, azi_star = 0.5*math.pi, -3*math.pi/4, 3*math.pi/4
        action = np.clip(action,-100.0,100.0).tolist() # TODO clip actions -  check type
        n_port, n_star, n_bow = action

        # Bow
        self.sim.val('THR1', 'MtcOn', 1, self.report_reset)
        self.sim.val('THR1', 'ThrustOrTorqueCmdMtc', n_bow, self.report_reset)
        self.sim.val('THR1', 'AzmCmdMtc', azi_bow, self.report_reset)

        # Stern, port
        self.sim.val('THR2', 'MtcOn', 1, self.report_reset)
        self.sim.val('THR2', 'ThrustOrTorqueCmdMtc', n_port, self.report_reset)
        self.sim.val('THR2', 'AzmCmdMtc', azi_port, self.report_reset)

        # Stern, starboard
        self.sim.val('THR3', 'MtcOn', 1, self.report_reset)
        self.sim.val('THR3', 'ThrustOrTorqueCmdMtc', n_star, self.report_reset)
        self.sim.val('THR3', 'AzmCmdMtc', azi_star, self.report_reset)

        return

    def calculate_reward(self):
        return sum( [e * c for e,c in zip(self.err.get_pose(), self.error_coefs)] )
        
    def is_terminal(self):
        ''' Using the boundaries of the error-frame to terminate episode if agent is outside of observation space'''
        for p,bound in zip(self.err.get_pose(),self.bounds):
            if abs(p) > bound:
                return True
        return False

class RevoltSimulator(object):
    ''' Initial state and full action space'''

    def __init__(self,sim,simple=False,observation_space=[10,10,np.pi],action_space=[np.pi,np.pi,100,100,100],**init):

        self.sim = sim
        self.reset(**init)
        self.err = observation_space
        self.a_space = action_space   

    def perform_action(self,action):
        ''' Actions should be a vector of [a1,a2,f1,f2,f3]'''

        # TODO does MtcOn need to be set to 1 each time?

        a1, a2, f1, f2, f3 = action # follows thesis convention
        azi_port, azi_star, n_port, n_star, n_bow = action # more readable

        # TODO clip actions        

        '''
        NOTE the enumeration of the thrusters in the simulator is different from the thesis
        The naming from the simulator, with the corresponding enumeration in the thesis is:
            - 'THR1' == 3: bow thruster
            - 'THR2' == 1: stern, portside
            - 'THR3' == 2: stern, starboard
        '''

        # Bow
        self.sim.val('THR1', 'MtcOn', 1, self.report_reset)
        self.sim.val('THR1', 'ThrustOrTorqueCmdMtc', n_bow, self.report_reset)
        self.sim.val('THR1', 'AzmCmdMtc', 0.5*math.pi, self.report_reset) # const

        # Stern, port
        self.sim.val('THR2', 'MtcOn', 1, self.report_reset)
        self.sim.val('THR2', 'ThrustOrTorqueCmdMtc', n_port, self.report_reset)
        self.sim.val('THR2', 'AzmCmdMtc', a1, self.report_reset)

        # Stern, starboard
        self.sim.val('THR3', 'MtcOn', 1, self.report_reset)
        self.sim.val('THR3', 'ThrustOrTorqueCmdMtc', n_star, self.report_reset)
        self.sim.val('THR3', 'AzmCmdMtc', a2, self.report_reset)

    def create_state(self):
        pose_ned = get_pose_3DOF(self.sim)
        self.err.update(pose_ned)
        error_pose_body = self.err.get_pose()
        vel_body = get_vel_3DOF(self.sim)
        return np.array(error_pose_body + vel_body) # np array of shape (6,)

    def calculate_reward(self):
        return 0
    
    def is_terminal(self):
        ''' Using the boundaries of the error-frame to terminate episode if agent is outside of observation space'''
        return True