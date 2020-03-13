import random
import math
import numpy as np
from utils_sim import get_pose_3DOF, get_vel_3DOF

class RevoltSimulator(object):

    def __init__(self,sim,observation_space,action_space,**init):
        self.sim = sim
        self.reset(**init)
        self.err = observation_space
        self.a_space = action_space
        self.report_reset = False

    def reset(self,**init):
        #set init values
        for modfeat in init:
            module, feature = modfeat.split('.')
            self.sim.val(module, feature, init[modfeat])
            
        #reset critical models to clear states from last episode
        self.sim.val('Hull', 'StateResetOn', 1, self.report_reset)
        self.sim.step(50) #min 50 steps should do it
        self.sim.val('Hull', 'StateResetOn', 0, self.report_reset)
        self.sim.val('THR2', 'MtcOn', 1, self.report_reset)
        self.sim.val('THR2', 'ThrustOrTorqueCmdMtc', 20, self.report_reset)
        self.sim.val('THR2', 'AzmCmdMtc', 0*math.pi, self.report_reset)
        self.sim.val('THR3', 'MtcOn', 1, self.report_reset)
        self.sim.val('THR3', 'ThrustOrTorqueCmdMtc', 20, self.report_reset)
        self.sim.val('THR3', 'AzmCmdMtc', 0*math.pi, self.report_reset)

    def step(self,action):
        # TODO these actions might has to be extracted from "inverse standardization"?

        self.act(action)

        self.sim.step()
        pose = get_pose_3DOF(self.sim)
        err_body = self.err.get_pose(pose)
        vel = get_vel_3DOF(self.sim)

        s_ = np.vstack(err_body,vel)
        r = self.calculate_reward()
        done = self.check_if_terminal()
        info = {'Closing in'}

        return s_, r, done, info

    def act(self,action):
        ''' Actions should be a vector of [a1,a2,f1,f2,f3]'''

        # TODO clip actions        
        # TODO does MtcOn need 

        a1, a2, f1, f2, f3 = action # follows thesis convention
        azi_port, azi_star, n_port, n_star, n_bow = action # more readable
        
        '''
        NOTE the enumeration of the thrusters in the simulator is different from the thesis
        The naming from the simulator, with the corresponding enumeration in the thesis is:
            - 'THR1' == 3: bow thruster
            - 'THR2' == 1: stern, portside
            - 'THR3' == 2: stern, starboard
        '''

        self.sim.val('THR1', 'MtcOn', 1, self.report_reset)
        self.sim.val('THR1', 'ThrustOrTorqueCmdMtc', 10, self.report_reset)
        self.sim.val('THR1', 'AzmCmdMtc', 0.5*math.pi, self.report_reset) # const

        self.sim.val('THR2', 'MtcOn', 1, self.report_reset)
        self.sim.val('THR2', 'ThrustOrTorqueCmdMtc', 20, self.report_reset)
        self.sim.val('THR2', 'AzmCmdMtc', 0*math.pi, self.report_reset)

        self.sim.val('THR3', 'MtcOn', 1, self.report_reset)
        self.sim.val('THR3', 'ThrustOrTorqueCmdMtc', 20, self.report_reset)
        self.sim.val('THR3', 'AzmCmdMtc', 0*math.pi, self.report_reset)

    def calculate_reward(self):
        return 0
    
    def check_if_terminal(self):
        ''' Using the boundaries of the error-frame to terminate episode if agent is outside of observation space'''
        return True