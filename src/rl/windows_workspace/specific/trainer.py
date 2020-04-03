import numpy as np
from specific.digitwin import DigiTwin
import time
from specific.misc.simtools import get_pose_on_radius, standardize_state
from specific.customEnv import Revolt,RevoltSimple
from specific.agents.ppo import PPO
import matplotlib.pyplot as plt
import datetime

SIM_CONFIG_PATH     = "C:\\Users\\simen\\Documents\\Utdanning\\GTK\\configuration"
SIM_PATH            = "C:\\Users\\simen\\Documents\\Utdanning\\GTK\\revoltsim\\bin\\revoltsim64.exe"
PYTHON_PORT_INITIAL = 25338
LOAD_SIM_CFG        = False
NUM_SIMULATORS      = 1
NUM_EPISODES        = 1000

class Trainer(object):
    '''
    Keeps track of all digitwins and its simulators + environments for a training session
    '''
    def __init__(self, n_sims = NUM_SIMULATORS, start = False, testing = False):
        assert isinstance(n_sims,int) and n_sims > 0, 'Number of simulators must be an integer'
        self._n_sims     = n_sims
        self._digitwins  = []*n_sims  # list of independent simulators
        if start:
            self.start_simulators()
            self.set_environments(env_type='simple',testing = testing)
        self._env_counter = 0

    def start_simulators(self,sim_path=SIM_PATH,python_port_initial=PYTHON_PORT_INITIAL,sim_cfg_path=SIM_CONFIG_PATH,load_cfg=LOAD_SIM_CFG):

        #Start up all simulators
        for sim_ix in range(self._n_sims):
            python_port = python_port_initial + sim_ix
            print("Open CS sim " + str(sim_ix) + " Python_port=" + str(python_port))
            self._digitwins.append(None) # Weird, by necessary order of commands
            self._digitwins[-1] = DigiTwin('Sim'+str(1+sim_ix), load_cfg, sim_path, sim_cfg_path, python_port)
        print("Connected to simulators and configuration loaded")

    def get_digitwins(self):
        return self._digitwins

    def set_environments(self,env_type='simple',testing=False):
        if env_type.lower() == 'simple':
            self._envs = [RevoltSimple(self._digitwins[i], testing=testing) for i in range(self._n_sims)]
        else:
            self._envs = [Revolt(self._digitwins[i], testing=testing) for i in range(self._n_sims)]

    def get_environments(self):
        return self._envs

    def env_fn(self):
        ''' This function is made for returning one environment at a time to the ppo algorithm'''
        env = self._envs[self._env_counter]
        print('Trainer returns environment no. {}'.format(self._env_counter))
        self._env_counter += 1
        return env