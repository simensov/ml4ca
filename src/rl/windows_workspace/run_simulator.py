# -*- coding: utf-8 -*-
'''
A test script for opening and running a (bad) actor inside a simulator
@author Simen Sem Oevereng, simensem@gmail.com
'''

from specific.trainer import Trainer

if __name__ == "__main__":
    from specific.sim_global_config import NUM_SIMULATORS, SIM_PATH, PYTHON_PORT_INITIAL, SIM_CONFIG_PATH, LOAD_SIM_CFG, NUM_EPISODES

    trainer = Trainer(NUM_SIMULATORS)
    trainer.start_simulators(SIM_PATH,PYTHON_PORT_INITIAL,SIM_CONFIG_PATH,LOAD_SIM_CFG)
    trainer.train(n_episodes = NUM_EPISODES)