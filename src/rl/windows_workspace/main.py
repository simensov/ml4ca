# -*- coding: utf-8 -*-
'''

@author Simen Sem Oevereng, simensem@gmail.com
'''

from trainer import Trainer


if __name__ == "__main__":
    
    #defs
    SIM_CONFIG_PATH     = "C:\\Users\\simen\\Documents\\Utdanning\\GTK\\configuration"
    SIM_PATH            = "C:\\Users\\simen\\Documents\\Utdanning\\GTK\\revoltsim\\bin\\revoltsim64.exe"
    PYTHON_PORT_INITIAL = 25338
    LOAD_SIM_CFG        = False
    NUM_SIMULATORS      = 1
    NUM_EPISODES        = 1000 # One sim, 300 episodes, 5000 steps ~ 12 hours with 100 Hz

    trainer = Trainer(NUM_SIMULATORS)
    trainer.start_simulators(SIM_PATH,PYTHON_PORT_INITIAL,SIM_CONFIG_PATH,LOAD_SIM_CFG)
    trainer.train(n_episodes = NUM_EPISODES)