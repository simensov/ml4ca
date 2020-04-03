# -*- coding: utf-8 -*-
'''
A test script for opening and running a (bad) actor inside a simulator
@author Simen Sem Oevereng, simensem@gmail.com
'''

from specific.trainer import Trainer

if __name__ == "__main__":
    from config import GLOBAL_SIM_ARGS as args
    trainer = Trainer(args.n_sims)
    trainer.start_simulators(args.sim_path,args.python_port_initial,args.sim_config_path,args.load_cfg)
    trainer.train(n_episodes = args.n_episodes)