import os
from specific.trainer import Trainer
from specific.customEnv import RevoltSimple

import numpy as np
import tensorflow as tf
import gym
import time
import spinup.algos.tf1.ppo.core as core
from spinup.algos.tf1.ppo.ppo import ppo
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

if __name__ == '__main__':
    from specific.sim_global_config import GLOBAL_SIM_ARGS as args 

    t = Trainer(args.n_sims)
    t.start_simulators(args.sim_path, args.python_port_initial, args.sim_cfg_path, args.load_cfg)
    digitwins = t.get_digitwins()
    env = RevoltSimple(digitwins[0])

    from ppo_config import PPO_ARGS as args
    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(lambda : env, actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)