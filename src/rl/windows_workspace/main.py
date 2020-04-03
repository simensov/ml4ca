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
    from config import PPO_ARGS as args

    print('Testing implementation with {} cores'.format(args.cpu))

    t = Trainer(n_sims = args.cpu, start = True)

    print('Number of simulators made: {}'.format(t._n_sims))

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(env_fn          = t.env_fn,
        actor_critic    = core.mlp_actor_critic,
        ac_kwargs       = dict(hidden_sizes=[args.hid]*args.l),
        gamma           = args.gamma,
        seed            = args.seed,
        steps_per_epoch = args.steps,
        epochs          = args.epochs,
        logger_kwargs   = logger_kwargs)

    '''
    Notes:
        - FIXED The state should be randomly initialized into the entire state space
        - It took ish 20 min running 103 epochs / TotalEnvInteracts (is an upper bound) of 103 000 on one CPU, without the policy looking to change.
    '''