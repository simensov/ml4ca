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

    assert int(args.steps / args.cpu) > args.max_ep_len, 'The number of steps (interations between the agent and environment per epoch) per process must be larger than the largest episode to avoid empty episodal returns'

    t = Trainer(n_sims = args.cpu, start = True)

    print('Number of simulators made: {}'.format(t._n_sims))

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    

    ppo(env_fn          = t.env_fn,
        actor_critic    = core.mlp_actor_critic,
        ac_kwargs       = dict(hidden_sizes=[args.hid]*args.l),
        seed            = args.seed,
        steps_per_epoch = args.steps,  # NB this is how many steps of experience should be gathered IN TOTAL (all CPUs combined) before PPO updates. 4000 with 4 cpus means each cpu collects 1000 steps. This last number must be larger than max episode length
        epochs          = args.epochs, # this is the total number of MINIBATCH COLLECTIONS AND PPO UPDATES - NOT NUMBER OF INTERACTIONS OR LENGTH OF EPISODES
        gamma           = args.gamma, # per mujoco experience, high dim robotics works better with lower gammas, e.g. < 0.99
        clip_ratio      = 0.2,
        pi_lr           = 3e-4,
        vf_lr           = 1e-3,
        train_pi_iters  = 100, # (80 originally)
        train_v_iters   = 100, # (80 originally)
        lam             = 0.97,
        max_ep_len      = 1000, # (1000 is good for 10 Hz steps) only affects how long each episode can be - not how many that are rolled out
        target_kl       = 0.05, # A rough estimate of what spinning up thinks is ok (0.01-0.05)
        logger_kwargs   = logger_kwargs,
        save_freq       = 10) # distance between epochs that the model is saved



    '''
    Notes:
        - FIXED The state should be randomly initialized into the entire state space
        - It took ish 20 min running 103 epochs / TotalEnvInteracts (is an upper bound) of 103 000 on one CPU, without the policy looking to change.

        Each process tries to separately collect int(batch_size / n_cpu) steps of interactions with the environment. 
        If this is shorter than a single episode, the logger gets an empty list for EpRet or EpLen in that process, and then complains when it tries to aggregate stats across processes. 
        That is the nature of the error. (And it doesn't happen in the first few epochs because the RL agent is so bad at playing, that its episodes terminate early and never get that long.) 
        As long as you pick n_cpu and batch_size so that the batch size per process is always greater than one episode length, you will not get this error.
    '''