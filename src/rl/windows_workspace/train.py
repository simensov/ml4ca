'''
train.py

Trains a ppo model with a configurable set of arguments

@author Simen Sem Oevereng, simensem@gmail.com
'''
from specific.trainer import Trainer
from spinup.algos.tf1.ppo.core import mlp_actor_critic
from spinup.algos.tf1.ppo.ppo import ppo
from spinup.utils.mpi_tools import mpi_fork
from spinup.utils.run_utils import setup_logger_kwargs
import tensorflow as tf

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid',            type=int,   default=32)     # Number of nodes in hidden layers
    parser.add_argument('--l',              type=int,   default=3)      # Number of hidden layers
    parser.add_argument('--gamma',          type=float, default=0.99)   # Discount factor (0.99) NOTE from author' mujoco experience, high dim robotics works better with lower gammas, e.g. < 0.99
    parser.add_argument('--lam',            type=float, default=0.97)   # Decay factor (0.97)
    parser.add_argument('--clip_ratio',     type=float, default=0.2)    # Allowance for policy ratio change per update (1 +- clip_ratio)
    parser.add_argument('--pi_lr',          type=float, default=3e-4)   # Policy network learning rate / initial step size for optimizer (3e-4)
    parser.add_argument('--vf_lr',          type=float, default=1e-3)   # Value function network learning rate (1e-3)
    parser.add_argument('--pi_epochs',      type=int,   default=80)     # Number of optimizer update steps on policy network per minibatch (80 originally)
    parser.add_argument('--vf_epochs',      type=int,   default=80)     # Number of optimizer update steps on value function network per minibatch (80 originally)
    parser.add_argument('--target_kl',      type=int,   default=0.01)   # Largest KL-divergence allowed for policy network updates per minibatch.  A rough estimate of what spinning up thinks is ok is (0.01-0.05)
    parser.add_argument('--seed',           type=int,   default=0)      # Random seed
    parser.add_argument('--cpu',            type=int,   default=1)      # Number of CPU's used during training
    parser.add_argument('--steps',          type=int,   default=1000)   # Number of steps during an entire episode for all processes combined
    parser.add_argument('--epochs',         type=int,   default=2000)   # Number of EPISODES
    parser.add_argument('--max_ep_len',     type=int,   default=1000)   # Number of steps per local episode # (1000 is lower bound for 10 Hz steps) only affects how long each episode can be - not how many that are rolled out
    parser.add_argument('--save_freq',      type=int,   default=10)     # Number of episodes between storage of actor-critic weights
    parser.add_argument('--exp_name',       type=str,   default='test') # Name of data storage area
    args = parser.parse_args()

    print('Testing implementation with {} cores'.format(args.cpu))
    assert args.cpu == 1 or int(args.steps / args.cpu) > args.max_ep_len, 'If n_cpu > 1: The number of steps (interations between the agent and environment per epoch) per process must be larger than the largest episode to avoid empty episodal returns'
    t = Trainer(n_sims = args.cpu, start = True) 
    mpi_fork(args.cpu)  # run parallel code with mpi 
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed,datestamp=False)
    actor_critic_kwargs = {'hidden_sizes' : [args.hid]*args.l,'activation' : tf.nn.leaky_relu}

    ppo(env_fn        = t.env_fn,            actor_critic  = mlp_actor_critic,
        ac_kwargs     = actor_critic_kwargs, seed          = args.seed,     steps_per_epoch = args.steps,
        epochs        = args.epochs,         gamma         = args.gamma,    clip_ratio      = args.clip_ratio,
        pi_lr         = args.pi_lr,          vf_lr         = args.vf_lr,    train_pi_iters  = args.pi_epochs,
        train_v_iters = args.vf_epochs,      lam           = args.lam,      max_ep_len      = args.max_ep_len,
        target_kl     = args.target_kl,      logger_kwargs = logger_kwargs, save_freq       = args.save_freq)