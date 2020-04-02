import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='CartPole-v1') # HalfCheetah-v2
parser.add_argument('--hid', type=int, default=64) # Number of nodes in hidden layers
parser.add_argument('--l', type=int, default=2) # Number of hidden layers
parser.add_argument('--gamma', type=float, default=0.99) # Discount factor
parser.add_argument('--seed', '-s', type=int, default=0) # Random seed
parser.add_argument('--cpu', type=int, default=1) # Number of CPU's used during training
parser.add_argument('--steps', type=int, default=1000) # Number of steps during an episode
parser.add_argument('--epochs', type=int, default=300) # Number of EPISODES
parser.add_argument('--exp_name', type=str, default='ppoReVolt') # Name of data storage area

PPO_ARGS = parser.parse_args()

parser = argparse.ArgumentParser()
parser.add_argument('--sim_cfg_path', type=str, default="C:\\Users\\simen\\Documents\\Utdanning\\GTK\\configuration") 
parser.add_argument('--sim_path', type=str, default="C:\\Users\\simen\\Documents\\Utdanning\\GTK\\revoltsim\\bin\\revoltsim64.exe")
parser.add_argument('--python_port_initial', type=int, default=25338)
parser.add_argument('--load_sim_cfg', type=bool, default=False)
parser.add_argument('--n_sims', '-s', type=int, default=1)
parser.add_argument('--n_episodes', type=int, default=1000)  # One sim, 300 episodes, 5000 steps ~ 12 hours with 100 Hz

GLOBAL_SIM_ARGS = parser.parse_args()