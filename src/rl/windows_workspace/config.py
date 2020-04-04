import argparse
'''
 TODO USE DATETIME HERE?
 TODO FIND BETTER WAY TO CHANGE VALUES FAST - USE JSON
'''

'''
GLOBAL VARS FOR THE LOADING SEQUENCE
'''
parser = argparse.ArgumentParser()
fpath = 'data\cputest\cputest_s0'
parser = argparse.ArgumentParser()
parser.add_argument('--fpath', type=str,default=fpath) # remove -- infront if wanting to use enter path from terminal
parser.add_argument('--len', '-l', type=int, default=0)
parser.add_argument('--episodes', '-n', type=int, default=100)
parser.add_argument('--norender', '-nr', action='store_true')
parser.add_argument('--itr', '-i', type=int, default=-1) # this allows for loading models from earlier epochs than the last one!
parser.add_argument('--deterministic', '-d', action='store_true')

GLOBAL_TEST_ARGS = parser.parse_args()

'''
GLOBAL VARS FOR THE PPO ALGORITHM
'''

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='CartPole-v1') # HalfCheetah-v2
parser.add_argument('--hid', type=int, default=64) # Number of nodes in hidden layers
parser.add_argument('--l', type=int, default=2) # Number of hidden layers
parser.add_argument('--gamma', type=float, default=0.99) # Discount factor
parser.add_argument('--seed', '-s', type=int, default=0) # Random seed
parser.add_argument('--cpu', type=int, default=2) # Number of CPU's used during training
parser.add_argument('--steps', type=int, default=3000) # Number of steps during an entire episode for all processes combined
parser.add_argument('--epochs', type=int, default=2000) # Number of EPISODES
parser.add_argument('--max_ep_len', type=int, default=1000) # Number of steps per episode IN TOTAL
parser.add_argument('--exp_name', type=str, default='cputestwithTWO') # Name of data storage area

PPO_ARGS = parser.parse_args()

# '''
# GLOBAL VARS FOR THE SIMULATOR
# '''
# parser = argparse.ArgumentParser()
# parser.add_argument('--sim_cfg_path', type=str, default="C:\\Users\\simen\\Documents\\Utdanning\\GTK\\configuration") 
# parser.add_argument('--sim_path', type=str, default="C:\\Users\\simen\\Documents\\Utdanning\\GTK\\revoltsim\\bin\\revoltsim64.exe")
# parser.add_argument('--python_port_initial', type=int, default=25338)
# parser.add_argument('--load_cfg', type=bool, default=False)
# parser.add_argument('--n_sims', '-s', type=int, default=1)
# parser.add_argument('--n_episodes', type=int, default=1000)  # One sim, 300 episodes, 5000 steps ~ 12 hours with 100 Hz

# GLOBAL_SIM_ARGS = parser.parse_args()
