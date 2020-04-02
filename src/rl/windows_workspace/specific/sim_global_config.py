

SIM_CONFIG_PATH     = "C:\\Users\\simen\\Documents\\Utdanning\\GTK\\configuration"
SIM_PATH            = "C:\\Users\\simen\\Documents\\Utdanning\\GTK\\revoltsim\\bin\\revoltsim64.exe"
PYTHON_PORT_INITIAL = 25338
LOAD_SIM_CFG        = False
NUM_SIMULATORS      = 1
NUM_EPISODES        = 1000

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--sim_cfg_path', type=str, default="C:\\Users\\simen\\Documents\\Utdanning\\GTK\\configuration") 
parser.add_argument('--sim_path', type=str, default="C:\\Users\\simen\\Documents\\Utdanning\\GTK\\revoltsim\\bin\\revoltsim64.exe")
parser.add_argument('--python_port_initial', type=int, default=25338)
parser.add_argument('--load_cfg', type=bool, default=False)
parser.add_argument('--n_sims', '-s', type=int, default=1)
parser.add_argument('--n_episodes', type=int, default=1000)  # One sim, 300 episodes, 5000 steps ~ 12 hours with 100 Hz
GLOBAL_SIM_ARGS = parser.parse_args()