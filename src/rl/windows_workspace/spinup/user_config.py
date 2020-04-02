import os
import os.path as osp

# Default neural network backend for each algo
# (Must be either 'tf1' or 'pytorch')
DEFAULT_BACKEND = {
    'vpg': 'tf1',
    'trpo': 'tf1',
    'ppo': 'tf1',
    'ddpg': 'tf1',
    'td3': 'tf1',
    'sac': 'tf1'
}

# Where experiment outputs are saved by default:
DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'data')

# Whether to automatically insert a date and time stamp into the names of
# save directories:
FORCE_DATESTAMP = False # TODO setting this to True creates trouble

# Whether GridSearch provides automatically-generated default shorthands:
DEFAULT_SHORTHAND = True

# Tells the GridSearch how many seconds to pause for before launching 
# experiments.
WAIT_BEFORE_LAUNCH = 5