import csv
import argparse
import numpy as np
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('--p', type=str, default='')
args = parser.parse_args()

stern_setpoints_path = 'bagfile__{}thrusterAllocation_stern_thruster_setpoints.csv'
stern_angles_path = 'bagfile__{}thrusterAllocation_pod_angle_input.csv'
bow_params_path = 'bagfile__{}bow_control.csv'
eta_path = path = 'bagfile__{}observer_eta_ned.csv'
ref_path = 'bagfile__{}reference_filter_state_desired.csv'

paths = [stern_setpoints_path, stern_angles_path, bow_params_path, eta_path, ref_path]

assert args.p != '', 'The script demands that prefix is passed as command line argument, e.g. " python3 add_prefix.py -p "RL" "'

for path in paths:

	openpath = path.format('')

	try:
		os.rename(openpath, path.format(args.p + '_'))
		print('Added new file prefix to {}'.format(openpath))
	except:
		print('Could not rename', openpath, ': could it have another name or not be in the folder?')

sys.exit()