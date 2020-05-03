import csv
import argparse
import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='')
args = parser.parse_args()

stern_setpoints_path = 'bagfile__{}thrusterAllocation_stern_thruster_setpoints.csv'
stern_angles_path = 'bagfile__{}thrusterAllocation_pod_angle_input.csv'
bow_params_path = 'bagfile__{}bow_control.csv'
eta_path = path = 'bagfile__{}observer_eta_ned.csv'

paths = [stern_setpoints_path, stern_angles_path, bow_params_path, eta_path]

assert args.p != '', 'The script demands that prefix is passed as command line argument, e.g. " python3 add_prefix.py -p "RL" "'

for path in paths:

	openpath = path.format('')
	try:
		r = csv.reader(open(openpath))
		print('Opened csv at {}'.format(openpath))
	except:
		print('Could not find any files at {}'.format(openpath))
		continue

	data = np.array(list(r))
	path = path.format(args.p + '_')
	writer = csv.writer(open(path, 'w'))
	writer.writerows(data)

	print('Added new file with prefix to {}'.format(path))

sys.exit()