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

appendix = args.p + '_' if args.p else ''

for path in paths:
	path = path.format(appendix)
	r = csv.reader(open(path))
	data = np.array(list(r))

	new_data = np.hstack((data, np.zeros((data.shape[0],1))))

	new_data[0,-1] = 'secs_since_start'
	start_time = float(new_data[1,0])

	for i,row in enumerate(new_data):
		if i > 0:
			diff = float(row[0]) - start_time
			row[-1] = str(diff / 10**9)

	writer = csv.writer(open(path, 'w'))
	writer.writerows(new_data)

sys.exit()