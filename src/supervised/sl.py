# Generate data from 

# TRY WITH BOTH FIXED AZIMUTHS AND ROTATING TO TEST PERFORMANCE!

import numpy as np
from math import sin, cos


# Order of thrusters: bow, stern(starboard), stern(port)
# Alfeim and Muggerud used port, starboard, bow representation

# Distance from CG, defined by bow,starboard,down oriented body frame
LX = [0 0.15 -0.15]
LY = [1.08 -1.12 -1.12]


def B(alpha):
	'''
	Returns an array of the effectiveness matrix

	:params:
		alpha - An 3x1 list/array of azimuth angles
	'''
	return np.array([[cos(alpha[0]), cos(alpha[1]), cos(alpha[2])],
					 [sin(alpha[0]), sin(alpha[1]), sin(alpha[2])],
					 [LX[0]*sin(alpha[0]) - LY[0]*cos(alpha[0]),LX[1]*sin(alpha[1]) - LY[1]*cos(alpha[1]), LX[2]*sin(alpha[2]) - LY[2]*cos(alpha[2])]
					 ])


def F(u):
	'''
	Calculate real force vector from applied control inputs. 
	The relation used was found from Alfheim and Muggerud, 2016, ch. 8.3.
	K1pm = K2pm = 2.7e-3
	K3m = 1.518e−3 for positive u, or 6.172e−4 for negative u 

	=> Fi = Ki|ui|ui
	u is given in percentages

	:params:
		u - A 3x1 vector of applied thruster forces, ranging from 0 - 100%
	'''
	K0 = K1 = 
	K2 = [, ]
	return np.array([[ 2.7e-3 * abs(u[0]) * u[0]],
					 [ 2.7e-3 * abs(u[1]) * u[1]],
					 [ 1.518e−3 * abs(u[2]) * u[2] if u[2] >= 0 else 6.172e−4 * abs(u[2]) * u[2]],
					 ])

# Create dataset. Constrain bow thruster to +-270 degrees
# Use radians