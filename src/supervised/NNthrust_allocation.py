#!/usr/bin/env python3

'''
ROS Node for using a neural network for solving the thrust allocation problem on the ReVolt model ship.
The neural network was trained with supervised learning, where a dataset was generated with all possible thruster inputs and azimuth angles, 
with the corresponding forces and moments generated. The neural network therefore approximates the pseudoinverse.

It was trained on the dataset where the azimuth angles were rotating from -pi to pi, but wrapped in sin(x/2) to keep the values within -1 and 1,
while still having each values representing a unique angle. Thruster inputs were scaled to -1,1 by dividing the thruster input on 100.0, as the inputs were given in percentages.
The corresponding forces as a function of angles and thruster input percentages was derived according to Alfheim and Muggerud, 2016.

See sl.py for more details on the dataset generation. There it is also understandable how the predictions of the network has to be rescaled.
Also, the input forces needs to be rescaled due to the scaling of them as inputs during training. See train.py for more details on the dataset augmentation.

@author: Simen Sem Oevereng, simensem@gmail.com. November 2019.
'''

import rospy
from custom_msgs.msg import someData # TODO find a suitable data type
import numpy as np 
import sys # for sys.exit()



def main():
	# Initialize publisher and node with rate
    pub = rospy.Publisher('RL4TA/thrusterCommands', someData, queue_size=10)
    rospy.init_node('RL4TA', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    ## Configure any of the variables
    try:
        # If initialization can go wrong
    except:
        rospy.logerr("Was not able to do what you probabliy wanted to. Calling sys.exit()")
        sys.exit()

    # Initialize Q-Learning object
    while not rospy.is_shutdown():
        # Run the node, e.g. ql.run(rate)

    # Perform any closing if necessary



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
