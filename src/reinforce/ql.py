#!/usr/bin/env python

'''
ROS Node for using the reinforcement learning scheme Tabular Q-Learning for solving thrust allocation on ReVolt

It contains:
	- The learning module itself
	- Storage functions for storing the Q-tables
	- Loading functions for reuse of previously trained Q-tables

Dependencies:


Author: Simen Sem Oevereng, December 2019, simensem@gmail.com
'''

# Ros 
import rospy
from custom_msgs.msg import someData # TODO find a suitable data type
import numpy as np 
import sys # for sys.exit()

class TabularQ():
	'''
	Stores Q-learning parameters and Q-table
	'''
	def __init__(self,states=np.array([[0,0,0]]).T,actions=np.array([[-1,0,1]]).T):
		self.states = states
		self.actions = actions
		self.q = dict([((s, a), 0.0) for s in states for a in actions])

    def copy(self):
        q_copy = TabularQ(self.states, self.actions)
        q_copy.q.update(self.q)
        return q_copy

    def set(self, s, a, v):
        self.q[(s,a)] = v

    def get(self, s, a):
        return self.q[(s,a)]

    def update(self, data):
        for i in range(len(data)):
            e = data[i] # e is a triple tuple of (previous state, action taken, target = reward + gamma*qmax)
            s = e[0]
            a = e[1]
            t = e[2]
            Qlast = self.q[(s,a)]
            self.q[(s,a)] += self.alpha * ( t - Qlast )


class Table2npy():

    def __init__(self,q):
        self.q = q
        self.filename = 'qtableentries.npy'

        # TODO addcheck to see if q is a numpy array or dict. If the latter, transform to numpy array: https://stackoverflow.com/questions/15579649/python-dict-to-numpy-structured-array
        # TODO adjust filename according to the current time in order to not overwrite an array without purpose

    def save(self):
    	'''
		Stores Q-Table for later convenience
    	'''
		np.save(self.filename, self.q)

	def load(self):
		self.q = np.load(self.filename)


class QLearning():

	def __init__(self,qtable=TabularQ(),gamma=0.99,epsilon=0.9,alpha=0.1):
		self.qtable = qtable
		self.gamma = gamma
		self.epsilon = epsilon
		self.alpha = alpha


	def epsilon_greedy(self,state):
	    '''
	    Returns the best action if a uniformly random picked number (0-1) is less than epsilon.
	    Else, it returns a randomly selected action.
	    Action returned an integer between 0 and 2, specialized for the pendulum.
	    '''

	    actions = self.qtable.actions
	    Qs = self.qtable[state,:]

	    a = 0
	    if np.random.uniform() < epsilon:
	        a = actions[np.argmax(Qs[:])]
	    else:
	        a = np.random.choice(actions)
	    return a


	def nearest_index(self,array,num):
	    '''
	    Returns the index of array which contains the value closest to num

	    Input:
	        - array     A Numpy array 
	        - num       A float
	    '''
	    return np.abs(array - num).argmin()


	def episodic_run(self):
		'''
		Runs one episode of Q-learning, using the previously updated. 
		An episode counts as the steps taken from an initial state until a terminal state
		'''
	    # Reset environment somehow TODO
	    (theta,dtheta), r, done = env.reset() # reset environment to a random state

	    # Ensure that initial state is not the terminal state
	    while not env.actions():
	        (theta,dtheta), r, done = env.reset()
	    
	    # Ensure that the states' indices are represented according to discretization
	    theta = nearest_index(ranges[0],theta)
	    dtheta = nearest_index(ranges[1],dtheta)

	    while not done:
	        # Take an action based on epsilon-greedy and observe result
	        a = epsilon_greedy_p(Q[theta,dtheta,:],env.actions(),epsilon)
	        (theta_n,dtheta_n), r, done = env.step(a)

	        # Ensure that the state's indices are represented according to discretization
	        theta_n = nearest_index(ranges[0],theta_n)
	        dtheta_n = nearest_index(ranges[1],dtheta_n)

	        # Update Q-table based on experience
	        Q_max = np.max(Q[theta_n,dtheta_n,:])
	        Q[theta,dtheta,a] += alpha * (r + gamma * Q_max - Q[theta,dtheta,a])

	        # Update state indices
	        theta = theta_n
	        dtheta = dtheta_n

	    return Q

# TODO implement the function that runs the learning through x no. of episodes, and stores the q-table after training
# TODO implement a predict function (just using the max operator on the loaded numpy array), so that the q-table can be used online after training.


'''
Main
'''

def main():
	# Initialize publisher and node with rate
    pub = rospy.Publisher('RL4TA/thrusterCommands', SomeData, queue_size=10)
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
