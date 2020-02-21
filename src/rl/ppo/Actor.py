import numpy as np 

class Actor():
    def __init__(self, alpha = 0.05, gamma = 0.95, epsilon = 0.1, decay = 0.9):
        self.alpha   = alpha
        self.e       = dict()   # dict of dicts: e[s][a] - needs check for e[s] before initializing e[s][a]
        self.PI      = dict()   # dict of dicts: PI[s][a] - needs initialization of all desirabilities if s has not been been visited before
        self.epsilon = epsilon  # epsilon-greedy
        self.decay   = decay    # eligibility decay
        self.gamma   = gamma    # discount rate

    def selectMove(self,s,env,epsilon=None):
        ''' Selects move in state s based on policy PI, using epsilon-greedy '''

        epsilon = self.epsilon if epsilon is None else epsilon # Allows for setting decaying epsilon or 0 etc.

        possible_actions = env.get_possible_hashed_moves()

        # Avoid the max operator iterating over empty dictionary
        if not possible_actions:
            return None 

        # If the state has not been visited yet, update with all possible actions with 0 desirability
        if s not in self.PI: 
            self.PI[s] = {action: 0 for action in possible_actions}

        # Get actions dictionary with a_i : desirability
        available_actions = self.PI[s]

        # Select action through epsilon greedy
        if np.random.rand() < epsilon:
            best_action = np.random.choice(list(available_actions.keys()))
        else:
            best_action = max(available_actions, key=lambda key: available_actions[key]) 
        
        return best_action

    def update(self,s,a,td_difference):
        ''' Update the actors' policy and eligibility of each state '''
        self.PI[s][a] += self.alpha * td_difference * self.e[s][a]
        self.e[s][a] = self.gamma * self.decay * self.e[s][a]

    def resetEligibilities(self):
        self.e = dict()