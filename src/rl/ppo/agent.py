from actor import Actor
from critic import Critic


class ActorCritic():
    def __init__(self,tabular_critic = True, lr_critic=0.05, decay=0.9, gamma=0.95, input_size = 16, layer_sizes=(15,15,1),lr_actor=0.05,eps_init=0.5):
        self.critic   = Critic(is_tabular=tabular_critic,alpha=lr_critic,input_size=input_size,layer_sizes=layer_sizes,decay=decay,gamma=gamma)
        self.actor    = Actor(alpha=lr_actor,decay=decay,gamma=gamma)
        self.eps_init = eps_init # initial epsilon for training
        self.gamma    = gamma

    def runEpisode(self,env,epsilon):
        ''' Goes through an episode of states and actions in the environment '''

        # Reset eligibilities at beginning of episode
        self.actor.resetEligibilities()
        self.critic.resetEligibilities()

        if env.is_done():
            return env.get_status()
        
        s = env.get_bytestring() # Set initial state
        a = self.actor.selectMove(s,env,epsilon) # Determine first move

        SAPs = [] # state,action-pairs encountered in this episode

        while not env.is_done():

            # Add state and action to the episode overview BEGINNING since the trace should be iterated going from most recent SAP-encountered
            SAPs.insert(0,(s,a))

            s_next, reward, done = env.perform_hashed_move(a)
                    
            # Actor, reset SAP eligibilities
            if s not in self.actor.e:
                self.actor.e[s] = dict()

            # Critic, reset eligibility trace as this s,a might have been encountered before, but now has a new reward to be traced down the trajectory
            self.actor.e[s][a] = 1 

            # Critic, temporal difference and reset eligibility (latter is only used if_tabular)
            td_difference = self.critic.calculateTD(s,s_next,reward)
            self.critic.e[s] = 1
            
            # Update with temporal difference + eligibility traces
            for s,a in SAPs:
                self.critic.update(s,td_difference)
                self.actor.update(s,a,td_difference)

            # Stop episode if a terminal state has been encountered, or carry on
            if done:
                break
            else:
                s = s_next
                a = self.actor.selectMove(s_next,env,epsilon)

        return env.get_status() # Return the 

    def train(self,env,episodes,display=False):

        info = 'tabular' if self.critic.is_tabular else 'neural'
        print('¤¤¤¤¤ Starting training with {} critic over {} episodes'.format(info,episodes))

        list_of_episodal_results = []
        best_reward_development = []

        for i in range(episodes):
            env.reset()
            print('Episode {}'.format(i)) if i % 50 == 0 else None
            epsilon = self.eps_init * np.exp(-5/episodes * i) + 0.001 # Decline epsilon as func of episodes (or just pick 0.1)
            
            result_param = self.runEpisode(env,epsilon)
            list_of_episodal_results.append(result_param)

            if i % 1 == 0: # Test current policy
                env.reset()
                rew = self.getRewardWithPolicy(env)
                best_reward_development.append((i,rew))

        return list_of_episodal_results, best_reward_development        

    def getRewardWithPolicy(self,env):
        accumulated_reward = 0
        while not env.is_done():
            current_state = env.get_bytestring()
            a = self.actor.selectMove(current_state,env,epsilon=0)
            env.perform_hashed_move(a) # perform most desirable move
            accumulated_reward += env.calculate_reward()

        return accumulated_reward


if __name__ == '__main__':
    agent = ActorCritic(tabular_critic=False)