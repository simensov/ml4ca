import numpy as np
from environment import RevoltSimulator

class Trainer(object):

    def __init__(self,
                 env,
                 agent,
                 episodes,
                 batch,
                 gamma):
        
        self.envs = [] # envs # list of independent environments
        self.agent = agent
        self.episodes = episodes
        self.batch_size = batch_size
        self.gamma = gamma

    def start_simulators(self):
        sims = []
        sim_semaphores = []
        NUM_SIMULATORS = len(self.envs)

        #Start up all simulators
        for sim_ix in range(NUM_SIMULATORS):
            python_port = PYTHON_PORT_INITIAL + sim_ix
            log("Open CS sim " + str(sim_ix) + " Python_port=" + str(python_port))
            sims.append(None)
            if not NON_SIM_DEBUG:
                sims[-1] = DigiTwin('Sim'+str(1+sim_ix), LOAD_SIM_CFG, SIM_PATH, SIM_CONFIG_PATH, python_port)
            sim_semaphores.append(threading.Semaphore())

            self.envs.append(RevoltSimulator(sims[-1]))

    def episode(self,episode_length : int = 5000):
        s = self.env.reset()
        states, actions, rewards = [], [], [] # todo could be stores as tuples
        episodal_reward = 0
        for t in range(episode_length):
            # env.render() happens automatically?
            a = self.agent.act(s)
            sn, r, done, info = self.env.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(r)

            s = sn
            episodal_reward += r

            if (t+1) % self.batch_size == 0 or t == episode_length-1: # update if batch size or if time steps has reached limit (important as DP operations dont have a terminal state)
                v_sn = self.agent.V(sn)

                discounted_rewards = []
                for reward in rewards[::-1]: # reverse the list
                    v_sn = reward + self.gamma * v_sn
                    discounted_rewards.append(v_sn)

                discounted_rewards.reverse()
                states_batch, actions_batch, reward_batch = np.vstack(states), np.vstack(actions), np.array(discounted_rewards)[:, np.newaxis]
                states, actions, rewards = [], [], []
                self.agent.update(states_batch, actions_batch, reward_batch) # Entranar el Cliente y el actor (Estado, acciones, discounted_r)
        return
    
    def train(self):

        for _ in range(self.episodes):
            self.episode()

