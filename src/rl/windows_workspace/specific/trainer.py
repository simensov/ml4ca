import numpy as np
from specific.digitwin import DigiTwin
import time
from specific.misc.simtools import get_pose_on_radius, standardize_state
from specific.customEnv import RevoltSimple
from specific.agents.ppo import PPO
import matplotlib.pyplot as plt
import datetime

class Trainer(object):
    # TODO Name this class something else like a DigiTwin handler or something. 

    def __init__(self, n_sims, start = False, args = None):
        
        assert isinstance(n_sims,int) and n_sims > 0, 'Number of simulators must be an integer'
        self._n_sims     = n_sims
        self._digitwins  = []*n_sims  # list of independent simulators
        self._gamma      = 0.99
        self._batch_size = 256
        self._ep_len = 4 * self._batch_size
        if start and args is not None:
            self.start_simulators(args.sim_path, args.python_port_initial, args.sim_cfg_path, args.load_cfg)

    def start_simulators(self,sim_path,python_port_initial,sim_cfg_path,load_cfg):
        #Start up all simulators
        for sim_ix in range(self._n_sims):
            python_port = python_port_initial + sim_ix
            print("Open CS sim " + str(sim_ix) + " Python_port=" + str(python_port))
            self._digitwins.append(None) # Weird, by necessary order of commands
            self._digitwins[-1] = DigiTwin('Sim'+str(1+sim_ix), load_cfg, sim_path, sim_cfg_path, python_port)
        print("Connected to simulators and configuration loaded")

    def get_digitwins(self):
        return self._digitwins      

    '''
    The functions below 
    '''
    def run_episode(self,env,agent,**init):

        s = env.reset(**init)
        states, actions, pred_actions, rewards = [], [], [], []
        episodal_reward = 0

        for t in range(self._ep_len):
            s = standardize_state(s)
            a_pi, a_old_pi = agent.act(s)
            sn, r, done, info = env.step(a_pi) # TODO done needs to be used so that batches don't contain random states next to each other
            states.append(s) 
            actions.append(a_pi)
            pred_actions.append(a_old_pi)
            rewards.append(r)
            s = sn
            episodal_reward += r

            if (t+1) % self._batch_size == 0 or t == self._ep_len-1: # update if batch size or if time steps has reached limit (important as DP operations dont have a terminal state)
                
                sn = standardize_state(sn)
                v_sn = agent.V(sn) # this is V(sT) from eq. 10 in the paper: the last state of the trajectory
                discounted_rewards = []
                for reward in rewards[::-1]: # reverse the list
                    v_sn = reward + self._gamma * v_sn
                    discounted_rewards.append(v_sn) # advantages are calculated by the agent, using its V-predictions

                discounted_rewards.reverse() 
                # TODO standardize returns: 
                # https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63
                # -> https://arxiv.org/abs/1506.02438

                batch = np.vstack(states), np.vstack(actions), np.vstack(pred_actions), np.vstack(discounted_rewards)
                states, actions, pred_actions, rewards = [], [], [], []
                advantages, actor_loss, critic_loss = agent.update(batch)
        
        return episodal_reward

    def train(self,n_episodes = 1000):

        # Setup envs and agents for training
        envs = [RevoltSimple(s) for s in self._digitwins]
        agents = [PPO(num_states=env.num_states,num_actions=env.num_actions) for env in envs]
        all_rewards = [[]*self._n_sims]

        time_begin = time.time()
        for ep_ix in range(n_episodes):
            for sim_ix in range(self._n_sims):
                N, E, Y = get_pose_on_radius() # get_random_pose()
                init = {'Hull.PosNED':[N,E],'Hull.PosAttitude':[0,0,Y]}

                print('''\ Ep: {} - Sim: {} '''.format(ep_ix+1,sim_ix+1))
                stime = time.time()
                episode_reward = self.run_episode(envs[sim_ix], agents[sim_ix],**init)
                print('''... took {:.2f} seconds, giving reward of {:.2f} '''.format(time.time() - stime, episode_reward))
                all_rewards[sim_ix].append(episode_reward)

        print('Entire training took: {} hh:mm:ss'.format(datetime.timedelta(seconds= time.time() - time_begin)))

        for rewards in all_rewards:
            plt.figure()
            plt.plot([i for i in range(n_episodes)], rewards)
            plt.xlabel('Episodes')
            plt.ylabel('Episodal rewards')
        plt.show()




# from inside the training loop
#     if THREADING:
#     sim_semaphores[sim_ix].acquire()
#     log("Locking sim" + str(sim_ix+1) + "/" + str(NUM_SIMULATORS))
#     t = threading.Thread(target=simulate_episode, args=[sims[sim_ix]], kwargs=init) if OLD_CODE else threading.Thread(target=episode_test, args=[envs[sim_ix],agents[sim_ix]], kwargs=init)
#     t.daemon = True
#     t.start()

#     # TODO hvis alle threads kunne ha delt en felles trajectory buffer hadde det vært awsome, slik som Spinning Up sin har!

# else:
#     if OLD_CODE:
#         simulate_episode(sims[sim_ix],**init)