# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:54:44 2020

@author: JONOM
"""

from digitwin import DigiTwin
import threading
from utils.log import log, forcelog
import math as m
import random as r
import time

from errorFrame import ErrorFrame 
# from utils.mathematics import *
from utils.simtools import get_pose_3DOF, get_vel_3DOF, get_random_pose
from utils.debug import print_pose
from environment import FixedThrusters
from agents.ppo import PPO
import numpy as np
import matplotlib.pyplot as plt 

#defs
SIM_CONFIG_PATH     = "C:\\Users\\simen\\Documents\\Utdanning\\GTK\\configuration"
SIM_PATH            = "C:\\Users\\simen\\Documents\\Utdanning\\GTK\\revoltsim\\bin\\revoltsim64.exe"
PYTHON_PORT_INITIAL = 25338
LOAD_SIM_CFG        = False
NON_SIM_DEBUG       = False
REPORT              = True #write out what is written to sim
REPORTRESETS        = False
NUM_SIMULATORS      = 1
THREADING           = False # = True if threading out sim process
NUM_EPISODES        = 30 # One sim, 300 episodes, 5000 steps ~ 12 hours
OLD_CODE            = False


def reset_sim(sim,**init):
    #set init values
    for modfeat in init:
        module, feature = modfeat.split('.')
        sim.val(module, feature, init[modfeat])
        
    #reset critical models to clear states from last episode
    sim.val('Hull', 'StateResetOn', 1, REPORTRESETS)
    sim.val('THR1', 'LinActuator', int(2), REPORTRESETS)
    sim.step(50) #min 50 steps should do it
    sim.val('Hull', 'StateResetOn', 0, REPORTRESETS)

    sim.val('THR1', 'MtcOn', 1, REPORTRESETS) # bow
    sim.val('THR1', 'AzmCmdMtc', 0.5*m.pi, REPORTRESETS)
    sim.val('THR1', 'ThrustOrTorqueCmdMtc', 0.0, REPORTRESETS) 
    sim.step(50) #min 50 steps should do it
    sim.val('THR2', 'MtcOn', 1, REPORTRESETS) # stern, portside
    sim.val('THR2', 'ThrustOrTorqueCmdMtc', -30, REPORTRESETS) 
    sim.val('THR2', 'AzmCmdMtc', 0*m.pi, REPORTRESETS)
    sim.val('THR3', 'MtcOn', 1, REPORTRESETS) # stern, starboard
    sim.val('THR3', 'ThrustOrTorqueCmdMtc', 30, REPORTRESETS) 
    sim.val('THR3', 'AzmCmdMtc', 0*m.pi, REPORTRESETS)
    return

'''
Relevant parameters and outputs:
    PARAMS: Hull.posAttitude [0,1,2] roll, pitch, yaw I guess
    OUTPUT: Hull.Eta [0,1,2,3,4,5] 
    OUTPUT: Hull.Nu [0,1,2,3,4,5] body velocity
    OUTPUT: Hull.SurgeSpeed and .SwaySpeed - compare to nu-vector?
    OUTPUT: Hull.Yaw 
'''
    
def simulate_episode(sim, **init):
    '''
    NB: adding / removing arguments in thus function might make it hard for the threading 
    '''
    reset_sim(sim,**init)
    steps = 4000 # TODO upper limit for PPO
    err = ErrorFrame(pos=get_pose_3DOF(sim)) # TODO passing reference point to this function?
    p_body = err.get_pose()

    for step in range(steps):
        # TODO RL: Here, the action choice has to be done
        sim.step() # Step the simulation 1 step or X steps digitwin.step(x)
        # Observe new state and reward. 
        pose = get_pose_3DOF(sim)
        p_body = err.get_pose(pose)
        vel = get_vel_3DOF(sim)
        print_pose(p_body,'Err') if step % 500 == 0 else None

def episode_test(env,**init):
    ''' Function to move into the trainer class '''
    episode_length = 5000 # trainer
    batch_size = 256 # trainer - training during episodes: n-step TD instead of 1-step or Monte Carlo
    gamma = 0.99 # trainer
    print("episode by ", env.sim.name)

    agent = PPO(num_states=env.num_states,num_actions=env.num_actions)
    s = env.reset(**init)
    states, actions, pred_actions, rewards = [], [], [], [] 
    episodal_reward = 0

    for t in range(episode_length):
        a_pi, a_old_pi = agent.act(s)
        a_pi = a_pi * 100.0 # scale agent's choice TODO this should be hidden somewhere
        sn, r, done, info = env.step(a_pi) # TODO done needs to be used so that batches don't contain random states next to each other
        states.append(s) # TODO standardize state
        actions.append(a_pi)
        pred_actions.append(a_old_pi)
        rewards.append(r)
        s = sn
        episodal_reward += r

        if (t+1) % batch_size == 0 or t == episode_length-1: # update if batch size or if time steps has reached limit (important as DP operations dont have a terminal state)
            
            v_sn = agent.V(sn) # this is V(sT) from eq. 10 in the paper: the last state of the trajectory
            discounted_rewards = []
            for reward in rewards[::-1]: # reverse the list
                v_sn = reward + gamma * v_sn
                discounted_rewards.append(v_sn) # advantages are calculated by the agent, using its V-predictions

            discounted_rewards.reverse() 
            # TODO standardize returns: 
            # https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63
            # -> https://arxiv.org/abs/1506.02438


            batch = np.vstack(states), np.vstack(actions), np.vstack(pred_actions), np.vstack(discounted_rewards)
            states, actions, pred_actions, rewards = [], [], [], []
            advantages, actor_loss, critic_loss = agent.update(batch)
    
    print(episodal_reward)
    return episodal_reward
        
'''
MAIN
'''

if __name__ == "__main__":

    # Vars
    sims = []
    sim_semaphores = []
    
    #Start up all simulators
    for sim_ix in range(NUM_SIMULATORS):
        python_port = PYTHON_PORT_INITIAL + sim_ix
        log("Open CS sim " + str(sim_ix) + " Python_port=" + str(python_port))
        sims.append(None)
        if not NON_SIM_DEBUG:
            sims[-1] = DigiTwin('Sim'+str(1+sim_ix), LOAD_SIM_CFG, SIM_PATH, SIM_CONFIG_PATH, python_port)
        sim_semaphores.append(threading.Semaphore())
        
    log("Connected to simulators and configuration loaded")

    envs = [FixedThrusters(s) for s in sims]
    all_rewards = [[]*NUM_SIMULATORS]

    for ep_ix in range(NUM_EPISODES):
        for sim_ix in range(NUM_SIMULATORS):
            N, E, Y = get_random_pose()
            print('Random pos set to {}'.format([N,E,Y*180/m.pi]))
            init = {'Hull.PosNED':[N,E],'Hull.PosAttitude':[0,0,Y],'THR1.LinActuator':2.0}

            if THREADING:
                sim_semaphores[sim_ix].acquire()
                log("Locking sim" + str(sim_ix+1) + "/" + str(NUM_SIMULATORS))
                # t = threading.Thread(target=simulate_episode, args=[sims[sim_ix]], kwargs=init)
                t = threading.Thread(target=episode_test, args=[envs[sim_ix]], kwargs=init)
                t.daemon = True
                t.start()

                # TODO hvis alle threads kunne ha delt en felles trajectory buffer hadde det vært awsome
    
            else:
                if OLD_CODE:
                    simulate_episode(sims[sim_ix],**init)
                else:
                    print('Episode {}'.format(ep_ix+1))
                    episode_reward = episode_test(envs[sim_ix],**init)
                    all_rewards[sim_ix].append(episode_reward)

    if not THREADING and not OLD_CODE:
        for rewards in all_rewards:
            print(rewards)
            plt.figure()
            plt.plot([i for i in range(NUM_EPISODES)], rewards)
            plt.xlabel('Episodes')
            plt.ylabel('Episodal rewards')
        plt.show()

    # TODO display performance from stored model
    