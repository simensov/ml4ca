# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:54:44 2020

@author: JONOM
"""

from digitwin import DigiTwin
import threading
from log import log, forcelog
import math as m
import random as r

# SIMENs CLASSES
from errorFrame import ErrorFrame 
# from utils_math import *
from utils_sim import get_pose_3DOF, get_vel_3DOF
from utils_debugging import print_pose
from environment import RevoltSimulator


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
NUM_EPISODES        = 15


def reset_sim(sim,**init):
    #set init values
    for modfeat in init:
        module, feature = modfeat.split('.')
        sim.val(module, feature, init[modfeat])
        
    #reset critical models to clear states from last episode
    sim.val('Hull', 'StateResetOn', 1, REPORTRESETS)
    sim.step(50) #min 50 steps should do it
    sim.val('Hull', 'StateResetOn', 0, REPORTRESETS)

    sim.val('THR1', 'MtcOn', 1, REPORTRESETS) # bow
    sim.val('THR1', 'ThrustOrTorqueCmdMtc', 0, REPORTRESETS) 
    sim.val('THR1', 'AzmCmdMtc', 0.5*m.pi, REPORTRESETS)
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
        # set inputs and parameters for next step
        # create state vector from measurements

        # TODO RL: Here, the initial action choice has to be done

        # Step the simulation 1 step or X steps digitwin.step(x)
        sim.step()

        # Observe new state and reward. 
        pose = get_pose_3DOF(sim)
        p_body = err.get_pose(pose)

        vel = get_vel_3DOF(sim)
        
        print_pose(p_body,'Err') if step % 500 == 0 else None
        print_pose(vel,'Vel') if step % 500 == 0 else None

def episode(env,agent,episode_length : int = 5000):
    BATCH_SIZE = 64
    GAMMA = 0.9

    s = env.reset()
    states, actions, rewards = [], [], [] # todo could be stores as tuples
    episodal_reward = 0
    for t in range(episode_length):
        # env.render() happens automatically?
        a = agent.choose_action(s)
        sn, r, done, info = env.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)

        s = sn
        episodal_reward += r

        if (t+1) % BATCH_SIZE == 0 or t == episode_length-1: # update if batch size or if time steps has reached limit (important as DP operations dont have a terminal state)
            v_sn = agent.V(sn)

            discounted_rewards = []
            for reward in rewards[::-1]: # reverse the list
                v_sn = reward + GAMMA * v_sn
                discounted_rewards.append(v_sn)

            discounted_rewards.reverse()
            states_batch, actions_batch, reward_batch = np.vstack(states), np.vstack(actions), np.array(discounted_rewards)[:, np.newaxis]
            states, actions, rewards = [], [], []
            ppo.update(states_batch, actions_batch, reward_batch) # Entranar el Cliente y el actor (Estado, acciones, discounted_r)


def get_random_pose():
    N = (r.random()-0.5)*20.0
    E = (r.random()-0.5)*20.0
    Y = (r.random()-0.5)*2*(m.pi)
    return N, E, Y

#MAIN...        
if __name__ == "__main__":

    #vars
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

    for ep_ix in range(NUM_EPISODES):
        for sim_ix in range(NUM_SIMULATORS):
            N, E, Y = get_random_pose()
            print('Random pos set to {}'.format([N,E,Y*180/m.pi]))
            init = {'Hull.PosNED':[N,E],'Hull.PosAttitude':[0,0,Y]}

            if THREADING:
                sim_semaphores[sim_ix].acquire()
                log("Locking sim" + str(sim_ix+1) + "/" + str(NUM_SIMULATORS))
                t = threading.Thread(target=simulate_episode, args=[sims[sim_ix]], kwargs=init)
                t.daemon = True
                t.start()
    
            else:
                simulate_episode(sims[sim_ix],**init)


