'''
Utilities used for extracting states from the revolt simulator
'''
import random
import math
import numpy as np
from specific.errorFrame import ErrorFrame 

# NOTE: sim is used for a DigiTwin object

def standardize_state(state, bounds=[1.0,1.0,1.0,1.0,1.0,1.0]):
    ''' Based on normalizing a symmetric state distribution. State is (6,) numpy from ErrorFrame. Normalized according to visual inspection. '''
    assert len(state) == len(bounds), 'The state and bounds are not of same length!'

    for i,b in enumerate(bounds):
        state[i] /= (1.0 * b) 

    return state


def reset_sim(sim,**init):
    #set init values
    for modfeat in init:
        module, feature = modfeat.split('.')
        sim.val(module, feature, init[modfeat])
        
    #reset critical models to clear states from last episode
    sim.val('Hull', 'StateResetOn', 1)
    sim.val('THR1', 'LinActuator', int(2))
    sim.step(50) #min 50 steps should do it
    sim.val('Hull', 'StateResetOn', 0)
    sim.val('THR1', 'MtcOn', 1) # bow
    sim.val('THR1', 'AzmCmdMtc', 0.5*math.pi)
    sim.val('THR1', 'ThrustOrTorqueCmdMtc', 0.0) 
    sim.step(50) #min 50 steps should do it
    sim.val('THR2', 'MtcOn', 1) # stern, portside
    sim.val('THR2', 'ThrustOrTorqueCmdMtc', -30) 
    sim.val('THR2', 'AzmCmdMtc', 0*math.pi)
    sim.val('THR3', 'MtcOn', 1) # stern, starboard
    sim.val('THR3', 'ThrustOrTorqueCmdMtc', 30) 
    sim.val('THR3', 'AzmCmdMtc', 0*math.pi)
    return

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
        # print_pose(p_body,'Err') if step % 500 == 0 else None

def get_pose_3DOF(sim):
    yaw = float(sim.val('Hull','Yaw'))
    eta_6D = list(sim.val('Hull','Eta'))
    return [eta_6D[0],eta_6D[1],yaw]

def get_vel_3DOF(sim):
    nu = list(sim.val('Hull','Nu'))
    return [nu[0],nu[1],nu[5]]

def get_3DOF_state(sim):
    return get_pose_3DOF(sim) + get_vel_3DOF(sim)

def get_average_GPS_measurements(sim):
    # accounting for three gps-modules
    gpsvals = [0,0]
    for i in range(1,4):
        stt = 'GPS' + str(int(i))
        gpspos = list(sim.val(stt,'NorthEastPosition'))
        gpsvals[0] += gpspos[0]; gpsvals[1] += gpspos[1]
    
    gpsvals[0] /= 3; gpsvals[1] /= 3 # take average
    return gpsvals

def get_random_pose():
    N = (random.random()-0.5)*20.0
    E = (random.random()-0.5)*20.0
    Y = (random.random()-0.5)*2*(math.pi)
    return N, E, Y

def get_pose_on_radius(r=3):
    # use polar coords to always get a position of radius r away from setpoint
    # nice for testing average rewards from each run after training, but not so nice for training due to bad exploration
    r = r
    theta = random.random()*2*math.pi # random angle between 
    E = r * math.cos(theta)
    N = r * math.sin(theta) # y-coord -> North
    Y = 0
    return N, E, Y

def get_pose_on_state_space(n=10,e=10,y=np.pi):
    N = random.uniform(-n,n)
    E = random.uniform(-e,e)
    Y = random.uniform(-y,y)
    return N, E, Y