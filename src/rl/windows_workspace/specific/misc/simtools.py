'''
Utilities used for extracting states from the revolt simulator
'''
import random
import math
import numpy as np
from specific.errorFrame import ErrorFrame 

# NOTE: sim is used for a DigiTwin object

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

def get_random_pose_on_radius(r=5, angle=5*np.pi/180):
    # use polar coords to always get a position of radius r away from setpoint
    # nice for testing average rewards from each run after training, but not so nice for training due to bad exploration
    theta = random.random()*2*math.pi # random angle between origin and the place on the circle to put the vessel. NOT the same as yaw angle
    E = r * math.cos(theta)
    N = r * math.sin(theta) # y-coord -> North
    Y = random.uniform(-angle,angle)
    return N, E, Y


def get_fixed_pose_on_radius(n, r=5, angle=5*np.pi/180):
    # use polar coords to always get a position of radius r away from setpoint
    # nice for testing average rewards from each run after training, but not so nice for training due to bad exploration
    thetas = [0.0, math.pi/4, math.pi/2, math.pi, 5*math.pi/4, 3*math.pi/2]
    angles = [0.0,   5.0,      0.0,        5.0,       0.0,       -5.0]
    
    if len(thetas) <= n: # Avoid n accessing unaccesable element, warn user about it
        print('n larger than 5 passed to fixed_points: starting on element 0')
        n = n % len(thetas) 

    NED_angle_to_unit_circle_angle = math.pi/2 - thetas[n]
    E = r * math.cos(NED_angle_to_unit_circle_angle)
    N = r * math.sin(NED_angle_to_unit_circle_angle) # y-coord -> North
    Y = angles[n] * math.pi / 180
    print('Fixed pose returns:', N, E, Y)
    return N, E, Y

def get_pose_on_state_space(bounds = [5,5,np.pi/18], fraction = 1.0):
    assert len(bounds) == 3, 'get_pose_on_state_space only sets 3dof eta'
    n, e, y = bounds[0] * fraction, bounds[1] * fraction, bounds[2] * fraction
    N = random.uniform(-n,n)
    E = random.uniform(-e,e)
    Y = random.uniform(-y,y)
    return N, E, Y

def get_vel_on_state_space(bounds = [2.2, 0.35, 0.60], fraction = 1.0):
    assert len(bounds) == 3, 'get_vel_on_state_space only sets 3dof nu'
    u, v, r = bounds[0] * fraction, bounds[1] * fraction, bounds[2] * fraction
    U = random.uniform(-u,u)
    V = random.uniform(-v,v)
    R = random.uniform(-r,r)
    return U,V,R


def str2bool(v):
    import argparse
    '''
        parser.add_argument("--nice", type=str2bool, nargs='?',
                                const=True, default=False,
                                help="Activate nice mode.")
        
        allows me to use:
        script --nice
        script --nice <bool>
    '''
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')