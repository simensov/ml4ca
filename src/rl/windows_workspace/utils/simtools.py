'''
Utilities used for extracting states from the revolt simulator
'''
import random
import math

# NOTE: sim is used for a DigiTwin object

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
    r = r
    theta = random.random()*2*math.pi # random angle between 
    E = r * math.cos(theta)
    N = r * math.sin(theta) # y-coord -> North
    Y = 0
    return N, E, Y