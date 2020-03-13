import numpy as np 

def rotation_matrix(a:float):
    return np.array([ [np.cos(a), -np.sin(a)],
                      [np.sin(a),  np.cos(a)]]) 

def clip(val,low,high) -> float:
    return max(min(val, high), low)


def wrap_angle(angle,deg = True):
    ''' Wrap an angle between -180 and 180 deg. deg == True means degrees, == False means radians'''
    ref = 180.0 if deg else np.pi
    return np.mod(angle + ref, 2*ref) - ref