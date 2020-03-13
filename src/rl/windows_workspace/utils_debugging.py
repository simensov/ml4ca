import math

def print_pose(vector:list, name=''):
    s2p = vector[:]
    s2p[2] *= 180/math.pi
    print(name + '{}'.format(["%0.2f" % s for s in s2p]))