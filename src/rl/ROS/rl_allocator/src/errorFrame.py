import numpy as np

def rotation_matrix(a):
    ''' a is an angle in radians'''
    return np.array([ [np.cos(a), -np.sin(a)],
                      [np.sin(a),  np.cos(a)]]) 

def wrap_angle(angle, deg = False):
    ''' Wrap angle between -180 and 180 deg. deg == True means degrees, == False means radians
        Handles if angle is a vector or list of angles. In both cases, a (x,) shaped numpy array is returned '''
    ref = 180.0 if deg else np.pi
    if isinstance(angle,np.array): # handle arrays
        ref = np.ones_like(angle) * ref
    elif isinstance(angle,list):
        angle = np.array(angle)
        ref = np.ones_like(angle) * ref

    return np.mod(angle + ref, 2*ref) - ref

class ErrorFrame(object):
    ''' Stores information about coordinates in a body error frame of a 3 DOF surface vessel '''

    def __init__(self,pos=[0,0,0],ref=[0,0,0]):
        '''
        :params:
            - pos   list    [north,east,heading] of the vessel
            - ref   list    [north,east,heading] of the reference point (TODO could be set point to avoid dependency on reference model)
        '''
        self.update(pos,ref)

    def get_pose(self,new_pose=None):
        if new_pose: self.update(new_pose)
        return self._error_coordinate

    def get_NED_pos(self):
        return self._pos
    
    def get_NED_ref(self):
        return self._ref

    def transform(self,pos=None):
        if pos: self.update(pos=pos)

        err = np.array([ [a - b for a, b in zip(self._pos,self._ref)] ]).T # full shape column vector of errors
        ang = wrap_angle(err[2,0]) # this is the same as smallest signed angle, see MSS Toolbox, ssa.m, by T.I.Fossen
        pos_bod = rotation_matrix(ang).T.dot(err[0:2,:]) # a 2,1 column vector
        self._error_coordinate = [pos_bod[0,0], pos_bod[1,0], ang] 

    def update(self,pos=None,ref=None):
        ''' Use already set values if no arguments are passed '''
        self._pos = pos if pos else self._pos
        self._ref = ref if ref else self._ref
        self.transform()