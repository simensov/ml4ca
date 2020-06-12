import numpy as np
from specific.misc.mathematics import rotation_matrix, wrap_angle

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
        rotation_angle = wrap_angle(self._pos[2]) # the yaw angle of the vehicle in NED frame, used in the rotation matrix
        pos_bod = rotation_matrix(rotation_angle).T.dot(err[0:2,:]) # a 2,1 column vector of body frame errors
        ang = wrap_angle(err[2,0]) # the deviation in yaw between current and desired angle, only wrapped in (-pi,pi)
        self._error_coordinate = [pos_bod[0,0], pos_bod[1,0], ang] 

    def update(self,pos=None,ref=None):
        ''' Use already set values if no arguments are passed '''
        self._pos = pos if pos else self._pos
        self._ref = ref if ref else self._ref
        self.transform()