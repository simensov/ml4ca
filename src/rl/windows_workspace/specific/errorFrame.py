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
        return

    def get_pose(self,new_pose=None):
        if new_pose: self.update(new_pose)
        return self._error_coordinate

    def transform(self,pos=None):
        if pos: self.update(pos=pos)

        # TODO clean this
        err = [a - b for a, b in zip(self._pos,self._ref)]
        ang = wrap_angle(err[2],deg=False) # this is the same as smallest signed angle, see MSS Toolbox, ssa.m, by T.I.Fossen
        pos_bod = ((rotation_matrix(ang).T).dot(np.array(err[0:2]).reshape(2,1))).reshape(2,).tolist() # a 2,1 vector

        self._error_coordinate = [pos_bod[0], pos_bod[1], ang]
        return 

    def update(self,pos=None,ref=None):
        ''' Use already set values if no arguments are passed '''
        self._pos = pos if pos else self._pos
        self._ref = ref if ref else self._ref
        self.transform()
        return