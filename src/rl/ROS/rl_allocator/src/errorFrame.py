#!/usr/bin/python3

import numpy as np
import time

def rotation_matrix(a):
    ''' a is an angle in radians - rotates FROM body coordinates to NED coordinates by dot product:
        NED_coord = R dot BODY_coord
        Remember: R^(-1) == R^T
    '''
    return np.array([ [np.cos(a), -np.sin(a)],
                      [np.sin(a),  np.cos(a)]]) 

def wrap_angle(angle, deg = False):
    ''' Wrap angle between -180 and 180 deg. deg == True means degrees, == False means radians
        Handles if angle is a vector or list of angles. In both cases, a (x,) shaped numpy array is returned '''
    ref = 180.0 if deg else np.pi

    if isinstance(angle,np.ndarray): # handle arrays
        ref = np.ones_like(angle) * ref
    elif isinstance(angle,list):
        angle = np.array(angle)
        ref = np.ones_like(angle) * ref

    return np.mod(angle + ref, 2*ref) - ref

class ErrorFrame(object):
    ''' Stores information about coordinates in a body error frame of a 3 DOF surface vessel '''

    def __init__(self,pos=[0,0,0],ref=[0,0,0],use_integral_effect = False):
        '''
        :params:
            - pos   list    [north,east,heading] of the vessel
            - ref   list    [north,east,heading] of the reference point (TODO could be set point to avoid dependency on reference model)
        '''
        self._use_integral_effect = use_integral_effect
        self._integrator = np.zeros(3)
        self._time_arrival = 0.0
    
        self.update(pos,ref)

    def get_pose(self,new_pose = None):
        if new_pose: self.update(new_pose)
        return self._error_coordinate

    def get_NED_pos(self):
        return self._pos
    
    def get_NED_ref(self):
        return self._ref

    def transform(self,pos=None):
        if pos: self.update(pos=pos)

        err = np.array([ [a - b for a, b in zip(self._pos,self._ref)] ]).T # full shape column vector of errors
        rotation_angle = wrap_angle(self._pos[2]) # this is the same as smallest signed angle, see MSS Toolbox, ssa.m, by T.I.Fossen
        pos_bod = rotation_matrix(rotation_angle).T.dot(err[0:2,:]) # a 2,1 column vector of the body frame errors in [surge,sway], using the NED angle for coordinate shift
        self._error_coordinate = [pos_bod[0,0], pos_bod[1,0], wrap_angle(err[2,0])] # body frame errors [surge,sway,yaw]

    def update(self,pos=None,ref=None):
        ''' Use already set values if no arguments are passed '''
        self._pos = pos if pos else self._pos
        self._ref = ref if ref else self._ref

        if self._use_integral_effect:
            err = np.array([a - b for a, b in zip(self._pos, self._ref)])
            if (np.abs(err[0]) > 3.0 or np.abs(err[1]) > 3.0 or np.abs(err[2]) > np.deg2rad(25)):
                # don't use integral effect when far away
                self._integrator = np.zeros(3)
                self._time_arrival = time.time()
            else:
                # use integral effect with windup guard of 0.5 meters and 0.01 rad = 0.5 degrees
                if (time.time() - self._time_arrival) > 10.0:
                    self._integrator = self._integrator + np.array([0.0002,0.002,0.0]) * err
                    self._integrator = np.clip(self._integrator,-np.array([0.5,0.5,0.01]),np.array([0.5,0.5,0.01]))
                
            self._pos = self._pos + self._integrator
            
            print(self._integrator)        

        self.transform()