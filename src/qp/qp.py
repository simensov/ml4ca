#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Tests with QP inspired by
https://github.com/stephane-caron/qpsolvers/blob/master/qpsolvers/quadprog_.py
https://scaron.info/blog/quadratic-programming-in-python.html
'''

import quadprog
import numpy as np
np.set_printoptions(precision=3)
from scipy.optimize import minimize
from quadprog import solve_qp
import time

'''
Test scipy minimize
'''

ly = [-0.15, 0.15, 0]
lx = [-1.12, -1.12, 1.08]

def B(a):
    '''
    Returns the effectiveness matrix B in tau = B*u, full shape numpy array (e.g. (3,3))

    :params:
        a - A (3,1) np.array of azimuth angles in radians
    '''

    a = a.reshape(3,)
    return np.array([[np.cos(a[0]), 									np.cos(a[1]), 										np.cos(a[2])],
                     [np.sin(a[0]), 									np.sin(a[1]), 										np.sin(a[2])],
                     [lx[0]*np.sin(a[0]) - ly[0]*np.cos(a[0]), 	lx[1]*np.sin(a[1]) - ly[1]*np.cos(a[1]), 	lx[2]*np.sin(a[2]) - ly[2]*np.cos(a[2])]
                     ])

if __name__=='__main__':

    ly = [-0.15, 0.15, 0]
    lx = [-1.12, -1.12, 1.08]
    TAU = np.array([[0.0, 0.0, 0.0]]).T
    
    s_t = [0,0,0,0,0] # [f1 f2 f3 a1 a2] - previous state
    
    def objective(x):
        # Objective to minimize (is quadratic) - does not constrain the angles, but it is possible to constraint the size depending on the previous angle in s_t
        # Decision variables for the optimization: x = [f1 f2 f3 a1 a2 s1 s2 s3]
        FandS = np.hstack( (x[0:3], x[5:]) )
        return (FandS.T).dot(FandS)
        #return (x.T).dot(x) # This will also minimize the angles themselves, but it is the ANGULAR RATES that should be minimized over time for wear and tear

    ### PHYSICAL CONSTRAINTS DEPENDENT ON THE THRUSTER SETUP: B(alpha)*F = tau
    # Since these are equality constrains, a slack variable is added and the goal is to minimize s^2 - added in the objective functions
    # The constraints are written as ||| B(alpha)*F - tau - s = 0 |||, so that if s is minimized, the produced forces are as close as possible to tau
    # scipy.optimize.minimize only takes scalar functions, so the each row has to be written seperately
    def c1(x):  return np.cos(x[3]) * x[0] + np.cos(x[4]) * x[1] + np.cos(np.pi/2) * x[2] - x[5] - float(TAU[0,0]) # top row
    def c2(x):  return np.sin(x[3]) * x[0] + np.sin(x[4]) * x[1] + np.sin(np.pi/2) * x[2] - x[6] - float(TAU[1,0]) # mid row
    def c3(x):  return (lx[0]*np.sin(x[3]) - ly[0]*np.cos(x[3])) * x[0] + (lx[1]*np.sin(x[4]) - ly[1]*np.cos(x[4])) * x[1] + (lx[2]*np.sin(np.pi/2) - ly[2]*np.cos(np.pi/2))*x[2] - x[7] - float(TAU[2,0]) # bottom row

    ### FORCE RATE CONSTRAINTS
    # Rate constraint on force increase/decrease: 
    # dfMin < dF < dFMax  becomes ||| dFMax - dF >= 0||| AND ||| -dFMin + dF >= 0 ||| where -dFmin is -(-dFMax) 
    # Note that e-3 added makes it possible for the rate to be zero! It just overlaps the other constraint, so it is all good - a good ol' optimization trick
    def c4(x): return 10.0 - (x[0] - s_t[0])
    def c5(x): return 10.0 + (x[0] - s_t[0])
    def c6(x): return 10.0 - (x[1] - s_t[1])
    def c7(x): return 10.0 + (x[1] - s_t[1])
    def c8(x): return 4.0 + (x[2] - s_t[2])
    def c9(x): return 4.0 - (x[2] - s_t[2])
    
    ### ANGULAR RATE CONSTRAINTS
    def c10(x): return np.pi/6 + (x[3] - s_t[3])
    def c11(x): return np.pi/6 - (x[3] - s_t[3])
    def c12(x): return np.pi/6 + (x[4] - s_t[4])
    def c13(x): return np.pi/6 - (x[4] - s_t[4])

    ### TODO the maximum force producable has not been added as a constraint. This is because the PID is constrained to take care of it.    

    # Gather all constrains
    con1  = {'type': 'eq', 'fun': c1}
    con2  = {'type': 'eq', 'fun': c2}
    con3  = {'type': 'eq', 'fun': c3}
    con4  = {'type': 'ineq', 'fun': c4}
    con5  = {'type': 'ineq', 'fun': c5}
    con6  = {'type': 'ineq', 'fun': c6}
    con7  = {'type': 'ineq', 'fun': c7}
    con8  = {'type': 'ineq', 'fun': c8}
    con9  = {'type': 'ineq', 'fun': c9}
    con10 = {'type': 'ineq', 'fun': c10}
    con11 = {'type': 'ineq', 'fun': c11}
    con12 = {'type': 'ineq', 'fun': c12}
    con13 = {'type': 'ineq', 'fun': c13}
        
    cons = ([con1,con2,con3,con4,con5,con6,con7,con8,con9,con10,con11,con12,con13])

    # x = [f1 f2 f3 a1 a2 s1 s2 s3]
    # np.inf is used to disable constrains, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.Bounds.html#scipy.optimize.Bounds
    s_bnd = 1.0
    # Bound decision variables for maximum force
    bnds = ((-25,25),(-25,25),(-6.1,14),(-2*np.pi,2*np.pi),(-2*np.pi,2*np.pi),(-s_bnd,s_bnd),(-s_bnd,s_bnd),(-s_bnd,s_bnd)) # TODO these has to be reset since the desired forces may be too large compared to the previous thruster states

    x0 = np.array([s_t[0],s_t[1],s_t[2],s_t[3],s_t[4],0,0,0]) # Initial value - changes with each time instance

    stime = time.time()
    solution = minimize(objective,x0,method='SLSQP',bounds=bnds,constraints=cons)
    x = solution.x

    # Clean solution for very small values
    x[np.where(np.abs(x) < 0.01)] = 0

    # Print forces (in N) and angles (in deg) in addition to the resulting slack variables
    print('F1: {:.3f}\nF2: {:.3f}\nF3: {:.3f}\na1: {:.3f}\na2: {:.3f}\ns1: {:.3f}\ns2: {:.3f}\ns3: {:.3f}\ntime: {:.5f}'.format(x[0],x[1],x[2],np.rad2deg(x[3]),np.rad2deg(x[4]), x[5],x[6],x[7], time.time() - stime ))

    angles = np.array([[x[3], x[4], np.pi/2]]).T
    forces = np.array([[x[0], x[1], x[2]]]).T

    print('Resulting tau: ')
    print(B(angles).dot(forces) )