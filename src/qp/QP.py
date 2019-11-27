#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Inspired by
https://github.com/stephane-caron/qpsolvers/blob/master/qpsolvers/quadprog_.py
https://scaron.info/blog/quadratic-programming-in-python.html
'''

import quadprog
import numpy as np
from scipy.optimize import minimize
from quadprog import solve_qp
import time

'''
Test scipy minimize
'''

if __name__=='__main__':

    ly = [-0.15, 0.15, 0]
    lx = [-1.12, -1.12, 1.08]
    TAU = np.array([[20.,0.,0.]]).T

    # Decision variables for the optimization
    # x = [f1 f2 f3 a1 a2 s1 s2 s3]

    def objective(x):
        'Just use 1 as weights for now - which is weird since the angles are in rad and forces are in N'
        return (x.T).dot(x)

    def c1(x):
        # eq c of top row of B*u - tau - s. Constraint is set to equal zero
        return np.cos(x[3]) * x[0] + np.cos(x[4]) * x[1] + np.cos(np.pi/2) * x[2] - x[5] - float(TAU[0,0])

    con1 = {'type': 'eq', 'fun': c1}    
        
    def c2(x):
        # eq c of mid row of B*u - tau - s. Constraint is set to equal zero
        return np.sin(x[3]) * x[0] + np.sin(x[4]) * x[1] + np.sin(np.pi/2) * x[2] - x[6] - float(TAU[1,0])

    con2 = {'type': 'eq', 'fun': c2}

    def c3(x):
        # eq c of bottom row of B*u - tau - s. Constraint is set to equal zero
        return (lx[0]*np.sin(x[3]) - ly[0]*np.cos(x[3])) * x[0] + (lx[1]*np.sin(x[4]) - ly[1]*np.cos(x[4])) * x[1] + (lx[2]*np.sin(np.pi/2) - ly[2]*np.cos(np.pi/2))*x[2] - x[7] - float(TAU[2,0])

    con3 = {'type': 'eq', 'fun': c3}

    # x = [f1 f2 f3 a1 a2 s1 s2 s3]
    # np.inf is used to disable constrains
    bnds = ((-25,25),(-25,25),(-6.1,14),(-np.pi,np.pi),(-np.pi,np.pi),(-1,1),(-1,1),(-1,1)) # TODO these has to be reset since the desired forces may be too large compared to the previous thruster states

    # Gather all constrains
    cons = ([con1,con2,con3])

    x0 = np.array([0,0,0,0,0,0,0,0]) # Initial value - change 

    stime = time.time()
    solution = minimize(objective,x0,method='SLSQP',bounds=bnds,constraints=cons)
    x = solution.x

    # Clean solution for very small values
    x[np.where(np.abs(x) < 0.01)] = 0

    # Print forces (in N) and angles (in deg) in addition to the resulting slack variables
    print('F1: {}\nF2: {}\nF3: {}\na1: {}\na2: {}\ns1: {}\ns2: {}\ns3: {}\ntime: {}'.format(x[0],x[1],x[2],np.rad2deg(x[3]),np.rad2deg(x[4]), x[5],x[6],x[7], time.time() - stime ))

'''
def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None, initvals=None):
    """
    Solve a Quadratic Program defined as:
        minimize
            (1/2) * x.T * P * x + q.T * x
        subject to
            G * x <= h
            A * x == b
    using quadprog <https://pypi.python.org/pypi/quadprog/>.
    Parameters
    ----------
    P : numpy.array
        Symmetric quadratic-cost matrix.
    q : numpy.array
        Quadratic-cost vector.
    G : numpy.array
        Linear inequality constraint matrix.
    h : numpy.array
        Linear inequality constraint vector.
    A : numpy.array, optional
        Linear equality constraint matrix.
    b : numpy.array, optional
        Linear equality constraint vector.
    initvals : numpy.array, optional
        Warm-start guess vector (not used).
    Returns
    -------
    x : numpy.array.
        Solution to the QP, if found, otherwise ``None``.
    Note
    ----
    The quadprog solver only considers the lower entries of `P`, therefore it
    will use a wrong cost function if a non-symmetric matrix is provided.
    """
    if initvals is not None:
        print("quadprog: note that warm-start values ignored by wrapper")
    qp_G = P
    qp_a = -q
    if A is not None:
        if G is None:
            qp_C = -A.T
            qp_b = -b
        else:
            qp_C = -vstack([A, G]).T
            qp_b = -hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T if G is not None else None
        qp_b = -h if h is not None else None
        meq = 0
    return solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]



    # MAIN
    M = array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
    P = dot(M.T, M)
    q = -dot(M.T, array([3., 2., 3.]))
    G = array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
    h = array([3., 2., -2.]).reshape((3,))


    s_time = time.time()

    for i in range(200):
        quadprog_solve_qp(P,)


    print(time.time() - s_time)

'''