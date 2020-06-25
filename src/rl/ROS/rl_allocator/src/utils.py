#!/usr/bin/python3

import time
import joblib
import os
import os.path as osp
import tensorflow as tFlow
import numpy as np 
from custom_msgs.msg import podAngle, SternThrusterSetpoints, bowControl
import rospy
import time

'''
### tensorflow specific
'''
def restore_tf_graph(sess, fpath):
    """
    Loads graphs saved by Logger.
    Will output a dictionary whose keys and values are from the 'inputs' 
    and 'outputs' dict you specified with logger.setup_tf_saver().
    Args:
        sess: A Tensorflow session.
        fpath: Filepath to save directory.
    Returns:
        A dictionary mapping from keys to tensors in the computation graph
        loaded from ``fpath``. 
    """
    tFlow.saved_model.loader.load(
                sess,
                [tFlow.saved_model.tag_constants.SERVING],
                fpath
            )
    model_info = joblib.load(osp.join(fpath, 'model_info.pkl'))
    graph = tFlow.get_default_graph()
    model = dict()
    model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['inputs'].items()})
    model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['outputs'].items()})
    return model

def load_tf_policy(fpath, itr, deterministic=False,n_hidden=None):
    """ Load a tensorflow policy """
    fname = osp.join(fpath, 'tf1_save'+itr)
    print('Loading policy from %s'%fname)
    sess = tFlow.Session()
    model = restore_tf_graph(sess, fname)

    # Get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using stochastic action op.')
        action_op = model['pi'] # loads entire stochastic policy, with noise
        if n_hidden:
            action_op = action_op.graph.get_tensor_by_name('pi/dense_{}/BiasAdd:0'.format(n_hidden)) # This extracts the mean components of the gaussian policy - like setting all noise to zero!

    # Return function for producing an action given a single state
    return lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

def load_policy(fpath, itr='last', deterministic=False, num_hidden_layers=None):
    """
    Load a policy from save.
    Not exceptionally future-proof, but it will suffice for basic uses of the 
    Spinning Up implementations.
    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, a NotImplementedError will be raised
    due to the removal of all torch-implementations in this thesis.
    """

    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value
        saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x)>8]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function
    get_action = load_tf_policy(fpath, itr, deterministic, n_hidden=num_hidden_layers)
    return get_action

'''
### Various helpful messages
'''
def create_publishable_messages(u, simulation = True):
    ''' Take in an array of actions and create publishable ROS messages for all thrusters.
    :params:
        - u (ndarray): a (6,) shaped array containing [n1,n2,n3,a1,a2,a3] in percentages and degrees
    '''
    thruster_percentages = u[:3]
    thruster_angles = u[3:]

    # Set messages as angles in degrees, and thruster inputs in RPM percentages
    pod_angle = podAngle()        
    pod_angle.port = float(np.rad2deg(thruster_angles[0]))
    pod_angle.star = float(np.rad2deg(thruster_angles[1]))

    stern_thruster_setpoints = SternThrusterSetpoints()
    stern_thruster_setpoints.port_effort = float(thruster_percentages[0])
    stern_thruster_setpoints.star_effort = float(thruster_percentages[1])
    
    bow_control = bowControl()
    bow_control.lin_act_bow = 2
    
    if simulation:
        bow_control.position_bow = int(np.rad2deg(thruster_angles[2]))
        bow_control.throttle_bow = float(thruster_percentages[2])
    else:
        bow_control.position_bow = int(45) # constant at 45 percent of 270 degrees (empirically found value during testing, 2. june 2020)
        bow_control.throttle_bow = np.clip( float(thruster_percentages[2]) * 2.5, -100.0, 100.0) # Empirically found value for increasing sensitivity of bow thruster thrust due to no response at low inputs. Clipped here to avoid too large values

    return pod_angle, stern_thruster_setpoints, bow_control

def shutdown_handler():
    ''' Function that publishes zero output to the thrusters if the node is shutdown before DP control mode has been deactivated '''
    rospy.loginfo('RL_allocator received signal to shutdown - publish 0.0 to thrusters for 5 seconds before closing')
    pub_stern_thruster_setpoints = rospy.Publisher("thrusterAllocation/stern_thruster_setpoints", SternThrusterSetpoints, queue_size=1)
    stern_thruster_setpoints = SternThrusterSetpoints(port_effort = float(0.0), star_effort = float(0.0))
    t = time.time()
    while (time.time() - t < 5.0): 
        pub_stern_thruster_setpoints.publish(stern_thruster_setpoints)    
