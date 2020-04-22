#!/usr/bin/python3

import time
import joblib
import os
import os.path as osp
import tensorflow as tFlow

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

def load_tf_policy(fpath, itr, deterministic=False):
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
        print('Using default action op.')
        action_op = model['pi']

    # Return function for producing an action given a single state
    return lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

def load_policy(fpath, itr='last', deterministic=False):
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
    get_action = load_tf_policy(fpath, itr, deterministic)

    return get_action
