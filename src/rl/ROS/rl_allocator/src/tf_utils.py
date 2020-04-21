import time
import joblib
import os
import os.path as osp
import tensorflow as tf

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
    tf.saved_model.loader.load(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                fpath
            )
    model_info = joblib.load(osp.join(fpath, 'model_info.pkl'))
    graph = tf.get_default_graph()
    model = dict()
    model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['inputs'].items()})
    model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['outputs'].items()})
    return model

def load_tf_policy(fpath, itr, deterministic=False):
    """ Load a tensorflow policy """
    fname = osp.join(fpath, 'tf1_save'+itr)
    print('\n\nLoading from %s.\n\n'%fname)
    sess = tf.Session()
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