import time
import joblib
import os
import os.path as osp
import tensorflow as tf
from spinup.utils.logx import EpochLogger, restore_tf_graph
import numpy as np


def load_policy_and_env(fpath, itr='last', deterministic=False):
    """
    Load a policy from save, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the 
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, a NotImplementedError will be raised
    due to the removal of all torch-implementations in this thesis.
    """

    # determine if tf save or pytorch save
    if any(['tf1_save' in x for x in os.listdir(fpath)]):
        backend = 'tf1'
    else:
        raise Exception('The tf1_save dir was not found')

    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'tf1':
            saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x)>8]

        elif backend == 'pytorch':
            raise NotImplementedError('All torch-implementations has been deleted')
            
        itr = '%d'%max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function
    if backend == 'tf1':
        get_action = load_tf_policy(fpath, itr, deterministic)
    else:
        raise NotImplementedError('All torch-implementations has been deleted')

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def load_tf_policy(fpath, itr, deterministic=False):
    """ Load a tensorflow policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'tf1_save'+itr)
    print('\n\nLoading from %s.\n\n'%fname)

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, fname)

    ### EXPERIMENT WITH TENSORBOARD. AFTER THIS, RUN 'tensorboard --logdir 'dir'
    # with tf.Session() as sess:
    #     writer = tf.summary.FileWriter("graphputputtest", sess.graph)
    #     print(sess.run(model))
    #     writer.close()

    # err = input('Pausing: press enter')
    ### EXPERIMENT

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    return get_action

def run_RL_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True, test_setpoint_changes = False):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    data = [[]] # A list of n_episodes number of lists. Each element in those lists are (observations, rewards)
    dataidx = 0
    
    ned_pos = [[]] # A list of n_episodes number of lists. Each element in those lists are (NED pos, NED ref)
    nedidx = 0

    action_data = [[]]
    actidx = 0

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    
    data[dataidx].append((o,r))
    ned_pos[nedidx].append((env.EF.get_NED_pos(), env.EF.get_NED_ref()))

    action_vec = np.zeros(len(env.default_actions))
    for key in env.default_actions:
        action_vec[key] = env.default_actions[key]

    action_init = np.copy(action_vec)
    action_data[actidx].append(action_vec)

    # Only used if test_setpoint_changes is set to True
    refs = [[5,0,0], [0,-5, 0], [0, 0, np.pi/2], [0, 5, np.pi/2], [-5,0,0] ]
    ref_ctr = 0
    
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        act = np.array(env.scale_and_clip(a))
        for i in range(len(act)):
            action_vec[env.act_2_act_map_inv[i]] = act[i]

        # Allow changes in setpoint changes
        if ep_len == int(max_ep_len / 2) and test_setpoint_changes:
            ref = refs[ref_ctr]; 
            print('Setting ref to {}'.format(ref))
            o, r, d, _ = env.step(np.zeros_like(a), new_ref=ref) # TODO resetting to zeros since this is what has been trained
            action_vec = np.copy(action_init)
            ref_ctr += 1
        else:
            o, r, d, _ = env.step(a)

        ep_ret += r
        ep_len += 1            
        
        data[dataidx].append((o,r))
        ned_pos[dataidx].append((env.EF.get_NED_pos(), env.EF.get_NED_ref()))
        action_data[dataidx].append(np.copy(action_vec))

        if d or (ep_len == max_ep_len):
            
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))

            none_ref = [0,0,0] if test_setpoint_changes else None
            o, r, d, ep_ret, ep_len = env.reset(new_ref=none_ref), 0, False, 0, 0
            n += 1

            if not n == num_episodes:
                data.append([]) # Add a new episode-list
                dataidx += 1 # Update counter to point to that list
                data[dataidx].append((o,r)) # Add initial state to that episode
                ned_pos.append([])
                ned_pos[dataidx].append((env.EF.get_NED_pos(), env.EF.get_NED_ref()))
                action_data.append([])
                action_data[dataidx].append(np.copy(action_init))

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()

    return data, ned_pos, action_data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')

    args = parser.parse_args()
    env, get_action = load_policy_and_env(args.fpath, 
                                          args.itr if args.itr >=0 else 'last',
                                          args.deterministic)
    run_policy(env, get_action, args.len, args.episodes, not(args.norender))