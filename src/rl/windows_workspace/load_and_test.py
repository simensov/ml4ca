    
if __name__ == '__main__':
    from spinup.utils.test_policy import load_policy_and_env, run_RL_policy
    from specific.trainer import Trainer
    from specific.customEnv import RevoltSimple, RevoltLimited
    from specific.misc.plotters import plot_policytest_data, plot_NED_data, plot_action_data
    import matplotlib.pyplot as plt
    import numpy as np

    ''' +++++++++++++++++++++++++++++++ '''
    '''       SETUP ALL ARGUMENTS       '''
    ''' +++++++++++++++++++++++++++++++ '''

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath',            type = str,           default = '')
    parser.add_argument('--len',            type = int,           default = 800) # EPISODE LENGTH
    parser.add_argument('--episodes',       type = int,           default = 5)
    parser.add_argument('--itr',            type = int,           default = -1) # this allows for loading models from earlier epochs than the last one!
    parser.add_argument('--realtime',       type = bool,          default = False) # doesnt really do anything since env.step() overwrites digitwin.setRealTimeMode(), so the sim speed is decided from how fast the cpu can loop the python code
    parser.add_argument('--env',            type = str,           default = 'simple')
    parser.add_argument('--sim',            type = int,           default = 0)
    parser.add_argument('--ext',            type = bool,          default = False)
    parser.add_argument('--plot',           type = bool,          default = True)
    parser.add_argument('--setpoints',      type = bool,          default = False) # Params for testing set point changes during policy
    parser.add_argument('--deterministic',  action='store_true')
    args = parser.parse_args()

    ''' +++++++++++++++++++++++++++++++ '''
    '''    LOAD POLICY AND START SIM    '''
    ''' +++++++++++++++++++++++++++++++ '''

    print('Loading environment for testing of algorithms')
    _, get_action = load_policy_and_env(args.fpath, args.itr if args.itr >=0 else 'last', args.deterministic) # Collect policy from saved tf_graph

    t = Trainer(n_sims = 1, start = True, testing = True, simulator_no = args.sim, 
                env_type = args.env, realtime = args.realtime, extended_state = args.ext)
    env = t.get_environments()[0]

    ''' +++++++++++++++++++++++++++++++ '''
    '''     RUN POLICY AND PLOT RES     '''
    ''' +++++++++++++++++++++++++++++++ '''
    
    data, ned_data, action_data = run_RL_policy(env, get_action, max_ep_len = env.max_ep_len, num_episodes = args.episodes, test_setpoint_changes = False)

    if args.plot:
        plot_policytest_data(args,data)
        plot_NED_data(args,ned_data)
        plot_action_data(args,action_data,env)
        plt.show()
