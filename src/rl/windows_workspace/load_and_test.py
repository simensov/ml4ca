    

    
if __name__ == '__main__':
    from spinup.utils.test_policy import load_policy_and_env, run_RL_policy
    from specific.trainer import Trainer
    from specific.customEnv import RevoltSimple, RevoltLimited
    from specific.misc.plotters import plot_policytest_data
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
    parser.add_argument('--norm',           type = bool,          default = False)
    parser.add_argument('--realtime',       type = bool,          default = False) # doesnt really do anything since env.step() overwrites digitwin.setRealTimeMode(), so the sim speed is decided from how fast the cpu can loop the python code
    parser.add_argument('--env',            type = str,           default = 'simple')
    parser.add_argument('--plot',           type = bool,          default = True)
    parser.add_argument('--setpoints',      type = bool,          default = False) # Params for testing set point changes during policy
    parser.add_argument('--deterministic',  action='store_true')
    args = parser.parse_args()

    ''' +++++++++++++++++++++++++++++++ '''
    '''    LOAD POLICY AND START SIM    '''
    ''' +++++++++++++++++++++++++++++++ '''

    print('Loading environment for testing of algorithms. Beware of normalized states or not!')
    _, get_action = load_policy_and_env(args.fpath, args.itr if args.itr >=0 else 'last', args.deterministic)

    t = Trainer(n_sims=1)
    t.start_simulators()
    if (args.env).lower() == 'simple':
        env = RevoltSimple(t.get_digitwins()[0], testing = True, realtime = args.realtime, norm_env = args.norm, max_ep_len = args.len) # NOTE norm_env must be set according to how the algorithm was trained
    elif (args.env).lower() == 'limited':
        env = RevoltLimited(t.get_digitwins()[0], testing = True, realtime = args.realtime, norm_env = args.norm) # NOTE norm_env must be set according to how the algorithm was trained
    else:
        raise Exception('The chosen env type is not applicable')

    ''' +++++++++++++++++++++++++++++++ '''
    '''     RUN POLICY AND PLOT RES     '''
    ''' +++++++++++++++++++++++++++++++ '''
    
    data, ned_data = run_RL_policy(env, get_action, max_ep_len = env.max_ep_len, num_episodes = args.episodes, test_setpoint_changes = False)

    if args.plot:
        plot_policytest_data(args,data)
        
