    

    
if __name__ == '__main__':
    from spinup.utils.test_policy import load_policy_and_env, run_policy
    from specific.trainer import Trainer
    from specific.customEnv import RevoltSimple

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str,default='')
    parser.add_argument('--len', '-l', type=int, default=0) # EPISODE LENGTH
    parser.add_argument('--episodes', '-n', type=int, default=10)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1) # this allows for loading models from earlier epochs than the last one!
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--norm', type=bool, default=False)

    args = parser.parse_args()

    print('Loading environment for testing of algorithms. Beware of normalized states or not!')
    _, get_action = load_policy_and_env(args.fpath, 
                                        args.itr if args.itr >=0 else 'last',
                                        args.deterministic)

    t = Trainer(n_sims=1)
    t.start_simulators()
    env = RevoltSimple(t.get_digitwins()[0], testing = True, realtime = False, norm_env = args.norm) # NOTE norm_env must be set according to how the algorithm was trained
    run_policy(env,get_action,max_ep_len=env.max_ep_len, num_episodes=5)
