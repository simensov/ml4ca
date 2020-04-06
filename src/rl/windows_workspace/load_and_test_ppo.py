    

    
if __name__ == '__main__':
    from spinup.utils.test_policy import load_policy_and_env, run_policy
    from specific.trainer import Trainer
    from specific.customEnv import RevoltSimple

    import argparse
    parser = argparse.ArgumentParser()
    fpath = 'data\ppoReVolt\ppoReVolt_s0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str,default=fpath) # remove -- infront if wanting to use enter path from terminal as requirement
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=5)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1) # this allows for loading models from earlier epochs than the last one!
    parser.add_argument('--deterministic', '-d', action='store_true')

    args = parser.parse_args()

    _, get_action = load_policy_and_env(args.fpath, 
                                        args.itr if args.itr >=0 else 'last',
                                        args.deterministic)

    t = Trainer(n_sims=1)
    t.start_simulators()
    env = RevoltSimple(t.get_digitwins()[0], testing = True, realtime = False)
    run_policy(env,get_action,max_ep_len=env.max_ep_len, num_episodes=5)
