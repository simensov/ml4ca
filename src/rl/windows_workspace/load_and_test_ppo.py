    

    
if __name__ == '__main__':
    from spinup.utils.test_policy import load_policy_and_env, run_policy
    import argparse
    import gym
    
    fpath = 'data\ppo\ppo_s0' # TODO this has to currently be entered from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str,default=fpath)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()

    _, get_action = load_policy_and_env(args.fpath, 
                                          args.itr if args.itr >=0 else 'last',
                                          args.deterministic)

    env = gym.make('CartPole-v1')

    run_policy(env,get_action)
