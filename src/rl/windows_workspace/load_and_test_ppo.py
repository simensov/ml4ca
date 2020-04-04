    

    
if __name__ == '__main__':
    from spinup.utils.test_policy import load_policy_and_env, run_policy
    from specific.trainer import Trainer
    from specific.customEnv import RevoltSimple
    
    from config import GLOBAL_TEST_ARGS, PPO_ARGS

    _, get_action = load_policy_and_env(GLOBAL_TEST_ARGS.fpath, 
                                        GLOBAL_TEST_ARGS.itr if GLOBAL_TEST_ARGS.itr >=0 else 'last',
                                        GLOBAL_TEST_ARGS.deterministic)

    t = Trainer(n_sims=1)
    t.start_simulators()
    env = RevoltSimple(t.get_digitwins()[0],testing=True)
    run_policy(env,get_action,max_ep_len=PPO_ARGS.steps,num_episodes=10)
