    

    
if __name__ == '__main__':
    from spinup.utils.test_policy import load_policy_and_env, run_policy
    from specific.trainer import Trainer
    from specific.customEnv import RevoltSimple, RevoltLimited
    import matplotlib.pyplot as plt
    import numpy as np

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath',            type = str,           default = '')
    parser.add_argument('--len',            type = int,           default = 1000) # EPISODE LENGTH
    parser.add_argument('--episodes',       type = int,           default = 5)
    parser.add_argument('--itr',            type = int,           default = -1) # this allows for loading models from earlier epochs than the last one!
    parser.add_argument('--norm',           type = bool,          default = False)
    parser.add_argument('--realtime',       type = bool,          default = False) # doesnt really do anything since env.step() overwrites digitwin.setRealTimeMode(), so the sim speed is decided from how fast the cpu can loop the python code
    parser.add_argument('--env',            type = str,           default = 'simple')
    parser.add_argument('--plot',           type = bool,           default = False)
    parser.add_argument('--deterministic',  action='store_true')
    args = parser.parse_args()

    print('Loading environment for testing of algorithms. Beware of normalized states or not!')
    _, get_action = load_policy_and_env(args.fpath, 
                                        args.itr if args.itr >=0 else 'last',
                                        args.deterministic)

    t = Trainer(n_sims=1)
    t.start_simulators()
    if (args.env).lower() == 'simple':
        env = RevoltSimple(t.get_digitwins()[0], testing = True, realtime = args.realtime, norm_env = args.norm, max_ep_len = args.len) # NOTE norm_env must be set according to how the algorithm was trained
    elif (args.env).lower() == 'limited':
        env = RevoltLimited(t.get_digitwins()[0], testing = True, realtime = args.realtime, norm_env = args.norm) # NOTE norm_env must be set according to how the algorithm was trained
    else:
        raise Exception('The chosen env type is not applicable')
    
    data = run_policy(env, get_action, max_ep_len = env.max_ep_len, num_episodes = args.episodes)

    if args.plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
        plt.xlabel('Time [s]')
        ax1.set_ylabel('$\sqrt{ {\~{x}}^2 + {\~{y}}^2 }$ [m]')
        ax2.set_ylabel('$\~{\psi}$ [deg]')
        ax3.set_ylabel('Immediate reward, $R_t$')

        fig, ax4 = plt.subplots()
        ax4.set_ylabel('Error surge')
        ax4.set_xlabel('Error sway')

        step_len = 0.001 if args.realtime else 0.1
        ep_no, ep_len = 0, 0
        for episode in data:
            ep_len = len(episode)
            eucl_dists, headings, rewards, steps = [], [], [], [i*step_len for i in range(ep_len)]
            pos = {'sway': [], 'surge' : []}
            for step in episode:
                state, reward = step
                eucl_dists.append( np.sqrt(state[0]**2 + state[1]**2) )
                pos['sway'].append(state[0])
                pos['surge'].append(state[1])
                headings.append( state[2] * 180 / np.pi)
                rewards.append(reward)
            
            ax1.plot(steps, eucl_dists, label='Run {}'.format(ep_no+1))
            ax1.plot(steps, [0 for _ in range(ep_len)], 'g--',label='Goal')
            ax2.plot(steps, headings, label='Run {}'.format(ep_no+1))
            ax2.plot(steps, [0 for _ in range(ep_len)], 'g--')
            ax3.plot(steps, rewards, label='Run {}'.format(ep_no+1))
            ax3.plot(steps, [1.1968268412042982 for _ in range(ep_len)], 'g--')

            ax4.plot(pos['surge'],pos['sway'],label='Run {}'.format(ep_no+1))
            ep_no += 1

        ax1.grid(True)
        ax1.legend(loc='best').set_draggable(True)
        ax2.grid(True)
        ax2.legend(loc='best').set_draggable(True)
        ax3.grid(True)
        ax3.legend(loc='best').set_draggable(True)
        ax4.grid(True)
        ax4.legend(loc='best').set_draggable(True)
        circle = plt.Circle((0, 0), radius=5, color='grey', fill=False)
        ax4.add_artist(circle)
        ax4.set_xlim((-5, 5))
        ax4.set_ylim((-5, 5))

        plt.show()
