import os
import copy
import pickle
import itertools
import time
import numpy as np
from argparse import ArgumentParser
import torch
import gym
from gym_minigrid.wrappers import *

from train_minigrid import evaluate_agent_bandits, evaluate_agent_oracle
from dqn import DQNAgentUnclear, DQNAgent
from evaluate_agent import evaluate_agent_rnd, evaluate_agent_max_q

from utils import load_checkpoint_minigrid, make_gif_minigrid_bandit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--episode_max_steps', type=int, default=100)
    parser.add_argument('--greedy_bandit', dest='greedy_bandit', default=False, action='store_true', help='Whether to use the greedy bandit.')
    parser.add_argument('--bandit_loss', dest='bandit_loss', type=str, default='mse', help='Bandit loss for Exp weights \in {nll, mse, hack}.')
    parser.add_argument('--bandit_lr', dest='bandit_lr', type=float, default=0.88, help='Bandit learning rate.')
    parser.add_argument('--bandit_decay', dest='bandit_decay', type=float, default=0.9, help='Bandit decay.')
    parser.add_argument('--bandit_step', dest='bandit_step', type=int, default=1, help='Number of steps taken between bandit arm pulls.')
    parser.add_argument('--bandit_epsilon', dest='bandit_epsilon', type=float, default=0.0, help='Eps greedy exploration in the ExpWeights bandit alg.')
    parser.add_argument('--grid_search', dest='grid_search', default=False, action='store_true', help='Whether to perorm grid search.')
    parser.add_argument('--id', nargs='+', dest='id', type=int, default=None, help='Whether to perform generalization exps on a specified seeds for parallelization.')
    parser.add_argument('--unclear', dest='unclear', default=False, action='store_true', help='Whether to use unclear/clear.')
    parser.add_argument('--no_state_bonus', dest='no_state_bonus', default=False, action='store_true', help='Whether to use StateBonus for unclear/clear.')
    parser.add_argument('--baseline_rnd', dest='baseline_rnd', default=False, action='store_true',
                        help='Whether to run the baselines.')
    parser.add_argument('--baseline_max', dest='baseline_max', default=False, action='store_true',
                        help='Whether to run the baselines.')
    parser.add_argument('--bandit', dest='bandit', default=False, action='store_true', help='Whether to use bandit for unclear.')
    parser.add_argument('--n_episodes', dest='n_episodes', type=int, default=16, help='Number of episodes to evaluate on.')
    parser.add_argument('--num_tasks', dest='num_tasks', type=int, default=100, help='Number of levels to evaluate on.')
    parser.add_argument('--num_base_tasks', dest='num_base_tasks', type=int, default=5, help='Number of levels trained on.')
    parser.add_argument('--env_seeds', nargs='+', dest='env_seeds', type=int, default=[100], help='The env seeds for env in {SimpleCrossing}. Needs to be same length as the number of tasks.')
    parser.add_argument('--env', dest='env', default='SC', type=str, help='The MiniGrid env')
    parser.add_argument('--make_gif', dest='make_gif', default=False, action='store_true', help='Whether to make a gif.')
    parser.add_argument('--tag', type=str, default='', help='unique str to tag tb.')

    args = parser.parse_args()
    params = vars(args)

    seed = params['seed']
    envs = []

    print(params)

    # Creating the environments
    if args.env == 'SC':
        env_str = ['MiniGrid-SimpleCrossingS9N1-v0'] * args.num_base_tasks
    elif args.env == "SC+DK":
        env_str = ['MiniGrid-SimpleCrossingS9N1-v0'] * args.num_base_tasks
        env_str[1] = 'MiniGrid-DoorKey-8x8-v0'
        env_str[3] = 'MiniGrid-DoorKey-8x8-v0'
    else:
        raise ValueError

    action_dims = []
    for i in range(args.num_base_tasks):
        env = gym.make(env_str[i])

        env = StateBonus(env)

        env = ImgObsWrapper(env)  # Get rid of the 'mission' field
        env = ReseedWrapper(env, [args.env_seeds[i]])  # Setting the seed for the e.g. FourRooms keeps the doors fixed.
        state_dim = env.observation_space.shape
        print("Max number of steps per episode: {}".format(env.max_steps))
        if args.env == 'SC':
            action_dim = 3
        else:
            action_dim = env.action_space.n  # https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/minigrid.py 0,1,2 left, right forward
        print("Size of the action space: {}".format(action_dim))
        action_dims.append(action_dim)
        envs.append(env)
    action_dim = max(action_dims)
    # Setup agent
    n_arms = args.num_base_tasks
    q_ewc_reg = 500
    exp_replay_capacity = 1000000
    episode_max_steps = params['episode_max_steps']
    n_episodes = params['n_episodes']
    if params['unclear']:
        agent = DQNAgentUnclear(
            seed, state_dim, action_dim, num_tasks=n_arms, lr=0.0000625, gamma=0.99,
            batchsize=32, hidden_size=200, q_ewc_reg=q_ewc_reg, bn=False, ddqn=True,
            duelling=True, pool_size=exp_replay_capacity, target_update=80,
            eps_decay=250000, huber=True, big_head=False, eps_gr_warm_start=False,
            dropout_prob=0.0, q_func_reg=0.0, uncert=False,
        )
    else:
        agent = DQNAgent(seed, state_dim, action_dim, num_tasks=1, lr=0.0000625, gamma=0.99, batchsize=32,
                         hidden_size=200, q_ewc_reg=0, bn=False, ddqn=True, duelling=True,
                         pool_size=1e6, target_update=80,
                         eps_decay=250000, huber=True, big_head=False, eps_gr_warm_start=False,
                         dropout_prob=0.0, q_func_reg=0.0, uncert=False)

    if params['unclear']:
        if args.env == 'SC':
            checkpoints = [
                {'ckpt_step_count': 11250006, 'ckpt_tag': 'unclear_t5_l500_ws_lr088_d090_eps_step_s101_155'},
                {'ckpt_step_count': 11250010, 'ckpt_tag': 'unclear_t5_l500_ws_lr088_d090_eps_step_s102_155'},
                {'ckpt_step_count': 11250009, 'ckpt_tag': 'unclear_t5_l500_ws_lr088_d090_eps_step_s103_155'},
                {'ckpt_step_count': 11250000, 'ckpt_tag': 'unclear_t5_l500_ws_lr088_d090_eps_step_s104_155'},
                {'ckpt_step_count': 11250013, 'ckpt_tag': 'unclear_t5_l500_ws_lr088_d090_eps_step_s105_155'},
                {'ckpt_step_count': 11250013, 'ckpt_tag': 'unclear_t5_l500_ws_lr088_d090_eps_step_s106_155'},
                {'ckpt_step_count': 11250011, 'ckpt_tag': 'unclear_t5_l500_ws_lr088_d090_eps_step_s107_155'},
                # {'ckpt_step_count': 8999999, 'ckpt_tag': 'unclear_new_t3_l500_ws50k_lr088_decay090_eps0_step1_s108'},
                {'ckpt_step_count': 11250002, 'ckpt_tag': 'unclear_t5_l500_ws_lr088_d090_eps_step_s109_155'},
                {'ckpt_step_count': 11250005, 'ckpt_tag': 'unclear_t5_l500_ws_lr088_d090_eps_step_s110_155'},
            ]

        elif args.env == "SC+DK":
            checkpoints= [
                {'ckpt_step_count': 11250015, 'ckpt_tag': 'unclear_sc+dk_t5_l500_ws_lr088_d090_eps_step_s105'},
                {'ckpt_step_count': 11250008, 'ckpt_tag': 'unclear_sc+dk_t5_l500_ws_lr088_d090_eps_step_s104'},
                {'ckpt_step_count': 11250007, 'ckpt_tag': 'unclear_sc+dk_t5_l500_ws_lr088_d090_eps_step_s101'},
                {'ckpt_step_count': 11250001, 'ckpt_tag': 'unclear_sc+dk_t5_l500_ws_lr088_d090_eps_step_s102'},
                {'ckpt_step_count': 11250001, 'ckpt_tag': 'unclear_sc+dk_t5_l500_ws_lr088_d090_eps_step_s103'},
                {'ckpt_step_count': 11250005, 'ckpt_tag': 'unclear_sc+dk_t5_l500_ws_lr088_d090_eps_step_s106'},
                {'ckpt_step_count': 11250009, 'ckpt_tag': 'unclear_sc+dk_t5_l500_ws_lr088_d090_eps_step_s107'},
                {'ckpt_step_count': 11250006, 'ckpt_tag': 'unclear_sc+dk_t5_l500_ws_lr088_d090_eps_step_s109'},
                {'ckpt_step_count': 11250005, 'ckpt_tag': 'unclear_sc+dk_t5_l500_ws_lr088_d090_eps_step_s108'},
                {'ckpt_step_count': 11250008, 'ckpt_tag': 'unclear_sc+dk_t5_l500_ws_lr088_d090_eps_step_s110'},
            ]
    else:
        if args.env == 'SC':
            checkpoints = [
                {'ckpt_step_count': 11250037, 'ckpt_tag': 'dqn_x5_s101_b4m'},
                {'ckpt_step_count': 11250003, 'ckpt_tag': 'dqn_x5_s102_b4m'},
                {'ckpt_step_count': 11250097, 'ckpt_tag': 'dqn_x5_s103_b4m'},
                {'ckpt_step_count': 11250057, 'ckpt_tag': 'dqn_x5_s104_b4m'},
                {'ckpt_step_count': 11250007, 'ckpt_tag': 'dqn_x5_s105_b4m'},
                {'ckpt_step_count': 11250064, 'ckpt_tag': 'dqn_x5_s106_b4m'},
                {'ckpt_step_count': 11250008, 'ckpt_tag': 'dqn_x5_s107_b4m'},
                {'ckpt_step_count': 11250001, 'ckpt_tag': 'dqn_x5_s108_b4m'},
                {'ckpt_step_count': 11250095, 'ckpt_tag': 'dqn_x5_s109_b4m'},
                {'ckpt_step_count': 11250075, 'ckpt_tag': 'dqn_x5_s110_b4m'},
            ]
        elif args.env == "SC+DK":
            checkpoints = [
                {'ckpt_step_count': 11250009, 'ckpt_tag': 'dqn_x5_sc+dk_b4m_s101'},
                {'ckpt_step_count': 11250066, 'ckpt_tag': 'dqn_x5_sc+dk_b4m_s102'},
                {'ckpt_step_count': 11250019, 'ckpt_tag': 'dqn_x5_sc+dk_b4m_s103'},
                {'ckpt_step_count': 11250007, 'ckpt_tag': 'dqn_x5_sc+dk_b4m_s104'},
                {'ckpt_step_count': 11250083, 'ckpt_tag': 'dqn_x5_sc+dk_b4m_s105'},
                {'ckpt_step_count': 11250036, 'ckpt_tag': 'dqn_x5_sc+dk_b4m_s106'},
                {'ckpt_step_count': 11250026, 'ckpt_tag': 'dqn_x5_sc+dk_b4m_s107'},
                {'ckpt_step_count': 11250003, 'ckpt_tag': 'dqn_x5_sc+dk_b4m_s108'},
                {'ckpt_step_count': 11250007, 'ckpt_tag': 'dqn_x5_sc+dk_b4m_s109'},
                {'ckpt_step_count': 11250002, 'ckpt_tag': 'dqn_x5_sc+dk_b4m_s110'},
            ]

    if params['id'] is not None:
        checkpoints = [checkpoints[i] for i in params['id']]

    # Bandit params
    bandit_loss = params['bandit_loss']
    greedy_bandit = params['greedy_bandit']
    bandit_lr = params['bandit_lr']
    bandit_decay = params['bandit_decay']
    bandit_epsilon = params['bandit_epsilon']
    bandit_step = params['bandit_step']
    grid_search = params['grid_search']
    tag = params['tag']
    bandit_learning_rates = list(np.linspace(0.60, 0.98, 20))
    bandit_decays = list(np.linspace(0.60, 0.98, 20))
    bandit_epsilons = [0, 0.1]
    bandit_steps = [1]

    hparams = list(itertools.product(*[bandit_learning_rates, bandit_decays, bandit_epsilons, bandit_steps]))

    # Unseen envs
    num_envs = 5 # 4 SC envs and 4 Rooms
    if grid_search:
        num_tasks = 20
        results = {'_'.join([str(item) for item in h]): np.zeros((len(checkpoints), num_envs, num_tasks)) for h in hparams}
        env_seeds = list(range(num_tasks))

    else:
        num_tasks = params['num_tasks']
        results, fpe = np.zeros((len(checkpoints), num_envs, num_tasks)), np.zeros((len(checkpoints), num_envs, num_tasks))
        results_rnd, fpe_rnd = np.zeros((len(checkpoints), num_envs, num_tasks)), np.zeros((len(checkpoints), num_envs, num_tasks))
        results_max_q, fpe_max_q = np.zeros((len(checkpoints), num_envs, num_tasks)), np.zeros((len(checkpoints), num_envs, num_tasks))
        results_max_q_a, fpe_max_q_a = np.zeros((len(checkpoints), num_envs, num_tasks)), np.zeros(
            (len(checkpoints), num_envs, num_tasks))
        env_seeds = list(range(20, num_tasks+20))

    # envs to evaluate on
    test_envs = []
    sc_difficulties = [1, 2, 3, 5]
    for i, n_walls in enumerate(sc_difficulties):
        if n_walls == 5:
            test_envs.append('MiniGrid-SimpleCrossingS11N{0}-v0'.format(int(n_walls)))
        else:
            test_envs.append('MiniGrid-SimpleCrossingS9N{0}-v0'.format(int(n_walls)))
    test_envs.append('MiniGrid-FourRooms-v0')


    for c, ckpt in enumerate(checkpoints):
        cumulative_timestep, envs, _, _, _, _ = load_checkpoint_minigrid(
            agent, ckpt['ckpt_step_count'], ckpt['ckpt_tag']
        )

        dones_per_episode, _, _ = evaluate_agent_oracle(
            copy.deepcopy(envs), agent, episode_max_steps=100, n_episodes=n_episodes,
            n_tasks=n_arms, dqn=True
        )

        print("Oracle performance (min) ", ' '.join(
            ["Task {0} dones {1}".format(i, np.min(item)) for i, item in dones_per_episode.items()]
        ))

        # Iterate over each env type and each level within the env type
        for i, env_str in enumerate(test_envs):

            envs = []
            env_strs = [env_str] * num_tasks
            for j in range(num_tasks):
                env = gym.make(env_strs[j])
                # Gen. results in the paper have a StateBonus in the new envs
                # The rewards from the StateBonus don't contribute to the numbers of
                # successes - but we might get better results by turning this off and
                # then the envs will resemble more the envs which they have trained on.
                if not params['no_state_bonus']:
                    env = StateBonus(env)
                env = ImgObsWrapper(env)  # Get rid of the 'mission' field
                env = ReseedWrapper(env, [env_seeds[j]])  # Setting the seed for the FourRooms keeps the doors fixed.
                envs.append(env)

            print("\n Eval on {} walls\n".format(env_str))

            if grid_search:

                for j, h in enumerate(hparams):

                    start = time.time()

                    bandit_lr, bandit_decay, bandit_epsilon, bandit_step = h
                    key = '_'.join([str(item) for item in h])

                    dones, _, _, _, _ = evaluate_agent_bandits(
                        copy.deepcopy(envs), agent, episode_max_steps, bandit_loss,
                        greedy_bandit, n_episodes=16, n_arms=n_arms,
                        debug=False, tag=ckpt['ckpt_tag'], step=cumulative_timestep,
                        lr=bandit_lr, decay=bandit_decay, epsilon=bandit_epsilon,
                        bandit_step=bandit_step
                    )

                    results[key][c, i, :] += list(dones.values())

                    print("Bandit performance (min) ", ' '.join(
                        ["Task {0} dones {1}".format(k, np.min(item)) for k, item in dones.items()][:10]
                    ))

                    with np.printoptions(precision=2, suppress=True):
                        print(
                            "Iter {0}, percentage done {1}, time taken {2}".format(
                            j, 100*(j+1)/len(hparams), time.time() - start
                            )
                        )

                # Dump results after each level evaluated
                with open('data/generalization_results_{0}.pickle'.format(tag), 'wb') as handle:
                    pickle.dump({'res': results},
                                handle, protocol=pickle.HIGHEST_PROTOCOL)

            else:

                if params['unclear']:
                    if params['bandit']:
                        dones, _, eval_fpe, _, _ = evaluate_agent_bandits(
                            copy.deepcopy(envs), agent, episode_max_steps, bandit_loss,
                            greedy_bandit, n_episodes=params['n_episodes'], n_arms=n_arms,
                            debug=False, tag=ckpt['ckpt_tag'] + "diff" + str(i), step=cumulative_timestep,
                            lr=bandit_lr, decay=bandit_decay, epsilon=bandit_epsilon,
                            bandit_step=bandit_step
                        )

                        if params['make_gif']:
                            make_gif_minigrid_bandit(copy.deepcopy(envs), agent, episode_max_steps, bandit_loss,
                                                     greedy_bandit, n_episodes=1, n_arms=n_arms,
                                                     tag=ckpt['ckpt_tag'] + "diff" + str(i), step=cumulative_timestep,
                                                     lr=bandit_lr, decay=bandit_decay, epsilon=bandit_epsilon,
                                                     bandit_step=bandit_step)

                        print("Performance (min) ", ' '.join(
                            ["Task {0} dones {1}".format(k, np.min(item)) for k, item in dones.items()][:10] # just show first 10
                        ))

                        results[c, i, :] += list(dones.values())
                        fpe[c, i, :] += np.mean(eval_fpe, 1)

                else:

                    dones, _, eval_fpe = evaluate_agent_oracle(
                        copy.deepcopy(envs), agent, episode_max_steps=episode_max_steps, n_episodes=n_episodes,
                        n_tasks=num_tasks, dqn=True
                    )

                    print("Performance (min) ", ' '.join(
                        ["Task {0} dones {1}".format(k, np.min(item)) for k, item in dones.items()][:10] # just show first 10
                    ))

                    results[c, i, :] += list(dones.values())
                    fpe[c, i, :] += np.mean(eval_fpe, 1)

                if params['baseline_rnd']:
                    dones_rnd, _, eval_fpe_rnd = evaluate_agent_rnd(copy.deepcopy(envs), agent, episode_max_steps,
                                                                    params['n_episodes'], n_arms)
                    results_rnd[c, i, :] += list(dones_rnd.values())
                    fpe_rnd[c, i, :] += np.mean(eval_fpe_rnd, 1)

                if params['baseline_max']:
                    dones_max_q, _, eval_fpe_max_q = evaluate_agent_max_q(copy.deepcopy(envs), agent, episode_max_steps,
                                                                          params['n_episodes'], n_arms,
                                                                          always_select=False)
                    results_max_q[c, i, :] += list(dones_max_q.values())
                    fpe_max_q[c, i, :] += np.mean(eval_fpe_max_q, 1)

                    dones_max_q, _, eval_fpe_max_q = evaluate_agent_max_q(copy.deepcopy(envs), agent, episode_max_steps,
                                                                          params['n_episodes'], n_arms,
                                                                          always_select=True)
                    results_max_q_a[c, i, :] += list(dones_max_q.values())
                    fpe_max_q_a[c, i, :] += np.mean(eval_fpe_max_q, 1)

        if not os.path.exists('data'):
            os.makedirs('data')

        if grid_search:
            with open('data/generalization_results_{0}.pickle'.format(tag), 'wb') as handle:
                pickle.dump({'res': results},
                            handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('data/generalization_results_{0}.pickle'.format(tag), 'wb') as handle:
                pickle.dump({'res': results, 'res_rnd': results_rnd, 'res_max_q': results_max_q,
                             'res_max_q_a': results_max_q_a, 'fpe': fpe, 'fpe_rnd': fpe_rnd,
                             'fpe_max_q': fpe_max_q, 'fpe_max_q_a': fpe_max_q_a},
                             handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()