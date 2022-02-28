from argparse import ArgumentParser
import copy

import gym
from gym_minigrid.wrappers import *

from train_agent_sequential import PendulumRewardFunction
from sac import SAC_Agent, SAC_Agent_BLR
from utils import MeanStdevFilter, make_gif, load_checkpoint

from dqn import DQNAgent, DQNAgentUnclear
from utils import load_checkpoint_minigrid, make_gif_minigrid_bandit
from train_minigrid import evaluate_agent_oracle, evaluate_agent_bandits

def smooth(scalars: list, weight: float) -> list:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

if __name__ == '__main__':

    # Pendulum
    parser = ArgumentParser()
    #parser.add_argument('--env', type=str, default='Pendulum-v0')
    #parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--use_obs_filter', dest='obs_filter', action='store_true')
    #parser.add_argument('--update_every_n_steps', type=int, default=1)
    #parser.add_argument('--n_random_actions', type=int, default=10000)
    #parser.add_argument('--n_collect_steps', type=int, default=1000)
    parser.add_argument('--update_blr_every_n_steps', type=int, default=1000)
    parser.add_argument('--n_blr_samples', type=int, default=10000)
    parser.add_argument('--blr', dest='blr', default=False, action='store_true')
    parser.add_argument('--prior_var', dest='prior_var', type=float, default=0.1, help='BLR prior var.')
    # parser.add_argument('--q_ewc_reg', dest='q_ewc_reg', type=float, default=0.0,
    #                     help='EWC Q-func regularisation strength.')
    parser.add_argument('--p_ewc_reg', dest='p_ewc_reg', type=float, default=0.0,
                        help='EWC policy regularisation strength.')
    parser.add_argument('--opt_angle', dest='opt_angle', type=int, default=45, help='Optimal pendulum angle.')
    parser.add_argument('--step_count', dest='step_count', type=int, help='Step to load checkpoint.')

    ##### Minigrid
    # Fixed params
    parser.set_defaults(action_bonus=False)
    parser.set_defaults(num_models=2)
    parser.set_defaults(obs_filter=False)
    parser.set_defaults(dqn=True)
    parser.set_defaults(duelling=True)
    parser.set_defaults(ddqn=True)
    parser.set_defaults(bn=False)
    parser.set_defaults(goal_reward=100)
    parser.set_defaults(min_exp=0.01)
    parser.set_defaults(eps_decay=250000)

    # env params
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--update_every_n_steps', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=4000)
    parser.add_argument('--gif_interval', type=int, default=40000)
    parser.add_argument('--save_interval', type=int, default=40000)
    parser.add_argument('--n_random_actions', type=int, default=10000)
    parser.add_argument('--n_collect_steps', type=int, default=128)
    parser.add_argument('--n_episodes', type=int, default=16)
    parser.add_argument('--episode_max_steps', type=int, default=100)
    parser.add_argument('--save_model', dest='save_model', action='store_true', default=False)
    parser.add_argument('--tag', type=str, default='', help='unique str to tag tb.')
    parser.add_argument('--load', dest='load', default=False, action='store_true', help='Whether to load a checkpoint.')
    parser.add_argument('--ckpt_tag', type=str, default='', help='Tag for checkpoint to load.')
    parser.add_argument('--ckpt_step_count', dest='ckpt_step_count', type=int, help='Step to load checkpoint.')
    parser.add_argument('--env', dest='env', default='MiniGrid-Empty-6x6-v0', type=str, help='The MiniGrid env')
    parser.add_argument('--logdir', dest='logdir', default='runs_minigrid', type=str, help='The logdir for TB.')
    parser.add_argument('--debug_grads', dest='debug_grads', default=False, action='store_true',
                        help='Whether to debug the policy and q function gradeints.')
    parser.add_argument('--debug_plots', dest='debug_plots', default=False, action='store_true',
                        help='Whether to debug the plotting functions.')
    parser.add_argument('--env_seeds', nargs='+', dest='env_seeds', type=int, default=[100],
                        help='The env seeds for env in {SimpleCrossing}. Needs to be same length as the number of tasks.')
    parser.add_argument('--state_bonus', dest='state_bonus', default=False, action='store_true', help='Whether to use a state bonus as an additional reward.')

    # RL params
    parser.add_argument('--grad_clip_norm', dest='grad_clip_norm', type=float, default=None)
    parser.add_argument('--exp_replay_capacity', dest='exp_replay_capacity', type=float, default=1e6)
    parser.add_argument('--huber', dest='huber', default=False, action='store_true',
                        help='Whether to use Huber loss for DQN agent.')

    # CL params
    parser.add_argument('--max_task_frames', type=int, default=1e6)
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--num_task_repeats', type=int, default=1)
    parser.add_argument('--q_ewc_reg', dest='q_ewc_reg', type=float, default=0.0,
                        help='EWC Q-func regularisation strength.')
    parser.add_argument('--unclear', dest='unclear', default=False, action='store_true',
                        help='Whether to use unclear DQN agent for CL.')
    parser.add_argument('--big_head', dest='big_head', default=False, action='store_true',
                        help='Whether to use a two layer network as a head for unclear for CL.')
    parser.add_argument('--eps_gr_warm_start', dest='eps_gr_warm_start', default=False, action='store_true',
                        help='Whether to reset eps greedy stretegy with twice the decay when seeing the same task again.')
    parser.add_argument('--eps_gr_warm_start_decay_rate', dest='eps_gr_warm_start_decay_rate', type=int, default=2,
                        help='The decay rate after having seen the task once.')
    parser.add_argument('--buffer_warm_start', dest='buffer_warm_start', default=False, action='store_true',
                        help='Whether to warm start the buffer when we see the same task again.')
    parser.add_argument('--buffer_warm_start_size', dest='buffer_warm_start_size', type=int, default=50000,
                        help='Size of the buffer used for warm starting.')
    parser.add_argument('--dropout_prob', dest='dropout_prob', type=float, default=0.0,
                        help='DQN CNN dropout probability.')
    parser.add_argument('--q_func_reg', dest='q_func_reg', type=float, default=0.0, help='Q functional reg.')
    parser.add_argument('--uncert', dest='uncert', default=False, action='store_true',
                        help='Whether to use learn variance train with NLL.')

    # Bandit params
    parser.add_argument('--bandits', dest='bandits', default=False, action='store_true',
                        help='Whether to use bandits to select correct head for task.')
    parser.add_argument('--greedy_bandit', dest='greedy_bandit', default=False, action='store_true',
                        help='Whether to use the greedy bandit.')
    parser.add_argument('--bandit_loss', dest='bandit_loss', type=str, default='mse',
                        help='Bandit loss for Exp weights \in {nll, mse, hack}.')
    parser.add_argument('--bandit_debug', dest='bandit_debug', default=False, action='store_true',
                        help='Debug the bandit at test time.')
    parser.add_argument('--bandit_lr', dest='bandit_lr', type=float, default=1.0, help='Bandit learning rate.')
    parser.add_argument('--bandit_decay', dest='bandit_decay', type=float, default=1.0, help='Bandit decay.')
    parser.add_argument('--bandit_step', dest='bandit_step', type=int, default=1,
                        help='Number of steps taken between bandit arm pulls.')
    parser.add_argument('--bandit_epsilon', dest='bandit_epsilon', type=float, default=0.0,
                        help='Eps greedy exploration in the ExpWeights bandit alg.')

    args = parser.parse_args()
    params = vars(args)

    seed = params['seed']
    if args.env == "Pendulum-v0":
        env = gym.make(params['env'])
        env = RescaleAction(env, -1, 1)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        if args.blr:
            agent = SAC_Agent_BLR(seed, state_dim, action_dim, blr_sample_size=args.n_blr_samples,
                                  prior_var=args.prior_var, grad_clip_norm=args.grad_clip_norm,
                                  num_models=args.num_models, num_tasks=args.num_tasks, q_ewc_reg_coef=args.q_ewc_reg,
                                  p_ewc_reg_coef=args.p_ewc_reg)
        else:
            agent = SAC_Agent(seed, state_dim, action_dim)

        state_filter = None

        reward_fnc = PendulumRewardFunction(0, args.opt_angle)
        # Load checkpoint
        load_checkpoint(agent, args.step_count, args.tag, args.blr)
        # Make gif
        make_gif(agent, env, args.step_count, state_filter, reward_fnc, n_tasks=2, tag=args.tag)

    else:
        if "SC" in args.env:
            # Creating the environments
            assert len(args.env_seeds) == args.num_tasks
            env_str = ['MiniGrid-SimpleCrossingS9N1-v0'] * args.num_tasks
        elif args.env == "SC+DK":
            assert len(args.env_seeds) == args.num_tasks
            env_str = ['MiniGrid-SimpleCrossingS9N1-v0'] * args.num_tasks
            env_str[1] = 'MiniGrid-DoorKey-8x8-v0'
            env_str[3] = 'MiniGrid-DoorKey-8x8-v0'
        elif "OOD" in args.env:
            env_str = ["MiniGrid-SimpleCrossingS11N5-v0"] * args.num_tasks
        else:
            raise ValueError

        test_envs = []
        action_dims = []
        for i in range(len(env_str)):
            env = gym.make(env_str[i])

            if args.state_bonus:
                env = StateBonus(env)

            if args.action_bonus:
                env = ActionBonus(env)

            env = ImgObsWrapper(env)  # Get rid of the 'mission' field
            env = ReseedWrapper(env, [args.env_seeds[i]])  # Setting the seed for the e.g. FourRooms keeps the doors fixed.
            state_dim = env.observation_space.shape
            print("Max number of steps per episode: {}".format(env.unwrapped.max_steps))
            if args.env == 'SC' or args.env == 'OOD':
                action_dim = 3
            else:
                action_dim = env.action_space.n  # https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/minigrid.py 0,1,2 left, right forward
            print("Size of the action space: {}".format(action_dim))
            action_dims.append(action_dim)
            test_envs.append(env)
        action_dim = max(action_dims)

        if args.unclear:
            agent = DQNAgentUnclear(seed, state_dim, action_dim, num_tasks=args.num_tasks, lr=0.0000625, gamma=0.99,
                                    batchsize=32, hidden_size=200,
                                    q_ewc_reg=args.q_ewc_reg, bn=args.bn, ddqn=args.ddqn, duelling=args.duelling,
                                    pool_size=args.exp_replay_capacity, target_update=80,
                                    eps_decay=args.eps_decay, huber=args.huber, big_head=args.big_head,
                                    eps_gr_warm_start=args.eps_gr_warm_start,
                                    eps_gr_warm_start_decay_rate=args.eps_gr_warm_start_decay_rate,
                                    dropout_prob=args.dropout_prob, q_func_reg=args.q_func_reg, uncert=args.uncert)
        else:
            agent = DQNAgent(seed, state_dim, action_dim, num_tasks=1, lr=0.0000625, gamma=0.99, batchsize=32,
                             hidden_size=200, q_ewc_reg=0, bn=args.bn, ddqn=args.ddqn, duelling=args.duelling,
                             pool_size=args.exp_replay_capacity, target_update=80,
                             eps_decay=args.eps_decay, huber=args.huber, dropout_prob=args.dropout_prob,
                             q_func_reg=args.q_func_reg, uncert=args.uncert)

        cumulative_timestep, envs, n_updates, i_episode, samples_number, task_number = load_checkpoint_minigrid(
            agent, params['ckpt_step_count'], params['ckpt_tag']
        )

        dones_per_episode, _, _ = evaluate_agent_oracle(
            copy.deepcopy(envs), agent, params['episode_max_steps'], n_episodes=params['n_episodes'],
            n_tasks=params['num_tasks'], dqn=params['dqn']
        )

        print("\t".join("Task {} dones {}".format(*item) for item in enumerate([np.min(dones_per_episode[i]) for i in range(len(dones_per_episode))])))

        dones, _, _, _, _ = evaluate_agent_bandits(
            copy.deepcopy(envs), agent, params['episode_max_steps'], params['bandit_loss'],
            params['greedy_bandit'], n_episodes=50, n_arms=params['num_tasks'],
            debug=True, tag=params['ckpt_tag'], step=cumulative_timestep,
            lr=params['bandit_lr'], decay=params['bandit_decay'],
            epsilon=params['bandit_epsilon'], bandit_step=params['bandit_step'],
        )

        # make_gif_minigrid_bandit(
        #     copy.deepcopy(test_envs), agent, params['episode_max_steps'], params['bandit_loss'],
        #     params['greedy_bandit'], n_episodes=1, n_arms=params['num_tasks'],
        #     tag=params['tag'], step=cumulative_timestep,
        #     lr=params['bandit_lr'], decay=params['bandit_decay'],
        #     epsilon=params['bandit_epsilon'], bandit_step=params['bandit_step'],
        #     bandit_debug=True,
        # )