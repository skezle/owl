import os
from datetime import datetime
import random
import time
from argparse import ArgumentParser
import collections
import copy

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from gym_minigrid.wrappers import *

from dqn import DQNAgent, DQNAgentOWL, MultiTaskDQN
from utils import Transition, make_checkpoint_minigrid, load_checkpoint_minigrid
from utils import make_gif_minigrid, make_qs_heat_map_minigrid, make_action_freq_heat_map_minigrid, BufferCache
from evaluate_agent import evaluate_agent_oracle, evaluate_agent_bandits, evaluate_agent_max_q, evaluate_agent_rnd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_agent_model_free_minigrid(agent, envs, params, action_dim):
    update_timestep = params['update_every_n_steps']
    seed = params['seed']
    max_task_frames = params['max_task_frames']
    num_tasks = params['num_tasks']
    num_repeats = params['num_task_repeats']
    log_interval = params['log_interval']
    gif_interval = params['gif_interval']
    save_interval = params['save_interval']
    n_random_actions = params['n_random_actions']
    n_episodes = params['n_episodes']
    episode_max_steps = params['episode_max_steps']
    n_collect_steps = params['n_collect_steps']
    save_model = params['save_model']
    owl = params['owl']
    multitask = params['multitask']
    goal_reward = params['goal_reward']
    tag = params['tag']
    load = params['load']
    ckpt_tag = params['ckpt_tag']
    ckpt_step_count = params['ckpt_step_count']
    buffer_warm_start = params['buffer_warm_start']
    exp_replay_capacity = params['exp_replay_capacity']
    q_func_reg = params['q_func_reg']
    bandit_eval = params['bandits']

    assert n_collect_steps > agent.batchsize, "We must initially collect as many steps as the batch size!"

    cumulative_timestep = 0
    n_updates = 0
    i_episode = 0
    samples_number = 0
    task_number = 0
    i_tasks_prev = 0
    goal_reached = 0
    episode_reward, timings = collections.deque(maxlen=10), collections.deque(maxlen=10)

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    start_time = time.time()
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(params['logdir'], current_time + '_' + tag)
    writer = SummaryWriter(log_dir=log_dir)

    cache = BufferCache(num_tasks)
    warm_start_size = params['buffer_warm_start_size']

    state_filter=None

    while True:
        time_step = 0
        i_episode += 1
        i_task = cumulative_timestep // max_task_frames
        # Switch tasks
        if i_task != i_tasks_prev:
            i_tasks_prev = i_task
            task_number = i_task % num_tasks

            print("\nEnv/Task number: {} Step: {}\n".format(task_number, cumulative_timestep + 1))

            # at the end of each task switch evaluation of the policy selection on baselines
            # create gif and heatmaps
            if owl:
                dones_rnd, _, _ = evaluate_agent_rnd(copy.deepcopy(envs), agent, episode_max_steps,
                                                     params['n_episodes'], num_tasks)

                dones_max_q, _, _ = evaluate_agent_max_q(copy.deepcopy(envs), agent, episode_max_steps,
                                                         params['n_episodes'], n_arms=num_tasks, always_select=False)

                dones_max_q_always, _, _ = evaluate_agent_max_q(copy.deepcopy(envs), agent, episode_max_steps,
                                                                params['n_episodes'], n_arms=num_tasks, always_select=True)

                for i in range(num_tasks):
                    writer.add_scalar('Reward/RND_Dones_{}'.format(i), list(dones_rnd.values())[i], cumulative_timestep)
                    writer.add_scalar('Reward/MAXQ_Dones_{}'.format(i), list(dones_max_q.values())[i], cumulative_timestep)
                    writer.add_scalar('Reward/MAXQa_Dones_{}'.format(i), list(dones_max_q_always.values())[i], cumulative_timestep)

                make_gif_minigrid(
                    copy.deepcopy(envs), agent, state_filter, tag, cumulative_timestep, episode_max_steps, n_episodes=1,
                    pause=0.1, dqn=params['dqn']
                )
                if "DK" not in params['env']:
                    make_action_freq_heat_map_minigrid(
                        copy.deepcopy(envs), agent, state_filter, tag, cumulative_timestep, episode_max_steps,
                        n_episodes=1, dqn=params['dqn']
                    )
                    make_qs_heat_map_minigrid(
                        copy.deepcopy(envs), agent, state_filter, tag, cumulative_timestep, n_episodes=1,
                        dqn=params['dqn']
                    )

            # Condition to terminate
            # also let's cache agent and envs for plotting later
            if i_task >= num_tasks*num_repeats:
                if save_model:
                    make_checkpoint_minigrid(
                        agent, copy.deepcopy(envs), cumulative_timestep, tag,
                        [cumulative_timestep, n_updates, i_episode, samples_number, task_number]
                    )
                return

            # owl oracle used to tell what task to train on
            # Change heads and warm start buffer or warm start the epsilon greedy strategy
            if owl:
                print("Manual switch")
                # sample data from the buffer
                if buffer_warm_start or q_func_reg > 0:
                    if len(agent.replay_pool) > warm_start_size:
                        memory = agent.replay_pool.get_list(warm_start_size)
                    else:
                        memory = agent.replay_pool.get_all_list()
                    cache.set((task_number - 1) % num_tasks, memory)
                else:
                    memory = None

                # Perform the reg. of the Q-func
                agent.set_task(task_number, q_reg=True, memory=memory)

                # Clear the buffer
                agent.replay_pool.clear_pool()

                # Warm start the buffer with data from the Cache
                if buffer_warm_start and i_task > 1:
                    memory = cache.get(task_number)
                    agent.replay_pool.set(memory)

                samples_number = 0
            elif multitask:
                print("Manual switch")
                agent.set_task(task_number, q_reg=False, update=True) # q_reg not used (shared in the eval functions though)
                samples_number = len(agent.replay_pool[task_number])


        # Set the task
        env = envs[int(task_number)]
        state = env.reset()

        done = False
        episode_max_steps = env.max_steps if env.max_steps < episode_max_steps else episode_max_steps

        while not done:
            start = time.time()
            time_step += 1
            samples_number += 1
            cumulative_timestep += 1
            if samples_number < n_random_actions:
                action = int(np.random.choice(action_dim))
            else:
                agent.eval()
                action = agent.get_action(state, state_filter=state_filter)
                action = int(action) # sometimes is an array...
                agent.train()

            # Take a step in the environment
            nextstate, reward, done, _ = env.step(action)

            # Truncate maximum steps per episode
            if time_step == episode_max_steps:
                done = True

            # If we reach the goal, let's boost the reward for reaching it
            real_done = False if time_step == episode_max_steps else done
            if real_done and reward > 0:
                reward = goal_reward
                goal_reached += 1

            # Logging
            episode_reward.append(reward / time_step)

            # Train agent
            if cumulative_timestep % update_timestep == 0 and samples_number >= n_collect_steps:
                q1_loss, eps, q_logvars, opt_timings = agent.optimize(update_timestep, i_episode)
                n_updates += 1
                timings.append(time.time() - start)

            # Push latest data to buffer
            if multitask:
                agent.replay_pool[task_number].push(Transition(state, action, reward, nextstate, done))
            else:
                agent.replay_pool.push(Transition(state, action, reward, nextstate, done))
            state = nextstate

            # Logging
            if cumulative_timestep % log_interval == 0 and samples_number > n_collect_steps:

                writer.add_scalar('Loss/Q1-func', q1_loss, cumulative_timestep)
                if isinstance(eps, dict):
                    for i, e in eps.items():
                        writer.add_scalar('Loss/epsilon_{}'.format(i), e, cumulative_timestep)
                else:
                    writer.add_scalar('Loss/epsilon', eps, cumulative_timestep)
                writer.add_scalar('Loss/time_wall', time.time() - start_time, cumulative_timestep)

                dones_per_episodes, return_per_episode, num_frames_per_episode = evaluate_agent_oracle(
                    copy.deepcopy(envs), agent, episode_max_steps, n_episodes=n_episodes, n_tasks=num_tasks,
                    dqn=params['dqn']
                )

                if bandit_eval:
                    dones_per_episodes_bandit, return_per_episode_bandit, num_frames_per_episode_bandit, corrects_bandit, bandit_logging = evaluate_agent_bandits(
                        copy.deepcopy(envs), agent, episode_max_steps, params['bandit_loss'], params['greedy_bandit'], n_episodes=n_episodes,
                        n_arms=num_tasks, debug=False, tag=tag, step=cumulative_timestep, lr=params['bandit_lr'],
                        decay=params['bandit_decay'], epsilon=params['bandit_epsilon'], bandit_step=params['bandit_step'],
                    )

                writer.add_scalar('Reward/Task', task_number, cumulative_timestep)
                for i in range(num_tasks):
                    r = return_per_episode[i, :]
                    fpe = num_frames_per_episode[i, :]
                    dones = dones_per_episodes[i]
                    writer.add_scalar('Reward/ORCL_Dones_{}'.format(i), dones, cumulative_timestep)
                    writer.add_scalar('Reward/ORCL_RPE_Mean_{}'.format(i), np.mean(r), cumulative_timestep)
                    writer.add_scalar('Reward/ORCL_FPE_Mean_{}'.format(i), np.mean(fpe), cumulative_timestep)

                # Log Nix and Weigand metrics
                if params['uncert']:
                    writer.add_scalar('Values/min_var', np.exp(agent.policy_net.network.min_logvar.cpu().detach().numpy()), cumulative_timestep)
                    writer.add_scalar('Values/max_var', np.exp(agent.policy_net.network.max_logvar.cpu().detach().numpy()), cumulative_timestep)
                    writer.add_histogram('Values/Q_vars', np.exp(q_logvars.cpu().detach().numpy()), cumulative_timestep)

                # Log bandit metrics
                if bandit_eval:
                    for i in range(num_tasks):
                        r = return_per_episode_bandit[i, :]
                        fpe = num_frames_per_episode_bandit[i, :]
                        dones = dones_per_episodes_bandit[i]
                        c = corrects_bandit[i]
                        writer.add_scalar('Reward/BNDT_Dones_{}'.format(i), dones, cumulative_timestep)
                        writer.add_scalar('Reward/BNDT_RPE_Mean_{}'.format(i), np.mean(r), cumulative_timestep)
                        writer.add_scalar('Reward/BNDT_FPE_Mean_{}'.format(i), np.mean(fpe), cumulative_timestep)
                        writer.add_scalar('Reward/BNDT_correct_arm_{}'.format(i), np.mean(c), cumulative_timestep)
                    if params['bandit_debug']:
                        for k, v in bandit_logging.items():
                            for i in range(num_tasks): # the task evaluated
                                for j in range(num_tasks): # the feedback from each arm
                                    writer.add_scalar('Bandit/Bandit_feedback_{}_task_{}_arm_{}'.format(k, i, j), np.nanmean(v[i, j, ...]), cumulative_timestep)

                # Log gradients
                if params['debug_grads']:
                    for n, p in agent.policy_net.named_parameters():
                        if p.grad is not None:
                            writer.add_histogram('q_grad_{}'.format(n), p.grad, n_updates)


                # Print output ot stdout
                with np.printoptions(precision=2, suppress=True):
                    print(
                        'Task {}, Episode {}, Samples {}, Test RPE mean: {}, Test RPE min: {}, Test RPE max: {},\
                        Train RPE: {:.3f}, #Goal Reached: {}, Test FPE mean: {}, Test FPE min: {}, Test FPE max: {},\
                        Number of Policy Updates: {}, Update time: {:.2f}'.format(task_number, i_episode, samples_number, np.mean(return_per_episode, 1),
                        np.min(return_per_episode, 1), np.max(return_per_episode, 1), np.mean(list(episode_reward)), goal_reached,
                        np.mean(num_frames_per_episode, 1), np.min(num_frames_per_episode, 1), np.max(num_frames_per_episode, 1),
                        n_updates, np.mean(list(timings)))
                        )

                # Debug bandit
                if bandit_eval and params['bandit_debug'] and load:

                    cumulative_timestep, envs, n_updates, i_episode, samples_number, task_number = load_checkpoint_minigrid(
                        agent, ckpt_step_count, ckpt_tag
                    )

                    dones_per_episode, _, _ = evaluate_agent_oracle(
                        copy.deepcopy(envs), agent, episode_max_steps, n_episodes=n_episodes, n_tasks=num_tasks,
                        dqn=params['dqn']
                    )

                    print("Task 0 dones {0}, Task 1 dones {1}".format(np.min(dones_per_episode[0]), np.min(dones_per_episode[1])))

                    _, _, _, _, _ = evaluate_agent_bandits(
                        copy.deepcopy(envs), agent, episode_max_steps, params['bandit_loss'],
                        params['greedy_bandit'], n_episodes=1, n_arms=num_tasks,
                        debug=params['bandit_debug'], tag=tag, step=cumulative_timestep,
                        lr=params['bandit_lr'], decay=params['bandit_decay'],
                        epsilon=params['bandit_epsilon'], bandit_step=params['bandit_step'],
                    )


def main():
    parser = ArgumentParser()

    # Fixed params
    parser.set_defaults(state_bonus=True)
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
    parser.add_argument('--action_bonus', dest='action_bonus', default=False, action='store_true', help='Whether to use action bonus.')
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
    parser.add_argument('--reverse', dest='reverse', default=False, action='store_true', help='Whether to reverse order of envs.')
    parser.add_argument('--debug_grads', dest='debug_grads', default=False, action='store_true', help='Whether to debug the policy and q function gradeints.')
    parser.add_argument('--debug_plots', dest='debug_plots', default=False, action='store_true', help='Whether to debug the plotting functions.')
    parser.add_argument('--env_seeds', nargs='+', dest='env_seeds', type=int, default=[100], help='The env seeds for env in {SimpleCrossing}. Needs to be same length as the number of tasks.')

    # RL params
    parser.add_argument('--grad_clip_norm', dest='grad_clip_norm', type=float, default=None)
    parser.add_argument('--exp_replay_capacity', dest='exp_replay_capacity', type=float, default=1e6)
    parser.add_argument('--huber', dest='huber', default=False, action='store_true', help='Whether to use Huber loss for DQN agent.')
    parser.add_argument('--lr', dest='lr', type=float, default=0.0000625, help='DQN learning rate.')

    # CL params
    parser.add_argument('--max_task_frames', type=int, default=1e6)
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--num_task_repeats', type=int, default=1)
    parser.add_argument('--q_ewc_reg', dest='q_ewc_reg', type=float, default=0.0, help='EWC Q-func regularisation strength.')
    parser.add_argument('--owl', dest='owl', default=False, action='store_true', help='Whether to use owl DQN agent for CL.')
    parser.add_argument('--big_head', dest='big_head', default=False, action='store_true', help='Whether to use a two layer network as a head for owl for CL.')
    parser.add_argument('--eps_gr_warm_start', dest='eps_gr_warm_start', default=False, action='store_true', help='Whether to reset eps greedy stretegy with twice the decay when seeing the same task again.')
    parser.add_argument('--eps_gr_warm_start_decay_rate', dest='eps_gr_warm_start_decay_rate', type=int, default=2, help='The decay rate after having seen the task once.')
    parser.add_argument('--buffer_warm_start', dest='buffer_warm_start', default=False, action='store_true', help='Whether to warm start the buffer when we see the same task again.')
    parser.add_argument('--buffer_warm_start_size', dest='buffer_warm_start_size', type=int, default=50000, help='Size of the buffer used for warm starting.')
    parser.add_argument('--dropout_prob', dest='dropout_prob', type=float, default=0.0, help='DQN CNN dropout probability.')
    parser.add_argument('--q_func_reg', dest='q_func_reg', type=float, default=0.0, help='Q functional reg.')
    parser.add_argument('--uncert', dest='uncert', default=False, action='store_true', help='Whether to use learn variance train with NLL.')
    parser.add_argument('--multitask', dest='multitask', default=False, action='store_true', help='Whether to train a Multi-Task DQN with sep buffers and sep heads per task.')

    # Bandit params
    parser.add_argument('--bandits', dest='bandits', default=False, action='store_true', help='Whether to use bandits to select correct head for task.')
    parser.add_argument('--greedy_bandit', dest='greedy_bandit', default=False, action='store_true', help='Whether to use the greedy bandit.')
    parser.add_argument('--bandit_loss', dest='bandit_loss', type=str, default='mse', help='Bandit loss for Exp weights \in {nll, mse, hack}.')
    parser.add_argument('--bandit_debug', dest='bandit_debug', default=False, action='store_true', help='Debug the bandit at test time.')
    parser.add_argument('--bandit_lr', dest='bandit_lr', type=float, default=1.0, help='Bandit learning rate.')
    parser.add_argument('--bandit_decay', dest='bandit_decay', type=float, default=1.0, help='Bandit decay.')
    parser.add_argument('--bandit_step', dest='bandit_step', type=int, default=1, help='Number of steps taken between bandit arm pulls.')
    parser.add_argument('--bandit_epsilon', dest='bandit_epsilon', type=float, default=0.0, help='Eps greedy exploration in the ExpWeights bandit alg.')

    args = parser.parse_args()
    params = vars(args)

    seed = params['seed']
    envs = []

    print(params)

    assert args.uncert != args.huber

    # Creating the environments
    if args.env == 'SC':
        assert len(args.env_seeds) == args.num_tasks
        env_str = ['MiniGrid-SimpleCrossingS9N1-v0'] * args.num_tasks
        if args.reverse: env_str.reverse()
    elif args.env == "DK":
        assert len(args.env_seeds) == args.num_tasks
        env_str = ['MiniGrid-DoorKey-8x8-v0'] * args.num_tasks
    elif args.env == "SC+DK":
        assert len(args.env_seeds) == args.num_tasks
        env_str = ['MiniGrid-SimpleCrossingS9N1-v0'] * args.num_tasks
        env_str[1] = 'MiniGrid-DoorKey-8x8-v0'
        env_str[3] = 'MiniGrid-DoorKey-8x8-v0'
    else:
        raise ValueError

    action_dims = []
    for i in range(args.num_tasks):
        env = gym.make(env_str[i])

        if args.state_bonus:
            env = StateBonus(env)

        if args.action_bonus:
            env = ActionBonus(env)

        env = ImgObsWrapper(env)  # Get rid of the 'mission' field
        env = ReseedWrapper(env, [args.env_seeds[i]]) # Setting the seed for the e.g. FourRooms keeps the doors fixed.
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

    if args.owl:
        agent = DQNAgentOWL(seed, state_dim, action_dim, num_tasks=args.num_tasks, lr=args.lr, gamma=0.99, batchsize=32, hidden_size=200,
                                q_ewc_reg=args.q_ewc_reg, bn=args.bn, ddqn=args.ddqn, duelling=args.duelling,
                                pool_size=args.exp_replay_capacity, target_update=80,
                                eps_decay=args.eps_decay, huber=args.huber, big_head=args.big_head, eps_gr_warm_start=args.eps_gr_warm_start,
                                eps_gr_warm_start_decay_rate=args.eps_gr_warm_start_decay_rate,
                                dropout_prob=args.dropout_prob, q_func_reg=args.q_func_reg, uncert=args.uncert)
    elif args.multitask:
        agent = MultiTaskDQN(seed, state_dim, action_dim, num_tasks=args.num_tasks, lr=args.lr, gamma=0.99,
                             batchsize=32, hidden_size=200,
                             bn=args.bn, ddqn=args.ddqn, duelling=args.duelling,
                             pool_size=args.exp_replay_capacity, target_update=80, eps_decay=args.eps_decay, huber=args.huber)
    else:
        agent = DQNAgent(seed, state_dim, action_dim, num_tasks=1, lr=args.lr, gamma=0.99, batchsize=32, hidden_size=200,
                         q_ewc_reg=0, bn=args.bn, ddqn=args.ddqn, duelling=args.duelling,
                         pool_size=args.exp_replay_capacity, target_update=80,
                         eps_decay=args.eps_decay, huber=args.huber, dropout_prob=args.dropout_prob,
                         q_func_reg=args.q_func_reg, uncert=args.uncert)

    train_agent_model_free_minigrid(agent=agent, envs=envs, params=params, action_dim=action_dim)

if __name__ == '__main__':
    main()
