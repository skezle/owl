import pdb
import copy
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

from online_learning import ExpWeights
from utils import plot_arr_trajectory_minigrid


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_agent_bandits(envs, agent, episode_max_steps, bandit_loss, greedy_bandit, n_episodes=16,
                           n_arms=2, debug=False, tag=None, step=None, lr=0.90, decay=0.90, epsilon=0.0,
                           bandit_step=1):
    agent.eval()
    task_idx = agent.get_task()
    n_tasks = len(envs)

    # Bandit debug
    feedback, arm = np.empty((n_tasks, n_arms, n_episodes, episode_max_steps+1)), np.empty((n_tasks, n_episodes, episode_max_steps+1))
    mses = np.empty((n_tasks, n_arms, n_episodes, episode_max_steps + 1))
    feedback[:], arm[:], mses[:] = np.nan, np.nan, 0

    # TB
    bandit_probs, bandit_p = np.empty((n_tasks, n_arms, n_episodes)), np.empty((n_tasks, n_arms, n_episodes, episode_max_steps+1))
    bandit_probs[:], bandit_p[:] = np.nan, np.nan
    dones, corrects = {i: 0 for i in range(n_tasks)}, {i: [] for i in range(n_tasks)}
    return_per_episode, num_frames_per_episode = np.zeros((n_tasks, n_episodes)), np.zeros((n_tasks, n_episodes))

    # Plotting Q
    h, w = envs[0].height, envs[0].width
    qs, q_vars, q_vars_episode = np.zeros((n_tasks, h, w)), np.zeros((n_tasks, h, w)), np.empty((n_tasks, n_arms, n_episodes, episode_max_steps+1))
    q_vars_episode[:] = np.nan

    # iterate through envs / Tasks
    for i in range(n_tasks):
        env = envs[i]
        # this will enable the 4R env to terminate without a realdone when reaching 100 steps
        episode_max_steps = min(episode_max_steps, env.max_steps - 1)
        reward_sum, correct, iter = 0, 0, 0
        freq = np.ones((h, w))
        for j in range(n_episodes):
            state = env.reset()
            x, y = env.agent_pos
            real_done, done = False, False
            logs_episode_return, logs_episode_num_frames, iter_episode = 0, 0, 0
            freq[y, x] += 1
            bandit = ExpWeights(arms=list(range(n_arms)), lr=lr, decay=decay, greedy=greedy_bandit, epsilon=epsilon)
            while not done:
                if iter_episode % bandit_step == 0:
                    idx = bandit.sample()
                arm[i, j, iter_episode] = idx
                bandit_p[i, :, j, iter_episode] = bandit.p
                iter += 1
                correct += 1 if idx == i else 0
                agent.set_task(idx, q_reg=False)
                action = agent.get_action(state, eval=True)
                action = int(action)  # sometimes is an array...
                nextstate, reward, done, _ = env.step(action)
                if logs_episode_num_frames == episode_max_steps:
                    done = True
                real_done = False if logs_episode_num_frames == episode_max_steps else done
                # get feedback for each arm - because we can easily.
                # We are comparing the main Q val to a fixed Q target which is chosen byt he bandit
                scores = []
                with torch.no_grad():
                    # DDQN
                    next_actions, _, _ = agent.policy_net(torch.Tensor(nextstate).to(device).unsqueeze(0), argmax=True)
                    _, next_actions_probs, _ = agent.target_net(torch.Tensor(nextstate).to(device).unsqueeze(0))
                    q_target = next_actions_probs.gather(1, next_actions)
                    value_target = reward + (1.0 - done) * agent.gamma * q_target.detach()
                for k in range(n_arms):
                    # iterate through the arms/heads to get feedback for the bandit
                    # Don't need to reset the agent with idx as it is not used, until the next round
                    agent.set_task(k, q_reg=False)
                    _, action_probs, log_vars = agent.policy_net(torch.Tensor(state).to(device).unsqueeze(0))
                    state_action_values = action_probs.gather(1, torch.Tensor(np.array([action])).long().view(1, -1).to(device))

                    # MSE feedback
                    mus_ = state_action_values.detach().cpu().numpy()
                    qs[i, y, x] += mus_
                    mse = np.sqrt(np.mean((mus_ - value_target.cpu().numpy()) ** 2))
                    mses[i, k, j, iter_episode] += mse
                    if bandit_loss == 'nll':
                        if agent.dropout_prob > 0:
                            mc_samples = 50
                            _mses = []
                            agent.train()  # make sure dropout is turned on
                            for _ in range(mc_samples):
                                _, action_probs, _ = agent.policy_net(torch.Tensor(state).to(device).unsqueeze(0))
                                state_action_value = action_probs.gather(
                                    1, torch.Tensor(np.array([action])).long().view(1,-1).to(device)
                                )
                                _mses.append((np.mean(state_action_value.detach().cpu().numpy()) - np.mean(
                                    value_target.cpu().numpy())) ** 2)
                            agent.eval()
                            nll = 0.5 * np.log(agent.data_var) + np.log(mc_samples) + np.log(2*np.pi) - np.log(
                                np.sum([np.exp(-0.5 * (1/agent.data_var) * y) for y in _mses]))
                            scores.append(min(-nll, 50))
                            feedback[i, k, j, iter_episode] = nll
                            q_vars[i, y, x] += nll
                        else:
                            assert agent.uncert
                            log_var_ = log_vars.gather(1, torch.Tensor(np.array([action])).long().view(1, -1).to(device))
                            var_ = np.exp(log_var_.detach().cpu().numpy()).mean()
                            nll = log_var_.detach().cpu().numpy().mean() + ((mus_ - value_target.cpu().numpy()) ** 2).mean() / var_
                            scores.append(min(-nll, 50))
                            feedback[i, k, j, iter_episode] = nll
                            q_vars[i, y, x] += var_
                            q_vars_episode[i, k, j, iter_episode] += var_
                    elif bandit_loss == 'mse':
                        scores.append(min(1/mse, 50))
                        feedback[i, k, j, iter_episode] = mse
                    else:
                        raise ValueError

                    x, y = env.agent_pos
                    freq[y, x] += 1

                state = nextstate

                logs_episode_return += reward
                logs_episode_num_frames += 1
                bandit.update_dists(scores)

                if real_done:
                    dones[i] += (1.0 / n_episodes)
                if done:
                    return_per_episode[i, j] = logs_episode_return / logs_episode_num_frames
                    num_frames_per_episode[i, j] = logs_episode_num_frames

                iter_episode += 1

            corrects[i].append(correct / iter)
            for m in range(len(bandit.p)):
                bandit_probs[i, m, j] = bandit.p[m] # last probability from the bandit

        # Normalization
        qs[i, :, :] /= freq
        q_vars[i, :, :] /= freq

    # Reset network to original task, head
    agent.set_task(task_idx, q_reg=False)
    agent.train()
    if debug:

        # plot activity traces
        # for k in range(n_episodes):
        #     fig, ax = plt.subplots(4, len(envs), figsize=(9, 6))
        #     for i in range(n_tasks): # tasks
        #         for j in range(n_tasks): # arms
        #             ax[0, i].plot(feedback[i, j, k, :], label="Arm {} ep {}".format(j, k), alpha=0.5)
        #             ax[1, i].plot(bandit_p[i, j, k, :], label="Arm {} ep {}".format(j, k), alpha=0.5)
        #             ax[2, i].plot(q_vars_episode[i, j, k, :], label="Arm {} ep {}".format(j, k), alpha=0.5)
        #             if j == 0:
        #                 ax[3, i].plot(arm[i, k, :], label="Episode: {}".format(k))
        #             ax[i, j].set_xlabel("Steps")
        #         ax[0, i].set_ylabel("Task {0} {1}".format(i, bandit_loss))
        #         ax[1, i].set_ylabel("Task {} Bandit probs".format(i))
        #         ax[2, i].set_ylabel("Task {} Arm var".format(i))
        #         ax[3, i].set_ylabel("Task {} Arm chosen".format(i))
        #         if bandit_loss == 'mse':
        #             ax[0, i].set_yscale('log')
        #
        #     plt.tight_layout()
        #     ax[0, 0].legend(loc='upper left', fontsize=8)
        #     ax[3, 0].legend(loc='upper left', fontsize=8)
        #
        #     plt.savefig('plots/bandit_debug_{0}_step_{1}_lr{2}_decay{3}_eps{4}_step{5}_ep{6}.pdf'.format(
        #         tag, step, bandit.lr, bandit.decay, bandit.epsilon, bandit_step, k
        #     ))

        fig, ax = plt.subplots(2, len(envs))
        for i in range(len(envs)):
            ax[0, i].boxplot([np.nansum(feedback[i, j, :, :], axis=1) for j in range(n_tasks)],
                             patch_artist=True, showmeans=False)
            ax[1, i].boxplot([np.nansum(mses[i, j, :, :], axis=1) for j in range(n_tasks)],
                             patch_artist=True, showmeans=False)

        plt.savefig('plots/bandit_debug_bp_{0}_step_{1}_lr{2}_decay{3}_eps{4}_step{5}.pdf'.format(
            tag, step, bandit.lr, bandit.decay, bandit.epsilon, bandit_step
        ))

        with open('plots/bandit_debug_bp_{0}_step_{1}_lr{2}_decay{3}_eps{4}_step{5}.pickle'.format(
                tag, step, bandit.lr, bandit.decay, bandit.epsilon, bandit_step
        ), 'wb') as handle:
            pickle.dump({'res': feedback},
                        handle, protocol=pickle.HIGHEST_PROTOCOL)

        #plot_arr_trajectory_minigrid(qs, h, w, len(envs), copy.deepcopy(envs), tag, step, 'qs')
        #plot_arr_trajectory_minigrid(q_vars, h, w, len(envs), copy.deepcopy(envs), tag, step, 'q_vars')
    return dones, return_per_episode, num_frames_per_episode, corrects, {
        'nlls': feedback if bandit_loss == 'nll' else np.empty(feedback.shape),
        'mses': feedback if bandit_loss == 'mse' else np.empty(feedback.shape),
        'bandit_prob': bandit_probs,
    }

def evaluate_agent_oracle(envs, agent, episode_max_steps, n_episodes=16, n_tasks=1, debug=False, dqn=True, tag=None, step=0):
    agent.eval()
    task_idx = agent.get_task()
    dones, return_per_episode, num_frames_per_episode = {i: 0 for i in range(n_tasks)}, np.zeros((n_tasks, n_episodes)), np.zeros((n_tasks, n_episodes))

    # Plotting the Q
    h, w = envs[0].height, envs[0].width
    qs = np.zeros((len(envs), h, w))
    if debug: pdb.set_trace()
    for i in range(n_tasks):
        env = envs[i]
        # this will enable the 4R env to terminate without a realdone when reaching 100 steps
        episode_max_steps = min(episode_max_steps, env.max_steps - 1)
        agent.set_task(i, q_reg=False)
        for j in range(n_episodes):
            state = env.reset()
            x, y = env.agent_pos
            real_done, done = False, False
            logs_done_counter, logs_episode_return, logs_episode_num_frames = 0, 0, 0
            while not done:
                action = agent.get_action(state, state_filter=None, deterministic=False, eval=True) # some args not used but match SAC for good measure
                action = int(action)  # sometimes is an array...
                nextstate, reward, done, _ = env.step(action)
                if logs_episode_num_frames == episode_max_steps:
                    done = True
                real_done = False if logs_episode_num_frames == episode_max_steps else done

                if dqn:
                    _, _q, _ = agent.policy_net(torch.Tensor(state).unsqueeze(0).to(device))
                    q = torch.max(_q).detach().cpu().numpy()
                else:
                    q1, q2 = agent.q_funcs(torch.Tensor(state).unsqueeze(0).to(device), action)  # q1, q2 \in [1, |A|] and torch.min(q1, q2) \in [1, |A|]
                    q = torch.max(torch.min(q1, q2)).detach().cpu().numpy()
                qs[i, y, x] = max(qs[i, y, x], q)  # max q \in |A|, picking min q value over all actions
                x, y = env.agent_pos

                state = nextstate

                logs_episode_return += reward
                logs_episode_num_frames += 1

                if real_done:
                    dones[i] += (1.0 / n_episodes)
                if done:
                    return_per_episode[i, j] = logs_episode_return / logs_episode_num_frames
                    num_frames_per_episode[i, j] = logs_episode_num_frames

    agent.set_task(task_idx, q_reg=False)
    agent.train()
    # Plotting the Q
    if debug:
        qs[:, 0, :] = np.nan
        qs[:, :, 0] = np.nan
        qs[:, h - 1, :] = np.nan
        qs[:, :, w - 1] = np.nan
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        for i in range(len(envs)):
            fig, ax = plt.subplots(1, 1, figsize=(9, 6))
            pos = ax.imshow(qs[i, :, :], cmap='Blues')
            ax.set_title('Q values')
            divider = make_axes_locatable(ax)
            cax1 = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(pos, cax=cax1)
            # Major ticks
            ax.set_xticks(np.arange(0, w, 1))
            ax.set_yticks(np.arange(0, h, 1))
            # Labels for major ticks
            ax.set_xticklabels(np.arange(1, w + 1, 1))
            ax.set_yticklabels(np.arange(1, h + 1, 1))
            # Minor ticks
            ax.set_xticks(np.arange(-.5, w, 1), minor=True)
            ax.set_yticks(np.arange(-.5, h, 1), minor=True)
            # Gridlines based on minor ticks
            ax.grid(which='minor', color='w', linestyle='-', linewidth=5)
            plt.tight_layout()
            plt.savefig('plots/{0}_{1}_Qs_task{2}.pdf'.format(tag, step, i))
    return dones, return_per_episode, num_frames_per_episode


def evaluate_agent_rnd(envs, agent, episode_max_steps, n_episodes, n_arms):
    agent.eval()
    task_idx = agent.get_task()
    n_tasks = len(envs)

    dones = {i: 0 for i in range(n_tasks)}
    return_per_episode, num_frames_per_episode = np.zeros((n_tasks, n_episodes)), np.zeros((n_tasks, n_episodes))

    # iterate through envs / Tasks
    for i in range(n_tasks):

        env = envs[i]
        # this will enable the 4R env to terminate without a realdone when reaching 100 steps
        episode_max_steps = min(episode_max_steps, env.max_steps - 1)
        reward_sum, correct, iter = 0, 0, 0

        for j in range(n_episodes):

            state = env.reset()
            real_done, done = False, False
            logs_episode_return, logs_episode_num_frames, iter_epsisode = 0, 0, 0

            while not done:

                idx = np.random.choice(range(0, n_arms))
                iter += 1
                agent.set_task(idx, q_reg=False)
                action = agent.get_action(state, eval=True)
                action = int(action)
                nextstate, reward, done, _ = env.step(action)
                if logs_episode_num_frames == episode_max_steps:
                    done = True
                real_done = False if logs_episode_num_frames == episode_max_steps else done

                state = nextstate

                logs_episode_return += reward
                logs_episode_num_frames += 1

                if real_done:
                    dones[i] += (1.0 / n_episodes)
                if done:
                    return_per_episode[i, j] = logs_episode_return / logs_episode_num_frames
                    num_frames_per_episode[i, j] = logs_episode_num_frames

                iter_epsisode += 1

    agent.set_task(task_idx, q_reg=False)
    agent.train()
    return dones, return_per_episode, num_frames_per_episode

def evaluate_agent_max_q(envs, agent, episode_max_steps, n_episodes, n_arms, always_select=False):

    agent.eval()
    task_idx = agent.get_task()
    n_tasks = len(envs)

    dones = {i: 0 for i in range(n_tasks)}
    return_per_episode, num_frames_per_episode = np.zeros((n_tasks, n_episodes)), np.zeros((n_tasks, n_episodes))

    for i in range(n_tasks):

        env = envs[i]

        # this will enable the 4R env to terminate without a realdone when reaching 100 steps
        episode_max_steps = min(episode_max_steps, env.max_steps - 1)

        for j in range(n_episodes):

            state = env.reset()
            real_done, done = False, False
            logs_episode_return, logs_episode_num_frames, iter_episode = 0, 0, 0
            q_vals = []
            # Evaluation loop
            while not done:

                # Pick the policy
                # Pick arm with the highest Q value
                # Don't need to take a step in the env
                if always_select:
                    q_vals = []
                if always_select or iter_episode == 0:
                    for k in range(n_arms):
                        # iterate through the arms/heads to get feedback for the bandit
                        # Don't need to reset the agent with idx as it is not used, until the next round
                        agent.set_task(k, q_reg=False)
                        action = agent.get_action(state, state_filter=None, deterministic=False,
                                                  eval=True)  # some args not used but match SAC for good measure
                        action = int(action)  # sometimes is an array...
                        _, action_probs, _ = agent.policy_net(torch.Tensor(state).to(device).unsqueeze(0))
                        state_action_values = action_probs.gather(1, torch.Tensor(np.array([action])).long().view(1,
                                                                                                                  -1).to(
                            device))
                        q_vals.append(np.mean(state_action_values.detach().cpu().numpy()))

                idx = np.argmax(q_vals)
                agent.set_task(idx, q_reg=False)


                action = agent.get_action(state, state_filter=None, deterministic=False,
                                          eval=True)  # some args not used but match SAC for good measure
                action = int(action)  # sometimes is an array...
                nextstate, reward, done, _ = env.step(action)
                if logs_episode_num_frames == episode_max_steps:
                    done = True
                real_done = False if logs_episode_num_frames == episode_max_steps else done
                state = nextstate

                logs_episode_return += reward
                logs_episode_num_frames += 1

                if real_done:
                    dones[i] += (1.0 / n_episodes)
                if done:
                    return_per_episode[i, j] = logs_episode_return / logs_episode_num_frames
                    num_frames_per_episode[i, j] = logs_episode_num_frames

                iter_episode += 1

    agent.set_task(task_idx, q_reg=False)
    agent.train()
    return dones, return_per_episode, num_frames_per_episode