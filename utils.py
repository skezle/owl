import itertools
import os
import random
from collections import deque, namedtuple
import pdb
import copy
import pickle
import gym
from gym_minigrid.wrappers import *

import numpy as np
import torch
from moviepy.editor import ImageSequenceClip
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from torch.nn.functional import softplus
from array2gif import write_gif

from online_learning import ExpWeights

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(105)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'nextstate', 'done'))
#Transition = namedtuple('Transition', ('state', 'action', 'reward', 'nextstate', 'done', 'qval'))


def smooth(scalars: list, weight: float) -> list:  # Weight between 0 and 1
    # EWMA smoothing - useful for plots
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed

def GaussianMSELoss(mu, logvar, target, logvar_loss=True):
    if logvar_loss:
        return (logvar + (target - mu) ** 2 / logvar.exp()).mean()
    else:
        return ((target - mu) ** 2).mean()

def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """
        From: https://github.com/openai/baselines/blob/master/baselines/common/schedules.py

        Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

class MeanStdevFilter():
    def __init__(self, shape, clip=3.0):
        self.eps = 1e-4
        self.shape = shape
        self.clip = clip
        self._count = 0
        self._running_sum = np.zeros(shape)
        self._running_sum_sq = np.zeros(shape) + self.eps
        self.mean = np.zeros(shape)
        self.stdev = np.ones(shape) * self.eps

    def update(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1,-1)
        self._running_sum += np.sum(x, axis=0)
        self._running_sum_sq += np.sum(np.square(x), axis=0)
        # assume 2D data
        self._count += x.shape[0]
        self.mean = self._running_sum / self._count
        self.stdev = np.sqrt(
            np.maximum(
                self._running_sum_sq / self._count - self.mean**2,
                 self.eps
                 ))
    
    def __call__(self, x):
        return np.clip(((x - self.mean) / self.stdev), -self.clip, self.clip)

    def invert(self, x):
        return (x * self.stdev) + self.mean

class BufferCache:
    def __init__(self, num_tasks):
        self._dict = {i: [] for i in range(num_tasks)}

    def set(self, i, mem):
        self._dict[int(i)] = mem

    def get(self, i):
        return self._dict[int(i)]

    def __str__(self):
        return str(["Task: {0}, size: {1}".format(t, len(l)) for t, l in self._dict.items()])

class ReplayPool:

    def __init__(self, capacity=1e6):
        self.capacity = int(capacity)
        self._memory = deque(maxlen=int(capacity))
        
    def push(self, transition: Transition):
        """ Saves a transition """
        self._memory.append(transition)
        
    def sample(self, batch_size: int) -> Transition:
        transitions = random.sample(self._memory, batch_size)
        return Transition(*zip(*transitions))

    def get(self, start_idx: int, end_idx: int) -> Transition:
        transitions = list(itertools.islice(self._memory, start_idx, end_idx))
        return Transition(*zip(*transitions))

    def get_all(self) -> Transition:
        return self.get(0, len(self._memory))

    def __len__(self) -> int:
        return len(self._memory)

    def clear_pool(self):
        self._memory.clear()

    def get_list(self, batch_size: int) -> list:
        return list(random.sample(self._memory, int(batch_size)))

    def get_all_list(self) -> list:
        return list(itertools.islice(self._memory, 0, len(self._memory)))

    def set(self, s: list):
        for item in s:
            self.push(item)


# Taken from: https://github.com/pytorch/pytorch/pull/19785/files
# The composition of affine + sigmoid + affine transforms is unstable numerically
# tanh transform is (2 * sigmoid(2x) - 1)
# Old Code Below:
# transforms = [AffineTransform(loc=0, scale=2), SigmoidTransform(), AffineTransform(loc=-1, scale=2)]
class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2. * (math.log(2.) - x - softplus(-2. * x))

# Code courtesy of JPH: https://github.com/jparkerholder
def make_gif(agent, env, step_count, state_filter, reward_fnc, n_tasks=1, maxsteps=1000, dqn=False, tag=""):
    envname = env.spec.id
    gif_name = '_'.join([envname, str(step_count)])
    # Cache current task so that we can reset it
    task_idx = agent.get_task()
    rwd_task_idx = reward_fnc.task_idx
    agent.eval()
    for i in range(n_tasks):
        state = env.reset()
        done = False
        steps = []
        rewards = []
        t = 0
        agent.set_task(i, q_reg=False, p_ewc_update=False)
        reward_fnc.set_task(i)
        while (not done) & (t< maxsteps):
            s = env.render('rgb_array')
            #pdb.set_trace()
            steps.append(s)
            if dqn:
                action = agent.get_action(state, state_filter=state_filter, deterministic=True, eval=True)
            else:
                action = agent.get_action(state, state_filter=state_filter, deterministic=True)
            nextstate, reward, done, _ = env.step(action)
            if reward_fnc is not None:
                reward = reward_fnc(state, action)
            state = nextstate
            rewards.append(reward)
            t += 1
        print('Final reward: {:.3f}'.format(np.sum(rewards) / t))
        clip = ImageSequenceClip(steps, fps=30)
        if not os.path.isdir('gifs'):
            os.makedirs('gifs')
        clip.write_gif('gifs/{0}_task{1}{2}.gif'.format(gif_name, i+1, "_" + tag if len(tag) > 0 else ""), fps=30)
        env.close()
    agent.set_task(task_idx, q_ewc_update=False, p_ewc_update=False)
    reward_fnc.set_task(rwd_task_idx)
    agent.train()
    print("Finished making gifs.")

def make_gif_minigrid(envs, agent, state_filter, tag, step, episode_max_steps, n_episodes=1, pause=0.1, dqn=False):
    print("Making gif.")
    agent.eval()
    task_idx = agent.get_task()
    for i in range(len(envs)):
        frames = []
        env = envs[i]
        env.render('human')
        agent.set_task(i, False)
        for episode in range(n_episodes):
            time_step = 0
            state = env.reset()
            while True:
                frames.append(np.moveaxis(env.render("rgb_array"), 2, 0))
                if dqn:
                    action = agent.get_action(state, state_filter=state_filter, deterministic=True, eval=True)
                else:
                    action = agent.get_action(state, state_filter=state_filter, deterministic=True)
                nextstate, reward, done, _ = env.step(action)
                state = nextstate

                if time_step == episode_max_steps:
                    done = True

                if done:
                    break

                time_step += 1
        print("Saving gif... ")
        write_gif(np.array(frames), "gifs/{0}_{1}_task_{2}.gif".format(tag, step, i), fps=1 / pause)
    agent.set_task(task_idx, False)
    agent.train()
    print("Done.")

def make_gif_minigrid_bandit(envs, agent, episode_max_steps, bandit_loss, greedy_bandit, n_episodes=1,
                             n_arms=2, tag=None, step=None, lr=0.90, decay=0.90, epsilon=0.0,
                             bandit_step=1, pause=0.1, bandit_debug=False):
    agent.eval()
    task_idx = agent.get_task()
    n_tasks = len(envs)

    mses = np.empty((n_tasks, n_arms, n_episodes, episode_max_steps + 1))
    bandit_p = np.empty((n_tasks, n_arms, n_episodes, episode_max_steps+1))
    arm_selected = np.empty((n_tasks, n_episodes, episode_max_steps+1))
    h = envs[0].height
    w = envs[0].width
    bandit_p[:], arm_selected[:], mses[:] = np.nan, np.nan, np.nan

    assert n_episodes == 1, "will plot multiple minigrids on top of each other."
    # iterate through envs / Tasks
    for i in range(n_tasks):
        env = envs[i]
        frames, _frames = [], []
        for j in range(n_episodes):
            state = env.reset()
            logs_episode_num_frames, iter_episode = 0, 0

            bandit = ExpWeights(arms=list(range(n_arms)), lr=lr, decay=decay, greedy=greedy_bandit, epsilon=epsilon)
            while True:
                if iter_episode % bandit_step == 0:
                    idx = bandit.sample()
                arm_selected[i, j, iter_episode] = idx
                bandit_p[i, :, j, iter_episode] = bandit.p

                # plt.savefig('plots/frame_{0}_{1}_task{2}_ep{3}_iter{4}.png'.format(step, tag, i, j, iter_episode))
                frames.append(np.moveaxis(env.render("rgb_array"), 2, 0))
                _frames.append(env.render("rgb_array"))

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
                    state_action_values = action_probs.gather(1, torch.Tensor(np.array([action])).long().view(1, -1).to(
                        device))

                    if bandit_loss == 'nll':
                        assert agent.uncert
                        mus_ = state_action_values.detach().cpu().numpy()
                        log_var_ = log_vars.gather(1, torch.Tensor(np.array([action])).long().view(1, -1).to(device))
                        nll = log_var_.detach().cpu().numpy().mean() + (
                                    (mus_ - value_target.cpu().numpy()) ** 2).mean() / np.exp(
                            log_var_.detach().cpu().numpy()).mean()
                        scores.append(min(-nll, 50))
                    elif bandit_loss == 'mse':
                        mus_ = state_action_values.detach().cpu().numpy()
                        mse = np.sqrt(np.mean((mus_ - value_target.cpu().numpy()) ** 2))
                        scores.append(min(1 / mse, 50))
                        mses[i, k, j, iter_episode] = mse
                    else:
                        raise ValueError

                state = nextstate

                logs_episode_num_frames += 1
                bandit.update_dists(scores)

                iter_episode += 1

                if done:
                    break
        if bandit_debug:
            # visualise MAB
            colors = {0: "dodgerblue", 1: "forestgreen", 2: "darkred", 3: "purple", 4: "darkorange"}
            l = len(frames)
            ncol = 8
            nrow = (l // ncol) + 1
            fig = plt.figure(figsize=(ncol + 1, nrow + 1))
            from matplotlib import gridspec
            gs = gridspec.GridSpec(nrow, ncol,
                                   wspace=0.05, hspace=0.05,
                                   top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                   left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
            n = 0
            for a in range(nrow):
                for b in range(ncol):
                    if n >= l:
                        im = _frames[-1]
                        alpha = 0.5
                    else:
                        im = _frames[n]
                        alpha = 1.0
                    ax = plt.subplot(gs[a, b], alpha=alpha)
                    ax.imshow(im)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_xticks([], minor=True)
                    ax.set_yticks([])
                    ax.set_yticks([], minor=True)
                    #ax.set_axis_off()
                    #ax.tick_params(color='green', labelcolor='green')
                    if n < l:
                        for spine in ax.spines.values():
                            spine.set_edgecolor(colors[arm_selected[i, 0, n]])
                            spine.set_linewidth(3.0)
                    n += 1
            plt.savefig('plots/frames_{0}_{1}_task{2}_ep{3}.png'.format(step, tag, i, j),
                        bbox_inches = 'tight', pad_inches = 0)

        write_gif(np.array(frames), "gifs/{0}_{1}_task_{2}.gif".format(tag, step, i), fps=1 / pause)
    # Reset network to original task, head
    agent.set_task(task_idx, q_reg=False)
    agent.train()

    if bandit_debug:

        # plot activity traces
        for i in range(n_tasks):  # tasks
            fig, ax = plt.subplots(2, 1, figsize=(3, 5))
            for j in range(n_tasks):  # arms
                for k in range(n_episodes):
                    ax[0].plot(smooth(mses[i, j, k, :], 0.6), label="Arm: {}".format(j+1), color=colors[j], alpha=0.8)
                    ax[1].plot(bandit_p[i, j, k, :], label="Arm: {}".format(j+1), color=colors[j], alpha=0.8)
                ax[1].set_xlabel("Steps")
            ax[0].set_ylabel("Task {} MSE".format(i+1))
            ax[1].set_ylabel("Task {} Bandit probs".format(i+1))
            ax[0].set_yscale('log')

            handles, labels = ax[0].get_legend_handles_labels()
            lgd = ax[0].legend(handles, labels, loc='upper right', bbox_to_anchor=(0, 0),
                       ncol=1, fancybox=True, shadow=True)
            lgd.get_frame().set_linewidth(1.0)
            for line in lgd.get_lines():
                line.set_linewidth(2.0)

            with open('plots/bandit_vis_{0}_step_{1}_lr{2}_decay{3}_eps{4}_step{5}_task{6}.pickle'.format(
                tag, step, bandit.lr, bandit.decay, bandit.epsilon, bandit_step, i
            ), 'wb') as handle:
                pickle.dump({'mses': mses, 'bandit_p': bandit_p}, handle, protocol=pickle.HIGHEST_PROTOCOL)

            plt.savefig('plots/bandit_debug_{0}_step_{1}_lr{2}_decay{3}_eps{4}_step{5}_task{6}.pdf'.format(
                tag, step, bandit.lr, bandit.decay, bandit.epsilon, bandit_step, i
            ), bbox_extra_artists=(lgd,))

def _wall_details(env):
    h = env.height
    w = env.width
    z = np.zeros((h-2, w-2))
    _ = env.reset()
    for i in range(h-2):
        for j in range(w-2):
            x, y = j+1, i+1
            fwd_cell = env.grid.get(*(x, y))
            if fwd_cell != None and fwd_cell.type == 'wall':
                z[i, j] = 1

    if np.count_nonzero(np.sum(z, 0) == h - 3) == 1:
        horizontal = False
    elif np.count_nonzero(np.sum(z, 1) == h - 3) == 1:
        horizontal = True
    else:
        raise ValueError

    if horizontal:
        x_door = np.argmin(np.sum(z, 0)) + 1
        y_door = np.argmax(np.sum(z, 1)) + 1
    else:
        x_door = np.argmin(np.sum(z, 1)) + 1
        y_door = np.argmax(np.sum(z, 0)) + 1

    return horizontal, z, x_door, y_door

def plot_arr_trajectory_minigrid(qs, h, w, num_envs, envs, tag, step, type):
    """
    qs: numpy array qs over trajectory
    h: height of the grid
    w: width of the grid
    num_envs: the number of envs
    tag: unique tag for saving
    step: step from
    """
    qs[:, 0, :] = np.nan
    qs[:, :, 0] = np.nan
    qs[:, h - 1, :] = np.nan
    qs[:, :, w - 1] = np.nan

    for i, env in enumerate(envs):
        _, walls, _, _ = _wall_details(env) # already deep copied
        qs[i, 1:h-1, 1:w-1][walls == 1] = np.nan

    assert type == 'qs' or type == 'q_vars'
    color_map = {'qs': 'Blues', 'q_vars': 'Greens'}
    titles = {'qs': 'Q values', 'q_vars': 'Q variance'}

    for i in range(num_envs):
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        pos = ax.imshow(qs[i, :, :], cmap=color_map[type])
        ax.set_title(titles[type])
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
        plt.savefig('plots/{0}_{1}_{2}_task{3}_bandit_debug.pdf'.format(tag, step, type, i))


def make_qs_heat_map_minigrid(envs, agent, state_filter, tag, step, n_episodes=1, debug=False, dqn=False):
    """
    Cover the entire grid with the agent and for each tile rotate 360deg for all states which the agent can see and get the max Q function for these.
    """
    agent.eval()
    task_idx = agent.get_task()
    h = envs[0].height
    w = envs[0].width
    qs, qs_after = np.zeros((len(envs), h, w)), np.zeros((len(envs), h, w))
    covered = np.zeros((len(envs), h, w))
    for i in range(len(envs)):
        env = envs[i]
        agent.set_task(i, False)
        for episode in range(n_episodes):
            state = env.reset()
            x, y = env.agent_pos
            actions = []
            rotate = [env.actions.left] * 4
            turn_right = [env.actions.right, env.actions.forward, env.actions.right]
            turn_left = [env.actions.left, env.actions.forward, env.actions.left]
            sweep_forward = [env.actions.forward] * (w-3) + [env.actions.left] * 2 + [env.actions.forward] * (w-3) + [env.actions.left] * 2
            # creating paths in env
            if "Empty" in env.env.spec.id:
                if (x, y) == (1, 1):
                    for j in range(h - 2):
                        actions += [env.actions.forward] * (w - 3)
                        if j % 2 == 0 and j < h - 3: # don't want to turn on last row
                            actions += turn_right
                        elif j % 2 == 1 and j < h - 3: # don't want to turn on last row
                            actions += turn_left
                elif (x, y) == (w - 2, 1):
                    for j in range(h - 2):
                        actions += [env.actions.forward] * (w - 3)
                        if j % 2 == 0 and j < h - 3: # don't want to turn on last row
                            actions += turn_left
                        elif j % 2 == 1 and j < h - 3: # don't want to turn on last row
                            actions += turn_right
                else:
                    raise ValueError
            elif "SimpleCrossing" in env.env.spec.id:
                horizontal, walls, x_door, y_door = _wall_details(copy.deepcopy(env))
                if not horizontal:
                    actions += [env.actions.right]
                j = 0
                while j < h-2:
                    to_door = [env.actions.forward] * (x_door - 1)
                    if horizontal and j + 1 == y_door - 1 and ((j%2==0 and horizontal) or (j%2==1 and not horizontal)):
                        actions += sweep_forward # grab Qs before entering door
                        actions += to_door + [env.actions.right, env.actions.forward, env.actions.forward, env.actions.right] \
                                           + to_door + [env.actions.right]*2 + [env.actions.forward]*(h-3) # go back on oneself to capture the rest of the tiles
                        j += 2 # move through door
                    elif not horizontal and j + 1 == y_door - 1 and ((j%2==0 and not horizontal) or (j%2==1 and horizontal)):
                        actions += sweep_forward # grab Qs before entering door
                        actions += to_door + [env.actions.left, env.actions.forward, env.actions.forward, env.actions.left] \
                                            + to_door + [env.actions.left]*2 + [env.actions.forward]*(h-3) # sweep backward
                        j += 2 # move through door
                    else:
                        actions += [env.actions.forward] * (w - 3)

                    if horizontal and j < h - 3:
                        if j % 2 == 0:
                            actions += turn_right
                        elif j % 2 == 1:
                            actions += turn_left
                    else:
                        if j % 2 == 0:
                            actions += turn_left
                        elif j % 2 == 1:
                            actions += turn_right

                    j += 1

            # interacting with env
            if debug:
                pdb.set_trace()
            for j, action in enumerate(actions): # 0, 1, 2 -> left, right, fwd
                full_rotation_actions = [action] + rotate
                for k, a in enumerate(full_rotation_actions):
                    nextstate, reward, done, _ = env.step(a)
                    if state_filter:
                        state = state_filter(state)
                    if dqn:
                        _, _q, _ = agent.policy_net(torch.Tensor(state).unsqueeze(0).to(device))
                        q = torch.max(_q).detach().cpu().numpy()
                    else:
                        q1, q2 = agent.q_funcs(torch.Tensor(state).unsqueeze(0).to(device), action) # q1, q2 \in [1, |A|] and torch.min(q1, q2) \in [1, |A|]
                        q = torch.max(torch.min(q1, q2)).detach().cpu().numpy()
                    qs[i, y, x] = max(qs[i, y, x], q) # max q \in |A|, picking min q value over all actions
                    covered[i, y, x] += 1
                    x, y = env.agent_pos
                    qs_after[i, y, x] = max(qs_after[i, y, x], q)  # max q \in |A|, picking min q value over all actions
                    state = nextstate

    agent.train()
    agent.set_task(task_idx, False)
    # normalise
    #qs = (qs - np.min(qs, (1, 2)).reshape(-1, 1, 1) ) / (np.max(qs, (1, 2)) - np.min(qs, (1, 2))).reshape(-1, 1, 1)
    qs[:, 0, :], qs_after[:, 0, :] = np.nan, np.nan
    qs[:, :, 0], qs_after[:, :, 0] = np.nan, np.nan
    qs[:, h - 1, :], qs_after[:, h - 1, :] = np.nan, np.nan
    qs[:, :, w - 1], qs_after[:, :, w - 1] = np.nan, np.nan
    for i, env in enumerate(envs):
        if "SimpleCrossing" in env.env.spec.id:
            _, walls, _, _ = _wall_details(copy.deepcopy(env))
            qs[i, 1:h-1, 1:w-1][walls == 1] = np.nan
            qs_after[i, 1:h - 1, 1:w - 1][walls == 1] = np.nan
    qs[qs == 0] = np.nan # lazy way of removing tiles which are not covered
    qs_after[qs_after == 0] = np.nan
    # plot
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

def make_action_freq_heat_map_minigrid(envs, agent, state_filter, tag, step, episode_max_steps, n_episodes=1, dqn=False):
    print("Making heatmap for actions")
    h = envs[0].height
    w = envs[0].width
    agent.eval()
    task_idx = agent.get_task()
    a_lr, a_fwd = np.zeros((len(envs), h, w)), np.zeros((len(envs), h, w))
    for i in range(len(envs)):
        env = envs[i]
        agent.set_task(i, False)
        for episode in range(n_episodes):
            state = env.reset()
            # dir = env.dir_vec # not needed
            # plt.imshow(env.render("rgb_array")); plt.savefig('test.png')
            pos = env.agent_pos
            x, y = pos[0], pos[1]
            a_lr[i, y, x] += 1
            a_fwd[i, y, x] += 1
            time_step = 0
            while True:
                if dqn:
                    action = agent.get_action(state, state_filter=state_filter, deterministic=True, eval=True)
                else:
                    action = agent.get_action(state, state_filter=state_filter, deterministic=True)
                action = int(action)
                nextstate, reward, done, _ = env.step(action)
                state = nextstate

                if time_step == episode_max_steps:
                    done = True

                pos = env.agent_pos
                x, y = pos[0], pos[1]
                # left/right
                if action == 0 or action == 1:
                    a_lr[i, y, x] += 1
                # fwd
                elif action == 2:
                    a_fwd[i, y, x] += 1
                else:
                    raise ValueError

                if done:
                    break

                time_step += 1

    agent.train()
    agent.set_task(task_idx, False)
    print("Printing heatmap")
    # normalise
    a_lr = a_lr / np.sum(a_lr, (1, 2)).reshape(-1, 1, 1)
    a_fwd = a_fwd / np.sum(a_fwd, (1, 2)).reshape(-1, 1, 1)
    # borders
    a_lr[:, 0, :], a_fwd[:, 0, :] = np.nan, np.nan
    a_lr[:, :, 0], a_fwd[:, :, 0] = np.nan, np.nan
    a_lr[:, h-1, :], a_fwd[:, h-1, :] = np.nan, np.nan
    a_lr[:, :, w-1], a_fwd[:, :, w-1] = np.nan, np.nan
    for i, env in enumerate(envs):
        if "SimpleCrossing" in env.env.spec.id:
            _, walls, _, _ = _wall_details(copy.deepcopy(env))
            a_lr[i, 1:h - 1, 1:w - 1][walls == 1] = np.nan
            a_fwd[i, 1:h - 1, 1:w - 1][walls == 1] = np.nan
    # plot
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    for i in range(len(envs)):
        fig, ax = plt.subplots(1, 2, figsize=(9, 6))
        pos1 = ax[0].imshow(a_lr[i,:,:], cmap='Blues')
        ax[0].set_title('left/right')
        divider = make_axes_locatable(ax[0])
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(pos1, cax=cax1)
        pos2 = ax[1].imshow(a_fwd[i,:,:], cmap='Greens')
        ax[1].set_title('forward')
        divider = make_axes_locatable(ax[1])
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(pos2, cax=cax2)
        for j in range(2):
            # Major ticks
            ax[j].set_xticks(np.arange(0, w, 1))
            ax[j].set_yticks(np.arange(0, h, 1))
            # Labels for major ticks
            ax[j].set_xticklabels(np.arange(1, w + 1, 1))
            ax[j].set_yticklabels(np.arange(1, h + 1, 1))
            # Minor ticks
            ax[j].set_xticks(np.arange(-.5, w, 1), minor=True)
            ax[j].set_yticks(np.arange(-.5, h, 1), minor=True)
            # Gridlines based on minor ticks
            ax[j].grid(which='minor', color='w', linestyle='-', linewidth=5)
        plt.tight_layout()
        plt.savefig('plots/{0}_{1}_task{2}.pdf'.format(tag, step, i))

def make_checkpoint(agent, step_count, tag, blr):
    q_funcs, target_q_funcs, policy, log_alpha = agent.q_funcs, agent.target_q_funcs, agent.policy, agent.log_alpha
    
    save_path = "checkpoints/model-{}-{}.pt".format(step_count, tag)

    if not os.path.isdir('checkpoints'):
        os.makedirs('checkpoints')

    torch.save({
        'double_q_state_dict': q_funcs.state_dict(),
        'target_double_q_state_dict': target_q_funcs.state_dict(),
        'policy_state_dict': policy.state_dict(),
        'log_alpha_state_dict': log_alpha,
        'q_funcs_blr_params': agent.q_funcs.get_blr_params() if blr else '',
        'target_q_funcs_blr_params': agent.q_funcs.get_blr_params() if blr else '',
    }, save_path)

def make_checkpoint_minigrid(agent, envs, step_count, tag, counters):
    policy_net, target_net = agent.policy_net, agent.target_net

    save_path = "checkpoints/model-{}-{}.pt".format(step_count, tag)

    if not os.path.isdir('checkpoints'):
        os.makedirs('checkpoints')

    torch.save({
        'q_state_dict': policy_net.state_dict(),
        'target_q_state_dict': target_net.state_dict(),
        'counters': counters,
        'envs': envs,
    }, save_path)

def load_checkpoint(agent, step_count, tag, blr):
    save_path = "checkpoints/model-{}-{}.pt".format(step_count, tag)
    checkpoint = torch.load(save_path, map_location=device)
    agent.q_funcs.load_state_dict(checkpoint['double_q_state_dict'])
    agent.target_q_funcs.load_state_dict(checkpoint['target_double_q_state_dict'])
    agent.policy.load_state_dict(checkpoint['policy_state_dict'])
    agent.log_alpha = checkpoint['log_alpha_state_dict']
    if blr:
        agent.q_funcs.set_blr_params(checkpoint['q_funcs_blr_params'])
        agent.target_q_funcs.set_blr_params(checkpoint['target_q_funcs_blr_params'])

def load_checkpoint_minigrid(agent, step_count, tag):
    save_path = "checkpoints/model-{}-{}.pt".format(step_count, tag)
    checkpoint = torch.load(save_path, map_location=device)
    agent.policy_net.load_state_dict(checkpoint['q_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_q_state_dict'])
    counters = checkpoint['counters']
    cumulative_timestep, n_updates, i_episode, samples_number, task_number = counters
    envs = checkpoint['envs']
    return cumulative_timestep, envs, n_updates, i_episode, samples_number, task_number

def show_envs(env_name, seeds=[0, 1]):
    if isinstance(env_name, list):
        s = 0
        for i, e in enumerate(env_name):
            env = gym.make(e)
            env = ImgObsWrapper(env)  # Get rid of the 'mission' field
            env = ReseedWrapper(env, [s])
            _ = env.reset()
            plt.imshow(env.render("rgb_array"))
            plt.savefig('plots/test_task_{}_{}.png'.format(e, s))
    else:
        s = list(range(seeds[0], seeds[1]))
        n_envs = len(s)
        for i in range(n_envs):
            env = gym.make(env_name)
            env = ImgObsWrapper(env)  # Get rid of the 'mission' field
            env = ReseedWrapper(env, [s[i]])
            _ = env.reset()
            plt.imshow(env.render("rgb_array"))
            plt.savefig('plots/test_task_{}.png'.format(s[i]))

if __name__ == '__main__':
    show_envs(env_name='MiniGrid-SimpleCrossingS9N1-v0', seeds=[0, 200])
    # show_envs(env_name=['MiniGrid-SimpleCrossingS9N1-v0', 'MiniGrid-SimpleCrossingS9N2-v0',
    #                     'MiniGrid-SimpleCrossingS9N3-v0', 'MiniGrid-SimpleCrossingS11N5-v0',
    #                     'MiniGrid-FourRooms-v0'])
    # from gym_minigrid.minigrid import *
    # import matplotlib
    # frames = []
    # s = [100]
    # n_envs = len(s)
    # fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    # for i in range(n_envs):
    #     env = gym.make('MiniGrid-FourRooms-v0', agent_pos=(16, 2), goal_pos=(2, 2))
    #     env = ImgObsWrapper(env)  # Get rid of the 'mission' field
    #     env = ReseedWrapper(env, [s[i]])
    #     env = ActionBonus(env)
    #     obs = env.reset()  # 'image', 'direction', 'mission', image \in 7x7x3
    #     ax.imshow(env.render("rgb_array"))
    #     ax.axis('off')
    #     plt.savefig('4R_{}.png'.format(s[i]))

    # actions = [env.actions.forward] * 5
    # i = 0
    # while True and i < len(actions):
    #     frames.append(np.moveaxis(env.render("rgb_array"), 2, 0))
    #     action = actions[i]
    #     obs, reward, done, _ = env.step(action)
    #     print("reward: {}".format(reward))
    #     print("done: {}".format(done))
    #     i += 1
    # frames.append(np.moveaxis(env.render("rgb_array"), 2, 0))
    # from array2gif import write_gif
    # write_gif(np.array(frames), "test.gif", fps=1)
