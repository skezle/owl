import copy
import pdb
import time
import math
import itertools
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
#from torchsummary import summary
from utils import ReplayPool, Transition, init_params, GaussianMSELoss
from dqn_ensemble import GaussianNoiseCNNNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_device(tensor, ipu=False):
    if ipu:
        return tensor
    else:
        return tensor.to(device)

eps = 1e-10


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)

class CNNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, num_tasks=1, hidden_size=64, bn=False, duelling=False, big_head=False, p=0.0):
        super(CNNNetwork, self).__init__()
        self.duelling = duelling
        self.big_head = big_head
        n = input_dim[0]
        m = input_dim[1]
        self.embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * hidden_size

        if bn:
            self.first = nn.Sequential(  # in [256, 3, 7, 7]
                nn.Conv2d(3, 16, (2, 2)), # out [-1, 16, 6, 6]
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)), # out [-1, 16, 3, 3]
                nn.Conv2d(16, 32, (2, 2)), # [-1, 32, 2, 2]
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)), # [-1, 64, 1, 1]
                nn.BatchNorm2d(64),
                nn.ReLU(),
                Flatten(), # [-1, 64]
                nn.Linear(64, hidden_size),
                nn.ReLU(),
            )
        else:
            self.first = nn.Sequential(  # in [256, 3, 7, 7]
                nn.Conv2d(3, 16, (2, 2)),  # out [-1, 16, 6, 6]
                nn.ReLU(),
                nn.Dropout(p=p),
                nn.MaxPool2d((2, 2)),  # out [-1, 16, 3, 3]
                nn.Conv2d(16, 32, (2, 2)),  # [-1, 32, 2, 2]
                nn.ReLU(),
                nn.Dropout(p=p),
                nn.Conv2d(32, 64, (2, 2)),  # [-1, 64, 1, 1]
                nn.ReLU(),
                nn.Dropout(p=p),
                Flatten(),  # [-1, 64]
                nn.Linear(64, hidden_size),
                nn.ReLU(),
                nn.Dropout(p=p),
            )
        # If dueling network have an advantage stream and a value stream
        self.last = nn.ModuleDict()
        if self.duelling:
            self.values_last = nn.ModuleDict()

        if self.big_head:
            for i in range(num_tasks):
                self.last[str(i)] = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                                  nn.ReLU(),
                                                  nn.Linear(hidden_size, output_dim))
                if self.duelling:
                    self.values_last[str(i)] = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                                             nn.ReLU(),
                                                             nn.Linear(hidden_size, 1))
        else:
            for i in range(num_tasks):
                self.last[str(i)] = nn.Linear(hidden_size, output_dim)  # output dim is the action space
                if self.duelling:
                    self.values_last[str(i)] = nn.Linear(hidden_size, 1)

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, x, task_idx=0):
        task_idx = int(task_idx)
        x = x.transpose(1, 3).transpose(2, 3)
        #print(summary(self.first, x.shape[1:]))
        x = self.first(x)
        if self.duelling:
            advantage_stream = self.last[str(task_idx)](x)
            value_stream = self.values_last[str(task_idx)](x)
            q_vals = value_stream + advantage_stream - advantage_stream.mean(1, keepdim=True)
        else:
            q_vals = self.last[str(task_idx)](x)
        return q_vals

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, lr, num_tasks=1, hidden_size=256, q_ewc_reg=0.0, q_func_reg=0.0, bn=False,
                 duelling=False, big_head=False, huber=False, p=0.0, uncert=False):
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.num_tasks = num_tasks
        self.huber = huber
        self.uncert = uncert

        if self.uncert:
            self.network = GaussianNoiseCNNNetwork(state_dim, action_dim, self.num_tasks, hidden_size, bn=bn, big_head=big_head, p=p)
        else:
            self.network = CNNNetwork(state_dim, action_dim, self.num_tasks, hidden_size, bn=bn, duelling=duelling, big_head=big_head, p=p)
        for param in self.network.parameters():
            param.requires_grad = True

        self.task_idx = 0
        self.q_ewc_reg = q_ewc_reg
        self.q_func_reg = q_func_reg
        self.f_reg_term = None
        self.regularization_terms = {}
        self.heads = ["{0}.{1}.{2}".format("last", a, b) for a, b in itertools.product(range(self.num_tasks), ['weight', 'bias'])] + \
                     ["{0}.{1}.{2}".format("values_last", a, b) for a, b in itertools.product(range(self.num_tasks), ['weight', 'bias'])]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if self.uncert:
            self.params = {n: p for n, p in self.named_parameters() if n not in ['min_logvar', 'max_logvar']}
        else:
            self.params = {n: p for n, p in self.named_parameters()}

    def forward(self, x, argmax=True, head=None):
        idx = self.task_idx if head is None else head
        if self.uncert:
            x, logvar = self.network.forward(x, int(idx))
        else:
            x = self.network.forward(x, int(idx))
            logvar = None
        mean = F.softmax(x, dim=1)
        if argmax:
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action = mean.max(1, keepdim=True)[1] # returns argmax i.e. the action
        else:
            prob_dist = torch.distributions.Categorical(mean)  # probs should be of size batch x classes
            action = prob_dist.sample()
        return action, mean, logvar

    def init_f_reg(self):
        # Save a copy to compute distillation outputs
        self.prev_model = copy.deepcopy(self)
        self.prev_model.to(device)

    def func_reg(self, state_batch):
        reg_loss = 0
        # self.f_reg_term - is not None due to being initialised by the init_f_reg function
        if self.f_reg_term:
            current_head = self.task_idx
            previous_head = (current_head - 1) % self.num_tasks
            _, prev_state_vals, _ = self.prev_model.forward(state_batch, head=previous_head)
            _, state_vals, _ = self.forward(state_batch, head=previous_head)  # vector over actions!
            reg_loss = self.q_func_reg * ((state_vals - prev_state_vals) ** 2).mean() # distillation loss
        return reg_loss

    def ewc_reg(self):
        reg_loss = 0
        if len(self.regularization_terms) > 0 and self.q_ewc_reg > 0:
            # diff items are diff task regs.
            for i, reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term['importance']
                task_param = reg_term['task_param']
                for n, p in self.params.items():
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
                reg_loss += task_reg_loss
        return self.q_ewc_reg * reg_loss

    def calculate_importance(self, state, action, value_target):
        # Backup the weight of current task
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()

        # Calculate the importance of weights for current task
        # Update the diagonal Fisher information
        # Initialize the importance matrix
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(0)  # zero initialized

        self.eval()
        # Accumulate the square of gradients
        # Batch size  = 1
        N = list(state.size())[0]
        task_heads = ["{0}.{1}.{2}".format(a, int(self.task_idx), b) for a, b in itertools.product(['logvar', 'out'], ['weight', 'bias'])]
        for i in range(N):
            s = to_device(state[i]) # state is an image!
            a = to_device(action[i].long().view(1, -1))
            v = to_device(value_target[i, :].view(1, -1))
            # Use groundtruth label, hence this is the empirical FI
            _, action_prob, logvar = self.forward(s.unsqueeze(0))
            state_action_value = action_prob.gather(1, a)
            if self.huber:
                loss = F.smooth_l1_loss(state_action_value, v)
            else: # mu, logvar, target, logvar_loss
                if self.uncert:
                    logvar_action_value = logvar.gather(1, a)
                    loss = GaussianMSELoss(state_action_value, logvar_action_value, v, logvar_loss=True)
                    loss += 0.01 * (self.network.max_logvar.sum() - self.network.min_logvar.sum())
                else:
                    loss = GaussianMSELoss(state_action_value, logvar_action_value, v, logvar_loss=False)
            self.optimizer.zero_grad()
            loss.backward()
            for n, p in importance.items():
                if self.params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
                    # pick specific head, "exclusive AND" logic used
                    if (str(n) in self.heads) == (str(n) in task_heads):
                        p += ((self.params[n].grad ** 2) * 1 / N)

        # Save the weight and importance of weights of current task
        self.regularization_terms[int(self.task_idx)] = {'importance': importance, 'task_param': task_param}
        self.train()

    def set_task(self, task_idx):
        self.task_idx = int(task_idx)

class DQNAgent:
    def __init__(self, seed, state_dim, action_dim, num_tasks, lr=3e-4, gamma=0.999, batchsize=256, hidden_size=56,
                 q_ewc_reg=0.0, ddqn=False, bn=False, duelling=False, pool_size=1e6, target_update=10, eps_end=0.05,
                 eps_decay=2000, huber=False, big_head=False, eps_gr_warm_start=False, eps_gr_warm_start_decay_rate=2,
                 dropout_prob=0.0, q_func_reg=0.0, data_var=1, uncert=False):

        self.gamma = gamma
        self.action_dim = action_dim
        self.batchsize = batchsize
        self.num_tasks = num_tasks
        self.task_idx = 0
        self.tasks = {i: 0 for i in range(num_tasks)}
        self.tasks[self.task_idx] += 1
        self.eps_start = 0.9
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.ddqn = ddqn
        self.bn = bn
        self.huber = huber
        self.eps_gr_warm_start = eps_gr_warm_start
        self.eps_gr_warm_start_decay_rate = eps_gr_warm_start_decay_rate
        self.ewc = True if q_ewc_reg >= 0 and q_func_reg == 0 else False # can be 0 for ablation purposes.
        self.func_reg = True if q_func_reg > 0 else False
        self.dropout_prob = dropout_prob
        self.data_var = data_var # Gaussian noise var
        self.uncert = uncert
        self.q_ewc_reg = q_ewc_reg
        self.q_func_reg = q_func_reg

        torch.manual_seed(seed)
        self.policy_net = Policy(state_dim, action_dim, lr, num_tasks, hidden_size=hidden_size, q_ewc_reg=q_ewc_reg, q_func_reg=q_func_reg, bn=bn, duelling=duelling,
                                 big_head=big_head, huber=huber, p=dropout_prob, uncert=self.uncert).to(device)

        self.target_net = Policy(state_dim, action_dim, lr, num_tasks, hidden_size=hidden_size, q_ewc_reg=q_ewc_reg, q_func_reg=q_func_reg, bn=bn, duelling=duelling,
                                 big_head=big_head, huber=huber, p=dropout_prob, uncert=self.uncert).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        self.replay_pool = ReplayPool(capacity=int(pool_size))
        self.steps_done = [0] * num_tasks
        self.eps_threshold = self.eps_start # init, will anneal in the get action function

    def eval(self):
        self.policy_net.eval()

    def train(self):
        self.policy_net.train()

    def get_action(self, state, state_filter=None, deterministic=False, eval=False):
        if eval:
            thres = 0
        else:
            if self.eps_gr_warm_start:
                if self.eps_gr_warm_start_decay_rate == 1:
                    decay = self.eps_decay
                elif self.eps_gr_warm_start_decay_rate > 1:
                    decay = self.eps_decay / self.tasks[int(self.task_idx)]
                self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done[int(self.task_idx)] / decay)
            else:
                self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done[int(self.task_idx)] / self.eps_decay)
            thres = self.eps_threshold
            self.steps_done[int(self.task_idx)] += 1

        sample = random.random()
        if sample > thres:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action, _, _ = self.policy_net(to_device(torch.Tensor(state).unsqueeze(0)), argmax=True) # if deterministic - argmax, else - sample
                return action
        else:
            return torch.tensor([[random.randrange(self.action_dim)]], device=device, dtype=torch.long)

    def optimize(self, n_updates, i_episode):
        q_loss = 0
        timings = {}
        # for i, p in enumerate(self.target_net.named_parameters()):
        #     print("{0}: {1}, {2}".format(i, p[0], p[1].shape))
        for i in range(n_updates):
            start = time.time()
            samples = self.replay_pool.sample(self.batchsize)

            state_batch = to_device(torch.FloatTensor(samples.state))
            nextstate_batch = to_device(torch.FloatTensor(samples.nextstate))
            # getting Type errors since some of the consituents of the replay buffer are
            # not numpy arrays of size (1, ) but of size (), better solution would be to
            # check for correct inputs when putting items into the replay buffer.
            try:
                action_batch = torch.tensor(samples.action, device=device, dtype=torch.long).unsqueeze(1)
            except TypeError or AttributeError:
                action_batch = torch.tensor(tuple(int(s) for s in samples.action), device=device, dtype=torch.long).unsqueeze(1)
            reward_batch = to_device(torch.FloatTensor(samples.reward).unsqueeze(1))
            done_batch = to_device(torch.FloatTensor(samples.done).unsqueeze(1))
            if 'batch' in timings:
                timings['batch'] += time.time() - start
            else:
                timings['batch'] = time.time() - start
            start = time.time()
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            _, action_probs, logvar = self.policy_net(state_batch)
            state_action_values = action_probs.gather(1, action_batch)
            if self.uncert:
                logvar_action_value = logvar.gather(1, action_batch)
            else:
                logvar_action_value = None
            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            with torch.no_grad():
                if self.ddqn:
                    next_actions_batch, _, _ = self.policy_net(nextstate_batch, argmax=True)
                    _, next_actions_probs, _ = self.target_net(nextstate_batch)
                    q_target = next_actions_probs.gather(1, next_actions_batch)
                else:
                    q_target, _, _ = self.target_net(nextstate_batch, argmax=True)
                value_target = reward_batch + (1.0 - done_batch) * self.gamma * q_target.detach()

            if 'fwd' in timings:
                timings['fwd'] += time.time() - start
            else:
                timings['fwd'] = time.time() - start
            start = time.time()

            if self.huber:
                loss = F.smooth_l1_loss(state_action_values, value_target)
            else: # mu, logvar, target, logvar_loss=True
                loss = GaussianMSELoss(state_action_values, logvar_action_value, value_target, logvar_loss=self.uncert)
                if self.uncert:
                    loss += 0.01 * (self.policy_net.network.max_logvar.sum() - self.policy_net.network.min_logvar.sum())

            if self.ewc:
                loss += self.policy_net.ewc_reg()
            elif self.func_reg:
                loss += self.policy_net.func_reg(state_batch)

            # Optimize the model
            self.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
            self.policy_optimizer.step()
            q_loss += loss.detach().item()

            if 'q_func_opt' in timings:
                timings['q_func_opt'] += time.time() - start
            else:
                timings['q_func_opt'] = time.time() - start
            start = time.time()
            if i_episode % self.target_update == 0:
                with torch.no_grad():
                    self.target_net.load_state_dict(self.policy_net.state_dict(), strict=False)

            if 'target_load' in timings:
                timings['target_load'] += time.time() - start
            else:
                timings['target_load'] = time.time() - start

        return q_loss, self.eps_threshold, logvar_action_value, {k: v/n_updates for k, v in timings.items()}

    def set_task(self, task_idx, q_reg=False):
        return None

    def get_task(self):
        return None

class DQNAgentOWL(DQNAgent):
    def __init__(self, seed, state_dim, action_dim, num_tasks, lr=3e-4, gamma=0.999, batchsize=256, hidden_size=56,
                 q_ewc_reg=0.0, ddqn=False, bn=False, duelling=False, pool_size=1e6, target_update=10, eps_end=0.05,
                 eps_decay=2000, huber=False, n_fisher_sample=60000, big_head=False, eps_gr_warm_start=False,
                 eps_gr_warm_start_decay_rate=2, dropout_prob=0.0, q_func_reg=0.0, uncert=False):

        super(DQNAgentOWL, self).__init__(seed=seed, state_dim=state_dim, action_dim=action_dim,
                 num_tasks=num_tasks, lr=lr, gamma=gamma, batchsize=batchsize, hidden_size=hidden_size,
                 q_ewc_reg=q_ewc_reg, ddqn=ddqn, bn=bn, duelling=duelling, pool_size=pool_size,
                 target_update=target_update, eps_end=eps_end, eps_decay=eps_decay, huber=huber, big_head=big_head,
                 eps_gr_warm_start=eps_gr_warm_start, eps_gr_warm_start_decay_rate=eps_gr_warm_start_decay_rate,
                 dropout_prob=dropout_prob, q_func_reg=q_func_reg, uncert=uncert)

        self.n_fisher_sample = n_fisher_sample

    def get_sample(self, sample_size):
        # gets sample from buffer and gets value of next state from Q-target
        if len(self.replay_pool) > sample_size:
            data = self.replay_pool.sample(sample_size)
        else:
            data = self.replay_pool.get_all()

        state_all = to_device(torch.FloatTensor(data.state))
        nextstate_all = to_device(torch.FloatTensor(data.nextstate))
        try:
            action_all = to_device(torch.FloatTensor(data.action))
        except TypeError:
            action_all = to_device(torch.FloatTensor(tuple(s.reshape(-1) for s in data.action)))
        rewards_all = to_device(torch.reshape(torch.FloatTensor(data.reward), (-1, 1)))
        assert len(rewards_all.shape) == 2
        done_all = to_device(torch.FloatTensor(data.done).unsqueeze(1))

        # Target is the outputs
        with torch.no_grad():
            if self.ddqn:
                next_actions_batch, _, _ = self.policy_net(nextstate_all, argmax=True)
                _, next_actions_probs, _ = self.target_net(nextstate_all)
                q_target = next_actions_probs.gather(1, next_actions_batch)
            else:
                q_target, _, _ = self.target_net(nextstate_all, argmax=True)
            value_target = rewards_all + (1.0 - done_all) * self.gamma * q_target.detach()

        return state_all, action_all, value_target

    def set_task(self, task_idx: int, q_reg=False, memory=None):
        # Changing heads
        self.task_idx = int(task_idx)
        # Performing the EWC update - adding new reg terms to the loss function
        assert self.ewc != self.func_reg
        if q_reg:
            if self.ewc:
                print("Performing Q-EWC update")
                state, action, value_target = self.get_sample(self.n_fisher_sample)
                if self.q_ewc_reg > 0:
                    self.policy_net.calculate_importance(state, action, value_target)
                self.tasks[self.task_idx] += 1
                if self.eps_gr_warm_start:
                    self.steps_done[self.task_idx] = 0 # if we are warm starting the eps greedy strategy let's reset the number fo steps taken to 0
            elif self.func_reg:
                print("Performing functional update")
                self.policy_net.init_f_reg()

        self.policy_net.set_task(int(self.task_idx))
        self.target_net.set_task(int(self.task_idx))

    def get_task(self) -> int:
        return self.task_idx


class MultiTaskDQN(DQNAgent):
    def __init__(self, seed, state_dim, action_dim, num_tasks, lr=3e-4, gamma=0.999, batchsize=256, hidden_size=56,
                 bn=False, ddqn=False, duelling=False, pool_size=1e6, target_update=10, eps_end=0.05,
                 eps_decay=2000, huber=False):

        super(MultiTaskDQN, self).__init__(seed=seed, state_dim=state_dim, action_dim=action_dim,
                 num_tasks=num_tasks, lr=lr, gamma=gamma, batchsize=batchsize, hidden_size=hidden_size,
                 ddqn=ddqn, bn=bn, duelling=duelling, pool_size=pool_size,
                 target_update=target_update, eps_end=eps_end, eps_decay=eps_decay, huber=huber)

        self.replay_pool = {i: ReplayPool(capacity=int(pool_size)) for i in range(num_tasks)}

    def optimize(self, n_updates, i_episode):
        q_loss = 0
        timings = {}

        # Choose which head to use from tasks which have been already seen
        rand = np.random.uniform()
        previous_task_idx = self.task_idx
        if rand > 0.25:
            idx = previous_task_idx
        else:
            other_policies = list(range(max([k for k, v in self.tasks.items() if v > 0]) + 1))
            # for Task 0 we will just sample Head 0 100% of the time.
            # for Task > 0 we will sample heads other than self.task_idx
            if self.task_idx > 0:
                other_policies.remove(previous_task_idx)
            idx = int(np.random.choice(other_policies))
        self.set_task(idx)

        for i in range(n_updates):
            samples = self.replay_pool[idx].sample(self.batchsize)

            state_batch = to_device(torch.FloatTensor(samples.state))
            nextstate_batch = to_device(torch.FloatTensor(samples.nextstate))
            try:
                action_batch = torch.tensor(samples.action, device=device, dtype=torch.long).unsqueeze(1)
            except TypeError or AttributeError:
                action_batch = torch.tensor(tuple(int(s) for s in samples.action), device=device,
                                            dtype=torch.long).unsqueeze(1)
            reward_batch = to_device(torch.FloatTensor(samples.reward).unsqueeze(1))
            done_batch = to_device(torch.FloatTensor(samples.done).unsqueeze(1))

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            _, action_probs, logvar = self.policy_net(state_batch)
            state_action_values = action_probs.gather(1, action_batch)

            logvar_action_value = None
            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            with torch.no_grad():
                if self.ddqn:
                    next_actions_batch, _, _ = self.policy_net(nextstate_batch, argmax=True)
                    _, next_actions_probs, _ = self.target_net(nextstate_batch)
                    q_target = next_actions_probs.gather(1, next_actions_batch)
                else:
                    q_target, _, _ = self.target_net(nextstate_batch, argmax=True)
                value_target = reward_batch + (1.0 - done_batch) * self.gamma * q_target.detach()

            if self.huber:
                loss = F.smooth_l1_loss(state_action_values, value_target)
            else:  # mu, logvar, target, logvar_loss=True
                loss = GaussianMSELoss(state_action_values, logvar_action_value, value_target, logvar_loss=False)

            # Optimize the model
            self.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
            self.policy_optimizer.step()
            q_loss += loss.detach().item()

            if i_episode % self.target_update == 0:
                with torch.no_grad():
                    self.target_net.load_state_dict(self.policy_net.state_dict(), strict=False)

        # Resetting the task idx to what it was before sampling
        self.set_task(previous_task_idx)

        # for TB logging
        all_eps_thresholds = {
            i: self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done[i] / self.eps_decay) for
            i in range(self.num_tasks)
        }

        return q_loss, all_eps_thresholds, logvar_action_value, {k: v / n_updates for k, v in timings.items()}

    def set_task(self, task_idx: int, q_reg=False, update=False):
        self.task_idx = task_idx
        self.policy_net.set_task(self.task_idx)
        self.target_net.set_task(self.task_idx)
        if update:
            self.tasks[self.task_idx] += 1

    def get_task(self) -> int:
        return int(self.task_idx)


