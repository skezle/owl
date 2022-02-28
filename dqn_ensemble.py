import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_params

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GaussianNoiseCNNNetwork(nn.Module):
    # This is Nix and Weigand DQN network
    # Duelling Network DQN
    def __init__(self, input_dim, output_dim, num_tasks=1, hidden_size=64,
                 bn=False, big_head=False, p=0.0):
        super(GaussianNoiseCNNNetwork, self).__init__()
        self.big_head = big_head
        n = input_dim[0]
        m = input_dim[1]
        self.embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * hidden_size

        class Flatten(nn.Module):
            def forward(self, x):
                return x.reshape(x.size(0), -1)

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
        self.logvar = nn.ModuleDict()
        self.values_last = nn.ModuleDict()

        if self.big_head:
            for i in range(num_tasks):
                self.last[str(i)] = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                                  nn.ReLU(),
                                                  nn.Linear(hidden_size, output_dim))
                self.values_last[str(i)] = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                                         nn.ReLU(),
                                                         nn.Linear(hidden_size, 1))
                self.logvar[str(i)] = nn.Linear(hidden_size, output_dim)
        else:
            for i in range(num_tasks):
                self.last[str(i)] = nn.Linear(hidden_size, output_dim)  # output dim is the action space
                self.values_last[str(i)] = nn.Linear(hidden_size, 1)
                self.logvar[str(i)] = nn.Linear(hidden_size, output_dim)

        # Initialize parameters correctly
        self.apply(init_params)

        # Normalise the logvar
        self.max_logvar = torch.nn.Parameter(torch.full((1,), 0.0, device=device))
        self.max_logvar.requires_grad = True
        self.min_logvar = torch.nn.Parameter(torch.full((1,), -10.0, device=device))
        self.min_logvar.requires_grad = True

    def forward(self, x, task_idx=0):
        task_idx = int(task_idx)
        x = x.transpose(1, 3).transpose(2, 3)
        #print(summary(self.first, x.shape[1:]))
        x = self.first(x)
        logvar = self.logvar[str(task_idx)](x)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        advantage_stream = self.last[str(task_idx)](x)
        value_stream = self.values_last[str(task_idx)](x)
        q_vals = value_stream + advantage_stream - advantage_stream.mean(1, keepdim=True)
        return q_vals, logvar

    def set_task(self, task_idx):
        self.task_idx = task_idx

class DQNEnsemble(nn.Module):
    pass
