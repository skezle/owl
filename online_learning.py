import numpy as np

class ExpWeights(object):

    def __init__(self,
                 arms=[0, 1],
                 lr=1,
                 window=20,  # we don't use this yet..
                 epsilon=0,  # set this above zero for
                 decay=1,
                 greedy=False):

        self.arms = arms
        self.l = {i: 0 for i in range(len(self.arms))}
        self.p = [0.5] * len(self.arms)
        self.arm = 0
        self.value = self.arms[self.arm]
        self.error_buffer = []
        self.window = window
        self.lr = lr
        self.epsilon = epsilon
        self.decay = decay
        self.greedy = greedy

        self.choices = [self.arm]
        self.data = []

    def sample(self):

        if np.random.uniform() > self.epsilon:
            self.p = [np.exp(x) for x in self.l.values()]
            self.p /= np.sum(self.p)  # normalize to make it a distribution
            try:
                self.arm = np.random.choice(range(0, len(self.p)), p=self.p)
            except ValueError:
                print("loss too large scaling")
                decay = self.lr * 0.1
                self.p = [np.exp(x * decay) for x in self.l.values()]
                self.p /= np.sum(self.p)  # normalize to make it a distribution
                self.arm = np.random.choice(range(0, len(self.p)), p=self.p)
        else:
            self.arm = int(np.random.uniform() * len(self.arms))

        self.value = self.arms[self.arm]
        self.choices.append(self.arm)

        return (self.value)

    def update_dists(self, feedback, norm=1):

        # feedback is a list

        for i in range(len(self.arms)):
            if self.greedy:
                self.l[i] *= self.decay
                self.l[i] += self.lr * feedback[i]
            else:
                self.l[i] *= self.decay
                self.l[i] += self.lr * (feedback[i] / max(np.exp(self.l[i]), 0.0001))

''' 
test

b = ExpWeights()

num_indices = 2 # this can be the number of heads

for _ in range(100):
    print("\* Round: {}".format(_))
    idx = b.sample()
    print("Using: {}".format(idx))
    reward = 10 # this will be the reward
    scores = [np.random.rand() for _ in range(num_indices)]
    print(scores)
    b.update_dists(scores)
'''