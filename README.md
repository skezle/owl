# Same State, Different Task: Continual Reinforcement Learning without Interference

This code accompanies the submission for [Same State Different Task: CRL without Interference](https://arxiv.org/abs/2106.02940). We show the dangers of using CLEAR/experience replay only as a means for preventing forgetting in CRL. This can occur when we ahve different tasks but the same or very similar states in an environment. We explore the relatively unexplained problem of *interference* in CRL, this is distinct from *forgetting*. We advocate the use of multi-heads together with a regularization of the shared feature extractor across tasks. Although using multiple heads as a means for preventing forgetting is nothing new: **we crucially employ it for preventing interference and modelling the multi-modallity of the tasks**. EWC and distillation are used to prevent forgetting in the shared feature extractor. 

## Quick Start

To run OWL on Minigrid with EWC:

`python train_minigrid.py --env=SC --num_tasks=5 --num_task_repeats=3 --max_task_frames=750000 --tag=owl_t5_l500_s101 --env_seeds 111 129 112 105 155 --exp_replay_capacity=1000000 --huber --owl --q_ewc_reg=500 --seed=101 --bandits --bandit_loss=mse --bandit_debug --bandit_lr=0.88 --bandit_decay=0.9 --bandit_epsilon=0 --bandit_step=1 --log_interval=8000 --buffer_warm_start --buffer_warm_start_size=50000 --logdir=logs`.

To run OWL on Minigrid with Distillation loss:

`python train_minigrid.py --env=SC --num_tasks=5 --num_task_repeats=3 --max_task_frames=750000 --tag=owl_t5_f100_s101 --env_seeds 111 129 112 105 155 --exp_replay_capacity=1000000 --huber --owl --q_func_reg=100 --seed=101 --bandits --bandit_loss=mse --bandit_debug --bandit_lr=0.88 --bandit_decay=0.9 --bandit_epsilon=0 --bandit_step=1 --log_interval=8000 --buffer_warm_start --buffer_warm_start_size=50000 --logdir=logs`

To run the CLEAR agent with DQN base RL algorithm:

`python train_minigrid.py --env=SC --num_tasks=5 --num_task_repeats=3 --max_task_frames=750000 --log_interval=8000 --env_seeds 111 129 112 105 128 --tag=dqn_x5_s101 --exp_replay_capacity=4000000 --seed=105 --huber --logdir=logs`

To run the full rehearsal baseline:

`python train_minigrid.py --env=SC --num_tasks=5 --num_task_repeats=3 --max_task_frames=750000 --log_interval=8000 --env_seeds 111 129 112 105 155 --tag=fr_x5_s101 --exp_replay_capacity=750000 --seed=101 --huber --logdir=logs  --multitask`

## Dependencies

- `gym==0.18.0`
- `gym-minigrid==1.0.2`

## Citation

```
@article{kessler2021same,
  title={Same state, different task: Continual reinforcement learning without interference},
  author={Kessler, Samuel and Parker-Holder, Jack and Ball, Philip and Zohren, Stefan and Roberts, Stephen J},
  journal={arXiv preprint arXiv:2106.02940},
  year={2021},
}
```
