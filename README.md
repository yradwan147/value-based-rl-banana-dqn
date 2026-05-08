# Value-Based RL — Banana Navigator (Double DQN + Dueling)

Final project for Udacity's *Deep Reinforcement Learning Nanodegree*
(nd893) — Project 1: Navigation.

A from-scratch DQN agent that solves Unity ML-Agents' **Banana**
environment: collect yellow bananas (+1 reward) and avoid blue ones
(−1 reward) in a finite-horizon episode. The environment is considered
solved when the rolling 100-episode score crosses **+13**.

## Project files

```
Navigation.ipynb   end-to-end solution notebook (run this)
Report.md          algorithm, hyper-parameters, and ideas for future work
model.py           QNetwork + DuelingQNetwork
agent.py           DQNAgent + ReplayBuffer (Double DQN, optional Dueling)
train.py           headless equivalent of the notebook (CLI)
README.md          you are here
.gitignore
```

The training run produces:

* `banana_qnet.pth` — solved Q-network weights
* `banana_scores.npy` — per-episode score array

## Environment setup

```bash
# 1. Python deps
pip install torch numpy matplotlib unityagents

# 2. Download the Banana environment binary from the project page:
#    https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation
#    (Linux 64-bit, Mac, Windows 64-bit, or headless versions are listed there.)
#    Unzip into the project folder.
```

## Running

### Notebook (recommended for the report)

```bash
jupyter notebook Navigation.ipynb
# Restart & Run All
```

The notebook plots the learning curve and runs one demo episode using
the saved weights at the end.

### CLI (recommended for headless training)

```bash
python3 train.py --env Banana_Linux/Banana.x86_64 --episodes 2000
```

## State / Action / Reward

| Dimension | Spec |
|---|---|
| State  | 37-dim vector (agent velocity + ray-cast features) |
| Action | 4 discrete: forward / backward / left / right |
| Reward | +1 yellow banana, −1 blue banana, 0 otherwise |
| Solve  | rolling 100-episode mean ≥ +13 |

## Algorithm overview

* **Double DQN** for unbiased TD targets.
* **Dueling head** (V + A) for sample-efficient value estimation.
* **Replay buffer** of 100k transitions.
* **Soft target update** (τ = 1e-3) every learning step.
* **ε-greedy** exploration with a 0.995 per-episode decay.

The full hyper-parameter table and architecture diagram are in
[`Report.md`](Report.md).

## Results

The agent solves the environment in roughly **400–550 episodes**
(seed-dependent). After training, the demo cell in the notebook
reliably scores in the high-teens during a single 300-step
evaluation episode.

## License

Educational submission for Udacity nd893. Unity Banana environment
© Unity / Udacity.
