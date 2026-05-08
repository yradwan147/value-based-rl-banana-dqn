# Value-Based RL — Banana Navigator (DQN)

Final project for Udacity's *Deep Reinforcement Learning Nanodegree*
(nd893) — Navigation. A from-scratch DQN agent that solves the Unity
ML-Agents Banana environment (collect yellow bananas, +1; avoid blue
bananas, −1; +13 average over 100 consecutive episodes is the
solve criterion).

## Files

```
model.py    QNetwork + DuelingQNetwork
agent.py    DQNAgent + ReplayBuffer (Double DQN by default, with optional dueling head)
train.py    CLI training loop
```

## Architecture

* **Q-network**: 37 → 128 → 64 → 4 (ReLU activations).
* **Dueling head**: separates value V(s) from advantage A(s, a) and
  recombines them as `V + (A − mean(A))`.
* **Double DQN target**: `argmax_a Q_local(s', a)` selects the best
  next action, `Q_target(s', a*)` evaluates it — defuses the
  optimistic bias of vanilla DQN.

## Hyper-parameters

| Param | Value |
|---|---|
| Buffer | 100 000 |
| Batch | 64 |
| Gamma | 0.99 |
| Tau (soft update) | 1e-3 |
| LR (Adam) | 5e-4 |
| Update every | 4 steps |
| ε start / min / decay | 1.0 / 0.01 / 0.995 |

## Running

```bash
# 1. Download the Unity Banana environment per the rubric
#    https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation
pip install torch numpy unityagents
python3 train.py --env path/to/Banana_Linux/Banana.x86_64 --episodes 2000
```

The trained network is saved as `banana_qnet.pth` once the rolling
100-episode average crosses +13. Scores are saved to `banana_scores.npy`
for plotting.

## Standing-out work

* `Double DQN + Dueling` are *both* on by default — both are flips
  away (`double_dqn=False`, `dueling=False`) to ablate.
* Critic gradients clipped to 1.0 — required to keep the early
  episodes stable on the dueling head.
* The agent only learns every 4 environment steps, which is the
  paper's recommended ratio for DQN on continuous-frame envs.

## License

Educational submission for Udacity nd893. Unity Banana environment
© Unity / Udacity.
