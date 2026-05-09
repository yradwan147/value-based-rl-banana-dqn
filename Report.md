# Project Report — Banana Navigator (Value-Based Deep RL)

## 1. Project setup

The agent operates inside Unity ML-Agents' **Banana** environment.

* **State space** — 37-dimensional vector containing the agent's velocity
  along with ray-cast distances + object types in front of it.
* **Action space** — 4 discrete actions: forward, backward, left, right.
* **Reward** — +1 for collecting a yellow banana, −1 for a blue one.
* **Solve criterion** — average score of **+13** over 100 consecutive
  episodes.

## 2. Learning algorithm

The agent in [`agent.py`](agent.py) uses **Double DQN with a Dueling head**:

* **Experience replay** — `deque(maxlen=100,000)`. Every interaction is
  pushed; mini-batches of 64 are sampled uniformly every 4 environment
  steps.
* **Target network** — soft-updated every learning step
  (`τ = 1e-3 · θ_local + (1 − τ) · θ_target`). This stabilises the
  bootstrapped TD targets and is the central trick of Mnih et al. 2015.
* **Double DQN** — when `double_dqn=True`, the next-state best action is
  selected with the **local** network and *evaluated* with the
  **target** network. This removes the well-known maximisation bias of
  vanilla DQN.
* **Dueling head** — when `dueling=True`, the network splits into a
  scalar value `V(s)` and a per-action advantage `A(s, a)` and recombines
  them as `Q(s, a) = V(s) + (A(s, a) − mean_a A(s, a))`. Concentrates
  capacity on state value when actions are ~equivalent.
* **ε-greedy** — `ε` starts at 1.0, decays multiplicatively by 0.995
  every episode and floors at 0.01.
* **Gradient clipping** — `clip_grad_norm_(1.0)` on the local Q-net is
  always on. Without it the dueling head's value-stream causes early
  loss spikes.

### Q-Network architecture

`DuelingQNetwork` (in [`model.py`](model.py)):

```
Input (37) ── Linear(128) ── ReLU
            ── Linear(64)  ── ReLU
                       ┌─ Linear(1)        → V(s)
                       └─ Linear(4)        → A(s, a)
                                              ⇒ Q(s,a) = V + (A − mean(A))
```

`QNetwork` (the non-dueling variant) replaces the value/advantage split
with a single `Linear(64, 4)` head and is exposed via the constructor
flag `dueling=False`.

### Hyper-parameters

| Parameter | Value | Notes |
|---|---|---|
| Replay buffer | 100,000 | enough for ~250 episodes worth of frames |
| Batch size | 64 | DQN paper default |
| Discount γ | 0.99 | |
| Target update τ | 1e-3 | soft update every learning step |
| Optimizer | Adam | |
| Learning rate | 5e-4 | |
| Update every | 4 env steps | matches Atari DQN cadence |
| ε start / min / decay | 1.0 / 0.01 / 0.995 | per-episode decay |
| Gradient clip | 1.0 (L2 norm) | always on |

## 3. Results

Trained locally on Apple Silicon (M3 Pro, MPS device) against the
macOS Unity Banana binary downloaded from the Udacity-provided S3
URL. **Solved in 505 episodes** (rolling-100 mean = 13.05 ≥ target of
13). Wall-clock total: 18.6 minutes on MPS.

Training-curve highlights (`banana_scores.npy`):

| Episode | Per-episode score | Rolling-100 mean | ε |
|---:|---:|---:|---:|
|  50 |  2.0 |  0.44 | 0.778 |
| 100 |  0.0 |  0.98 | 0.606 |
| 150 |  8.0 |  2.95 | 0.471 |
| 200 |  4.0 |  5.38 | 0.367 |
| 250 | 11.0 |  7.20 | 0.286 |
| 300 | 10.0 |  8.44 | 0.222 |
| 350 |  9.0 |  9.35 | 0.173 |
| 400 | 15.0 | 10.77 | 0.135 |
| 450 | 13.0 | 11.80 | 0.105 |
| 500 | 16.0 | 12.88 | 0.082 |
| **505** | 17.0 | **13.05** *(env solved)* | 0.080 |

Saved artefacts:

* `banana_dqn.pth` — final Q-network weights at solve time
* `banana_scores.npy` — full per-episode score history (505 points)
* `banana_training_curve.png` — rendered learning curve (per-episode
  + rolling-100 + +13 threshold line)
* `Navigation.ipynb` — executable training notebook (mirrors `train.py`)

## 4. Ideas for future work

* **Prioritised experience replay** (Schaul et al. 2016) — sample
  transitions in proportion to their TD error to spend more updates on
  hard examples. The current uniform sampler is the dominant
  performance limit at convergence.
* **Distributional / categorical DQN (C51)** — predict the full
  return distribution rather than its mean. Has been shown to learn
  faster and match a higher final score on the same hardware.
* **Rainbow** — combine prioritised replay, distributional Q,
  multi-step bootstrap, noisy nets, and dueling/double DQN. The
  paper's claim that the combination is more than the sum of its
  parts is worth verifying on Banana, where each piece on its own is
  already implementable in a couple of hundred lines.
* **n-step returns** with `n = 3` — typically gives a 50–70 episode
  speedup at no implementation cost.
* **Hyper-parameter sweep** — the rate of update (`update_every`) and
  the target soft-update τ are not deeply tuned; a small Optuna sweep
  would likely shave a few hundred episodes off the solve.

## 5. Reproducibility

```bash
pip install torch numpy matplotlib unityagents
# Download Banana env per the Udacity project page (Linux/Mac/Win/Headless)
python3 train.py --env Banana_Linux/Banana.x86_64 --episodes 2000
```

`train.py` is the headless equivalent of the notebook — same agent and
hyper-parameters, but writes the checkpoint and scores instead of
plotting them. Either path solves the environment.
