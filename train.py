"""Train a DQN agent on the Unity Banana navigation environment.

Usage:
    python3 train.py --env Banana_Linux/Banana.x86_64 --episodes 2000
"""
import argparse
from collections import deque

import numpy as np
import torch
from unityagents import UnityEnvironment

from agent import DQNAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--target",   type=float, default=13.0)
    ap.add_argument("--save", default="banana")
    args = ap.parse_args()

    env = UnityEnvironment(file_name=args.env)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    info = env.reset(train_mode=True)[brain_name]
    state_size = len(info.vector_observations[0])
    action_size = brain.vector_action_space_size
    print(f"state_size={state_size}, action_size={action_size}")

    agent = DQNAgent(state_size, action_size)
    scores_window = deque(maxlen=100)
    all_scores = []
    eps, eps_min, eps_decay = 1.0, 0.01, 0.995

    for ep in range(1, args.episodes + 1):
        info  = env.reset(train_mode=True)[brain_name]
        state = info.vector_observations[0]
        score = 0.0
        while True:
            action = agent.act(state, eps)
            info = env.step(action)[brain_name]
            ns, r, done = info.vector_observations[0], info.rewards[0], info.local_done[0]
            agent.step(state, action, r, ns, int(done))
            state, score = ns, score + r
            if done:
                break
        eps = max(eps_min, eps_decay * eps)
        scores_window.append(score)
        all_scores.append(score)
        avg = np.mean(scores_window)
        print(f"Ep {ep:4d}\tscore={score:.1f}\trolling100={avg:.2f}\teps={eps:.3f}", end="\r")
        if ep % 100 == 0:
            print(f"\nEpisode {ep}: rolling-100 mean = {avg:.2f}")
        if avg >= args.target and len(scores_window) == 100:
            print(f"\nSolved in {ep} episodes — saving checkpoint")
            torch.save(agent.qnet_local.state_dict(), f"{args.save}_qnet.pth")
            break

    env.close()
    np.save(f"{args.save}_scores.npy", np.array(all_scores))


if __name__ == "__main__":
    main()
