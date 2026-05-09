"""DQN agent — Double DQN + (optional) Dueling head + ε-greedy + replay buffer."""
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork, DuelingQNetwork


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


class ReplayBuffer:
    Experience = namedtuple("E", ("state", "action", "reward", "next_state", "done"))

    def __init__(self, capacity=int(1e5), batch_size=64, seed=27):
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        random.seed(seed)

    def add(self, *exp):
        self.memory.append(self.Experience(*exp))

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        s  = torch.from_numpy(np.vstack([e.state      for e in batch])).float().to(DEVICE)
        a  = torch.from_numpy(np.vstack([e.action     for e in batch])).long().to(DEVICE)
        r  = torch.from_numpy(np.vstack([e.reward     for e in batch])).float().to(DEVICE)
        ns = torch.from_numpy(np.vstack([e.next_state for e in batch])).float().to(DEVICE)
        d  = torch.from_numpy(np.vstack([e.done       for e in batch]).astype(np.uint8)).float().to(DEVICE)
        return s, a, r, ns, d

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, state_size=37, action_size=4,
                 lr=5e-4, gamma=0.99, tau=1e-3,
                 buffer_size=int(1e5), batch_size=64,
                 update_every=4, dueling=True, double_dqn=True):
        Net = DuelingQNetwork if dueling else QNetwork
        self.qnet_local  = Net(state_size, action_size).to(DEVICE)
        self.qnet_target = Net(state_size, action_size).to(DEVICE)
        self.optimizer   = optim.Adam(self.qnet_local.parameters(), lr=lr)

        self.memory       = ReplayBuffer(buffer_size, batch_size)
        self.batch_size   = batch_size
        self.gamma        = gamma
        self.tau          = tau
        self.update_every = update_every
        self.action_size  = action_size
        self.double_dqn   = double_dqn
        self.t_step       = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) >= self.batch_size:
            self._learn(self.memory.sample())

    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.qnet_local.eval()
        with torch.no_grad():
            q = self.qnet_local(state)
        self.qnet_local.train()
        if random.random() > eps:
            return int(np.argmax(q.cpu().numpy()))
        return random.randrange(self.action_size)

    def _learn(self, batch):
        s, a, r, ns, d = batch
        if self.double_dqn:
            best_actions = self.qnet_local(ns).argmax(dim=1, keepdim=True)
            q_next = self.qnet_target(ns).gather(1, best_actions)
        else:
            q_next = self.qnet_target(ns).max(dim=1, keepdim=True)[0]
        q_target = r + self.gamma * q_next * (1 - d)
        q_expected = self.qnet_local(s).gather(1, a)
        loss = F.mse_loss(q_expected, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet_local.parameters(), 1.0)
        self.optimizer.step()
        self._soft_update()

    def _soft_update(self):
        for s, t in zip(self.qnet_local.parameters(), self.qnet_target.parameters()):
            t.data.copy_(self.tau * s.data + (1.0 - self.tau) * t.data)
