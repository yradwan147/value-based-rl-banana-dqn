"""Q-network for the Banana navigation DQN agent."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """3-layer MLP. Input is the 37-dim ray-cast state; output is the
    Q-value for each of the four actions."""
    def __init__(self, state_size=37, action_size=4,
                 fc1=128, fc2=64, seed=27):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingQNetwork(nn.Module):
    """Separate value- and advantage- streams (Wang et al. 2016)."""
    def __init__(self, state_size=37, action_size=4,
                 fc1=128, fc2=64, seed=27):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.value     = nn.Linear(fc2, 1)
        self.advantage = nn.Linear(fc2, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.value(x)
        a = self.advantage(x)
        return v + (a - a.mean(dim=1, keepdim=True))
