import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean_layer = nn.Linear(128, action_size)
        self.std_layer = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        std = F.softplus(self.std_layer(x))  
        return mean, std

class Value(nn.Module):
    def __init__(self, state_size):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

def compute_returns(rewards, dones, gamma=0.99):
    returns = []
    R = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            R = 0
        R = reward + gamma * R
        returns.insert(0, R)
    return returns

def trainppo(policy, value, optimizer, trajectories, gamma=0.99, epsilon=0.2, entropy_coef=0.01):
    states, actions, rewards, log_probs, dones, next_states = zip(*trajectories)


    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    old_log_probs = torch.tensor(log_probs, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)


    returns = compute_returns(rewards, dones, gamma)
    returns = torch.tensor(returns, dtype=torch.float32)
    values = value(states).squeeze()
    advantages = (returns - values).detach()


    means, stds = policy(states)
    dist = torch.distributions.Normal(means, stds)
    new_log_probs = dist.log_prob(actions).sum(dim=1)
    ratios = torch.exp(new_log_probs - old_log_probs)

    surrogate1 = ratios * advantages
    surrogate2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surrogate1, surrogate2).mean()


    entropy = dist.entropy().mean()
    policy_loss -= entropy_coef * entropy


    value_loss = F.mse_loss(values, returns)

    optimizer.zero_grad()
    loss = policy_loss + value_loss
    loss.backward()
    optimizer.step()

def generate_trajectories(policy, env, num_trajectories):
    trajectories = []
    for _ in range(num_trajectories):
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                mean, std = policy(torch.tensor(state, dtype=torch.float32))
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample().numpy()
                log_prob = dist.log_prob(torch.tensor(action, dtype=torch.float32)).sum().item()
            next_state, reward, done, _ = env.step(action)
            trajectories.append((state, action, reward, log_prob, done, next_state))
            state = next_state
    return trajectories
