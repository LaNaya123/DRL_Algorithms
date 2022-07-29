# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def safe_mean(arr):
     return np.nan if len(arr) == 0 else round(np.mean(arr), 2)

def swap_and_flatten(arr):
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
    
def obs_to_tensor(x):
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    return torch.from_numpy(x).float()

def clip_grad_norm_(module, max_grad_norm):
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)

def compute_gae_advantage(rewards, values, dones, last_value, gamma=0.95, gae_lambda=0.95):
    advantages = []
    gae_advantage = 0

    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_value = last_value
        else:
            next_value = values[i+1]
        delta = rewards[i] + gamma * (1 - dones[i]) * next_value - values[i]
        gae_advantage = delta + gamma * gae_lambda * (1 - dones[i]) * gae_advantage
        advantages.append(gae_advantage)
    return torch.FloatTensor(advantages[::-1]).view(-1, 1)

def compute_td_target(rewards, dones, last_value, gamma=0.95):
    target_values = []
    td_target = last_value
    for i in reversed(range(len(rewards))):
        td_target = rewards[i] + gamma * (1 - dones[i]) * td_target
        target_values.append(td_target)
    return torch.FloatTensor(target_values[::-1]).view(-1, 1)
    
class Mish(nn.Module):
    def __init__(self): 
        super().__init__()
        
    def forward(self, x): 
        return self._mish(x)
    
    def _mish(self, x):
        return x * torch.tanh(F.softplus(x))
    
class OrnsteinUhlenbeckNoise():
    def __init__(self, mu, sigma=0.1, theta=0.1, dt=0.01):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x_prev = np.zeros_like(self.mu)
        
    def reset(self):
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x