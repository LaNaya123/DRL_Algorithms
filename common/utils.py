# -*- coding: utf-8 -*-
from typing import Union, Tuple
from common.envs import Monitor, VecEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym.spaces as spaces

def safe_mean(arr: np.ndarray) -> float:
    return np.nan if len(arr) == 0 else round(np.mean(arr), 2)

def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
    shape = arr.shape
    if len(shape) < 3:
        shape = shape + (1,)
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
    
def obs_to_tensor(x: np.ndarray) -> torch.Tensor:
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    return torch.from_numpy(x).float()

def clip_grad_norm_(module: nn.Module, max_grad_norm: float) -> None:
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)

def compute_gae_advantage(rewards: torch.Tensor, 
                          values: torch.Tensor, 
                          dones: torch.Tensor, 
                          last_value: Union[int, torch.Tensor], 
                          gamma: float = 0.95, 
                          gae_lambda: float = 0.95
                         ) -> torch.Tensor:
    
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

def compute_td_target(rewards: torch.Tensor, 
                      dones: torch.Tensor, 
                      last_value: Union[int, torch.Tensor], 
                      gamma: float = 0.95
                     ) -> torch.Tensor:
    
    target_values = []
    td_target = last_value
    for i in reversed(range(len(rewards))):
        td_target = rewards[i] + gamma * (1 - dones[i]) * td_target
        target_values.append(td_target)
    return torch.FloatTensor(target_values[::-1]).view(-1, 1)

def compute_rtg(rewards: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    rtg = np.zeros_like(rewards)
    rtg[-1] = rewards[-1]
    
    for i in reversed(range(len(rewards)-1)):
        rtg[i] = rewards[i] + gamma * rtg[i+1]
    return rtg

def evaluate_policy(model: nn.Module, env: Union[Monitor, VecEnv], num_eval_episodes: int = 10) -> Tuple[float, float, float, float]:
    num_envs = env.num_envs

    episode_rewards = []
    episode_lengths = []
    
    episode_counts = np.zeros(num_envs, dtype="int")
    episode_counts_target = np.array([(num_eval_episodes + i) // num_envs for i in range(num_envs)], dtype="int")
    
    current_rewards = np.zeros(num_envs)
    current_lengths = np.zeros(num_envs, dtype="int")
    
    observations = env.reset()
    
    while (episode_counts < episode_counts_target).any():
        if isinstance(observations, int):
            observations = [observations]

        actions = model.predict(obs_to_tensor(observations))
        if isinstance(env, spaces.Box):
            actions = np.clip(actions, env.action_space.low.min(), env.action_space.high.max())
        
        observations, rewards, dones, infos = env.step(actions)

        current_rewards += rewards
        current_lengths += 1
        
        for i in range(num_envs):
            if isinstance(dones, bool):
                if dones:
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
            else:
                if dones[i]:
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    
    mean_rewards = np.mean(episode_rewards)
    std_rewards = np.std(episode_rewards)
        
    mean_lengths = np.mean(episode_lengths)
    std_lengths = np.std(episode_lengths)
    
    return (mean_rewards, std_rewards, mean_lengths, std_lengths)
 
class Mish(nn.Module):
    def __init__(self): 
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self._mish(x)
    
    def _mish(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))
    
class OrnsteinUhlenbeckNoise():
    def __init__(self, mu: np.ndarray, sigma: float = 0.1, theta: float = 0.1, dt: float = 0.01):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x_prev = np.zeros_like(self.mu)
        
    def reset(self) -> None:
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self) -> np.ndarray:
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x