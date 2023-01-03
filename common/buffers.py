# -*- coding: utf-8 -*-
from typing import Union, Tuple
import torch 
import torch.multiprocessing as mp
import numpy as np
import random
from collections import deque
from common.utils.utils import swap_and_flatten

class RolloutBuffer():
    def __init__(self, buffer_size: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        
        self.buffer = deque(maxlen=buffer_size)
        
    def reset(self) -> None:
        self.buffer.clear()
        
    def add(self, transition: Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray], np.ndarray, Union[bool, np.ndarray]]) -> None:
        self.buffer.append(transition)
        
    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs, actions, rewards, next_obs, dones = [], [], [], [], []
        
        for ob, action, reward, next_ob, done in self.buffer:
            
            if len(ob.shape) == 1:
                ob = ob[np.newaxis, :]
            if len(action.shape) == 1:
                action = action[np.newaxis, :]
            if not isinstance(reward, np.ndarray):
                reward = [reward]
            if len(next_ob.shape) == 1:
                next_ob = next_ob[np.newaxis, :]
            if not isinstance(done, np.ndarray):
                done = [done]
                
            obs.append(ob)
            actions.append(action)
            rewards.append(reward)
            next_obs.append(next_ob)
            dones.append(done)
        
        obs = torch.FloatTensor(swap_and_flatten(np.asarray(obs))).to(self.device)
        actions = torch.FloatTensor(swap_and_flatten(np.asarray(actions))).to(self.device)
        rewards = torch.FloatTensor(swap_and_flatten(np.asarray(rewards))).to(self.device)
        next_obs = torch.FloatTensor(swap_and_flatten(np.asarray(next_obs))).to(self.device)
        dones = torch.FloatTensor(swap_and_flatten(np.asarray(dones))).to(self.device)
        
        return (obs, actions, rewards, next_obs, dones)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
class ReplayBuffer():
    def __init__(self, buffer_size: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        
        self.buffer = deque(maxlen=buffer_size)

    def add(self, transition: Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray], np.ndarray, Union[bool, np.ndarray]]) -> None:
        self.buffer.append(transition)
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            samples = random.sample(self.buffer, batch_size)
        except:
            raise ValueError("The batch size must be less than learning start:)")
        
        obs, actions, rewards, next_obs, dones = [], [], [], [], []
        
        for ob, action, reward, next_ob, done in samples:
            if isinstance(ob, int):
                ob = np.array([ob])
            if len(ob.shape) == 1:
                ob = ob[np.newaxis, :]
                
            if isinstance(action, int):
                action = [action]
            elif len(action.shape) == 1:
                action = action[np.newaxis, :]
             
            if not isinstance(reward, np.ndarray):
                reward = [reward]
            
            if isinstance(next_ob, int):
               next_ob = np.array([next_ob])
            if len(next_ob.shape) == 1:
                next_ob = next_ob[np.newaxis, :]
                
            if not isinstance(done, np.ndarray):
                done = [done]
                
            obs.append(ob)
            actions.append(action)
            rewards.append(reward)
            next_obs.append(next_ob)
            dones.append(done)
        
        obs = torch.FloatTensor(swap_and_flatten(np.asarray(obs))).to(self.device)
        actions = torch.FloatTensor(swap_and_flatten(np.asarray(actions))).to(self.device)
        rewards = torch.FloatTensor(swap_and_flatten(np.asarray(rewards))).to(self.device)
        next_obs = torch.FloatTensor(swap_and_flatten(np.asarray(next_obs))).to(self.device)
        dones = torch.FloatTensor(swap_and_flatten(np.asarray(dones))).to(self.device)
        
        return (obs, actions, rewards, next_obs, dones)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
class SharedReplayBuffer():
    def __init__(self, buffer_size: int, num_envs: int, observation_dim: int):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        
        self.observations = torch.zeros((buffer_size, num_envs, observation_dim)).share_memory_()
        self.actions = torch.zeros((buffer_size, num_envs, 1)).share_memory_()
        self.rewards = torch.zeros((buffer_size, num_envs)).share_memory_()
        self.next_observations =  torch.zeros((buffer_size, num_envs, observation_dim)).share_memory_()
        self.dones = torch.zeros((buffer_size, num_envs)).share_memory_()
        
        self.pos = mp.Value("i", 0)
        self.full = False
        
    def add(self, transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]) -> None:
        pos = self.pos.value
        
        obs = transition[0]
        act = transition[1]
        rew = transition[2]
        next_obs = transition[3]
        done = transition[4]
        
        obs = obs[np.newaxis, :]
        act = act[np.newaxis, :]
        rew = np.array([rew])
        done = np.array([done])
            
        self.observations[pos] = torch.from_numpy(obs)
        self.actions[pos] = torch.from_numpy(act)
        self.rewards[pos] = torch.from_numpy(rew)
        self.next_observations[pos] = torch.from_numpy(next_obs)
        self.dones[pos] = torch.from_numpy(done)
        
        self.pos.value += 1
        
        if self.pos.value == self.buffer_size:
            self.pos.value = 0
            self.full = True
            
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.full:
            batch_inds = np.random.randint(0, self.buffer_size, size=batch_size)
        else:
            batch_inds = np.random.randint(0, self.pos.value)
            
        env_inds = np.random.randint(0, self.num_envs, size=batch_size)
        
        observations = self.observations[batch_inds, env_inds, :]
        actions = self.actions[batch_inds, env_inds, :]
        rewards = self.rewards[batch_inds, env_inds]
        next_observations = self.next_observations[batch_inds, env_inds, :]
        dones = self.dones[batch_inds, env_inds]
        
        rewards = rewards.unsqueeze(dim=1)
        dones = dones.unsqueeze(dim=1)

        return (observations, actions, rewards, next_observations, dones)
    
    def __len__(self) -> int:
        if self.full:
            return self.buffer_size
        else:
            return self.pos.value