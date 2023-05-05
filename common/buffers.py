# -*- coding: utf-8 -*-
from typing import Union, Tuple
import torch 
import numpy as np
import random
from collections import deque
from common.utils import swap_and_flatten

class RolloutBuffer():
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        
        self.buffer = deque(maxlen=buffer_size)
        
    def reset(self) -> None:
        self.buffer.clear()
        
    def add(self, transition: Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray], np.ndarray, Union[bool, np.ndarray]]) -> None:
        self.buffer.append(transition)
        
    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs, actions, rewards, next_obs, dones, log_probs = [], [], [], [], [], []
        
        for ob, action, reward, next_ob, done, log_prob in self.buffer:
            
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
                
            if len(log_prob.shape) == 1:
                log_prob = log_prob[np.newaxis, :]
                
            obs.append(ob)
            actions.append(action)
            rewards.append(reward)
            next_obs.append(next_ob)
            dones.append(done)
            log_probs.append(log_prob)
        
        obs = torch.FloatTensor(swap_and_flatten(np.asarray(obs)))
        actions = torch.FloatTensor(swap_and_flatten(np.asarray(actions)))
        rewards = torch.FloatTensor(swap_and_flatten(np.asarray(rewards)))
        next_obs = torch.FloatTensor(swap_and_flatten(np.asarray(next_obs)))
        dones = torch.FloatTensor(swap_and_flatten(np.asarray(dones)))
        log_probs = torch.FloatTensor(swap_and_flatten(np.asarray(log_probs)))
        
        return (obs, actions, rewards, next_obs, dones, log_probs)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
class ReplayBuffer():
    def __init__(self, buffer_size: int, n_steps: int = 1, gamma: float = 0.99):
        self.buffer_size = buffer_size
        self.n_steps = n_steps
        self.gamma = gamma
        
        self.buffer = deque(maxlen=buffer_size)
        
        self.n_steps_buffer = deque(maxlen=n_steps)
        
    def _update_n_steps_info(self) -> Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray], np.ndarray, Union[bool, np.ndarray]]:
        state, action = self.n_steps_buffer[0][:2]
        reward, next_state, done = self.n_steps_buffer[-1][-3:]

        for transition in reversed(list(self.n_steps_buffer)[:-1]):
            r, next_s, d = transition[-3:]

            reward = r + self.gamma * reward * (1 - d)
            
            next_state, done = (next_s, d) if d else (next_state, done)

        return (state, action, reward, next_state, done)

    def add(self, transition: Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray], np.ndarray, Union[bool, np.ndarray]]) -> None:
        self.n_steps_buffer.append(transition)
        
        if len(self.n_steps_buffer) >= self.n_steps:
            transition = self._update_n_steps_info()
        
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
        
        obs = torch.FloatTensor(swap_and_flatten(np.asarray(obs)))
        actions = torch.FloatTensor(swap_and_flatten(np.asarray(actions)))
        rewards = torch.FloatTensor(swap_and_flatten(np.asarray(rewards)))
        next_obs = torch.FloatTensor(swap_and_flatten(np.asarray(next_obs)))
        dones = torch.FloatTensor(swap_and_flatten(np.asarray(dones)))
        
        return (obs, actions, rewards, next_obs, dones)
    
    def __len__(self) -> int:
        return len(self.buffer)