# -*- coding: utf-8 -*-
import torch 
import numpy as np
import random
from collections import deque
from common.utils import swap_and_flatten

class RolloutBuffer():
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        
    def reset(self):
        self.buffer.clear()
        
    def add(self, transition):
        self.buffer.append(transition)
        
    def get(self):
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
        
        obs = torch.FloatTensor(swap_and_flatten(np.asarray(obs)))
        actions = torch.FloatTensor(swap_and_flatten(np.asarray(actions)))
        rewards = torch.FloatTensor(swap_and_flatten(np.asarray(rewards)))
        next_obs = torch.FloatTensor(swap_and_flatten(np.asarray(next_obs)))
        dones = torch.FloatTensor(swap_and_flatten(np.asarray(dones)))
        
        return obs, actions, rewards, next_obs, dones
    
    
class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        
    def add(self, transition):
        self.buffer.append(transition)
        
    def sample(self, batch_size):
        try:
            samples = random.sample(self.buffer, batch_size)
        except:
            raise ValueError("The batch size must be greater than learning start:)")
        
        obs, actions, rewards, next_obs, dones = [], [], [], [], []
        
        for ob, action, reward, next_ob, done in samples:
            
            if len(ob.shape) == 1:
                ob = ob[np.newaxis, :]
                
            if isinstance(action, int):
                action = [action]
            elif len(action.shape) == 1:
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
        
        obs = torch.FloatTensor(swap_and_flatten(np.asarray(obs)))
        actions = torch.FloatTensor(swap_and_flatten(np.asarray(actions)))
        rewards = torch.FloatTensor(swap_and_flatten(np.asarray(rewards)))
        next_obs = torch.FloatTensor(swap_and_flatten(np.asarray(next_obs)))
        dones = torch.FloatTensor(swap_and_flatten(np.asarray(dones)))
        
        return obs, actions, rewards, next_obs, dones