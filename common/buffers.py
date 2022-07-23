# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 13:51:47 2022

@author: lanaya
"""
import torch 
import numpy as np
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