# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 13:51:47 2022

@author: lanaya
"""
import torch 

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
            
        obs = torch.FloatTensor(obs)
        actions = torch.FloatTensor(actions).view(-1, 1)
        rewards = torch.FloatTensor(rewards).view(-1, 1)
        next_obs = torch.FloatTensor(next_obs)
        dones = torch.FloatTensor(dones).view(-1, 1)
        
        return obs, actions, rewards, next_obs, dones