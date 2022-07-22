# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 13:12:55 2022

@author: lanaya
"""
class Monitor():
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.rewards = []
        
    def reset(self):
        observation = self.env.reset()
        return observation 
    
    def step(self, action):
        next_observation, reward, done, info = self.env.step(action)
        
        self.rewards.append(reward)
        
        if done:
            ep_r = sum(self.rewards)
            ep_l = len(self.rewards)
            
            episode_info = {"episode returns": round(ep_r, 2), "episode length": ep_l}
            
            info["episode"] = episode_info
            
            next_observation = self.env.reset()
            
            self.rewards.clear()
            
        return next_observation, reward, done, info
    
    def close(self):
        self.env.close()