# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 23:36:28 2022

@author: lanaya
"""
import numpy as np
import random
import torch
from collections import deque
from common.models import ContinuousActor, Critic
from common.buffers import RolloutBuffer
from common.utils import obs_to_tensor, safe_mean
    
class OnPolicyAlgorithm():
    def __init__(self, 
                 env, 
                 rollout_steps,
                 total_timesteps, 
                 actor_kwargs=None,
                 critic_kwargs=None,
                 td_method="td_lambda",
                 verbose=1,
                 log_interval=10,
                 device="auto",
                 seed=None,
                 ):
        
        self.env = env
        self.rollout_steps = rollout_steps
        self.total_timesteps = total_timesteps
        self.actor_kwargs = {} if actor_kwargs is None else actor_kwargs
        self.critic_kwargs = {} if critic_kwargs is None else critic_kwargs
        self.td_method = td_method
        self.verbose = verbose
        self.log_interval = log_interval
        self.seed = seed
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        if self.verbose > 0:
            print(f"Using the device: {self.device}")

        self._setup_model()
        
        self._setup_param()
        
        self._set_seed()

    def _setup_model(self):
        observation_dim = self.env.observation_space.shape[0]
        num_action = self.env.action_space.shape[0]

        self.actor = ContinuousActor(observation_dim, num_action, **self.actor_kwargs)

        self.critic = Critic(observation_dim, **self.critic_kwargs)
        
        if self.verbose > 0:
            print(self.actor)
            print(self.critic)
        
        self.buffer = RolloutBuffer(self.rollout_steps)
        
        self.obs = self.env.reset()
    
    def _setup_param(self):
        self.episode_info_buffer = deque(maxlen=4)
        
        self.num_episode = 0
        
        self.current_timesteps = 0
        
    def _set_seed(self):
        if self.seed is None:
            return
        if self.verbose > 0:
            print(f"Setting the random seed to {self.seed}")
        
        random.seed(self.seed)
        
        np.random.seed(self.seed)

        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.env.seed(self.seed)
    
    def _update_episode_info(self, infos):
        if isinstance(infos, dict):
            infos = [infos]
        for info in infos:
            episode_info = info.get("episode")
            if episode_info is not None:
                self.episode_info_buffer.append(episode_info)
                self.num_episode += 1
                
    def rollout(self):
        self.buffer.reset()
        
        for i in range(self.rollout_steps):
            dists = self.actor(obs_to_tensor(self.obs))
            action = dists.sample().detach().numpy()
            action_clipped = np.clip(action, self.env.action_space.low.min(), self.env.action_space.high.max())

            next_obs, reward, done, info = self.env.step(action_clipped)
            
            self.buffer.add((self.obs, action, reward, next_obs, done))
            
            self.obs = next_obs
            
            self.current_timesteps += 1
            
            self._update_episode_info(info)
            
    def train(self):
        raise NotImplementedError
    
    def learn(self):
        training_iterations = 0
        
        while self.current_timesteps < self.total_timesteps:
            
            self.rollout()
            
            self.train()
            
            training_iterations += 1
            
            if training_iterations % self.log_interval == 0 and self.verbose > 0:
                 print("episode", self.num_episode,
                       "episode_reward_mean", safe_mean([ep_info["episode returns"] for ep_info in self.episode_info_buffer]),
                       )