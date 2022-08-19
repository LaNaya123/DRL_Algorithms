# -*- coding: utf-8 -*-
import numpy as np
import random
import torch
from gym import spaces
from collections import deque
from common.models import VPGActor, ACKTRActor, VCritic
from common.buffers import RolloutBuffer
from common.utils import obs_to_tensor, safe_mean
from common.logger import Logger
    
class OnPolicyAlgorithm():
    def __init__(self, 
                 env, 
                 rollout_steps,
                 total_timesteps, 
                 actor_kwargs,
                 critic_kwargs,
                 td_method,
                 gamma,
                 gae_lambda,
                 max_grad_norm, 
                 verbose,
                 log_dir,
                 log_interval,
                 device,
                 seed,
                 ):
        
        self.env = env
        self.rollout_steps = rollout_steps
        self.total_timesteps = total_timesteps
        self.actor_kwargs = {} if actor_kwargs is None else actor_kwargs
        self.critic_kwargs = {} if critic_kwargs is None else critic_kwargs
        self.td_method = td_method
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.verbose = verbose
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.seed = seed

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        if self.verbose > 0:
            print(f"Using the device: {self.device}")
        
        self._set_seed()
        
        self._setup_model()
        
        self._setup_param()
        
        self._setup_logger()
        
    def _setup_model(self):
        raise NotImplementedError("You have to overwrite this method in your own algorithm:)")
    
    def _setup_param(self):
        self.episode_info_buffer = deque(maxlen=4)
        
        self.num_episode = 0
        
        self.current_timesteps = 0
        
    def _setup_logger(self):
        if self.log_dir is None:
            self.logger = None
        else:
            self.logger = Logger(self.log_dir)
        
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
                
                if self.logger is not None:
                    if self.logger.log_count == 0:
                        data = list(episode_info.keys())
                        data = ["num_episode"] + data
                        self.logger.write(data)
                    data = list(episode_info.values())
                    data = [self.num_episode] + data
                    self.logger.write(data)
                    self.logger.log_count += 1
                    
    def rollout(self):
        raise NotImplementedError("You have to overwrite this method in your own algorithm:)")
            
    def train(self):
        raise NotImplementedError("You have to overwrite this method in your own algorithm:)")
    
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
        if self.logger is not None:
            self.logger.close()
                 
class OffPolicyAlgorithm():
    def __init__(self, 
                 env, 
                 rollout_steps,
                 total_timesteps, 
                 gradient_steps,
                 learning_start,
                 buffer_size,
                 batch_size,
                 target_update_interval,
                 gamma,
                 verbose,
                 log_dir,
                 log_interval,
                 device,
                 seed,
                 ):
        
        self.env = env
        self.rollout_steps = rollout_steps
        self.total_timesteps = total_timesteps
        self.gradient_steps = gradient_steps
        self.learning_start = learning_start
        self.buffer_size = buffer_size
        self.batch_size=batch_size
        self.target_update_interval = target_update_interval
        self.gamma = gamma
        self.verbose = verbose
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.seed = seed
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        if self.verbose > 0:
            print(f"Using the device: {self.device}")
        
        self._set_seed()
        
        self._setup_model()
        
        self._setup_param()
        
        self._setup_logger()
        
    def _setup_model(self):
        raise NotImplementedError("You have to overwrite this method in your own algorithm:)")
    
    def _setup_param(self):
        self.episode_info_buffer = deque(maxlen=10)
        
        self.num_episode = 0
        
        self.current_timesteps = 0
    
    def _setup_logger(self):
        if self.log_dir is None:
            self.logger = None
        else:
            self.logger = Logger(self.log_dir)
            
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
                
                if self.logger is not None:
                    if self.logger.log_count == 0:
                        data = list(episode_info.keys())
                        data = ["num_episode"] + data
                        self.logger.write(data)
                    data = list(episode_info.values())
                    data = [self.num_episode] + data
                    self.logger.write(data)
                    self.logger.log_count += 1
                
    def rollout(self):
        raise NotImplementedError("You have to overwrite this method in your own algorithm:)")
            
    def train(self):
        raise NotImplementedError("You have to overwrite this method in your own algorithm:)")
    
    def learn(self):
        self.training_iterations = 0
        
        while self.current_timesteps < self.total_timesteps:
            
            self.rollout()
            
            if self.current_timesteps > 0 and self.current_timesteps > self.learning_start:
                
                for _ in range(self.gradient_steps):
                    
                    self.train()
            
                    self.training_iterations += 1
            
            if self.training_iterations % self.log_interval == 0 and self.verbose > 0:
                 print("episode", self.num_episode,
                       "episode_reward_mean", safe_mean([ep_info["episode returns"] for ep_info in self.episode_info_buffer]),
                       )
                 
        if self.logger is not None:
            self.logger.close()