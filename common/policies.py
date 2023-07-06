# -*- coding: utf-8 -*-
import numpy as np
import random
import torch
from typing import Union, Optional, Any, Dict
from collections import deque
from common.envs import Monitor, VecEnv
from common.utils import safe_mean
from torch.utils.tensorboard import SummaryWriter

class OnPolicyAlgorithm():
    def __init__(self, 
                 env: Union[Monitor, VecEnv],
                 rollout_steps: int,
                 total_timesteps: int, 
                 actor_kwargs: Dict[str, Any],
                 critic_kwargs: Dict[str, Any],
                 td_method: str,
                 gamma: float,
                 gae_lambda: float,
                 max_grad_norm: Optional[float],
                 log_dir: Optional[str],
                 log_interval: int,
                 verbose: int,
                 seed: Optional[int],
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
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.verbose = verbose
        self.seed = seed
        
        self._set_seed()
        
        self._setup_model()
        
        self._setup_param()
        
        self._setup_logger()
        
    def _setup_model(self) -> None:
        raise NotImplementedError("You have to overwrite this method in your own algorithm:)")
    
    def _setup_param(self) -> None:
        self.episode_info_buffer = deque(maxlen=4)
        
        self.num_episodes = 0
        
        self.current_timesteps = 0
        
        self.training_iterations = 0
        
    def _setup_logger(self) -> None:
        if self.log_dir is None:
            self.logger = None
        else:
            self.logger = SummaryWriter(self.log_dir)
        
    def _set_seed(self) -> None:
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
    
    def _update_episode_info(self, infos) -> None:
        if isinstance(infos, dict):
            infos = [infos]
        for info in infos:
            episode_info = info.get("episode")
            if episode_info is not None:
                self.episode_info_buffer.append(episode_info)
                self.num_episodes += 1
                    
    def _rollout(self) -> None:
        raise NotImplementedError("You have to overwrite this method in your own algorithm:)")
            
    def _train(self) -> None:
        raise NotImplementedError("You have to overwrite this method in your own algorithm:)")
    
    def learn(self) -> None:
        while self.current_timesteps < self.total_timesteps:
            
            self._rollout()
            
            self._train()
            
            self.training_iterations += 1
            
            if self.training_iterations % self.log_interval == 0:
                if self.logger:
                    self.logger.add_scalar("episode", safe_mean([ep_info["episode returns"] for ep_info in self.episode_info_buffer]), self.num_episodes)
                    self.logger.add_scalar("policy_loss", self.policy_loss, self.training_iterations)
                    self.logger.add_scalar("value_loss", self.value_loss, self.training_iterations)
                
                if self.verbose > 0:
                    print("episode", self.num_episodes,
                          "episode_reward_mean", safe_mean([ep_info["episode returns"] for ep_info in self.episode_info_buffer]),
                         )
                    
        if self.logger is not None:
            self.logger.close()
               
    def save(self, path: str) -> None:
        raise NotImplementedError("You have to overwrite this method in your own algorithm:)")
        
    def load(self, path: str) -> None:
        raise NotImplementedError("You have to overwrite this method in your own algorithm:)")
                 
class OffPolicyAlgorithm():
    def __init__(self, 
                 env: Union[Monitor, VecEnv], 
                 rollout_steps: int,
                 total_timesteps: int, 
                 gradient_steps: int,
                 n_steps: int,
                 learning_start: int,
                 buffer_size: int,
                 batch_size: int,
                 target_update_interval: int,
                 gamma: float,
                 max_grad_norm: Optional[float],
                 log_dir: Optional[str],
                 log_interval: int,
                 verbose: int,
                 seed: Optional[int],
                 ):
        
        self.env = env
        self.rollout_steps = rollout_steps
        self.total_timesteps = total_timesteps
        self.gradient_steps = gradient_steps
        self.n_steps = n_steps
        self.learning_start = learning_start
        self.buffer_size = buffer_size
        self.batch_size=batch_size
        self.target_update_interval = target_update_interval
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.verbose = verbose
        self.seed = seed
        
        self._set_seed()

        self._setup_model()
        
        self._setup_param()
        
        self._setup_logger()
        
    def _setup_model(self) -> None:
        raise NotImplementedError("You have to overwrite this method in your own algorithm:)")
    
    def _setup_param(self) -> None:
        self.episode_info_buffer = deque(maxlen=10)
        
        self.num_episodes = 0
        
        self.current_timesteps = 0
        
        self.training_iterations = 0
    
    def _setup_logger(self) -> None:
        if self.log_dir is None:
            self.logger = None
        else:
            self.logger = SummaryWriter(self.log_dir)
            
    def _set_seed(self) -> None:
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
    
    def _update_episode_info(self, infos) -> None:
        if isinstance(infos, dict):
            infos = [infos]
        for info in infos:
            episode_info = info.get("episode")
            if episode_info is not None:
                self.episode_info_buffer.append(episode_info)
                self.num_episodes += 1
                
    def _rollout(self) -> None:
        raise NotImplementedError("You have to overwrite this method in your own algorithm:)")
            
    def _train(self) -> None:
        raise NotImplementedError("You have to overwrite this method in your own algorithm:)")
    
    def learn(self) -> None:
        while self.current_timesteps < self.total_timesteps:
            
            self._rollout()
            
            if self.current_timesteps > self.learning_start:
                
                for _ in range(self.gradient_steps):
                    
                    self._train()
            
                    self.training_iterations += 1
            
            if self.training_iterations % self.log_interval == 0:
                if self.logger:
                    self.logger.add_scalar("episode", self.episode_reward_mean, self.num_episodes)
                    self.logger.add_scalar("policy_loss", self.policy_loss, self.training_iterations())
                    self.logger.add_scalar("value_loss", self.value_loss, self.training_iterations())
                
                if self.verbose > 0:
                    print("episode", self.num_episodes,
                          "episode_reward_mean", safe_mean([ep_info["episode returns"] for ep_info in self.episode_info_buffer]),
                         )
                 
        if self.logger is not None:
            self.logger.close()
            
    def save(self, path: str) -> None:
        raise NotImplementedError("You have to overwrite this method in your own algorithm:)")
        
    def load(self, path: str) -> None:
        raise NotImplementedError("You have to overwrite this method in your own algorithm:)")