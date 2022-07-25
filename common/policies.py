# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 23:36:28 2022

@author: lanaya
"""
import numpy as np
import random
import torch
from gym import spaces
from collections import deque
from common.models import ContinuousActor, Critic, DeepQNetwork
from common.buffers import RolloutBuffer, ReplayBuffer
from common.utils import obs_to_tensor, safe_mean
    
class OnPolicyAlgorithm():
    def __init__(self, 
                 env, 
                 rollout_steps,
                 total_timesteps, 
                 actor_kwargs=None,
                 critic_kwargs=None,
                 td_method="td_lambda",
                 gamma=0.99,
                 gae_lambda=0.95,
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
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.verbose = verbose
        self.log_interval = log_interval
        self.seed = seed
        print(self.seed, "1123123")
        print(log_interval)
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
        
        if isinstance(self.env.action_space, spaces.Discrete):
            num_action = self.env.action_space.n
        elif isinstance(self.env.action_space, spaces.Box):
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
            
            self.current_timesteps += self.env.num_envs
            
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
                 
class OffPolicyAlgorithm():
    def __init__(self, 
                 env, 
                 rollout_steps,
                 total_timesteps, 
                 qnet_kwargs=None,
                 actor_kwargs=None,
                 critic_kwargs=None,
                 learning_start=1000,
                 buffer_size=10000,
                 batch_size=256,
                 target_update_interval=20,
                 gamma=0.99,
                 verbose=1,
                 log_interval=10,
                 mode="value_based",
                 device="auto",
                 seed=None,
                 ):
        
        self.env = env
        self.rollout_steps = rollout_steps
        self.total_timesteps = total_timesteps
        self.qnet_kwargs = {} if qnet_kwargs is None else qnet_kwargs
        self.actor_kwargs = {} if actor_kwargs is None else actor_kwargs
        self.critic_kwargs = {} if critic_kwargs is None else critic_kwargs
        self.learning_start = learning_start
        self.buffer_size = buffer_size
        self.batch_size=batch_size
        self.target_update_interval = target_update_interval
        self.gamma = gamma
        self.verbose = verbose
        self.log_interval = log_interval
        self.mode = mode
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
        
        if isinstance(self.env.action_space, spaces.Discrete):
            num_action = self.env.action_space.n
        elif isinstance(self.env.action_space, spaces.Box):
            num_action = self.env.action_space.shape[0]
        
        if self.mode == "value_based":
            self.qnet = DeepQNetwork(observation_dim, num_action, **self.qnet_kwargs)
            self.target_qnet = DeepQNetwork(observation_dim, num_action, **self.qnet_kwargs)
            self.target_qnet.load_state_dict(self.qnet.state_dict())
            
            if self.verbose > 0:
                print(self.qnet)
        else:
            self.actor = ContinuousActor(observation_dim, num_action, **self.actor_kwargs)

            self.critic = Critic(observation_dim, **self.critic_kwargs)
        
            if self.verbose > 0:
                print(self.actor)
                print(self.critic)
        
        self.buffer = ReplayBuffer(self.buffer_size)
        
        self.obs = self.env.reset()
    
    def _setup_param(self):
        self.episode_info_buffer = deque(maxlen=10)
        
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
        for i in range(self.rollout_steps):
            if self.mode == "value_based":
                q = self.qnet(obs_to_tensor(self.obs))
                coin = random.random()
                if coin < 0.1:
                    clipped_action = random.randint(0, self.env.action_space.n - 1)
                else:
                    clipped_action = q.argmax().item()
                action = clipped_action
    
            else:
                dists = self.actor(obs_to_tensor(self.obs))
                action = dists.sample().detach().numpy()
                clipped_action = np.clip(action, self.env.action_space.low.min(), self.env.action_space.high.max())

            next_obs, reward, done, info = self.env.step(clipped_action)
            
            self.buffer.add((self.obs, action, reward, next_obs, done))
            
            self.obs = next_obs
            
            self.current_timesteps += self.env.num_envs
            
            self._update_episode_info(info)
            
    def train(self):
        raise NotImplementedError
    
    def learn(self):
        self.training_iterations = 0
        
        while self.current_timesteps < self.total_timesteps:
            
            self.rollout()
            
            if self.current_timesteps > 0 and self.current_timesteps > self.learning_start:
                
                self.train()
            
                self.training_iterations += 1
            
            if self.training_iterations % self.log_interval == 0 and self.verbose > 0:
                 print("episode", self.num_episode,
                       "episode_reward_mean", safe_mean([ep_info["episode returns"] for ep_info in self.episode_info_buffer]),
                       )