# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
from typing import Any, Union, Optional, List, Dict
import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.envs import Monitor
from ppo import PPO
from common.utils.utils import Mish, obs_to_tensor
from common.utils.models import CuriosityModel

class ICM(PPO):
    def __init__(self, 
                 env: Monitor, 
                 rollout_steps: int = 16, 
                 total_timesteps: int = 1e5, 
                 actor_kwargs: Optional[Dict[str, Any]] = None,
                 critic_kwargs: Optional[Dict[str, Any]] = None,
                 td_method: str = "td_lambda",
                 num_epochs: int = 10,
                 clip_range: float = 0.2,
                 ent_coef: float = 0.0,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 beta: float = 0.01, 
                 max_grad_norm: Optional[float] = None,
                 verbose: int = 1,
                 log_dir: Optional[str] = None,
                 log_interval: int = 100,
                 device: str = "auto",
                 seed: Optional[int] = None,
                 ):
        
        super(ICM, self).__init__(
                 env, 
                 rollout_steps, 
                 total_timesteps, 
                 actor_kwargs,
                 critic_kwargs,
                 td_method,
                 num_epochs,
                 clip_range,
                 ent_coef,
                 gamma,
                 gae_lambda, 
                 max_grad_norm,
                 verbose,
                 log_dir,
                 log_interval,
                 device,
                 seed,
                 )
        
        self.beta = beta
        
    def _setup_model(self) -> None:
        super()._setup_model()
        
        self.icm = CuriosityModel(self.observation_dim, self.num_actions)
        
    
    def _compute_intrinsic_reward(self, 
                                  obs: Union[np.ndarray, torch.Tensor], 
                                  next_obs: Union[np.ndarray, torch.Tensor], 
                                  acts: Union[np.ndarray, torch.Tensor]
                                 ) -> float:
        obs = obs_to_tensor(obs)
        next_obs = obs_to_tensor(next_obs)
        acts = obs_to_tensor(acts)
        
        encoded_obs_, pred_obs_, pred_acts = self.icm(obs, next_obs, acts)
        
        intrinsic_reward = F.mse_loss(pred_obs_, encoded_obs_)
        
        return intrinsic_reward
    
    def _rollout(self):
        self.buffer.reset()
        
        with torch.no_grad():
            for i in range(self.rollout_steps):

                dists = self.actor(obs_to_tensor(self.obs).to(self.device))
            
                action = dists.sample().cpu().detach().numpy()

                clipped_action = np.clip(action, self.env.action_space.low.min(), self.env.action_space.high.max())

                next_obs, reward, done, info = self.env.step(clipped_action)
                
                intrinsic_reward = self._compute_intrinsic_reward(self.obs, next_obs, clipped_action).item()
                augmented_reward = reward + self.beta * intrinsic_reward
                
                self.buffer.add((self.obs, action, augmented_reward, next_obs, done))
            
                self.obs = next_obs
            
                self.current_timesteps += self.env.num_envs
            
                self._update_episode_info(info)
                
    def _train(self):
        obs, actions, rewards, next_obs, dones = super()._train()
        
        encoded_next_obs, pred_next_obs, pred_actions = self.icm(obs, next_obs, actions)
        
        inverse_loss = F.mse_loss(pred_actions, actions)
        
        forward_loss = F.mse_loss(pred_next_obs, encoded_next_obs)
        
        loss = inverse_loss + forward_loss

        self.icm.optimizer.zero_grad()
        loss.backward()
        self.icm.optimizer.step()        
        
if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    env = Monitor(env)
    icm = ICM(env, 
              rollout_steps=8, 
              total_timesteps=2e5, 
              actor_kwargs={"activation_fn": Mish}, 
              critic_kwargs={"activation_fn": Mish},
              beta = 0,
              td_method="td_lambda",
              num_epochs=1,
              log_dir=None,
              seed=7,
             )
    icm.learn()