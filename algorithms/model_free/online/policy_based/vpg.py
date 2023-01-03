# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
from typing import Any, Dict, Optional, Union
import gym
import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
from common.envs import Monitor, VecEnv
from common.models import VPGActor, VCritic
from common.buffers import RolloutBuffer
from common.policies import OnPolicyAlgorithm
from common.utils.utils import Mish, clip_grad_norm_, compute_gae_advantage, compute_td_target, obs_to_tensor

class VPG(OnPolicyAlgorithm):
    def __init__(self, 
                 env: Union[Monitor, VecEnv], 
                 rollout_steps: int = 16, 
                 total_timesteps: int = 1e5, 
                 actor_kwargs: Optional[Dict[str, Any]] = None,
                 critic_kwargs: Optional[Dict[str, Any]] = None,
                 td_method: str = "td_lambda",
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 max_grad_norm: Optional[float] = 0.5,
                 auxiliary_buffer_size: Optional[int] = None,
                 verbose: int = 1,
                 log_dir: Optional[str] = None,
                 log_interval: int = 100,
                 device: str = "auto",
                 seed: Optional[int] = None,
                 ):
        
        super(VPG, self).__init__(
              env, 
              rollout_steps, 
              total_timesteps, 
              actor_kwargs,
              critic_kwargs,
              td_method,
              gamma,
              gae_lambda,
              max_grad_norm,
              auxiliary_buffer_size,
              verbose, 
              log_dir,
              log_interval,
              device,
              seed,
            )
    
    def _setup_model(self) -> None:
        observation_dim = self.env.observation_space.shape[0]
        
        if isinstance(self.env.action_space, spaces.Discrete):
            num_action = self.env.action_space.n
        elif isinstance(self.env.action_space, spaces.Box):
            num_action = self.env.action_space.shape[0]

        self.actor = VPGActor(observation_dim, num_action, **self.actor_kwargs).to(self.device)

        self.critic = VCritic(observation_dim, **self.critic_kwargs).to(self.device)
        
        if self.verbose > 0:
            print(self.actor)
            print(self.critic)
        
        self.buffer = RolloutBuffer(self.rollout_steps, self.device)
        
        self.obs = self.env.reset()
        
    def _rollout(self) -> None:
        self.buffer.reset()
        
        with torch.no_grad():
            for i in range(self.rollout_steps):
                dists = self.actor(obs_to_tensor(self.obs).to(self.device))
            
                action = dists.sample().cpu().detach().numpy()

                action_clipped = np.clip(action, self.env.action_space.low.min(), self.env.action_space.high.max())

                next_obs, reward, done, info = self.env.step(action_clipped)
            
                self.buffer.add((self.obs, action, reward, next_obs, done))
            
                self.obs = next_obs
            
                self.current_timesteps += self.env.num_envs
            
                self._update_episode_info(info)
            
    def _train(self) -> None:
        obs, actions, rewards, next_obs, dones = self.buffer.get()
            
        assert isinstance(obs, torch.Tensor) and obs.shape[1] == self.env.observation_space.shape[0]
        assert isinstance(actions, torch.Tensor) and actions.shape[1] == self.env.action_space.shape[0]
        assert isinstance(rewards, torch.Tensor) and rewards.shape[1] == 1
        assert isinstance(next_obs, torch.Tensor) and next_obs.shape[1] == self.env.observation_space.shape[0]
        assert isinstance(dones, torch.Tensor) and dones.shape[1] == 1
            
        values = self.critic(obs)
            
        if  self.td_method == "td":
            target_values = rewards + self.gamma * self.critic(next_obs) * (1 - dones)
                
            advantages = target_values - values
            
        elif self.td_method == "td_n":
            if dones[-1]:
                last_value = 0
            else:
                last_value = self.critic(next_obs[-1])
            
            target_values = compute_td_target(rewards, dones, last_value, gamma=self.gamma)
            target_values = target_values.to(self.device)
                
            advantages = target_values - values
                
        elif self.td_method == "td_lambda":
            if dones[-1]:
                last_value = 0
            else:
                last_value = self.critic(next_obs[-1])
            
            advantages = compute_gae_advantage(rewards, values, dones, last_value, gamma=self.gamma, gae_lambda=self.gae_lambda)
            advantages = advantages.to(self.device)
            
            target_values = advantages + values
                
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        dists = self.actor(obs)
            
        log_probs = dists.log_prob(actions)
             
        actor_loss = -(log_probs * advantages.detach()).mean()
            
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
            
        if self.max_grad_norm:
            clip_grad_norm_(self.actor.optimizer, self.max_grad_norm)
            
        self.actor.optimizer.step()
        
        critic_loss = F.mse_loss(target_values.detach(), values)
            
        self.critic.optimizer.zero_grad()
            
        critic_loss.backward()
            
        if self.max_grad_norm:
            clip_grad_norm_(self.critic.optimizer, self.max_grad_norm)
            
        self.critic.optimizer.step()
            
if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    env = Monitor(env)
    vpg = VPG(env, 
              rollout_steps=8, 
              total_timesteps=2e5, 
              actor_kwargs={"activation_fn": Mish}, 
              critic_kwargs={"activation_fn": Mish},
              td_method="td_lambda",
              max_grad_norm=0.5,
              log_dir=None,
              seed=14,
             )
    vpg.learn()