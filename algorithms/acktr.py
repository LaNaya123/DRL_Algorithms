# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from common.envs import Monitor, VecEnv
from common.models import ACKTRActor, ACKTRCritic
from common.buffers import RolloutBuffer
from common.policies import OnPolicyAlgorithm
from common.utils import Mish, clip_grad_norm_, compute_gae_advantage, compute_td_target, obs_to_tensor
from common.kfac import KFACOptimizer

class ACKTR(OnPolicyAlgorithm):
    def __init__(self, 
                 env, 
                 rollout_steps, 
                 total_timesteps, 
                 actor_kwargs=None,
                 critic_kwargs=None,
                 td_method="td_lambda",
                 gamma=0.99,
                 gae_lambda=0.95,
                 max_grad_norm=None,
                 verbose=1,
                 log_dir=None,
                 log_interval=100,
                 device="auto",
                 seed=12,
                 ):
        
        super(ACKTR, self).__init__(
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
            )
    
    def _setup_model(self):
        observation_dim = self.env.observation_space.shape[0]
        
        if isinstance(self.env.action_space, spaces.Discrete):
            num_action = self.env.action_space.n
        elif isinstance(self.env.action_space, spaces.Box):
            num_action = self.env.action_space.shape[0]

        self.actor = ACKTRActor(observation_dim, num_action, **self.actor_kwargs)

        self.critic = ACKTRCritic(observation_dim, **self.critic_kwargs)
        
        if self.verbose > 0:
            print(self.actor)
            print(self.critic)
        
        self.buffer = RolloutBuffer(self.rollout_steps)
        
        self.obs = self.env.reset()
        
    def rollout(self):
        self.buffer.reset()
        
        with torch.no_grad():
            
            for i in range(self.rollout_steps):
                obs = obs_to_tensor(self.obs)
                
                dists = self.actor(obs)
                
                action = dists.sample().detach().numpy()
                if not self.env.is_vec:
                    action = action.squeeze(axis=0)

                action_clipped = np.clip(action, self.env.action_space.low.min(), self.env.action_space.high.max())
            
                next_obs, reward, done, info = self.env.step(action_clipped)
            
                self.buffer.add((self.obs, action, reward, next_obs, done))
            
                self.obs = next_obs
            
                self.current_timesteps += self.env.num_envs
            
                self._update_episode_info(info)
        
    def train(self):
            obs, actions, rewards, next_obs, dones = self.buffer.get()
            
            assert isinstance(obs, torch.Tensor) and obs.shape[1] == self.env.observation_space.shape[0]
            assert isinstance(actions, torch.Tensor) and actions.shape[1] == self.env.action_space.shape[0]
            assert isinstance(rewards, torch.Tensor) and rewards.shape[1] == 1
            assert isinstance(next_obs, torch.Tensor) and next_obs.shape[1] == self.env.observation_space.shape[0]
            assert isinstance(dones, torch.Tensor) and dones.shape[1] == 1
            
            values = self.critic(obs)
            
            if self.td_method == "td":
                with torch.no_grad():
                    target_values = rewards + self.gamma * self.critic(next_obs) * (1 - dones)
                    advantages = target_values - values
            
            elif self.td_method == "td_n":
                if dones[-1]:
                    last_value = 0
                else:
                    with torch.no_grad():
                        last_value = self.critic(next_obs[-1])
            
                target_values = compute_td_target(rewards, dones, last_value, gamma=self.gamma)
                
                advantages = target_values - values
                
            elif self.td_method == "td_lambda":
                if dones[-1]:
                    last_value = 0
                else:
                    with torch.no_grad():
                        last_value = self.critic(next_obs[-1])
            
                advantages = compute_gae_advantage(rewards, values, dones, last_value, gamma=self.gamma, gae_lambda=self.gae_lambda)
                
                target_values = advantages + values
                
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            dists = self.actor(obs)
            
            log_probs = dists.log_prob(actions)
             
            if self.actor.optimizer.steps % self.actor.optimizer.Ts == 0:
                self.actor.zero_grad()
                
                pg_fisher_loss = log_probs.mean()
                
                self.actor.optimizer.acc_stats = True
                pg_fisher_loss.backward(retain_graph=True)
                self.actor.optimizer.acc_stats = False

            actor_loss = -(log_probs * advantages.detach()).mean()
            
            self.actor.optimizer.zero_grad()
            
            actor_loss.backward()
            
            if self.max_grad_norm is not None:
                clip_grad_norm_(self.actor.optimizer, self.max_grad_norm)
            
            self.actor.optimizer.step()
            
            if self.critic.optimizer.steps % self.critic.optimizer.Ts == 0:
                self.critic.zero_grad()
                
                value_noise = torch.randn(values.size())
                
                sample_values = values + value_noise
                
                vf_fisher_loss = -0.5 * F.smooth_l1_loss(values, sample_values.detach())
                
                self.critic.optimizer.acc_stats = True
                vf_fisher_loss.backward(retain_graph=True)
                self.critic.optimizer.acc_stats = False

            critic_loss = F.smooth_l1_loss(target_values.detach(), values)
            
            self.critic.optimizer.zero_grad()
            
            critic_loss.backward()
            
            if self.max_grad_norm is not None:
                clip_grad_norm_(self.actor.optimizer, self.max_grad_norm)

            self.critic.optimizer.step()
            
if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    env = Monitor(env)
    env = VecEnv(env, num_envs=4)
    acktr = ACKTR(env, 
              rollout_steps=8, 
              total_timesteps=5e5, 
              actor_kwargs={"activation_fn": nn.ReLU}, 
              critic_kwargs={"activation_fn": nn.ReLU},
              max_grad_norm=0.05,
              td_method="td_lambda",
              log_dir=None,
              seed=14,
             )
    acktr.learn()