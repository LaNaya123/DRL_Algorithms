# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Union, Optional, Dict
import common.models as models
from common.envs import Monitor, VecEnv
from common.buffers import ReplayBuffer
from common.policies import OffPolicyAlgorithm
from common.utils import Mish, clip_grad_norm_, obs_to_tensor, evaluate_policy

class SAC(OffPolicyAlgorithm):
    def __init__(
                 self, 
                 env: Union[Monitor, VecEnv], 
                 rollout_steps: int = 16,
                 total_timesteps: int = 1e6, 
                 gradient_steps: int = 4,
                 n_steps: int = 1,
                 learning_start: int = 1000,
                 buffer_size: int = 10000,
                 batch_size: int = 256,
                 target_update_interval: int = 20,
                 actor_kwargs: Optional[Dict[str, Any]] = None,
                 critic_kwargs: Optional[Dict[str, Any]] = None,
                 alpha: float = 0.01,
                 tau: float = 0.95,
                 gamma: float = 0.99,
                 max_grad_norm: Optional[float] = 0.5,
                 log_dir: Optional[str] = None,
                 log_interval: int = 10,
                 verbose: int = 1,
                 seed: Optional[int] = None,
                ):
        
        self.actor_kwargs = {} if actor_kwargs is None else actor_kwargs
        self.critic_kwargs = {} if critic_kwargs is None else critic_kwargs
        self.alpha = alpha        
        self.tau = tau 
        
        super(SAC, self).__init__( 
                 env, 
                 rollout_steps,
                 total_timesteps, 
                 gradient_steps,
                 n_steps,
                 learning_start,
                 buffer_size,
                 batch_size,
                 target_update_interval,
                 gamma,
                 max_grad_norm,
                 log_dir, 
                 log_interval,
                 verbose,
                 seed,
             )

    def _setup_model(self):
        self.observation_dim = self.env.observation_space.shape[0]

        self.num_actions = self.env.action_space.shape[0]

        self.policy_net = models.VPG(self.observation_dim, self.num_actions, **self.actor_kwargs)

        self.value_net = models.Q1(self.observation_dim, self.num_actions, **self.critic_kwargs)
        self.target_value_net = models.Q1(self.observation_dim, self.num_actions, **self.critic_kwargs)
        self.target_value_net.load_state_dict(self.value_net.state_dict())
            
        if self.verbose > 0:
            print(self.policy_net)
            print(self.value_net)
        
        self.buffer = ReplayBuffer(self.buffer_size)
            
        self.obs = self.env.reset()
            
    def _polyak_update(self, online: nn.Module, target: nn.Module) -> None:
        for target_param, param in zip(target.parameters(), online.parameters()):
            target_param.data.copy_(param.data * (1.0 - self.tau) + target_param.data * self.tau)
        
    def _rollout(self) -> None:
        for i in range(self.rollout_steps):
            dist = self.policy_net(obs_to_tensor(self.obs))
            
            action = dist.sample().detach().numpy()
            
            clipped_action = np.clip(action, self.env.action_space.low.min(), self.env.action_space.high.max())
            
            next_obs, reward, done, info = self.env.step(clipped_action)
            
            self.buffer.add((self.obs, action, reward/100., next_obs, done))
            
            self.obs = next_obs
            
            self.current_timesteps += self.env.num_envs
            
            self._update_episode_info(info)
                
    def _train(self) -> None:
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        assert isinstance(obs, torch.Tensor) and obs.shape[1] == self.env.observation_space.shape[0]
        assert isinstance(actions, torch.Tensor) and actions.shape[1] == self.env.action_space.shape[0]
        assert isinstance(rewards, torch.Tensor) and rewards.shape[1] == 1
        assert isinstance(next_obs, torch.Tensor) and next_obs.shape[1] == self.env.observation_space.shape[0]
        assert isinstance(dones, torch.Tensor) and dones.shape[1] == 1
        
        with torch.no_grad():
            next_dist = self.policy_net(next_obs)
            next_a = next_dist.sample()
            next_log_prob = next_dist.log_prob(next_a).sum(dim=1, keepdim=True)
            next_q = self.target_value_net(next_obs, next_a)
                        
        target_q = rewards + self.gamma * (1 - dones) * next_q - self.alpha * next_log_prob
        value_loss = F.smooth_l1_loss(self.value_net(obs, actions), target_q)

            
        self.value_net.optimizer.zero_grad()
        value_loss.backward()
        if self.max_grad_norm:
            clip_grad_norm_(self.value_net.optimizer, self.max_grad_norm)
        self.value_net.optimizer.step()
            
        dist = self.policy_net(obs)
        a = dist.rsample()
        log_prob = dist.log_prob(a).sum(dim=1, keepdim=True)
        q = self.value_net(obs, a)

        policy_loss = (self.alpha * log_prob - q).mean()

        self.policy_net.optimizer.zero_grad()
        policy_loss.backward()
        if self.max_grad_norm:
            clip_grad_norm_(self.policy_net.optimizer, self.max_grad_norm)
        self.policy_net.optimizer.step()
                    
        if self.training_iterations % self.target_update_interval == 0:
            self._polyak_update(self.value_net, self.target_value_net)
                
    def save(self, path: str) -> None:
        state_dict = self.policy_net.state_dict()
        
        with open(path, "wb") as f:
            torch.save(state_dict, f)
        
        if self.verbose > 0:
            print("The sac model has been saved successfully")
    
    def load(self, path: str) -> nn.Module:
        with open(path, "rb") as f:
            state_dict = torch.load(f)
            
            self.policy_net = models.VPG(self.observation_dim, self.num_actions, **self.actor_kwargs)
            self.policy_net.load_state_dict(state_dict)
            self.policy_net = self.policy_net
 
        if self.verbose > 0:
            print("The sac model has been loaded successfully")
            
        return self.policy_net
    
        
if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    env = Monitor(env)
    
    sac = SAC(env, 
              total_timesteps=1e2, 
              gradient_steps=4,
              rollout_steps=16, 
              n_steps=1,
              actor_kwargs={"activation_fn": Mish, "optimizer_kwargs":{"lr":1e-4}}, 
              critic_kwargs={"activation_fn": Mish, "optimizer_kwargs":{"lr":1e-3}},
              learning_start=500,
              buffer_size=10000,
              batch_size=32,
              target_update_interval=1,
              log_dir=None,
              log_interval=80,
              seed=1,
             )
    
    sac.learn()    
    sac.save("./model.ckpt")
    sac = sac.load("./model.ckpt")
    print(evaluate_policy(sac, env))