# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Union
from algorithms.policy_based.ddpg import DDPG
import common.models as models
from common.envs import Monitor, VecEnv
from common.buffers import ReplayBuffer
from common.utils import OrnsteinUhlenbeckNoise, Mish, evaluate_policy

class TD3(DDPG):
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
                 policy_delay: int = 2,
                 ou_noise: Optional[OrnsteinUhlenbeckNoise] = None,
                 tau: float = 0.95,
                 gamma: float = 0.99,
                 max_grad_norm: Optional[float] = 0.5,
                 log_dir: Optional[str] = None,
                 log_interval: int = 10,
                 verbose: int = 1,
                 seed: Optional[int] = None,
                ):
    
        super(TD3, self).__init__( 
                 env, 
                 rollout_steps,
                 total_timesteps, 
                 gradient_steps,
                 n_steps,
                 learning_start,
                 buffer_size,
                 batch_size,
                 target_update_interval,
                 actor_kwargs,
                 critic_kwargs,
                 ou_noise,
                 tau,
                 gamma,
                 max_grad_norm,
                 log_dir,
                 log_interval,
                 verbose,
                 seed,
             )
        
        self.policy_delay = policy_delay
        
    def _setup_model(self) -> None:
        self.observation_dim = self.env.observation_space.shape[0]
        
        self.num_actions = self.env.action_space.shape[0]
        
        self.policy_net = models.DDPG(self.observation_dim, self.num_actions, **self.actor_kwargs)
        self.target_policy_net = models.DDPG(self.observation_dim, self.num_actions, **self.actor_kwargs)
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        
        self.value_net1 = models.Q1(self.observation_dim, self.num_actions, **self.critic_kwargs)
        self.target_value_net1 = models.Q1(self.observation_dim, self.num_actions, **self.critic_kwargs)
        self.target_value_net1.load_state_dict(self.value_net1.state_dict())
        
        self.value_net2 = models.Q1(self.observation_dim, self.num_actions, **self.critic_kwargs)
        self.target_value_net2 = models.Q1(self.observation_dim, self.num_actions, **self.critic_kwargs)
        self.target_value_net2.load_state_dict(self.value_net2.state_dict())
            
        if self.verbose > 0:
            print(self.policy_net)
            print(self.value_net1)

        self.buffer = ReplayBuffer(self.buffer_size)
        
        self.obs = self.env.reset()
        
    def _train(self) -> None:
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        assert isinstance(obs, torch.Tensor) and obs.shape[1] == self.env.observation_space.shape[0]
        assert isinstance(actions, torch.Tensor) and actions.shape[1] == self.env.action_space.shape[0]
        assert isinstance(rewards, torch.Tensor) and rewards.shape[1] == 1
        assert isinstance(next_obs, torch.Tensor) and next_obs.shape[1] == self.env.observation_space.shape[0]
        assert isinstance(dones, torch.Tensor) and dones.shape[1] == 1
        
        with torch.no_grad():
            next_a = self.target_policy_net(next_obs)
            
            next_q1 = self.target_value_net1(next_obs, next_a)
            next_q2 = self.target_value_net2(next_obs, next_a)
            
            next_q = torch.min(torch.cat([next_q1, next_q2], dim=1), dim=1, keepdim=True)[0]

            target_q = rewards + self.gamma * (1 - dones) * next_q
            
        critic_loss1 = F.smooth_l1_loss(self.value_net1(obs, actions), target_q)
        self.value_net1.optimizer.zero_grad()
        critic_loss1.backward()
        self.value_net1.optimizer.step()
        
        critic_loss2 = F.smooth_l1_loss(self.value_net2(obs, actions), target_q)
        self.value_net2.optimizer.zero_grad()
        critic_loss2.backward()
        self.value_net2.optimizer.step()
        
        if self.training_iterations % self.policy_delay == 0:
            actor_loss = -self.value_net1(obs, self.policy_net(obs)).mean()
            self.policy_net.optimizer.zero_grad()
            actor_loss.backward()
            self.policy_net.optimizer.step()
            
        if self.training_iterations % self.target_update_interval == 0:
            self._polyak_update(self.policy_net, self.target_policy_net)
            self._polyak_update(self.value_net1, self.target_value_net1)
            self._polyak_update(self.value_net2, self.target_value_net2)

    def save(self, path: str) -> None:
        state_dict = self.policy_net.state_dict()
        
        with open(path, "wb") as f:
            torch.save(state_dict, f)
        
        if self.verbose >= 1:
            print("The td3 model has been saved successfully")
    
    def load(self, path: str) -> nn.Module:
        with open(path, "rb") as f:
            state_dict = torch.load(f)
            
            self.policy_net = models.DDPG(self.observation_dim, self.num_actions, **self.actor_kwargs)
            self.policy_net.load_state_dict(state_dict)
            self.policy_net = self.policy_net
 
        if self.verbose >= 1:
            print("The td3 model has been loaded successfully")
            
        return self.policy_net
    
        
if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    env = Monitor(env)
    env = VecEnv(env, num_envs=1)
    
    ou_noise = OrnsteinUhlenbeckNoise(np.zeros(env.action_space.shape[0]))
    
    td3 = TD3(env, 
              total_timesteps=1e6, 
              gradient_steps=4,
              rollout_steps=8, 
              n_steps=1,
              learning_start=1000,
              buffer_size=10000,
              batch_size=64,
              target_update_interval=1,
              policy_delay=1,
              log_dir=None,
              log_interval=80,
              seed=12,
              actor_kwargs={"hidden_size":32, "activation_fn": Mish, "optimizer_kwargs":{"lr":5e-4}}, 
              critic_kwargs={"activation_fn": Mish, "optimizer_kwargs":{"lr":1e-3}},
              ou_noise=ou_noise)
    
    td3.learn()
    td3.save("./model.ckpt")
    td3 = td3.load("./model.ckpt")
    print(evaluate_policy(td3, env))