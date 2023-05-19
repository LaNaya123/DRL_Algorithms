# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
from typing import Any, Dict, Optional, Union
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from common.envs import Monitor, VecEnv
import common.models as models
from common.buffers import RolloutBuffer
from common.policies import OnPolicyAlgorithm
from common.optimizers import KFACOptimizer
from common.utils import clip_grad_norm_, compute_gae_advantage, compute_td_target, obs_to_tensor, evaluate_policy

class ACKTR(OnPolicyAlgorithm):
    def __init__(self, 
                 env: Union[Monitor, VecEnv], 
                 rollout_steps: int = 16, 
                 total_timesteps: int = 1e5, 
                 actor_kwargs: Optional[Dict[str, Any]] = None,
                 critic_kwargs: Optional[Dict[str, Any]] = None,
                 td_method: str = "td_lambda",
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 max_grad_norm: Optional[float] = None,
                 log_dir: str = None,
                 log_interval: int = 100,
                 verbose: int = 1,
                 seed: int = 0,
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
            log_dir,
            log_interval,
            verbose, 
            seed,
            )
        
    def _setup_model(self) -> None:
        self.observation_dim = self.env.observation_space.shape[0]
        
        if isinstance(self.env.action_space, spaces.Discrete):
            self.num_actions = self.env.action_space.n
        elif isinstance(self.env.action_space, spaces.Box):
            self.num_actions = self.env.action_space.shape[0]

        self.policy_net = models.ACKTR(self.observation_dim, self.num_actions, **self.actor_kwargs)
        
        if not self.critic_kwargs:
            self.critic_kwargs = {"optimizer": KFACOptimizer, "optimizer_kwargs":{"lr": 0.25}}
        else:
            if "optimizer" not in self.critic_kwargs:
                self.critic_kwargs["optimizer"] = KFACOptimizer
            else:
                if self.critic_kwargs["optimizer"] != KFACOptimizer:
                    raise TypeError("The optimizer must be K-FAC optimizer")
                    
            if "optimizer_kwargs" not in self.critic_kwargs: 
                self.critic_kwargs["optimizer_kwargs"] = {"lr": 0.25}
            else:
                if "lr" not in self.critic_kwargs["optimizer_kwargs"]:
                    self.critic_kwargs["optimizer_kwargs"]["lr"] = 0.25
        self.value_net = models.V(self.observation_dim, **self.critic_kwargs)
        
        if self.verbose > 0:
            print(self.policy_net)
            print(self.value_net)
        
        self.buffer = RolloutBuffer(self.rollout_steps)
        
        self.obs = self.env.reset()
        
    def _rollout(self) -> None:
        self.buffer.reset()
        
        with torch.no_grad():
            
            for i in range(self.rollout_steps):
                dist = self.policy_net(obs_to_tensor(self.obs))
                
                action = dist.sample().detach()
                
                log_prob = dist.log_prob(action).numpy()
                
                action = action.numpy()

                if not self.env.is_vec:
                    action = action.squeeze(axis=0)
                    
                clipped_action = np.clip(action, self.env.action_space.low.min(), self.env.action_space.high.max())
            
                next_obs, reward, done, info = self.env.step(clipped_action)
            
                self.buffer.add((self.obs, action, reward, next_obs, done, log_prob))
            
                self.obs = next_obs
            
                self.current_timesteps += self.env.num_envs
            
                self._update_episode_info(info)
        
    def _train(self) -> None:
            obs, actions, rewards, next_obs, dones, log_probs = self.buffer.get()
            
            assert isinstance(obs, torch.Tensor) and obs.shape[1] == self.env.observation_space.shape[0]
            assert isinstance(actions, torch.Tensor) and actions.shape[1] == self.env.action_space.shape[0]
            assert isinstance(rewards, torch.Tensor) and rewards.shape[1] == 1
            assert isinstance(next_obs, torch.Tensor) and next_obs.shape[1] == self.env.observation_space.shape[0]
            assert isinstance(dones, torch.Tensor) and dones.shape[1] == 1
            
            values = self.value_net(obs)
            
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
                target_values = target_values.to(self.device)
                
                advantages = target_values - values
                
            elif self.td_method == "td_lambda":
                if dones[-1]:
                    last_value = 0
                else:
                    with torch.no_grad():
                        last_value = self.value_net(next_obs[-1])
            
                advantages = compute_gae_advantage(rewards, values, dones, last_value, gamma=self.gamma, gae_lambda=self.gae_lambda)
                
                target_values = advantages + values
                
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            dists = self.policy_net(obs)
            
            log_probs = dists.log_prob(actions)
             
            if self.policy_net.optimizer.steps % self.policy_net.optimizer.Ts == 0:
                self.policy_net.zero_grad()
                
                pg_fisher_loss = log_probs.mean()
                
                self.policy_net.optimizer.acc_stats = True
                pg_fisher_loss.backward(retain_graph=True)
                self.policy_net.optimizer.acc_stats = False

            self.policy_loss = -(log_probs * advantages.detach()).mean()
            
            self.policy_net.optimizer.zero_grad()
            
            self.policy_loss.backward()
            
            if self.max_grad_norm is not None:
                clip_grad_norm_(self.policy_net.optimizer, self.max_grad_norm)
            
            self.policy_net.optimizer.step()
            
            if self.value_net.optimizer.steps % self.value_net.optimizer.Ts == 0:
                self.value_net.zero_grad()
                
                value_noise = torch.randn(values.size())
                
                sample_values = values + value_noise
                
                vf_fisher_loss = -0.5 * F.smooth_l1_loss(values, sample_values.detach())
                
                self.value_net.optimizer.acc_stats = True
                vf_fisher_loss.backward(retain_graph=True)
                self.value_net.optimizer.acc_stats = False

            self.value_loss = F.smooth_l1_loss(target_values.detach(), values)
            
            self.value_net.optimizer.zero_grad()
            
            self.value_loss.backward()
            
            if self.max_grad_norm is not None:
                clip_grad_norm_(self.value_net.optimizer, self.max_grad_norm)
            
            self.value_net.optimizer.step()
            
    def save(self, path: str) -> None:
        state_dict = self.policy_net.state_dict()
        
        with open(path, "wb") as f:
            torch.save(state_dict, f)
        
        if self.verbose >= 1:
            print("The acktr model has been saved successfully")
    
    def load(self, path: str) -> nn.Module:
        with open(path, "rb") as f:
            state_dict = torch.load(f)
            
            self.policy_net = models.ACKTR(self.observation_dim, self.num_actions, **self.actor_kwargs)
            self.policy_net.load_state_dict(state_dict)
            self.policy_net = self.policy_net
 
        if self.verbose >= 1:
            print("The acktr model has been loaded successfully")
            
        return self.policy_net

if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    env = Monitor(env)
    env = VecEnv(env, num_envs=1)
    
    acktr = ACKTR(env, 
              rollout_steps=8, 
              total_timesteps=1e2,
              critic_kwargs={"activation_fn": nn.ReLU},
              max_grad_norm=0.5,
              td_method="td_lambda",
              log_dir="C:/Users/lanaya/Desktop",
              seed=22,
             )
    
    acktr.learn()
    acktr.save("./model.pkl")
    acktr = acktr.load("./model.pkl")
    print(evaluate_policy(acktr, env, num_eval_episodes=2))