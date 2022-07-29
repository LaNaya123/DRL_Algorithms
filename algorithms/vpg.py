# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
import gym
import torch
import torch.nn.functional as F
from common.envs import Monitor, VecEnv
from common.policies import OnPolicyAlgorithm
from common.utils import Mish, clip_grad_norm_, compute_gae_advantage, compute_td_target
    
class VPG(OnPolicyAlgorithm):
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
                 log_interval=100,
                 device="auto",
                 seed=12,
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
            verbose, 
            log_interval,
            device,
            seed,
            )
        
    def train(self):
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
                
                advantages = target_values - values
                
            elif self.td_method == "td_lambda":
                if dones[-1]:
                    last_value = 0
                else:
                    last_value = self.critic(next_obs[-1])
            
                advantages = compute_gae_advantage(rewards, values, dones, last_value, gamma=self.gamma, gae_lambda=self.gae_lambda)
                
                target_values = advantages + values
                
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            dists = self.actor(obs)
            
            log_probs = dists.log_prob(actions)
             
            actor_loss = -(log_probs * advantages.detach()).mean()
            
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
        
            critic_loss = F.mse_loss(target_values.detach(), values)
            
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(self.critic.optimizer, 0.5)
            self.critic.optimizer.step()
            
if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    env = Monitor(env)
    vpg = VPG(env, 
              rollout_steps=16, 
              total_timesteps=5e5, 
              actor_kwargs={"activation_fn": Mish}, 
              critic_kwargs={"activation_fn": Mish},
              td_method="td_lambda",
              seed=5,
             )
    vpg.learn()