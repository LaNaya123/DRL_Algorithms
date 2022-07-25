# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:06:14 2022

@author: lanaya
"""

import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.envs import Monitor, VecEnv
from common.policies import OffPolicyAlgorithm
from common.utils import Mish, clip_grad_norm_, compute_gae_advantage, compute_td_target
    
class DQN(OffPolicyAlgorithm):
    def __init__(self, 
                 env, 
                 rollout_steps,
                 total_timesteps, 
                 qnet_kwargs=None,
                 learning_start=1000,
                 buffer_size=10000,
                 batch_size=256,
                 target_update_interval=20,
                 gamma=0.99,
                 verbose=1,
                 log_interval=10,
                 device="auto",
                 seed=None,
                ):
        
        super(DQN, self).__init__(
                 env, 
                 rollout_steps,
                 total_timesteps, 
                 qnet_kwargs,
                 None,
                 None,
                 learning_start,
                 buffer_size,
                 batch_size,
                 target_update_interval,
                 gamma,
                 verbose,
                 log_interval,
                 "value_based",
                 device,
                 seed,
                )
        
    def train(self):

            obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
            
            actions = actions.type("torch.LongTensor")
            
            assert isinstance(obs, torch.Tensor) and obs.shape[1] == self.env.observation_space.shape[0]
            if isinstance(self.env.action_space, spaces.Discrete):
                assert isinstance(actions, torch.Tensor) and actions.shape[1] == 1
            elif isinstance(self.env.action_space, spaces.Box):
                assert isinstance(actions, torch.Tensor) and actions.shape[1] == self.env.action_space.shape[0]
            assert isinstance(rewards, torch.Tensor) and rewards.shape[1] == 1
            assert isinstance(next_obs, torch.Tensor) and next_obs.shape[1] == self.env.observation_space.shape[0]
            assert isinstance(dones, torch.Tensor) and dones.shape[1] == 1
            
            q_next = self.target_qnet(next_obs)
            q_next = q_next.max(dim=1, keepdim=True)[0]
            
            q_target = rewards + self.gamma * (1 - dones) * q_next
            
            q_values = self.qnet(obs)
            
            q_a = q_values.gather(1,actions)

            loss = F.smooth_l1_loss(q_a, q_target)

            self.qnet.optimizer.zero_grad()
            loss.backward()
            self.qnet.optimizer.step()
            
            if self.training_iterations % self.target_update_interval == 0:
                self.target_qnet.load_state_dict(self.qnet.state_dict())

        
            
if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = Monitor(env)
    dqn = DQN(env, 
              total_timesteps=5e5, 
              rollout_steps=8, 
              qnet_kwargs={"activation_fn": Mish, "optimizer_kwargs":{"lr":1e-3}}, 
              learning_start=100,
              buffer_size=2000,
              batch_size=64,
              log_interval=20,
              seed=1,)
    dqn.learn()