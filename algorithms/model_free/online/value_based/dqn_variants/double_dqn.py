# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
from typing import Any, Dict, Optional, Union, Tuple
import gym
import gym.spaces as spaces
import random
import numpy as np
import torch
import torch.nn.functional as F
from dqn import DQN
from common.envs import Monitor, VecEnv
from common.models import DeepQNetwork
from common.buffers import ReplayBuffer
from common.utils.utils import Mish, obs_to_tensor
    
class DDQN(DQN):
    def __init__(self, 
                 env: Union[Monitor, VecEnv], 
                 rollout_steps: int = 16,
                 total_timesteps: int = 1e6, 
                 gradient_steps: int = 4,
                 qnet_kwargs: Optional[Dict[str, Any]] = None,
                 learning_start: int = 1000,
                 buffer_size: int = 10000,
                 batch_size: int = 256,
                 target_update_interval: int = 20,
                 gamma: float = 0.99,
                 exploration_initial_eps: float = 0.2,
                 exploration_final_eps: float = 0.05,
                 exploration_decay_steps: int = 10000,
                 verbose: int = 1,
                 log_dir: Optional[str] = None,
                 log_interval: int = 10,
                 device: str = "auto",
                 seed: Optional[int] = None,
                ):
        
        super(DDQN, self).__init__(
                 env, 
                 rollout_steps,
                 total_timesteps, 
                 gradient_steps,
                 qnet_kwargs,
                 learning_start,
                 buffer_size,
                 batch_size,
                 target_update_interval,
                 gamma,
                 exploration_initial_eps,
                 exploration_final_eps,
                 exploration_decay_steps,
                 verbose,
                 log_dir,
                 log_interval,
                 device,
                 seed,
                )

    def _train(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
            
        actions = actions.type("torch.LongTensor").to(self.device)
            
        assert isinstance(obs, torch.Tensor) and obs.shape[1] == self.observation_dim
        assert isinstance(actions, torch.Tensor) and actions.shape[1] == 1
        assert isinstance(rewards, torch.Tensor) and rewards.shape[1] == 1
        assert isinstance(next_obs, torch.Tensor) and next_obs.shape[1] == self.observation_dim
        assert isinstance(dones, torch.Tensor) and dones.shape[1] == 1
            
        q_next = self.qnet(next_obs)        
        next_acts = q_next.max(dim=1, keepdim=True)[1]
        q_next = self.target_qnet(next_obs)
        q_next = q_next.gather(dim=1, index=next_acts)
        
        q_target = rewards + self.gamma * (1 - dones) * q_next
            
        q_values = self.qnet(obs)
            
        q_a = q_values.gather(1, actions)

        loss = F.smooth_l1_loss(q_a, q_target)

        self.qnet.optimizer.zero_grad()
        loss.backward()
        self.qnet.optimizer.step()

        if self.training_iterations % self.target_update_interval == 0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())
                
        return (obs, actions, rewards, next_obs, dones)

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = Monitor(env)
    #env = VecEnv(env, num_envs=4)
    ddqn = DDQN(env, 
              rollout_steps=8,
              total_timesteps=1e6,
              gradient_steps=1,
              qnet_kwargs={"activation_fn": Mish, "optimizer_kwargs":{"lr":1e-3}}, 
              learning_start=500,
              buffer_size=5000,
              batch_size=64,
              log_dir=None,
              log_interval=20,
              seed=2,)
    ddqn.learn()