# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
from typing import Any, Optional, List, Dict
import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.envs import Monitor
from dqn import DQN
from common.utils.utils import Mish, obs_to_tensor
from common.utils.models import SiameseNet

class NGU(DQN):
    def __init__(self, 
                 env: Monitor, 
                 rollout_steps: int = 16,
                 total_timesteps: int = 1e6, 
                 gradient_steps: int = 4,
                 learning_start: int = 1000,
                 buffer_size: int = 10000,
                 batch_size: int = 256,
                 target_update_interval: int = 20,
                 gamma: float = 0.99,
                 beta: float = 0.0001,
                 verbose: int = 1,
                 log_dir: Optional[str] = None,
                 log_interval: int = 10,
                 device: str = "auto",
                 seed: Optional[int] = None,
                 qnet_kwargs: Optional[Dict[str, Any]] = None,
                 exploration_initial_eps: float = 0.2,
                 exploration_final_eps: float = 0.05,
                 exploration_decay_steps: int = 10000,
                ):
        
        super(NGU, self).__init__(
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
        
        self.beta = beta
        
    def _setup_model(self) -> None:
        super()._setup_model()
        self.embedding_model = SiameseNet(
                    self.observation_dim,
                    self.num_actions
                    )
        
        self.episodic_memory = [self.embedding_model.embedding(obs_to_tensor([self.obs]))]
    
    def _compute_intrinsic_reward(self,
                                  controllable_state: torch.Tensor,
                                  k: int = 10,
                                  kernel_cluster_distance: float = 0.008,
                                  kernel_epsilon: float = 0.0001,
                                  c: float = 0.001,
                                  sm: int = 8
                                 ) -> float:
    
        state_dist = [(state, torch.dist(state, controllable_state)) for state in self.episodic_memory]
        state_dist.sort(key=lambda x: x[1])
        state_dist = state_dist[:k]
    
        dist = [d[1].item() for d in state_dist]
        dist = np.array(dist)
        dist = dist / (np.mean(dist) + 1e-7)
        dist = np.max(dist - kernel_cluster_distance, 0)
    
        kernel = kernel_epsilon / (dist + kernel_epsilon)
    
        s = np.sqrt(np.sum(kernel)) + c

        if np.isnan(s) or s > sm:
            return 0
    
        return 1 / s

    def rollout(self) -> None:
        for i in range(self.rollout_steps):
            q = self.qnet(obs_to_tensor([self.obs]).to(self.device))
            
            coin = random.random()
            if coin < self.current_eps:
                action = [random.randint(0, self.env.action_space.n - 1) for _ in range(self.env.num_envs)]
                action = np.asarray(action)[:, np.newaxis]
            else:
                action = q.argmax(dim=-1, keepdim=True).cpu().detach().numpy()

            next_obs, reward, done, info = self.env.step(action)
        
            if done:
                real_obs = info["real_terminal_state"]
                    
                controllable_state = self.embedding_model.embedding(obs_to_tensor([real_obs]))  
            else:
                controllable_state = self.embedding_model.embedding(obs_to_tensor([next_obs]))
                    
            intrinsic_reward = self._compute_intrinsic_reward(controllable_state)
            
            if done:
                self.episodic_memory = [self.embedding_model.embedding(obs_to_tensor([next_obs]))]
            else:
                self.episodic_memory.append(controllable_state)
            
            augmented_reward = reward + self.beta * intrinsic_reward

            self.buffer.add((self.obs, action, augmented_reward, next_obs, done))
                    
            self.obs = next_obs
            
            self.current_timesteps += self.env.num_envs
            
            self._update_episode_info(info)
            
            self._update_exploration_eps()
            
                
    def train(self) -> None:
        obs, actions, rewards, next_obs, dones = super().train()

        preds = self.embedding_model(obs, next_obs)
        labels = F.one_hot(actions.squeeze(dim=1), preds.shape[1]).type("torch.FloatTensor")
        embedding_loss = nn.MSELoss()(preds, labels)

        self.embedding_model.optimizer.zero_grad()
        embedding_loss.backward()
        self.embedding_model.optimizer.step()
  
    
if __name__ == "__main__":
    env = gym.make("FrozenLake-v1")
    env = Monitor(env)
    ngu = NGU(env, 
              total_timesteps=1e6,
              gradient_steps=2,
              rollout_steps=8,
              qnet_kwargs={"activation_fn": Mish, "optimizer_kwargs":{"lr":1e-3}}, 
              learning_start=500,
              buffer_size=5000,
              batch_size=64,
              beta=0.00001,
              log_dir=None,
              log_interval=20,
              seed=2,)
    ngu.learn()