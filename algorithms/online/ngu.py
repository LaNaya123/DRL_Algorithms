# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
from typing import Any, Optional, Union, Dict
import gym
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.envs import Monitor, VecEnv
from dqn import DQN
from common.utils import Mish, SiameseNet, obs_to_tensor, compute_ngu_intrinsic_reward

class NGU(DQN):
    def __init__(self, 
                 env: Union[Monitor, VecEnv], 
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
                 learning_start,
                 buffer_size,
                 batch_size,
                 target_update_interval,
                 gamma,
                 verbose,
                 log_dir,
                 log_interval,
                 device,
                 seed,
                 qnet_kwargs,
                 exploration_initial_eps,
                 exploration_final_eps,
                 exploration_decay_steps,
            )
        
        self.beta = beta
        
    def _setup_model(self):
        super()._setup_model()
        self.embedding_model = SiameseNet(
                    self.observation_dim,
                    self.num_actions
                    )
        
        self.episodic_memory = [self.embedding_model.embedding(obs_to_tensor([self.obs]))]

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
                    
            intrinsic_reward = compute_ngu_intrinsic_reward(self.episodic_memory, controllable_state)
            
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