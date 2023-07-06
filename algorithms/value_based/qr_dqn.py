# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Union, Tuple
from algorithms.value_based.dqn import DQN
from common.envs import Monitor, VecEnv
import common.models as models
from common.buffers import ReplayBuffer
from common.utils import Mish, obs_to_tensor, clip_grad_norm_, evaluate_policy


class QRDQN(DQN):
    def __init__(self, 
                 env: Union[Monitor, VecEnv], 
                 rollout_steps: int = 16,
                 total_timesteps: int = 1e6, 
                 gradient_steps: int = 4,
                 n_steps: int = 1,
                 qnet_kwargs: Optional[Dict[str, Any]] = None,
                 learning_start: int = 1000,
                 buffer_size: int = 10000,
                 batch_size: int = 256,
                 target_update_interval: int = 20,
                 gamma: float = 0.99,
                 max_grad_norm: Optional[float] = 0.5,
                 num_quantiles: int = 10,
                 exploration_initial_eps: float = 0.2,
                 exploration_final_eps: float = 0.05,
                 exploration_decay_steps: int = 10000,
                 log_dir: Optional[str] = None,
                 log_interval: int = 10,
                 verbose: int = 1,
                 seed: Optional[int] = None,
                ):
        
        self.num_quantiles = num_quantiles
        tau = np.linspace(0.0, 1.0, self.num_quantiles + 1)[1:]
        self.tau = (np.linspace(0.0, 1.0, self.num_quantiles + 1)[:-1] + tau)/2
        
        super(QRDQN, self).__init__(
                 env, 
                 rollout_steps,
                 total_timesteps, 
                 gradient_steps,
                 n_steps,
                 qnet_kwargs,
                 learning_start,
                 buffer_size,
                 batch_size,
                 target_update_interval,
                 gamma,
                 max_grad_norm,
                 exploration_initial_eps,
                 exploration_final_eps,
                 exploration_decay_steps,
                 log_dir,
                 log_interval,
                 verbose,
                 seed,
            )
        
    def _setup_model(self) -> None:
        self.observation_dim = self.env.observation_space.shape[0]

        self.num_actions = self.env.action_space.n
        
        self.q_net = models.QRDQN(self.observation_dim, self.num_actions, self.num_quantiles, **self.qnet_kwargs)
        self.target_q_net = models.QRDQN(self.observation_dim, self.num_actions, self.num_quantiles, **self.qnet_kwargs)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
            
        if self.verbose >= 1:
            print(self.q_net) 

        self.buffer = ReplayBuffer(self.buffer_size, self.n_steps)
        
        self.obs = self.env.reset()
        
    def _rollout(self) -> None:
        for i in range(self.rollout_steps):
            quantiles = self.q_net(obs_to_tensor(self.obs))

            coin = random.random()
            if coin < self.current_eps:
                action = [random.randint(0, self.env.action_space.n - 1) for _ in range(self.env.num_envs)]
                action = np.asarray(action)[:, np.newaxis]
            else:
                q = quantiles.mean(dim=2)
                action = q.argmax(dim=1, keepdim=True).detach().numpy()

            next_obs, reward, done, info = self.env.step(action)

            self.buffer.add((self.obs, action, reward, next_obs, done))

            self.obs = next_obs
            
            self.current_timesteps += self.env.num_envs
            
            self._update_episode_info(info)
            
            self._update_exploration_eps()
            
    def _train(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
            
        actions = actions.type("torch.LongTensor")
            
        assert isinstance(obs, torch.Tensor) and obs.shape[1] == self.observation_dim
        assert isinstance(actions, torch.Tensor) and actions.shape[1] == 1
        assert isinstance(rewards, torch.Tensor) and rewards.shape[1] == 1
        assert isinstance(next_obs, torch.Tensor) and next_obs.shape[1] == self.observation_dim
        assert isinstance(dones, torch.Tensor) and dones.shape[1] == 1
            
        q_next = self.target_q_net(next_obs).detach()
        a_next = q_next.mean(dim=2).argmax(dim=1)
        q_next = torch.stack([q_next[i].index_select(0, a_next[i]) for i in range(self.batch_size * self.env.num_envs)]).squeeze(dim=1) 

        q_target = rewards + self.gamma * (1 - dones) * q_next
        q_target = q_target.unsqueeze(dim=1)
        
        q = self.q_net(obs)
        q = torch.stack([q[i].index_select(0, actions[i]) for i in range(self.batch_size * self.env.num_envs)]).squeeze(dim=1) 
        q = q.unsqueeze(dim=2)

        self.tau = torch.FloatTensor(self.tau).view(1, -1, 1)
        
        weights = torch.abs(self.tau - (q_target - q).le(0).float())
        
        huber_loss = F.smooth_l1_loss(q, q_target, reduction="none")
        
        loss = (weights * huber_loss).mean()

        self.q_net.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm:
            clip_grad_norm_(self.q_net.optimizer, self.max_grad_norm)
        self.q_net.optimizer.step()

        if self.training_iterations % self.target_update_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
    
    def save(self, path: str) -> None:
        state_dict = self.q_net.state_dict()
        
        with open(path, "wb") as f:
            torch.save(state_dict, f)
        
        if self.verbose >= 1:
            print("The qr_dqn model has been saved successfully")
    
    def load(self, path: str) -> nn.Module:
        with open(path, "rb") as f:
            state_dict = torch.load(f)
            
            self.q_net = models.QRDQN(self.observation_dim, self.num_actions, self.num_quantiles, **self.qnet_kwargs)
            self.q_net.load_state_dict(state_dict)
            self.q_net = self.q_net
 
        if self.verbose >= 1:
            print("The qr_dqn model has been loaded successfully")
            
        return self.q_net
        
if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = Monitor(env)
    env = VecEnv(env, num_envs=1)
    
    qr_dqn = QRDQN(env, 
              rollout_steps=8,
              total_timesteps=1e6,
              gradient_steps=1,
              n_steps=1,
              qnet_kwargs={"activation_fn": Mish, "optimizer_kwargs":{"lr":1e-3}}, 
              learning_start=500,
              buffer_size=5000,
              batch_size=64,
              max_grad_norm=None,
              log_dir=None,
              log_interval=20,
              seed=7,)
    
    qr_dqn.learn()
    qr_dqn.save("./model.ckpt")
    qr_dqn = qr_dqn.load("./model.ckpt")    
    print(evaluate_policy(qr_dqn, env))     