# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
import gym
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Union, Tuple
from algorithms.value_based.dqn import DQN
from common.envs import Monitor, VecEnv
import common.models as models
from common.buffers import ReplayBuffer
from common.utils import Mish, obs_to_tensor, evaluate_policy
    
class C51(DQN):
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
                 num_atoms: int = 10,
                 v_min: int = 0,
                 v_max: int = 200,
                 exploration_initial_eps: float = 0.2,
                 exploration_final_eps: float = 0.05,
                 exploration_decay_steps: int = 10000,
                 log_dir: Optional[str] = None,
                 log_interval: int = 10,
                 verbose: int = 1,
                 seed: Optional[int] = None,
                ):
        
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.v_step = (v_max - v_min) / (num_atoms - 1)
        v_range = np.linspace(v_min, v_max, num_atoms)
        self.v_range = torch.FloatTensor(v_range)
        
        super(C51, self).__init__(
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
        
        self.q_net = models.C51(self.observation_dim, self.num_actions, self.v_range, self.num_atoms, **self.qnet_kwargs)
        self.target_q_net = models.C51(self.observation_dim, self.num_actions, self.v_range, self.num_atoms, **self.qnet_kwargs)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
            
        if self.verbose > 0:
            print(self.q_net) 

        self.buffer = ReplayBuffer(self.buffer_size, self.n_steps)
        
        self.obs = self.env.reset()
        
    def _rollout(self) -> None:
        for i in range(self.rollout_steps):
            v_dist = self.q_net(obs_to_tensor(self.obs))

            coin = random.random()
            if coin < self.current_eps:
                action = [random.randint(0, self.env.action_space.n - 1) for _ in range(self.env.num_envs)]
                action = np.asarray(action)[:, np.newaxis]
            else:
                q = torch.sum(v_dist * self.v_range.view(1, 1, -1), dim=2)
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
            
        v_probs_next = self.target_q_net(next_obs).detach()
        
        q_next = torch.sum(v_probs_next * self.v_range.view(1, 1, -1), dim=2)
        
        a_next = q_next.max(dim=1)[1]
        
        v_probs_next = torch.stack([v_probs_next[i].index_select(0, a_next[i]) for i in range(self.batch_size * self.env.num_envs)]).squeeze(1)
        
        v_dist_target = rewards + self.gamma * (1 - dones) * self.v_range.unsqueeze(0)
        v_dist_target = torch.clamp(v_dist_target, self.v_min, self.v_max)
        
        v_dist_pos = (v_dist_target - self.v_min) / self.v_step
        
        lower_bound = torch.floor(v_dist_pos).type("torch.LongTensor")
        upper_bound = torch.ceil(v_dist_pos).type("torch.LongTensor")
        
        v_probs_target = torch.zeros((self.batch_size * self.env.num_envs, self.num_atoms))
        
        for i in range(self.batch_size * self.env.num_envs):
            for j in range(self.num_atoms):
                v_probs_target[i, lower_bound[i, j]] += v_probs_next[i][j] * (upper_bound[i][j] - v_dist_pos[i][j])
                v_probs_target[i, upper_bound[i, j]] += v_probs_next[i][j] * (v_dist_pos[i][j] - lower_bound[i][j])
        
            
        v_probs_eval = self.q_net(obs)
        v_probs_eval = torch.stack([v_probs_eval[i].index_select(0, actions[i][0]) for i in range(self.batch_size * self.env.num_envs)]).squeeze(1)

        loss = torch.mean(v_probs_target * (-torch.log(v_probs_eval + 1e-8)))

        self.q_net.optimizer.zero_grad()
        loss.backward()
        self.q_net.optimizer.step()

        if self.training_iterations % self.target_update_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
    
    def save(self, path: str) -> None:
        state_dict = self.q_net.state_dict()
        
        with open(path, "wb") as f:
            torch.save(state_dict, f)
        
        if self.verbose >= 1:
            print("The c51 model has been saved successfully")
    
    def load(self, path: str) -> nn.Module:
        with open(path, "rb") as f:
            state_dict = torch.load(f)
            
            self.q_net = models.C51(self.observation_dim, self.num_actions, self.v_range, self.num_atoms, **self.qnet_kwargs)
            self.q_net.load_state_dict(state_dict)
            self.q_net = self.q_net
 
        if self.verbose >= 1:
            print("The c51 model has been loaded successfully")
            
        return self.q_net
        
if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = Monitor(env)
    env = VecEnv(env, num_envs=4)
    
    c51 = C51(env, 
              rollout_steps=8,
              total_timesteps=1e4,
              gradient_steps=1,
              n_steps=1,
              qnet_kwargs={"activation_fn": Mish, "optimizer_kwargs":{"lr":1e-3}}, 
              learning_start=500,
              buffer_size=5000,
              batch_size=64,
              log_dir=None,
              log_interval=20,
              seed=7,)
    
    c51.learn()
    c51.save("./model.ckpt")
    c51 = c51.load("./model.ckpt")    
    print(evaluate_policy(c51, env))