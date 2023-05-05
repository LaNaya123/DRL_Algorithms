# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
from typing import Any, Dict, Optional, Union, Tuple
import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.envs import Monitor, VecEnv, CliprewardEnv
from common.policies import OffPolicyAlgorithm
import common.models as models
from common.buffers import ReplayBuffer
from common.utils import Mish, obs_to_tensor, evaluate_policy
    
class DQN(OffPolicyAlgorithm):
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
                 exploration_initial_eps: float = 0.2,
                 exploration_final_eps: float = 0.05,
                 exploration_decay_steps: int = 10000,
                 log_dir: Optional[str] = None,
                 log_interval: int = 10,
                 verbose: int = 1,
                 seed: Optional[int] = None,
                ):

        self.qnet_kwargs = qnet_kwargs if qnet_kwargs else {}
        
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_decay_steps = exploration_decay_steps
        self.current_eps = exploration_initial_eps
        
        super(DQN, self).__init__(
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
                 log_dir,
                 log_interval,
                 verbose,
                 seed,
                )
        
    def _setup_model(self) -> None:
        self.observation_dim = self.env.observation_space.shape[0]
        
        self.num_actions = self.env.action_space.n
        
        self.policy_net = models.DQN(self.observation_dim, self.num_actions, **self.qnet_kwargs)
        self.target_policy_net = models.DQN(self.observation_dim, self.num_actions, **self.qnet_kwargs)
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())
            
        if self.verbose > 0:
            print(self.policy_net) 

        self.buffer = ReplayBuffer(self.buffer_size, self.n_steps)
        
        self.obs = self.env.reset()
        
    def _update_exploration_eps(self) -> None:
        if self.current_eps == self.exploration_final_eps:
            return 
        self.current_eps = self.exploration_initial_eps - ((self.exploration_initial_eps - self.exploration_final_eps) / self.exploration_decay_steps) * self.current_timesteps
        self.current_eps = max(self.current_eps, self.exploration_final_eps)

    def _rollout(self) -> None:
        for i in range(self.rollout_steps):
            q = self.policy_net(obs_to_tensor(self.obs))
            
            coin = random.random()
            if coin < self.current_eps:
                action = [random.randint(0, self.env.action_space.n - 1) for _ in range(self.env.num_envs)]
                action = np.asarray(action)[:, np.newaxis]
            else:
                action = q.argmax(dim=-1, keepdim=True).cpu().detach().numpy()

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
            
        q_next = self.target_policy_net(next_obs)
        q_next = q_next.max(dim=1, keepdim=True)[0]
            
        q_target = rewards + self.gamma * (1 - dones) * q_next
            
        q_values = self.policy_net(obs)
            
        q_a = q_values.gather(1, actions)

        loss = F.smooth_l1_loss(q_a, q_target)

        self.policy_net.optimizer.zero_grad()
        loss.backward()
        self.policy_net.optimizer.step()

        if self.training_iterations % self.target_update_interval == 0:
            self.target_policy_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path: str) -> None:
        state_dict = self.policy_net.state_dict()
        
        with open(path, "wb") as f:
            torch.save(state_dict, f)
        
        if self.verbose >= 1:
            print("The dqn model has been saved successfully")
    
    def load(self, path: str) -> nn.Module:
        with open(path, "rb") as f:
            state_dict = torch.load(f)
            
            self.policy_net = models.DQN(self.observation_dim, self.num_actions, **self.qnet_kwargs)
            self.policy_net.load_state_dict(state_dict)
 
        if self.verbose >= 1:
            print("The dqn model has been loaded successfully")
            
        return self.policy_net
        
if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = CliprewardEnv(env)
    env = Monitor(env)
    #env = VecEnv(env, num_envs=4)
    dqn = DQN(env, 
              rollout_steps=8,
              total_timesteps=3e4,
              gradient_steps=1,
              n_steps=1,
              qnet_kwargs={"activation_fn": Mish, "optimizer_kwargs":{"lr":1e-3}}, 
              learning_start=500,
              buffer_size=5000,
              batch_size=64,
              log_dir=None,
              log_interval=20,
              seed=7,)
    
    dqn.learn()
    
    dqn.save("./model.ckpt")
    model = dqn.load("./model.ckpt")
    
    print(evaluate_policy(dqn.policy_net, env))