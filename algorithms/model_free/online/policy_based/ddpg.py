# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
from typing import Any, Dict, Optional, Union
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.envs import Monitor, VecEnv
from common.policies import OffPolicyAlgorithm
from common.models import DDPGActor, QCritic
from common.buffers import ReplayBuffer
from common.utils.functionality import OrnsteinUhlenbeckNoise, Mish, obs_to_tensor, evaluate_policy

class DDPG(OffPolicyAlgorithm):
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
                 gamma: float = 0.99,
                 verbose: int = 1,
                 log_dir: Optional[str] = None,
                 log_interval: int = 10,
                 device: str = "auto",
                 seed: Optional[int] = None,
                 actor_kwargs: Optional[Dict[str, Any]] = None,
                 critic_kwargs: Optional[Dict[str, Any]] = None,
                 ou_noise: Optional[OrnsteinUhlenbeckNoise] = None,
                 tau: float = 0.95,
                ):
        
        self.actor_kwargs = {} if actor_kwargs is None else actor_kwargs
        self.critic_kwargs = {} if critic_kwargs is None else critic_kwargs
        self.ou_noise = ou_noise
        self.tau = tau 
        
        super(DDPG, self).__init__(
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
                 verbose,
                 log_dir, 
                 log_interval,
                 device,
                 seed,
            )
        
    def _setup_model(self) -> None:
        self.observation_dim = self.env.observation_space.shape[0]
        
        self.num_actions = self.env.action_space.shape[0]
        
        self.policy_net = DDPGActor(self.observation_dim, self.num_actions, **self.actor_kwargs).to(self.device)
        self.target_policy_net = DDPGActor(self.observation_dim, self.num_actions, **self.actor_kwargs).to(self.device)
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        
        self.value_net = QCritic(self.observation_dim, self.num_actions, **self.critic_kwargs).to(self.device)
        self.target_value_net = QCritic(self.observation_dim, self.num_actions, **self.critic_kwargs).to(self.device)
        self.target_value_net.load_state_dict(self.value_net.state_dict())
            
        if self.verbose > 0:
            print(self.policy_net)
            print(self.value_net)

        self.buffer = ReplayBuffer(self.buffer_size, self.device)
        
        self.obs = self.env.reset()
        
    def _polyak_update(self, online, target) -> None:
        for target_param, param in zip(target.parameters(), online.parameters()):
            target_param.data.copy_(param.data * (1.0 - self.tau) + target_param.data * self.tau)
        
    def _rollout(self) -> None:
        for i in range(self.rollout_steps):
            action = self.policy_net(obs_to_tensor(self.obs).to(self.device)).cpu().detach().numpy()
            
            action *= self.env.action_space.high
            
            if self.ou_noise:
                action = action + self.ou_noise()
            
            next_obs, reward, done, info = self.env.step(action)
            
            self.buffer.add((self.obs, action, reward/100., next_obs, done))
            
            self.obs = next_obs
            
            self.current_timesteps += self.env.num_envs
            
            self._update_episode_info(info)
    
    def _train(self) -> None:
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        assert isinstance(obs, torch.Tensor) and obs.shape[1] == self.env.observation_space.shape[0]
        assert isinstance(actions, torch.Tensor) and actions.shape[1] == self.env.action_space.shape[0]
        assert isinstance(rewards, torch.Tensor) and rewards.shape[1] == 1
        assert isinstance(next_obs, torch.Tensor) and next_obs.shape[1] == self.env.observation_space.shape[0]
        assert isinstance(dones, torch.Tensor) and dones.shape[1] == 1
        
        with torch.no_grad():
            next_a = self.target_policy_net(next_obs)
            next_q = self.target_value_net(next_obs, next_a)
            target_q = rewards + self.gamma * (1 - dones) * next_q
            
        critic_loss = F.smooth_l1_loss(self.value_net(obs, actions), target_q)
            
        self.value_net.optimizer.zero_grad()
        critic_loss.backward()
        self.value_net.optimizer.step()
    
        actor_loss = -self.value_net(obs,self.policy_net(obs)).mean()
            
        self.policy_net.optimizer.zero_grad()
        actor_loss.backward()
        self.policy_net.optimizer.step()
            
        if self.training_iterations % self.target_update_interval == 0:
            self._polyak_update(self.policy_net, self.target_policy_net)
            self._polyak_update(self.value_net, self.target_value_net)
    
    def save(self, path: str) -> None:
        state_dict = self.policy_net.state_dict()
        
        with open(path, "wb") as f:
            torch.save(state_dict, f)
        
        if self.verbose >= 1:
            print("The ddpg model has been saved successfully")
    
    def load(self, path: str) -> nn.Module:
        with open(path, "rb") as f:
            state_dict = torch.load(f)
            
            self.policy_net = DDPGActor(self.observation_dim, self.num_actions, **self.actor_kwargs)
            self.policy_net.load_state_dict(state_dict)
            self.policy_net = self.policy_net.to(self.device)
 
        if self.verbose >= 1:
            print("The ddpg model has been loaded successfully")
            
        return self.policy_net
    
        
if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    env = Monitor(env)
    #env = VecEnv(env, num_envs=4)
    ou_noise = OrnsteinUhlenbeckNoise(np.zeros(env.action_space.shape[0]))
    ddpg = DDPG(env, 
              total_timesteps=1e4, 
              gradient_steps=4,
              rollout_steps=8, 
              n_steps=1,
              learning_start=500,
              buffer_size=10000,
              batch_size=64,
              target_update_interval=1,
              log_dir=None,
              log_interval=80,
              seed=12,
              actor_kwargs={"hidden_size":32, "activation_fn": Mish, "optimizer_kwargs":{"lr":5e-4}}, 
              critic_kwargs={"activation_fn": Mish, "optimizer_kwargs":{"lr":1e-3}},
              ou_noise=ou_noise)
    
    ddpg.learn()
    
    ddpg.save("./model.ckpt")
    model = ddpg.load("./model.ckpt")
    
    print(evaluate_policy(ddpg.policy_net, env))