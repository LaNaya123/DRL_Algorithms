# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
from typing import Any, Optional, Dict 
import numpy as np
import torch
import torch.nn.functional as F
import gym
import random
from algorithms.model_free.online.policy_based.ppo import PPO
from common.envs import Monitor
from common.models import VPGActor, VCritic
from common.buffers import RolloutBuffer
from common.utils.utils import Mish, obs_to_tensor, compute_gae_advantage

class AWR():
    def __init__(
                 self,
                 env: Monitor,
                 buffer: Optional[RolloutBuffer] = None,
                 buffer_size: Optional[int] = None,
                 num_iterations: int = 1000,
                 batch_size: int = 64, 
                 eval_interval: int = 10,
                 eval_episodes: int = 5,
                 gamma: int = 0.99,
                 gae_lambda: float = 0.95,
                 beta: float = 2.5,
                 verbose: int = 1,
                 log_dir: Optional[str] = None,
                 log_interval: int = 10,
                 device: str = "auto",
                 seed: Optional[int] = None,
                 actor_kwargs: Optional[Dict[str, Any]] = None,
                 critic_kwargs: Optional[Dict[str, Any]] = None,
                ):
        
        self.env = env
        self.buffer = buffer
        self.buffer_size = buffer_size
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.beta = beta
        self.verbose = verbose
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.seed = seed
        self.actor_kwargs = {} if actor_kwargs is None else actor_kwargs
        self.critic_kwargs = {} if critic_kwargs is None else critic_kwargs
    
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        if self.verbose > 0:
            print(f"Using the device: {self.device}")
            
        self._setup_seed()
        self._setup_model()
        
    def _setup_seed(self):
        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.env.seed(self.seed)
            if self.verbose > 0:
                print(f"Setting the random seed to {self.seed}")
           
    def _setup_model(self):
        observation_dim = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.shape[0]
        
        self.actor = VPGActor(observation_dim, num_actions, **self.actor_kwargs).to(self.device)
        
        self.critic = VCritic(observation_dim, **self.critic_kwargs).to(self.device)
        
        if self.verbose > 0:
            print(self.actor)
            print(self.critic)
            
    def _generate_buffer(self, **kwargs) -> None:
        ppo = PPO(self.env, total_timesteps=5e3, num_epochs=1, auxiliary_buffer_size=5000, **kwargs)
        
        ppo.learn()
        
        self.buffer = ppo.auxiliary_buffer
            
    def _step(self, obs):
        with torch.no_grad():
            dists = self.actor(obs_to_tensor(obs).to(self.device))
            
            action = dists.sample().cpu().detach().numpy()

            clipped_action = np.clip(action, self.env.action_space.low.min(), self.env.action_space.high.max())
            
            return clipped_action
    
    def _train(self):
        obs, actions, rewards, next_obs, dones = self.buffer.get()
            
        assert isinstance(obs, torch.Tensor) and obs.shape[1] == self.env.observation_space.shape[0]
        assert isinstance(actions, torch.Tensor) and actions.shape[1] == self.env.action_space.shape[0]
        assert isinstance(rewards, torch.Tensor) and rewards.shape[1] == 1
        assert isinstance(next_obs, torch.Tensor) and next_obs.shape[1] == self.env.observation_space.shape[0]
        assert isinstance(dones, torch.Tensor) and dones.shape[1] == 1
        
        values = self.critic(obs)
        
        if dones[-1]:
            last_value = 0
        else:
            last_value = self.critic(next_obs[-1])
            
        advantages = compute_gae_advantage(rewards, values, dones, last_value, gamma=self.gamma, gae_lambda=self.gae_lambda)
        advantages = advantages.to(self.device)

        target_values = advantages + values
                
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for iteration in range(self.num_iterations):
            indices = random.sample(range(len(obs)), self.batch_size)
            obs = obs[indices]
            actions = actions[indices]
            target_values = target_values[indices].detach()
            advantages = advantages[indices]
            
            values = self.critic(obs)
            
            critic_loss = F.mse_loss(values, target_values)
            
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            
            weights = torch.exp(advantages / self.beta)
            
            dists = self.actor(obs)
            
            log_probs = dists.log_prob(actions)
            log_probs = log_probs.sum(dim=1, keepdim=True)
            
            actor_loss = (-log_probs * weights).mean()
            
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            
            if iteration != 0 and iteration % self.eval_interval == 0:
                self.evaluate()
                
    def learn(self, **kwargs) -> None:
        if self.buffer is not None:
            self._train()
        else:
            self._generate_buffer(**kwargs)
            self._train()
    
    def evaluate(self) -> None:
        avg_returns = 0.
        
        for _ in range(self.eval_episodes):
            obs = self.env.reset()
            done = False
            
            while not done:
                act = self._step(obs)
                obs, r, done, _ = self.env.step(act)
                avg_returns += r
        
        avg_returns /= self.eval_episodes
        
        if self.verbose > 0:
            print(f"episodes_reward_mean: {avg_returns:.2f}")
        
        return avg_returns     
            
            
if __name__ == "__main__":    
    env = gym.make("Pendulum-v1")
    env = Monitor(env)
    
    awr = AWR(env, 
              buffer=None, 
              num_iterations=10000,
              batch_size=32,
              eval_episodes=1,
              seed=9, 
              actor_kwargs={"activation_fn": Mish, "optimizer_kwargs":{"lr":1e-4}}, 
              critic_kwargs={"activation_fn": Mish, "optimizer_kwargs":{"lr":1e-3}}
             )
    
    awr.learn()         