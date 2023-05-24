# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional
from common.envs import Monitor
from common.policies import OffPolicyAlgorithm
from common.models import VPG, Q2
from common.utils import Mish, obs_to_tensor, evaluate_policy

class ACER(OffPolicyAlgorithm):
    def __init__(
                 self, 
                 env: Monitor, 
                 rollout_steps: int = 16,
                 total_timesteps: int = 1e6, 
                 gradient_steps: int = 4,
                 n_steps: int = 1,
                 learning_start: int = 1000,
                 actor_kwargs: Optional[Dict[str, Any]] = None,
                 critic_kwargs: Optional[Dict[str, Any]] = None,
                 buffer_size: int = 10000,
                 gamma: float = 0.99,
                 tau: float = 0.95,
                 c: int = 1,
                 log_dir: Optional[str] = None,
                 log_interval: int = 10,
                 verbose: int = 1,
                 seed: Optional[int] = None,

                ):
        
        self.actor_kwargs = {} if actor_kwargs is None else actor_kwargs
        self.critic_kwargs = {} if critic_kwargs is None else critic_kwargs
        self.tau = tau 
        self.c = c
        
        super(ACER, self).__init__(
                 env, 
                 rollout_steps,
                 total_timesteps, 
                 gradient_steps,
                 n_steps,
                 learning_start,
                 buffer_size,
                 1,
                 None,
                 gamma,
                 log_dir, 
                 log_interval,
                 verbose,
                 seed,
            )
    
    def _setup_model(self) -> None:
        self.observation_dim = self.env.observation_space.shape[0]
        
        self.num_actions = self.env.action_space.n
        
        self.policy_net = VPG(self.observation_dim, self.num_actions, action_space="Discrete", **self.actor_kwargs)
        
        self.value_net = Q2(self.observation_dim, self.num_actions, **self.critic_kwargs)
        
        if self.verbose > 0:
            print(self.policy_net)
            print(self.value_net)

        self.buffer = []
        
        self.obs = self.env.reset()
    
    def _rollout(self) -> None:
        self.trajectory = []
        
        with torch.no_grad():
            for i in range(self.rollout_steps):
                dist = self.policy_net(obs_to_tensor(self.obs))
                
                prob = dist.probs.detach().numpy()
                
                action = dist.sample().item()
            
                next_obs, reward, done, info = self.env.step(action)
            
                self.trajectory.append((self.obs, action, reward, next_obs, done, prob))
                
                self.obs = next_obs
            
                self.current_timesteps += self.env.num_envs
            
                self._update_episode_info(info)
            
                if done:
                    break
            
        self.buffer.append(self.trajectory)
        
    def _train(self) -> None:
        trajectory = random.sample(self.buffer, self.batch_size)[0]

        obs, actions, rewards, next_obs, dones, probs = zip(*trajectory)
        
        obs = torch.FloatTensor(obs)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_obs = torch.FloatTensor(next_obs)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        probs = torch.FloatTensor(probs)
        
        assert isinstance(obs, torch.Tensor) and obs.shape[1] == self.observation_dim
        assert isinstance(actions, torch.Tensor) and actions.shape[1] == 1
        assert isinstance(rewards, torch.Tensor) and rewards.shape[1] == 1
        assert isinstance(next_obs, torch.Tensor) and next_obs.shape[1] == self.observation_dim
        assert isinstance(dones, torch.Tensor) and dones.shape[1] == 1
        assert isinstance(dones, torch.Tensor) and probs.shape[1] == self.num_actions
        
        pi = self.policy_net(obs).probs
        pi_a = pi.gather(1, actions)
        
        q = self.value_net(obs)
        q_a = q.gather(1, actions)
        
        v = (q * pi).sum(dim=1, keepdim=True).detach()
        
        rho = pi.detach() / probs
        rho_a = rho.gather(1, actions)
        rho_a_clip = rho_a.clamp(max=self.c)
        
        q_ret = v[-1] * dones[-1]
        
        q_ret_buffer = []
        for i in reversed(range(len(rewards))):
            q_ret = rewards[i] + self.gamma * q_ret
            q_ret_buffer.append(q_ret)
            q_ret = rho_a_clip[i] * (q_ret - q_a[i]) + v[i]
        q_ret_buffer.reverse()
        q_ret = torch.FloatTensor(q_ret_buffer).unsqueeze(1)
            
        critic_loss = F.smooth_l1_loss(q_a, q_ret)
            
        self.value_net.optimizer.zero_grad()
        critic_loss.backward()
        self.value_net.optimizer.step()
        
        truncation_loss = rho_a_clip * torch.log(pi_a) * (q_ret - v)

        bias_correction_loss = (1-self.c/rho).clamp(min=0) * pi.detach() * torch.log(pi) * (q.detach() - v)

        actor_loss = (truncation_loss + bias_correction_loss.sum(dim=1)).mean()
            
        self.policy_net.optimizer.zero_grad()
        actor_loss.backward()
        self.policy_net.optimizer.step()
    
    def save(self, path: str) -> None:
        state_dict = self.policy_net.state_dict()
        
        with open(path, "wb") as f:
            torch.save(state_dict, f)
        
        if self.verbose >= 1:
            print("The acer model has been saved successfully")
    
    def load(self, path: str) -> nn.Module:
        with open(path, "rb") as f:
            state_dict = torch.load(f)
            
            self.policy_net = VPG(self.observation_dim, self.num_actions, action_space="Discrete", **self.actor_kwargs)
            self.policy_net.load_state_dict(state_dict)
            self.policy_net = self.policy_net
 
        if self.verbose >= 1:
            print("The acer model has been loaded successfully")
            
        return self.policy_net
            
if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = Monitor(env)
    
    acer = ACER(env, 
              rollout_steps=8, 
              total_timesteps=3e4,
              actor_kwargs={"activation_fn": Mish}, 
              critic_kwargs={"activation_fn": Mish},
              log_dir=None,
              seed=7,
             )
    
    acer.learn()
    acer.save("./model.pkl")
    acer = acer.load("./model.pkl")
    print(evaluate_policy(acer, env, num_eval_episodes=2))