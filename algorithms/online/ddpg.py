# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
import gym
import numpy as np
import torch
import torch.nn.functional as F
from common.envs import Monitor, VecEnv
from common.policies import OffPolicyAlgorithm
from common.models import DDPGActor, QCritic
from common.buffers import ReplayBuffer
from common.utils import OrnsteinUhlenbeckNoise, Mish, obs_to_tensor

class DDPG(OffPolicyAlgorithm):
    def __init__(
                 self, 
                 env, 
                 rollout_steps=16,
                 total_timesteps=1e6, 
                 gradient_steps=4,
                 learning_start=1000,
                 buffer_size=10000,
                 batch_size=256,
                 target_update_interval=20,
                 gamma=0.99,
                 verbose=1,
                 log_dir=None,
                 log_interval=10,
                 device="auto",
                 seed=None,
                 actor_kwargs=None,
                 critic_kwargs=None,
                 ou_noise=None,
                 tau=0.95,
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
        
    def _setup_model(self):
        observation_dim = self.env.observation_space.shape[0]
        
        num_action = self.env.action_space.shape[0]
        
        self.actor = DDPGActor(observation_dim, num_action, **self.actor_kwargs)
        self.target_actor = DDPGActor(observation_dim, num_action, **self.actor_kwargs)
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        self.critic = QCritic(observation_dim, num_action, **self.critic_kwargs)
        self.target_critic = QCritic(observation_dim, num_action, **self.critic_kwargs)
        self.target_critic.load_state_dict(self.critic.state_dict())
            
        if self.verbose > 0:
            print(self.actor)
            print(self.critic)

        self.buffer = ReplayBuffer(self.buffer_size)
        
        self.obs = self.env.reset()
        
    def _polyak_update(self, online, target):
        for target_param, param in zip(target.parameters(), online.parameters()):
            target_param.data.copy_(param.data * (1.0 - self.tau) + target_param.data * self.tau)
        
    def rollout(self):
        for i in range(self.rollout_steps):
            action = self.actor(obs_to_tensor(self.obs)).detach().numpy()
            
            action *= self.env.action_space.high
            
            if self.ou_noise:
                action = action + self.ou_noise()
            
            next_obs, reward, done, info = self.env.step(action)
            
            self.buffer.add((self.obs, action, reward/100., next_obs, done))
            
            self.obs = next_obs
            
            self.current_timesteps += self.env.num_envs
            
            self._update_episode_info(info)
    
    def train(self):
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        assert isinstance(obs, torch.Tensor) and obs.shape[1] == self.env.observation_space.shape[0]
        assert isinstance(actions, torch.Tensor) and actions.shape[1] == self.env.action_space.shape[0]
        assert isinstance(rewards, torch.Tensor) and rewards.shape[1] == 1
        assert isinstance(next_obs, torch.Tensor) and next_obs.shape[1] == self.env.observation_space.shape[0]
        assert isinstance(dones, torch.Tensor) and dones.shape[1] == 1
        
        with torch.no_grad():
            next_a = self.target_actor(next_obs)
            next_q = self.target_critic(next_obs, next_a)
            target_q = rewards + self.gamma * (1 - dones) * next_q
            
        critic_loss = F.smooth_l1_loss(self.critic(obs, actions), target_q)
            
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
    
        actor_loss = -self.critic(obs,self.actor(obs)).mean()
            
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
            
        if self.training_iterations % self.target_update_interval == 0:
            self._polyak_update(self.actor, self.target_actor)
            self._polyak_update(self.critic, self.target_critic)
        
if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    env = Monitor(env)
    #env = VecEnv(env, num_envs=4)
    ou_noise = OrnsteinUhlenbeckNoise(np.zeros(env.action_space.shape[0]))
    ddpg = DDPG(env, 
              total_timesteps=2e4, 
              gradient_steps=4,
              rollout_steps=8, 
              learning_start=500,
              buffer_size=10000,
              batch_size=64,
              target_update_interval=1,
              log_dir=None,
              log_interval=80,
              seed=5,
              actor_kwargs={"activation_fn": Mish, "optimizer_kwargs":{"lr":5e-4}}, 
              critic_kwargs={"activation_fn": Mish, "optimizer_kwargs":{"lr":1e-3}},
              ou_noise=ou_noise)
    ddpg.learn()
    print(len(ddpg.buffer))