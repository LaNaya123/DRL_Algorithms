# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
from algorithms.online.ddpg import DDPG
from common.envs import Monitor, VecEnv
from common.models import BCQActor, QCritic
from common.utils import VAE
from common.utils import OrnsteinUhlenbeckNoise, Mish, clip_grad_norm_

class BCQ():
    def __init__(
                 self,
                 env,
                 buffer=None, 
                 num_iterations=1000,
                 num_samples=10,
                 batch_size=64,
                 target_update_interval=1,
                 eval_interval=10,
                 eval_episodes=5,
                 gamma=0.99,
                 tau=0.95,
                 lmbda=0.75,
                 verbose=1,
                 log_dir=None,
                 log_interval=10,
                 device="auto",
                 seed=None,
                 actor_kwargs=None,
                 critic_kwargs=None,
                ):
        
        self.env = env
        self.buffer = buffer
        self.num_iterations = num_iterations
        self.num_samples = num_samples
        self.batch_size=batch_size
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.gamma = gamma
        self.tau = tau
        self.lmbda = lmbda
        self.verbose = verbose
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.seed = seed
        self.actor_kwargs = {} if actor_kwargs is None else actor_kwargs
        self.critic_kwargs = {} if critic_kwargs is None else critic_kwargs
        
        if seed:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.env.seed(seed)
            
        observation_dim = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.shape[0]
        max_action = self.env.action_space.high
        
        self.actor = BCQActor(observation_dim, num_actions, max_action, **self.actor_kwargs)
        self.target_actor = BCQActor(observation_dim, num_actions, max_action, **self.actor_kwargs)
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        self.critic1 = QCritic(observation_dim, num_actions, **self.critic_kwargs)
        self.target_critic1 = QCritic(observation_dim, num_actions, **self.critic_kwargs)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        
        self.critic2 = QCritic(observation_dim, num_actions, **self.critic_kwargs)
        self.target_critic2 = QCritic(observation_dim, num_actions, **self.critic_kwargs)
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        self.vae = VAE(observation_dim, num_actions, num_actions * 2, self.env.action_space.high)
        self.vae_optimizer = optim.Adam(self.vae.parameters())
        
        if self.verbose > 0:
            print(self.actor)
            print(self.critic1)
    
    def _generate_buffer(self, **kwargs):
            ddpg = DDPG(self.env, **kwargs)
            ddpg.learn()
            self.buffer = ddpg.buffer
        
    def _polyak_update(self, online, target):
        for target_param, param in zip(target.parameters(), online.parameters()):
            target_param.data.copy_(param.data * (1.0 - self.tau) + target_param.data * self.tau)
    
    def _act(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs.reshape(1, -1)).repeat(100, 1)
            acts = self.actor(obs, self.vae.decode(obs))
            q = self.critic1(obs, acts)
            ind = q.argmax(0)
            
        return acts[ind].numpy().flatten()
            
    def _train(self):
        for iteration in range(self.num_iterations):
            obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

            assert isinstance(obs, torch.Tensor) and obs.shape[1] == self.env.observation_space.shape[0]
            assert isinstance(actions, torch.Tensor) and actions.shape[1] == self.env.action_space.shape[0]
            assert isinstance(rewards, torch.Tensor) and rewards.shape[1] == 1
            assert isinstance(next_obs, torch.Tensor) and next_obs.shape[1] == self.env.observation_space.shape[0]
            assert isinstance(dones, torch.Tensor) and dones.shape[1] == 1
            
            sample_actions, mean, std = self.vae(obs, actions)
            vae_loss = F.mse_loss(sample_actions, actions) -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            
            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()
            
            with torch.no_grad():
                next_obs = torch.repeat_interleave(next_obs, self.num_samples, 0)
            
                target_acts = self.target_actor(next_obs, self.vae.decode(next_obs))

                target_q1 = self.target_critic1(next_obs, target_acts)
                
                target_q2 = self.target_critic2(next_obs, target_acts)
                
                target_q = self.lmbda * torch.min(target_q1, target_q2) + (1. - self.lmbda) * torch.max(target_q1, target_q2)
                target_q = target_q.reshape(self.batch_size, -1).max(1)[0].reshape(-1, 1)
                target_q = rewards + (1 - dones) * self.gamma * target_q
            
            q1 = self.critic1(obs, actions)
            q2 = self.critic2(obs, actions)
            
            critic1_loss = F.mse_loss(q1, target_q)
            critic2_loss = F.mse_loss(q2, target_q)
            
            self.critic1.optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1.optimizer.step()
            
            self.critic2.optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2.optimizer.step()
            
            sampled_actions = self.vae.decode(obs)
            perturbed_actions = self.actor(obs, sampled_actions)
            
            actor_loss = -self.critic1(obs, perturbed_actions).mean()
            
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            
            if iteration % self.target_update_interval == 0:
                self._polyak_update(self.actor, self.target_actor)
                self._polyak_update(self.critic1, self.target_critic1)
                self._polyak_update(self.critic2, self.target_critic2)
                
            if iteration != 0 and iteration % self.eval_interval == 0:
                self.inference()
                
    def learn(self, **kwargs):
        if self.buffer is not None:
            self._train()
        else:
            self._generate_buffer(**kwargs)
            self._train()
    
    def inference(self):
        avg_returns = 0.
        
        for _ in range(self.eval_episodes):
            obs = self.env.reset()
            done = False
            
            while not done:
                act = self._act(obs)
                obs, r, done, _ = self.env.step(act)
                avg_returns += r
        
        avg_returns /= self.eval_episodes
        
        if self.verbose > 0:
            print(f"episodes_reward_mean: {avg_returns:.2f}")
        
        return avg_returns
        
if __name__ == "__main__":
    import pickle
    
    env = gym.make("Pendulum-v1")
    env = Monitor(env)
    
    ou_noise = OrnsteinUhlenbeckNoise(np.zeros(env.action_space.shape[0]))
    
    ddpg = DDPG(env, 
              total_timesteps=3e4, 
              gradient_steps=4,
              rollout_steps=8, 
              learning_start=500,
              buffer_size=20000,
              batch_size=64,
              target_update_interval=1,
              log_dir=None,
              log_interval=80,
              seed=5,
              actor_kwargs={"activation_fn": Mish, "optimizer_kwargs":{"lr":5e-4}}, 
              critic_kwargs={"activation_fn": Mish, "optimizer_kwargs":{"lr":1e-3}},
              ou_noise=ou_noise)
    
    ddpg.learn()     
    
    buffer = ddpg.buffer
    
    with open("buffer.pkl", "wb") as f:
        pickle.dump(buffer, f)
        
    with open("buffer.pkl","rb") as f:
        buffer = pickle.load(f)
        
    bcq = BCQ(env, 
              buffer, 
              num_iterations=3000,
              batch_size=128,
              eval_episodes=1,
              seed=9, 
              actor_kwargs={"activation_fn": Mish, "optimizer_kwargs":{"lr":1e-4}}, 
              critic_kwargs={"activation_fn": Mish, "optimizer_kwargs":{"lr":1e-3}},)
    bcq.learn()