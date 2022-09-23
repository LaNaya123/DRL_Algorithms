# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
from typing import Any, Dict, Optional, Union
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
import random
from common.envs import Monitor
from common.policies import OffPolicyAlgorithm
from common.models import DeepQNetwork
from common.buffers import SharedReplayBuffer
from common.utils import Mish, obs_to_tensor, safe_mean
        

class APEX(OffPolicyAlgorithm):
    def __init__(
                 self, 
                 env: Monitor, 
                 num_iterations: int = 1000,
                 buffer_size: int = 10000,
                 batch_size: int = 256,
                 target_update_interval: int = 20,
                 gamma: float = 0.99,
                 verbose: int = 1,
                 log_dir: Optional[str] = None,
                 log_interval: int = 10,
                 seed: Optional[int] = None,
                 qnet_kwargs: Optional[Dict[str, Any]] = None,
                 exploration_initial_eps: float = 0.2,
                 exploration_final_eps: float = 0.05,
                 exploration_decay_steps: int = 10000,
                 num_actors: int = 4,
                 sychronize_freq: int = 2,
                ):
    
        self.qnet_kwargs = qnet_kwargs
        
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_decay_steps = exploration_decay_steps
        self.current_eps = exploration_initial_eps
        
        self.num_iterations = num_iterations
        
        self.num_actors = num_actors
        
        self.sychronize_freq = sychronize_freq
        
        super(APEX, self).__init__(
                 env, 
                 None,
                 None, 
                 None,
                 None,
                 buffer_size,
                 batch_size,
                 target_update_interval,
                 gamma,
                 verbose,
                 log_dir,
                 log_interval,
                 "cpu",
                 seed,
                )
        
    def _setup_model(self) -> None:
        observation_dim = self.env.observation_space.shape[0]
        
        num_actions = self.env.action_space.n
        
        num_envs = self.env.num_envs
        
        self.actor = DeepQNetwork(observation_dim, num_actions, **self.qnet_kwargs)
        
        self.learner = DeepQNetwork(observation_dim, num_actions, **self.qnet_kwargs)
        
        self.target_qnet = DeepQNetwork(observation_dim, num_actions, **self.qnet_kwargs)
        self.target_qnet.load_state_dict(self.learner.state_dict())
        
        if self.verbose > 0:
            print(self.actor) 
        
        self.obs = self.env.reset()
        
        self.buffer = SharedReplayBuffer(self.buffer_size, num_envs, observation_dim)
        
    def _setup_param(self) -> None:
        manager = mp.Manager()
        
        self.shared_dict = manager.dict()
        
        self.episode_info_buffer = manager.list()
        
        self.num_episode = manager.Value("i", 0)
        
        self.current_timesteps = manager.Value("i", 0)
        
        self.training_iterations = manager.Value("i", 0)
        
        self.lock = manager.Lock()
        
    def _update_episode_info(self, infos) -> None:
        if isinstance(infos, dict):
            infos = [infos]
            
        for info in infos:
            episode_info = info.get("episode")
            with self.lock:
                if episode_info is not None:
                    if len(self.episode_info_buffer) == 10:
                        self.episode_info_buffer.pop(0)

                    self.episode_info_buffer.append(episode_info)
            
                    self.num_episode.value += 1
        
    def _update_exploration_eps(self) -> None:
        if self.current_eps == self.exploration_final_eps:
            return 
        self.current_eps = self.exploration_initial_eps - ((self.exploration_initial_eps - self.exploration_final_eps) / self.exploration_decay_steps) \
                           * self.current_timesteps.value
        self.current_eps = max(self.current_eps, self.exploration_final_eps)
        
    def rollout(self) -> None:
        while self.training_iterations.value < self.num_iterations:
            q = self.actor(obs_to_tensor(self.obs))
            
            coin = random.random()
            if coin < self.current_eps:
                action = [random.randint(0, self.env.action_space.n - 1) for _ in range(self.env.num_envs)]
                action = np.asarray(action)[:, np.newaxis]
            else:
                action = q.argmax(dim=-1, keepdim=True).detach().numpy()
        
            next_obs, reward, done, info = self.env.step(action)

            self.buffer.add((self.obs, action, reward, next_obs, done))

            self.obs = next_obs
            
            self.current_timesteps.value += self.env.num_envs
            
            self._update_episode_info(info)

            self._update_exploration_eps()

            if self.training_iterations.value != 0 and self.training_iterations.value % self.sychronize_freq == 0:
                self.actor.load_state_dict(self.shared_dict["model_state_dict"])
    
    def train(self) -> None:
        while self.training_iterations.value < self.num_iterations:
            if len(self.buffer) < self.batch_size:
                continue
            else:
                obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
            
            actions = actions.type("torch.LongTensor")
            
            assert isinstance(obs, torch.Tensor) and obs.shape[1] == self.env.observation_space.shape[0]
            assert isinstance(actions, torch.Tensor) and actions.shape[1] == 1
            assert isinstance(rewards, torch.Tensor) and rewards.shape[1] == 1
            assert isinstance(next_obs, torch.Tensor) and next_obs.shape[1] == self.env.observation_space.shape[0]
            assert isinstance(dones, torch.Tensor) and dones.shape[1] == 1
            
            q_next = self.target_qnet(next_obs)
            q_next = q_next.max(dim=1, keepdim=True)[0]
            
            q_target = rewards + self.gamma * (1 - dones) * q_next
            
            q_values = self.learner(obs)
            
            q_a = q_values.gather(1, actions)

            loss = F.smooth_l1_loss(q_a, q_target)

            self.learner.optimizer.zero_grad()
            loss.backward()
            self.learner.optimizer.step()
            
            self.shared_dict["model_state_dict"] = self.learner.state_dict()
        
            self.training_iterations.value += 1
        
            if self.training_iterations.value % self.target_update_interval == 0:
                self.target_qnet.load_state_dict(self.learner.state_dict())

            if self.training_iterations.value % self.log_interval == 0 and self.verbose > 0:
                print("episode", self.num_episode.value,
                      "episode_reward_mean", safe_mean([ep_info["episode returns"] for ep_info in self.episode_info_buffer]),
                     )
            
    def learn(self) -> None:
        p = mp.Process(target=self.train, args=(()))
        p.start()
        
        for i in range(self.num_actors):
            p = mp.Process(target=self.rollout, args=(()))
            p.start()
            p.join()
        
if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = Monitor(env)
    apex = APEX(env, 
                num_iterations=2.5e4,
                qnet_kwargs={"activation_fn": Mish, "optimizer_kwargs":{"lr":1e-3}}, 
                buffer_size=5000,
                batch_size=128,
                target_update_interval=40,
                log_dir=None,
                log_interval=20,
                seed=1018,
                num_actors=4,
                sychronize_freq=1)

    apex.learn()