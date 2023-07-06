# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
import ray  
import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Union, Tuple
from collections import deque
import common.models as models
from common.envs import Monitor
from common.utils import Mish, obs_to_tensor, clip_grad_norm_, swap_and_flatten, evaluate_policy, safe_mean

ray.init(ignore_reinit_error=True, runtime_env={"working_dir":"C:/Users/lanaya/Desktop/DRLAlgorithms"})

@ray.remote
class Actor():
    def __init__(self, 
                 actor_id: int, 
                 observation_dim: int, 
                 num_actions: int, 
                 qnet_kwargs: Optional[Dict[str, Any]], 
                 buffer, 
                 params_server,
                 env,
                 update_steps,
                 actor_update_interval,
                 exploration_initial_eps,
                 exploration_final_eps,
                 exploration_decay_steps,
                 log_interval,
                 verbose,
                 ):
        self.actor_id = actor_id
        self.observation_dim = observation_dim
        self.num_actions = num_actions
        self.qnet_kwargs = qnet_kwargs if qnet_kwargs else {}

        self.q_net = models.DQN(observation_dim, num_actions, **qnet_kwargs)
        self.target_q_net = models.DQN(observation_dim, num_actions, **qnet_kwargs)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.buffer = buffer
        
        self.params_server = params_server
        
        self.env = env
        self.obs = env.reset()
        
        self.update_steps = update_steps
        
        self.actor_update_interval = actor_update_interval
        
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_decay_steps = exploration_decay_steps
        self.current_eps = exploration_initial_eps
        
        self.current_timesteps = 0
        
        self.num_episodes = 0
        self.episode_info_buffer = deque(maxlen=10)
        
        self.log_interval = log_interval
        
        self.verbose = verbose
            
    def _update_exploration_eps(self) -> None:
        if self.current_eps == self.exploration_final_eps:
            return 
        self.current_eps = self.exploration_initial_eps - ((self.exploration_initial_eps - self.exploration_final_eps) / self.exploration_decay_steps) * self.current_timesteps
        self.current_eps = max(self.current_eps, self.exploration_final_eps)
    
    def _update_params(self):
        new_params = ray.get(self.params_server._get_params.remote())
        
        for param, new_param in zip(self.q_net.parameters(), new_params):
            param.data.copy_(new_param)
            
    def _update_episode_info(self, infos) -> None:
        if isinstance(infos, dict):
            infos = [infos]
        for info in infos:
            episode_info = info.get("episode")
            if episode_info is not None:
                self.episode_info_buffer.append(episode_info)
                self.num_episodes += 1
            
    def _run(self):
        update_steps = ray.get(self.params_server._get_update_steps.remote())
        
        with torch.no_grad():
            while update_steps < self.update_steps:
                q = self.q_net(obs_to_tensor(self.obs))
            
                coin = random.random()
                if coin < self.current_eps:
                    action = [random.randint(0, self.env.action_space.n - 1) for _ in range(self.env.num_envs)]
                    action = np.asarray(action)[:, np.newaxis]
                else:
                    action = q.argmax(dim=-1, keepdim=True).cpu().detach().numpy()

                next_obs, reward, done, info = self.env.step(action)

                self.buffer._add.remote((self.obs, action, reward, next_obs, done))

                self.obs = next_obs
            
                self.params_server._update_timesteps.remote(self.env.num_envs)
            
                self._update_episode_info(info)
            
                self._update_exploration_eps()
            
                update_steps = ray.get(self.params_server._get_update_steps.remote())
                if update_steps % self.actor_update_interval == 0:
                    self._update_params()  
                if update_steps % self.log_interval == 0:
                    if self.verbose >= 1:
                        print("actor_id", self.actor_id,
                              "episode", self.num_episodes,
                              "episode_reward_mean", safe_mean([ep_info["episode returns"] for ep_info in self.episode_info_buffer]),
                             )
@ray.remote
class Learner():
    def __init__(self, 
                 observation_dim, 
                 num_actions, 
                 qnet_kwargs, 
                 buffer, 
                 params_server,
                 batch_size,
                 update_steps,
                 learning_start,
                 target_update_interval,
                 gamma,
                 max_grad_norm,
                ):
        
        self.observation_dim = observation_dim
        self.num_actions = num_actions
        self.qnet_kwargs = qnet_kwargs if qnet_kwargs else {}
        
        self.q_net = models.DQN(self.observation_dim, self.num_actions, **self.qnet_kwargs)
        self.target_q_net = models.DQN(observation_dim, num_actions, **self.qnet_kwargs)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.buffer = buffer
        
        self.params_server = params_server
        
        self.params_list = list(self.q_net.state_dict())
        
        self.batch_size = batch_size
        
        self.update_steps = update_steps
        
        self.learning_start = learning_start
        
        self.target_update_interval = target_update_interval
        
        self.gamma = gamma
        
        self.max_grad_norm = max_grad_norm

    def _run(self) -> None:
        update_steps = ray.get(self.params_server._get_update_steps.remote())

        while update_steps < self.update_steps:
            current_timesteps = ray.get(self.params_server._get_timesteps.remote())
            if current_timesteps >= self.learning_start:
                obs, actions, rewards, next_obs, dones = ray.get(self.buffer._sample.remote(self.batch_size))
            
                actions = actions.type("torch.LongTensor")
            
                assert isinstance(obs, torch.Tensor) and obs.shape[1] == self.observation_dim
                assert isinstance(actions, torch.Tensor) and actions.shape[1] == 1
                assert isinstance(rewards, torch.Tensor) and rewards.shape[1] == 1
                assert isinstance(next_obs, torch.Tensor) and next_obs.shape[1] == self.observation_dim
                assert isinstance(dones, torch.Tensor) and dones.shape[1] == 1
            
                q_next = self.target_q_net(next_obs)
                q_next = q_next.max(dim=1, keepdim=True)[0]
            
                q_target = rewards + self.gamma * (1 - dones) * q_next
            
                q_values = self.q_net(obs)
            
                q_a = q_values.gather(1, actions)

                loss = F.smooth_l1_loss(q_a, q_target)

                self.q_net.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm:
                     clip_grad_norm_(self.q_net.optimizer, self.max_grad_norm)
                self.q_net.optimizer.step()
            
                self._update_params_server()
            
                update_steps = ray.get(self.params_server._get_update_steps.remote())

                if update_steps % self.target_update_interval == 0:

                    self.target_q_net.load_state_dict(self.q_net.state_dict())
            
    def _update_params_server(self) -> None:
        params = []
        
        params_state_dict = self.q_net.state_dict()
        
        for param in self.params_list:
            params.append(params_state_dict[param])
            
        self.params_server._update_params.remote(params)
        
    def _get_qnet(self):
        return self.q_net
        
@ray.remote 
class ParameterServer():
    def __init__(self):
        self.params = []
        self.update_steps = 0
        self.current_timesteps = 0

    def _get_params(self):
        return self.params

    def _get_update_steps(self):
        return self.update_steps
    
    def _get_timesteps(self):
        return self.current_timesteps
    
    def _update_params(self, params):
        if len(self.params) < len(params):
            for param in params:
                self.params.append(param)
                
        else:
            for i in range(len(params)):
                self.params[i] = params[i]
        
        self.update_steps += 1
    
    def _update_timesteps(self, timesteps):
        self.current_timesteps += timesteps
        
@ray.remote
class ReplayBuffer():
    def __init__(self, buffer_size, n_steps, gamma):
        self.buffer_size = buffer_size
        self.n_steps = n_steps
        self.gamma = gamma
        
        self.buffer = deque(maxlen=buffer_size)
        
        self.n_steps_buffer = deque(maxlen=n_steps)

    def _update_n_steps_info(self) -> Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray], np.ndarray, Union[bool, np.ndarray]]:
        state, action = self.n_steps_buffer[0][:2]
        reward, next_state, done = self.n_steps_buffer[-1][-3:]

        for transition in reversed(list(self.n_steps_buffer)[:-1]):
            r, next_s, d = transition[-3:]

            reward = r + self.gamma * reward * (1 - d)
            
            next_state, done = (next_s, d) if d else (next_state, done)

        return (state, action, reward, next_state, done)

    def _add(self, transition: Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray], np.ndarray, Union[bool, np.ndarray]]) -> None:
        self.n_steps_buffer.append(transition)
        
        if len(self.n_steps_buffer) >= self.n_steps:
            transition = self._update_n_steps_info()
        
            self.buffer.append(transition)
        
    def _sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            samples = random.sample(self.buffer, batch_size)
        except:
            raise ValueError("The batch size must be less than learning start:)")
        
        obs, actions, rewards, next_obs, dones = [], [], [], [], []
        
        for ob, action, reward, next_ob, done in samples:
            if isinstance(ob, int):
                ob = np.array([ob])
            if len(ob.shape) == 1:
                ob = ob[np.newaxis, :]
                
            if isinstance(action, int):
                action = [action]
            elif len(action.shape) == 1:
                action = action[np.newaxis, :]
             
            if not isinstance(reward, np.ndarray):
                reward = [reward]
            
            if isinstance(next_ob, int):
               next_ob = np.array([next_ob])
            if len(next_ob.shape) == 1:
                next_ob = next_ob[np.newaxis, :]
                
            if not isinstance(done, np.ndarray):
                done = [done]
                
            obs.append(ob)
            actions.append(action)
            rewards.append(reward)
            next_obs.append(next_ob)
            dones.append(done)
        
        obs = torch.FloatTensor(swap_and_flatten(np.asarray(obs)))
        actions = torch.FloatTensor(swap_and_flatten(np.asarray(actions)))
        rewards = torch.FloatTensor(swap_and_flatten(np.asarray(rewards)))
        next_obs = torch.FloatTensor(swap_and_flatten(np.asarray(next_obs)))
        dones = torch.FloatTensor(swap_and_flatten(np.asarray(dones)))
        
        return (obs, actions, rewards, next_obs, dones)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    
class APEX():
    def __init__(self, 
                 env: Monitor, 
                 update_steps: int = 16,
                 gradient_steps: int = 4,
                 n_steps: int = 1,
                 num_actors: int = 1,
                 qnet_kwargs: Optional[Dict[str, Any]] = None,
                 learning_start: int = 1000,
                 buffer_size: int = 10000,
                 batch_size: int = 256,
                 actor_update_interval: int = 10,
                 target_update_interval: int = 20,
                 gamma: float = 0.99,
                 max_grad_norm: Optional[float] = 0.5,
                 exploration_initial_eps: float = 0.2,
                 exploration_final_eps: float = 0.05,
                 exploration_decay_steps: int = 10000,
                 verbose: int = 1,
                 log_dir: Optional[str] = None,
                 log_interval: int = 10,
                 seed: Optional[int] = None,
                ):
        
        self.env = env
        self.update_steps = update_steps
        self.gradient_steps = gradient_steps
        self.n_steps = n_steps
        self.num_actors = num_actors
        self.qnet_kwargs = qnet_kwargs
        self.learning_start =learning_start
        self.buffer_size =buffer_size
        self.batch_size = batch_size
        self.actor_update_interval = actor_update_interval
        self.target_update_interval = target_update_interval
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps =exploration_final_eps
        self.exploration_decay_steps =exploration_decay_steps
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.verbose = verbose
        self.seed = seed
        
        self._setup_model()
        
    def _setup_model(self):
        self.params_server = ParameterServer.remote()
        
        self.buffer = ReplayBuffer.remote(self.buffer_size, self.n_steps, self.gamma)
        
        self.observation_dim = self.env.observation_space.shape[0]
        
        self.num_actions = self.env.action_space.n
        
        self.actors = [Actor.remote(
            i, self.observation_dim, self.num_actions, self.qnet_kwargs, 
            self.buffer, self.params_server, self.env, self.update_steps,
            self.actor_update_interval, self.exploration_initial_eps, 
            self.exploration_final_eps, self.exploration_decay_steps, 
            self.log_interval, self.verbose) for i in range(self.num_actors)
                      ]
        
        self.learner = Learner.remote(
            self.observation_dim, self.num_actions, self.qnet_kwargs, 
            self.buffer, self.params_server, self.batch_size,
            self.update_steps, self.learning_start, self.target_update_interval,
            self.gamma, self.max_grad_norm,
            )
            
        self.obs = self.env.reset()
        
    def learn(self):
        procs = self.actors + [self.learner]
        
        proc_refs = [proc._run.remote() for proc in procs]
        
        ready_refs, _ = ray.wait(proc_refs, num_returns=len(proc_refs))
        
    def save(self, path: str) -> None:
        q_net = ray.get(self.learner._get_qnet.remote())
        state_dict = q_net.state_dict()
        
        with open(path, "wb") as f:
            torch.save(state_dict, f)
        
        if self.verbose >= 1:
            print("The apex model has been saved successfully")
    
    def load(self, path: str) -> nn.Module:
        with open(path, "rb") as f:
            state_dict = torch.load(f)
            
            self.q_net = models.DQN(self.observation_dim, self.num_actions, **self.qnet_kwargs)
            self.q_net.load_state_dict(state_dict)
            self.q_net = self.q_net
 
        if self.verbose >= 1:
            print("The apex model has been loaded successfully")
            
        return self.q_net
        
if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = Monitor(env)

    apex = APEX(env, 
                update_steps=10,
                gradient_steps=4,
                n_steps=1,
                num_actors=4,
                qnet_kwargs={"activation_fn": Mish, "optimizer_kwargs":{"lr":1e-3}}, 
                learning_start=500,
                buffer_size=10000,
                batch_size=64,
                max_grad_norm=None,
                log_dir=None,
                log_interval=20,
                verbose=1,
                seed=1,)
    
    apex.learn()
    apex.save("./model.ckpt")
    apex = apex.load("./model.ckpt")    
    print(evaluate_policy(apex, env))