# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 13:12:55 2022

@author: lanaya
"""
import multiprocessing as mp
import numpy as np

def _worker(worker_remote, parent_remote, env):
    parent_remote.close()
    
    while True:
        try:
            cmd, data = worker_remote.recv()
            
            if cmd == "seed":
                env.seed(data)

            elif cmd == "reset":
                observation = env.reset()
                worker_remote.send(observation)
            
            elif cmd == "step":
                observation, reward, done, info = env.step(data)
                if done:
                    observation = env.reset()
                worker_remote.send((observation, reward, done, info))

            elif cmd == "close":
                env.close()
                worker_remote.close()
                break
            
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        
        except EOFError:
            break
        
class VecEnv():
    def __init__(self, env, num_envs=4, start_method=None):
        self.envs = [env] * num_envs
        self.num_envs = num_envs
        
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"

        ctx = mp.get_context(start_method)
        
        self.parent_remotes, self.worker_remotes = zip(*[ctx.Pipe() for _ in range(num_envs)])
        
        self.processes = []
        
        for worker_remote, parent_remote, env in zip(self.worker_remotes, self.parent_remotes, self.envs):
            args = (worker_remote, parent_remote, env)
            
            process = ctx.Process(target=_worker, args=args, daemon=True)
            
            process.start()
            
            self.processes.append(process)
            
            worker_remote.close()
            
    def seed(self, seed):
        for i, parent_remote in enumerate(self.parent_remotes):
            parent_remote.send(("seed", seed + i))
    
    def reset(self):
        for parent_remote in self.parent_remotes:
            parent_remote.send(("reset", None))
        observation = [parent_remote.recv() for parent_remote in self.parent_remotes]
        return np.stack(observation)
    
    def step(self, action):
        for parent_remote, action in zip(self.parent_remotes, action):
            parent_remote.send(("step", action))
        
        results = [parent_remote.recv() for parent_remote in self.parent_remotes]
        next_observation, reward, done, info = zip(*results)
        return np.stack(next_observation), np.stack(reward), np.stack(done), info
        
    def close(self):
        for parent_remote in zip(self.parent_remotes):
            parent_remote.send(("close", None))
        
class Monitor():
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.rewards = []
    
    def seed(self, seed):
        self.env(seed)
        
    def reset(self):
        observation = self.env.reset()
        return observation 
    
    def step(self, action):
        next_observation, reward, done, info = self.env.step(action)
        
        self.rewards.append(reward)
        
        if done:
            ep_r = sum(self.rewards)
            ep_l = len(self.rewards)
            
            episode_info = {"episode returns": round(ep_r, 2), "episode length": ep_l}
            
            info["episode"] = episode_info
            info["terminal_state"] = next_observation
            
            next_observation = self.env.reset()
            
            self.rewards.clear()
            
        return next_observation, reward, done, info
    
    def close(self):
        self.env.close()