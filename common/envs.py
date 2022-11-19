# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional, Union, Tuple
import multiprocessing as mp
import numpy as np
import gym
from gym import spaces

class Monitor():
    def __init__(self, env: gym.Env):
        self.env = env
        self.num_envs = 1
        self.is_vec = False
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.rewards = []
    
    def seed(self, seed: int) -> None:
        self.env.seed(seed)
        
    def reset(self) -> np.ndarray:
        observation = self.env.reset()
        return observation 
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, dict]:
        if isinstance(self.action_space, spaces.Discrete) and isinstance(action, np.ndarray):
            action = action.item()

        next_observation, reward, done, info = self.env.step(action)
        
        self.rewards.append(reward)
        
        if done:
            ep_r = sum(self.rewards)
            ep_l = len(self.rewards)
            
            episode_info = {"episode returns": round(ep_r, 2), "episode length": ep_l}
            
            info["episode"] = episode_info
            info["real_terminal_state"] = next_observation
            
            next_observation = self.env.reset()
            
            self.rewards.clear()
            
        return (next_observation, reward, done, info)
    
    def close(self) -> None:
        self.env.close()

def _worker(worker_remote, parent_remote, env: Monitor) -> None:
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
    def __init__(self, env: Monitor, num_envs: int = 4, start_method: Optional[str] = None):
        self.envs = [env] * num_envs
        self.num_envs = num_envs
        self.is_vec = True
        
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
            
    def seed(self, seed: int) -> None:
        for i, parent_remote in enumerate(self.parent_remotes):
            parent_remote.send(("seed", seed + i))
    
    def reset(self) -> np.ndarray:
        for parent_remote in self.parent_remotes:
            parent_remote.send(("reset", None))
        observation = [parent_remote.recv() for parent_remote in self.parent_remotes]
        return np.stack(observation)
    
    def step(self, action) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        for parent_remote, action in zip(self.parent_remotes, action):
            parent_remote.send(("step", action))
        
        results = [parent_remote.recv() for parent_remote in self.parent_remotes]
        next_observation, reward, done, info = zip(*results)
        return (np.stack(next_observation), np.stack(reward), np.stack(done), info)
        
    def close(self) -> None:
        for parent_remote in zip(self.parent_remotes):
            parent_remote.send(("close", None))
            

class ChainEnv(gym.Env):
    def __init__(self):
        super(ChainEnv, self).__init__()
        self.states = (0, 1, 2, 3, 4) 
        
        self.actions = (0, 1)
        
        self.rewards = (-1, -1, 10, -1, -1)
        
        self.transitions = [
            [[0.9, 0.1], [0.1, 0.9], [0, 0], [0, 0], [0, 0]],
            [[0.9, 0.1], [0, 0], [0.1, 0.9], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],  
            [[0, 0], [0, 0], [0.9, 0.1], [0, 0], [0.1, 0.9]],
            [[0, 0], [0, 0], [0, 0], [0.9, 0.1], [0.1, 0.9]],
            ]   
        
        self.observation_space = spaces.Discrete(5)
        
        self.action_space = spaces.Discrete(2)