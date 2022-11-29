# -*- coding: utf-8 -*-
from typing import Tuple
import torch
import numpy as np
import random
import os
import pickle
import gym
from torch.utils.data import Dataset
from collections import defaultdict

def get_dataset(dataset_dir: str) -> None:
    os.makedirs(dataset_dir)
    
    for env_name in ["halfcheetah", "hopper", "walker2d"]:
        for dataset_type in ['medium', 'medium-expert', 'medium-replay']:
            dataset_name = f'{env_name}-{dataset_type}-v2'
            pkl_file_path = os.path.join(dataset_dir, dataset_name)
            
            env = gym.make(dataset_name)
            
            dataset = env.get_dataset()
            
            N = dataset['rewards'].shape[0]
            
            if "timeout" in dataset:
                use_timeout = True
            else:
                use_timeout = False
            
            trajectories = []
            episode_data = defaultdict(list)
            episode_timestep = 0
            
            for i in range(N):
                done = bool(dataset["terminals"][i])
                if use_timeout:
                    last_timestep = dataset["timeout"][i]
                else:
                    last_timestep = (episode_timestep == 1000 - 1)
                    
                for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
                    episode_data[k].append(dataset[k][i])
                
                if done or last_timestep:
                    episode_timestep = 0
                    for k in episode_data:
                        episode_data[k] = np.array(episode_data[k])
                    trajectories.append(episode_data)
                    episode_data = defaultdict(list)
                
                episode_timestep += 1
            
            with open(f"{pkl_file_path}.pkl", "wb") as f:
                pickle.dump(trajectories, f)
                
class TrajectoryDataset(Dataset):
    def __init__(self, dataset_dir: str, seq_len: int):
        self.seq_len = seq_len
        
        with open(dataset_dir, "rb") as f:
            self.trajectories = pickle.load(f)
        
        for traj in self.trajectories:
            traj["returns_to_go"] = compute_rtg(traj["rewards"])
            
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        traj = self.trajectories[index]
        traj_len = traj['observations'].shape[0]
        
        if traj_len >= self.seq_len:
            start_idx = random.randint(0, traj_len - self.seq_len)

            observations = torch.from_numpy(traj['observations'][start_idx : start_idx + self.seq_len])
            actions = torch.from_numpy(traj['actions'][start_idx : start_idx + self.seq_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][start_idx : start_idx + self.seq_len])
            timesteps = torch.arange(start=start_idx, end=start_idx+self.seq_len, step=1)

            traj_masks = torch.ones(self.seq_len, dtype=torch.long)
        
        else:
            padding_len = self.seq_len - traj_len

            observations = torch.from_numpy(traj['observations'])
            observations = torch.cat([observations,
                                      torch.zeros(([padding_len] + list(observations.shape[1:])),
                                     dtype=observations.dtype)],
                                     dim=0)

            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat([actions,
                                 torch.zeros(([padding_len] + list(actions.shape[1:])),
                                dtype=actions.dtype)],
                                dim=0)

            returns_to_go = torch.from_numpy(traj['returns_to_go'])
            returns_to_go = torch.cat([returns_to_go,
                                       torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
                                      dtype=returns_to_go.dtype)],
                                      dim=0)

            timesteps = torch.arange(start=0, end=self.seq_len, step=1)

            traj_masks = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)

        return (timesteps, observations, actions, returns_to_go, traj_masks)
        
    @property
    def observation_dim(self) -> int:
        return self.trajectories[0]["observations"].shape[1]
    
    @property
    def num_actions(self) -> int:
        return self.trajectories[0]["actions"].shape[1]
    
    def __len__(self) -> int:
        return len(self.trajectories)