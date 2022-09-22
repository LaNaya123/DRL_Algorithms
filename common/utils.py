# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math 
import os
import pickle
import gym
from torch.utils.data import Dataset
from collections import defaultdict

def safe_mean(arr):
    return np.nan if len(arr) == 0 else round(np.mean(arr), 2)

def swap_and_flatten(arr):
    shape = arr.shape
    if len(shape) < 3:
        shape = shape + (1,)
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
    
def obs_to_tensor(x):
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    return torch.from_numpy(x).float()

def clip_grad_norm_(module, max_grad_norm):
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)

def compute_gae_advantage(rewards, values, dones, last_value, gamma=0.95, gae_lambda=0.95):
    advantages = []
    gae_advantage = 0

    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_value = last_value
        else:
            next_value = values[i+1]
        delta = rewards[i] + gamma * (1 - dones[i]) * next_value - values[i]
        gae_advantage = delta + gamma * gae_lambda * (1 - dones[i]) * gae_advantage
        advantages.append(gae_advantage)
    return torch.FloatTensor(advantages[::-1]).view(-1, 1)

def compute_td_target(rewards, dones, last_value, gamma=0.95):
    target_values = []
    td_target = last_value
    for i in reversed(range(len(rewards))):
        td_target = rewards[i] + gamma * (1 - dones[i]) * td_target
        target_values.append(td_target)
    return torch.FloatTensor(target_values[::-1]).view(-1, 1)

def compute_rtg(rewards, gamma=0.99):
    rtg = np.zeros_like(rewards)
    rtg[-1] = rewards[-1]
    
    for i in reversed(range(len(rewards)-1)):
        rtg[i] = rewards[i] + gamma * rtg[i+1]
    return rtg
        
def get_dataset(dataset_dir):
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
       
class Mish(nn.Module):
    def __init__(self): 
        super().__init__()
        
    def forward(self, x): 
        return self._mish(x)
    
    def _mish(self, x):
        return x * torch.tanh(F.softplus(x))
    
class OrnsteinUhlenbeckNoise():
    def __init__(self, mu, sigma=0.1, theta=0.1, dt=0.01):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x_prev = np.zeros_like(self.mu)
        
    def reset(self):
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

class TrajectoryDataset(Dataset):
    def __init__(self, dataset_dir, seq_len):
        self.seq_len = seq_len
        
        with open(dataset_dir, "rb") as f:
            self.trajectories = pickle.load(f)
        
        for traj in self.trajectories:
            traj["returns_to_go"] = compute_rtg(traj["rewards"])
            
    def __getitem__(self, index):
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

        return timesteps, observations, actions, returns_to_go, traj_masks
        
    @property
    def observation_dim(self):
        return self.trajectories[0]["observations"].shape[1]
    
    @property
    def num_actions(self):
        return self.trajectories[0]["actions"].shape[1]
    
    def __len__(self):
        return len(self.trajectories)
    
    
class MaskedAttention(nn.Module):
    def __init__(self, hidden_size, seq_len, num_heads, dropout_prob=0.2):
        super(MaskedAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        
        self.proj = nn.Linear(hidden_size, hidden_size)
        
        self.attn_dropout = nn.Dropout(dropout_prob)
        self.proj_dropout = nn.Dropout(dropout_prob)
        
        ones = torch.ones((seq_len, seq_len))
        attn_mask = torch.tril(ones).view(1, 1, seq_len, seq_len)
        self.register_buffer('attn_mask',attn_mask)
        
    def forward(self, x):
        B, T, C = x.shape 
        N, D = self.num_heads, C // self.num_heads 

        q = self.q(x).view(B, T, N, D).transpose(1,2)
        k = self.k(x).view(B, T, N, D).transpose(1,2)
        v = self.v(x).view(B, T, N, D).transpose(1,2)
        
        attn = q @ k.transpose(2,3) / math.sqrt(D)
        attn = attn.masked_fill(self.attn_mask[...,:T,:T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn @ v)
        
        attn = attn.transpose(1, 2).contiguous().view(B,T,N*D)

        x = self.proj_dropout(self.proj(attn))
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, hidden_size, seq_len, num_heads, dropout_prob):
        super(DecoderBlock, self).__init__()
        
        self.attn = MaskedAttention(hidden_size, seq_len, num_heads, dropout_prob)
        
        self.mlp = nn.Sequential(
                nn.Linear(hidden_size, 4*hidden_size),
                nn.GELU(),
                nn.Linear(4*hidden_size, hidden_size),
                nn.Dropout(dropout_prob),
            )
        
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x + self.attn(x) 
        x = self.ln1(x)
        x = x + self.mlp(x) 
        x = self.ln2(x)
        return x
    
class DecisionTransformer(nn.Module):
    def __init__(self, observation_dim, num_actions, num_blocks, hidden_size, seq_len,
                 num_heads, dropout_prob, max_timestep=4096):
        super(DecisionTransformer, self).__init__()

        self.observation_dim = observation_dim
        self.num_actions = num_actions
        self.hidden_size = hidden_size

        input_seq_len = 3 * seq_len
        
        decoder_blocks = [DecoderBlock(hidden_size, input_seq_len, num_heads, dropout_prob) for _ in range(num_blocks)]
        
        self.transformer = nn.Sequential(*decoder_blocks)

        self.ln = nn.LayerNorm(hidden_size)
        
        self.embed_timestep = nn.Embedding(max_timestep, hidden_size)
        self.embed_rtg = torch.nn.Linear(1, hidden_size)
        self.embed_obs = torch.nn.Linear(observation_dim, hidden_size)
        self.embed_act = torch.nn.Linear(num_actions, hidden_size)
        
        self.predict_rtg = nn.Linear(hidden_size, 1)
        self.predict_obs = nn.Linear(hidden_size, observation_dim)
        self.predict_act = nn.Linear(hidden_size, num_actions)

    def forward(self, timesteps, observations, actions, returns_to_go):

        B, T, _ = observations.shape

        time_embeddings = self.embed_timestep(timesteps)

        obs_embeddings = self.embed_obs(observations) + time_embeddings
        act_embeddings = self.embed_act(actions) + time_embeddings
        rtg_embeddings = self.embed_rtg(returns_to_go) + time_embeddings

        embeddings = torch.stack((rtg_embeddings, obs_embeddings, act_embeddings), dim=1
                        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.hidden_size)

        embeddings = self.ln(embeddings)
        
        hidden_states = self.transformer(embeddings)

        hidden_states = hidden_states.reshape(B, T, 3, self.hidden_size).permute(0, 2, 1, 3)

        rtg_preds = self.predict_rtg(hidden_states[:,2])    
        obs_preds = self.predict_obs(hidden_states[:,2])    
        act_preds = self.predict_act(hidden_states[:,1])  
        return rtg_preds, obs_preds, act_preds

class VAE(nn.Module):
    def __init__(self, observation_dim, num_actions, latent_dim, max_action):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.max_action = max_action
        
        self.e1 = nn.Linear(observation_dim + num_actions, 750)
        self.e2 = nn.Linear(750, 750)
        
        self.mean = nn.Linear(750, latent_dim)
        
        self.log_std = nn.Linear(750, latent_dim)
        
        self.d1 = nn.Linear(observation_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, num_actions)
    
    def forward(self, observation, action):
        z = F.relu(self.e1(torch.cat([observation, action], 1)))
        z = F.relu(self.e2(z))
        
        mean = self.mean(z)
        
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        
        z = mean + std * torch.randn_like(std)
        
        a = self.decode(observation, z)
        
        return a, mean, std
    
    def decode(self, observation, device, z=None):
        if z is None:
            z = torch.randn((observation.shape[0], self.latent_dim)).clamp(-0.5,0.5).to(device)
            
        a = F.relu(self.d1(torch.cat([observation, z], 1)))
        a = F.relu(self.d2(a))
        a = F.tanh(self.d3(a)) * self.max_action
        return a