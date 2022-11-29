# -*- coding: utf-8 -*-
from typing import Optional, Type, Any, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math 

class MaskedAttention(nn.Module):
    def __init__(self, hidden_size: int, seq_len: int, num_heads: int, dropout_prob: float = 0.2):
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, hidden_size: int, seq_len: int, num_heads: int, dropout_prob: float = 0.2):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) 
        x = self.ln1(x)
        x = x + self.mlp(x) 
        x = self.ln2(x)
        return x
    
class DecisionTransformer(nn.Module):
    def __init__(self, observation_dim: int, num_actions: int, num_blocks: int, hidden_size: int, 
                 seq_len: int, num_heads: int, dropout_prob: float = 0.2, max_timestep: int = 4096):
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

    def forward(self, timesteps: torch.Tensor, observations: torch.Tensor, 
                actions: torch.Tensor, returns_to_go: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

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
        return (rtg_preds, obs_preds, act_preds)

class VAE(nn.Module):
    def __init__(self, observation_dim: int, num_actions: int, latent_dim: int, max_action: int):
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
    
    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = F.relu(self.e1(torch.cat([observation, action], 1)))
        z = F.relu(self.e2(z))
        
        mean = self.mean(z)
        
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        
        z = mean + std * torch.randn_like(std)
        
        a = self.decode(observation, z)
        
        return (a, mean, std)
    
    def decode(self, observation: torch.Tensor, device: torch.device, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        if z is None:
            z = torch.randn((observation.shape[0], self.latent_dim)).clamp(-0.5,0.5).to(device)
            
        a = F.relu(self.d1(torch.cat([observation, z], 1)))
        a = F.relu(self.d2(a))
        a = F.tanh(self.d3(a)) * self.max_action
        return a
    
class SiameseNet(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 num_actions: int,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3}
                 ):
        super(SiameseNet, self).__init__()
        
        self.fc1 = nn.Linear(observation_dim, 64)
        
        self.fc2 = nn.Linear(64, 64)
        
        self.classifier = nn.Linear(128, num_actions)
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
        
    def embedding(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
        
    def forward(self, x1, x2):
        x1 = self.embedding(x1)
        
        x2 = self.embedding(x2)
        
        x = torch.cat([x1, x2], dim=-1)
        
        x = self.classifier(x)
        
        x = self.softmax(x)
        
        return x 
    
class BootstrappedHead(nn.Module):
    def __init__(self, input_size: int, num_actions: int, hidden_size: int = 64):
        super(BootstrappedHead, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
class CuriosityModel(nn.Module):
    def __init__(self, observation_dim: int, num_actions: int, hidden_size: int = 64):
        super(CuriosityModel, self).__init__()
        
        self.observation_dim = observation_dim
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        
        self.feature = nn.Sequential(
                       nn.Linear(observation_dim, hidden_size),
                       nn.Linear(hidden_size, hidden_size),
                       nn.Linear(hidden_size, hidden_size)
                       )
        
        self.inverse_model = nn.Sequential(
                             nn.Linear(hidden_size * 2, hidden_size),
                             nn.Linear(hidden_size, hidden_size),
                             nn.Linear(hidden_size, num_actions)
                             )
        
        self.forward_model = nn.Sequential(
                       nn.Linear(hidden_size + num_actions, hidden_size),
                       nn.Linear(hidden_size, hidden_size),
                       nn.Linear(hidden_size, hidden_size)
                       )
        
        self.optimizer = optim.Adam(self.parameters())
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s, s_, a = x1, x2, x3

        encoded_s = self.feature(s)
        encoded_s_ = self.feature(s_)
        
        pred_actions = self.inverse_model(torch.cat((encoded_s, encoded_s_), dim=-1))
        
        pred_states = self.forward_model(torch.cat((encoded_s, a), dim=-1))
        
        return (encoded_s_, pred_states, pred_actions)