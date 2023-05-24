# -*- coding: utf-8 -*-
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
from typing import Any, Type, Dict
from common.optimizers import AddBias, KFACOptimizer

class ACKTR(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 num_actions: int, 
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 0.25},
                ):
        
        super(ACKTR, self).__init__()
        
        self.observation_dim = observation_dim
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        
        self.model = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, num_actions),
            )
        
        self.logstds = AddBias(torch.zeros(num_actions))
        
        self.optimizer = KFACOptimizer(self, **optimizer_kwargs)
    
    def forward(self, x: torch.Tensor) -> distributions.Normal:
        means = self.model(x)

        zeros = torch.zeros(means.size())
        logstds = self.logstds(zeros)
        stds = torch.clamp(logstds.exp(), 7e-4, 50)
        
        return distributions.Normal(means, stds)
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            means = self.model(x)
        
            zeros = torch.zeros(means.size())
            logstds = self.logstds(zeros)
            stds = torch.clamp(logstds.exp(), 7e-4, 50)
        
        normal_dist = distributions.Normal(means, stds)
        action = normal_dist.sample().detach().numpy()
        return action
    
class C51(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 num_actions: int, 
                 v_range: np.ndarray,
                 num_atoms: int = 51, 
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3}
                ):
        
        super(C51, self).__init__()
        
        self.observation_dim = observation_dim
        self.num_actions = num_actions
        self.v_range = v_range
        self.num_atoms = num_atoms
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        
        self.model = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, num_actions * self.num_atoms),
            )
        
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v_probs = self.model(x)
        v_probs = F.softmax(v_probs.view(-1, self.num_actions, self.num_atoms), dim=2)
        return v_probs
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            v_prob = self.model(x)
            v_prob = F.softmax(v_prob.view(-1, self.num_actions, self.num_atoms), dim=2)
            q = torch.sum(v_prob * self.v_range.view(1, 1, -1), dim=2)
            action = q.argmax(dim=1, keepdim=True).numpy()
        return action
            
class DQN(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 num_actions: int, 
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3}
                ):
        super(DQN, self).__init__()
        
        self.observation_dim = observation_dim
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        
        self.model = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, num_actions),
            )
        
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.model(x)
        return q
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            q = self.model(x)
            
            action = q.argmax(dim=-1, keepdim=True).detach().numpy()
        return action
    
class DDPG(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 num_actions: int,
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 3e-4}
                ):
        
        super(DDPG, self).__init__()
        
        self.observation_dim = observation_dim
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        
        self.model = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, num_actions),
            nn.Tanh(),
            )
        
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            x = self.model(x).cpu().detach().numpy()
        return x

class DuelingDQN(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 num_actions: int, 
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3}
                ):
        super(DuelingDQN, self).__init__()
        
        self.observation_dim = observation_dim
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn

        self.backbone = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            activation_fn(),
            )
        
        self.adv = nn.Linear(hidden_size, num_actions)
        
        self.v = nn.Linear(hidden_size, 1)
        
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
    
        adv = self.adv(x)
        v = self.v(x)
        
        return v + adv - adv.mean()
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            x = self.backbone(x)
            adv = self.adv(x)
            v = self.v(x)
        
        q = v + adv - adv.mean()
        a = q.argmax(dim=-1, keepdim=True).cpu().detach().numpy()
        return a 

class QRDQN(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 num_actions: int, 
                 num_quantiles: int = 10, 
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3}
                ):
        
        super(QRDQN, self).__init__()
        
        self.observation_dim = observation_dim
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        
        self.model = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, num_actions * self.num_quantiles),
            )
        
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantiles = self.model(x)
        quantiles = quantiles.view(-1, self.num_actions, self.num_quantiles)
        return quantiles
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            quantiles = self.model(x)
            quantiles = quantiles.view(-1, self.num_actions, self.num_quantiles)
            q = quantiles.mean(dim=2)
            action = q.argmax(dim=1, keepdim=True).numpy()
        return action
    
class VPG(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 num_actions: int, 
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 3e-4},
                ):
        
        super(VPG, self).__init__()
        
        self.observation_dim = observation_dim
        self.num_action = num_actions
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        
        self.model = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, num_actions),
            )
            
        logstds_param = nn.Parameter(torch.full((num_actions,), 0.1))
        self.register_parameter("logstds", logstds_param)
          
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        means = self.model(x)
         
        stds = torch.clamp(self.logstds.exp(), 7e-4, 50)
        
        return distributions.Normal(means, stds) 
           
    def predict(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            mean = self.model(x)
            
            std = torch.clamp(self.logstds.exp(), 7e-4, 50)
            
            dist = distributions.Normal(mean, std) 
            
            action = dist.sample().cpu().detach().numpy()
               
        return action
      
class Q1(nn.Module):
    def __init__(self, 
                 observation_dim: int,
                 num_actions: int, 
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh, 
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 3e-4}
                ):
        
        super(Q1, self).__init__()
        
        self.observation_dim = observation_dim
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn

        self.fc_s = nn.Linear(observation_dim, hidden_size)
        self.fc_a = nn.Linear(num_actions, hidden_size)
        self.fc_q1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_q2 = nn.Linear(hidden_size, 1)
        
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)

    def forward(self, x_s: torch.Tensor, x_a: torch.Tensor) -> torch.Tensor:
        h_s = self.activation_fn()(self.fc_s(x_s))
        h_a = self.activation_fn()(self.fc_a(x_a))
        h = torch.cat([h_s,h_a], dim=1)
        x = self.activation_fn()(self.fc_q1(h))
        x = self.fc_q2(x)
        return x
    
class Q2(nn.Module):
    def __init__(self, 
                 observation_dim: int,
                 num_actions: int, 
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh, 
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 3e-4}
                ):
        
        super(Q2, self).__init__()
        
        self.observation_dim = observation_dim
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        
        self.model = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, num_actions),
            )
        
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
    
class V(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh, 
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3},
                ):
        
        super(V, self).__init__()
        
        self.observation_dim = observation_dim
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
    
        self.model = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, 1),
            )
        
        if optimizer == KFACOptimizer:
            self.optimizer = optimizer(self, **optimizer_kwargs)
        else:
            self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x