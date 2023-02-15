# -*- coding: utf-8 -*-
from typing import Any, Union, Optional, Type, List, Dict, Tuple
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
import random
from common.utils.optimizers import AddBias, KFACOptimizer
from common.utils.models import BootstrappedHead

class ACKTRActor(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 num_actions: int, 
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 0.25},
                ):
        
        super(ACKTRActor, self).__init__()
        
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
    
    def forward(self, x: torch.Tensor, device: torch.device) -> distributions.Normal:
        means = self.model(x)
        
        zeros = torch.zeros(means.size()).to(device)
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
        a = normal_dist.sample().cpu().detach().numpy()
        if len(a.shape) == 2:
            a = a.squeeze(axis=0)
        return a
        
class BCQActor(nn.Module):
    def __init__(self,
                 observation_dim: int,
                 num_actions: int,
                 max_action: torch.Tensor,
                 hidden_size: int = 64,
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 3e-4},
                 max_perturbation: float = 0.05,
                ):
        
        super(BCQActor, self).__init__()
        
        self.observation_dim = observation_dim
        self.num_actions = num_actions
        self.max_action = max_action
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.max_perturbation = max_perturbation
        
       
        self.model = nn.Sequential(
            nn.Linear(observation_dim + num_actions, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, num_actions),
            nn.Tanh(),
            )
        
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
    
    def forward(self, x_s: torch.Tensor, x_a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_s, x_a], dim=1)
        x = self.model(x)
        x = self.max_perturbation * self.max_action * x
        x = (x + x_a).clamp(-self.max_action, self.max_action)
        return x

class BootstrappedQNetwork(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 num_actions: int, 
                 num_heads: int = 10,
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3}
                ):
        super(BootstrappedQNetwork, self).__init__()
        
        self.observation_dim = observation_dim
        self.num_actions = num_actions
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn

        self.backbone = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            )
            
        self.heads = nn.ModuleList([BootstrappedHead(hidden_size, num_actions) for k in range(num_heads)])
        
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
    
    def forward(self, x: torch.Tensor, k: Optional[int] = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = self.backbone(x)
            
        if k is not None:
            x = self.heads[k](x)
        else:
            x = [head(x) for head in self.heads]
        return x    

    def predict(self, x: torch.Tensor) -> np.ndarray:
        k = random.randint(0, self.num_heads-1)
        
        with torch.no_grad():
            x = self.backbone(x)
            q = self.heads[k](x)

            a = q.argmax(dim=-1, keepdim=True).cpu().detach().numpy()
            return a
    
class DeepQNetwork(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 num_actions: int, 
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3}
                ):
        super(DeepQNetwork, self).__init__()
        
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
            
        a = q.argmax(dim=-1, keepdim=True).cpu().detach().numpy()
        return a
        

class DDPGActor(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 num_actions: int,
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 3e-4}
                ):
        
        super(DDPGActor, self).__init__()
        
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

class DuelingQNetwork(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 num_actions: int, 
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3}
                ):
        super(DuelingQNetwork, self).__init__()
        
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

class RecurrentQNetwork(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 num_actions: int, 
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3}
                ):
        super(RecurrentQNetwork, self).__init__()
        
        self.observation_dim = observation_dim
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn

        self.lstm = nn.LSTM(observation_dim + num_actions + 1, hidden_size, batch_first=True),
        self.head = nn.Sequential(
            activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, num_actions),
            )
        
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
    
    def forward(self, x: torch.Tensor, hs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        _, hidden_states = self.lstm(x, hs)
        x = self.head(hidden_states[0])
        return x, hidden_states 
    
class VPGActor(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 num_action: int, 
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 3e-4}
                ):
        
        super(VPGActor, self).__init__()
        
        self.observation_dim = observation_dim
        self.num_action = num_action
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        
        self.model = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, num_action),
            )
        
        logstds_param = nn.Parameter(torch.full((num_action,), 0.1))
        self.register_parameter("logstds", logstds_param)
        
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        means = self.model(x)
         
        stds = torch.clamp(self.logstds.exp(), 7e-4, 50)
        
        return torch.distributions.Normal(means, stds) 
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        means = self.model(x)
        stds = torch.clamp(self.logstds.exp(), 7e-4, 50)
        dists = torch.distributions.Normal(means, stds) 
        a = dists.sample().cpu().detach().numpy()
        return a
        

class ACKTRCritic(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh, 
                 optimizer_kwargs: Dict[str, Any] = {"lr": 0.25},
                ):
        
        super(ACKTRCritic, self).__init__()
        
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
        
        self.optimizer = KFACOptimizer(self, **optimizer_kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
    
class QCritic(nn.Module):
    def __init__(self, 
                 observation_dim: int,
                 num_action: int, 
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh, 
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 3e-4}
                ):
        
        super(QCritic, self).__init__()
        
        self.observation_dim = observation_dim
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn

        self.fc_s = nn.Linear(observation_dim, hidden_size)
        self.fc_a = nn.Linear(num_action, hidden_size)
        self.fc_q1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_q2 = nn.Linear(hidden_size, 1)
        
        self.optimizer = optim.Adam(self.parameters(), **optimizer_kwargs)

    def forward(self, x_s: torch.Tensor, x_a: torch.Tensor) -> torch.Tensor:
        h_s = self.activation_fn()(self.fc_s(x_s))
        h_a = self.activation_fn()(self.fc_a(x_a))
        h = torch.cat([h_s,h_a], dim=1)
        x = self.activation_fn()(self.fc_q1(h))
        x = self.fc_q2(x)
        return x
    
class VCritic(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh, 
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3},
                ):
        
        super(VCritic, self).__init__()
        
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
        
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x