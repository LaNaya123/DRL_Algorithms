# -*- coding: utf-8 -*-
from typing import Any, Union, Optional, Type, List, Dict
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from common.kfac import AddBias, KFACOptimizer
from common.utils import BootstrappedHead

class ACKTRActor(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 num_actions: int, 
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 net_arch: Optional[List[int]] = None,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 0.25},
                ):
        
        super(ACKTRActor, self).__init__()
        
        self.observation_dim = observation_dim
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.net_arch = net_arch
        
        if net_arch is not None:
            in_features = observation_dim
            self.model = nn.ModuleList()
            for i, out_features in enumerate(net_arch):
                self.model.append(nn.Linear(in_features, out_features))
                if i != len(net_arch) - 1:
                    self.model.append(activation_fn())
                in_features = out_features
        
        else:
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
        if self.net_arch is not None:
            for i, layer in enumerate(self.model):
                x = layer(x)
            means = x
        else:
            means = self.model(x)
        
        zeros = torch.zeros(means.size()).to(device)
        logstds = self.logstds(zeros)
         
        stds = torch.clamp(logstds.exp(), 7e-4, 50)
        return distributions.Normal(means, stds) 

class BCQActor(nn.Module):
    def __init__(self,
                 observation_dim: int,
                 num_actions: int,
                 max_action: torch.Tensor,
                 hidden_size: int = 64,
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 net_arch: Optional[List[int]] = None,
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
        self.net_arch = net_arch
        self.max_perturbation = max_perturbation
        
        if net_arch is not None:
            in_features = observation_dim
            self.model = nn.ModuleList()
            for i, out_features in enumerate(net_arch):
                self.model.append(nn.Linear(in_features, out_features))
                if i != len(net_arch) - 1:
                    self.model.append(activation_fn())
                else:
                    self.model.append(nn.Tanh())
                in_features = out_features
        else:
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
        if self.net_arch is not None:
            for i, layer in enumerate(self.model):
                x = layer(x)
        else: 
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
                 net_arch: Optional[List[int]] = None,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3}
                ):
        super(BootstrappedQNetwork, self).__init__()
        
        self.observation_dim = observation_dim
        self.num_actions = num_actions
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.net_arch = net_arch
        
        if net_arch is not None:
            in_features = observation_dim
            self.backbone = nn.ModuleList()
            for i, out_features in enumerate(net_arch):
                self.model.append(nn.Linear(in_features, out_features))
                if i != len(net_arch) - 1:
                    self.model.append(activation_fn())
                in_features = out_features
            
            self.heads = nn.ModuleList([BootstrappedHead(out_features, num_actions) for k in range(num_heads)])
        
        else:
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
        if self.net_arch is not None:
            for i, layer in enumerate(self.backbone):
                x = layer(x)
        else:
            x = self.backbone(x)
            
        if k is not None:
            x = self.heads[k](x)
        else:
            x = [head(x) for head in self.heads]
        return x        
    
class DeepQNetwork(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 num_actions: int, 
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 net_arch: Optional[List[int]] = None,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3}
                ):
        super(DeepQNetwork, self).__init__()
        
        self.observation_dim = observation_dim
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.net_arch = net_arch
        
        if net_arch is not None:
            in_features = observation_dim
            self.model = nn.ModuleList()
            for i, out_features in enumerate(net_arch):
                self.model.append(nn.Linear(in_features, out_features))
                if i != len(net_arch) - 1:
                    self.model.append(activation_fn())
                in_features = out_features
        
        else:
            self.model = nn.Sequential(
                nn.Linear(observation_dim, hidden_size),
                activation_fn(),
                nn.Linear(hidden_size, hidden_size),
                activation_fn(),
                nn.Linear(hidden_size, num_actions),
                )
        
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net_arch is not None:
            for i, layer in enumerate(self.model):
                x = layer(x)
        else:
            x = self.model(x)
        return x

class DDPGActor(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 num_action: int,
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 net_arch: Optional[List[int]] = None,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 3e-4}
                ):
        
        super(DDPGActor, self).__init__()
        
        self.observation_dim = observation_dim
        self.num_action = num_action
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.net_arch = net_arch
        
        if net_arch is not None:
            in_features = observation_dim
            self.model = nn.ModuleList()
            for i, out_features in enumerate(net_arch):
                self.model.append(nn.Linear(in_features, out_features))
                if i != len(net_arch) - 1:
                    self.model.append(activation_fn())
                else:
                    self.model.append(nn.Tanh())
                in_features = out_features
        
        else:
            self.model = nn.Sequential(
                nn.Linear(observation_dim, hidden_size),
                activation_fn(),
                nn.Linear(hidden_size, hidden_size),
                activation_fn(),
                nn.Linear(hidden_size, num_action),
                nn.Tanh(),
                )
        
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net_arch is not None:
            for i, layer in enumerate(self.model):
                x = layer(x)
        else:
            x = self.model(x)
        return x
    
class RecurrentQNetwork(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 num_action: int, 
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 net_arch: Optional[List[int]] = None,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3}
                ):
        super(RecurrentQNetwork, self).__init__()
        
        self.observation_dim = observation_dim
        self.num_action = num_action
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.net_arch = net_arch
        
        if net_arch is not None:
            in_features = observation_dim
            self.model = nn.ModuleList()
            for i, out_features in enumerate(net_arch):
                if i == 0:
                    self.model.append(nn.LSTM(in_features, out_features, batch_first=True))
                else:
                    self.model.append(nn.Linear(in_features, out_features))
                if i != len(net_arch) - 1:
                    self.model.append(activation_fn())
                in_features = out_features
        
        else:
            self.model = nn.Sequential(
                nn.LSTM(observation_dim, hidden_size, batch_first=True),
                activation_fn(),
                nn.Linear(hidden_size, hidden_size),
                activation_fn(),
                nn.Linear(hidden_size, num_action),
                )
        
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net_arch is not None:
            for i, layer in enumerate(self.model):
                x = layer(x)
        else:
            x = self.model(x)
        return x
    
class VPGActor(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 num_action: int, 
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 net_arch: Optional[List[int]] = None,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 3e-4}
                ):
        
        super(VPGActor, self).__init__()
        
        self.observation_dim = observation_dim
        self.num_action = num_action
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.net_arch = net_arch
        
        if net_arch is not None:
            in_features = observation_dim
            self.model = nn.ModuleList()
            for i, out_features in enumerate(net_arch):
                self.model.append(nn.Linear(in_features, out_features))
                if i != len(net_arch) - 1:
                    self.model.append(activation_fn())
                in_features = out_features
        
        else:
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
        if self.net_arch is not None:
            for i, layer in enumerate(self.model):
                x = layer(x)
            means = x
        else:
            means = self.model(x)
         
        stds = torch.clamp(self.logstds.exp(), 7e-4, 50)
        return torch.distributions.Normal(means, stds) 

class ACKTRCritic(nn.Module):
    def __init__(self, 
                 observation_dim: int, 
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh, 
                 net_arch: Optional[List[int]] = None,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 0.25},
                ):
        
        super(ACKTRCritic, self).__init__()
        
        self.observation_dim = observation_dim
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.net_arch = net_arch
        
        if net_arch is not None:
            in_features = observation_dim
            self.model = nn.ModuleList()
            for i, out_features in enumerate(net_arch):
                self.model.append(nn.Linear(in_features, out_features))
                if i != len(net_arch) - 1:
                    self.model.append(activation_fn())
                in_features = out_features
        else:
            self.model = nn.Sequential(
                nn.Linear(observation_dim, hidden_size),
                activation_fn(),
                nn.Linear(hidden_size, hidden_size),
                activation_fn(),
                nn.Linear(hidden_size, 1),
            )
        
        self.optimizer = KFACOptimizer(self, **optimizer_kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net_arch is not None:
            for i, layer in enumerate(self.model):
                x = layer(x)
        else:
            x = self.model(x)
        return x
    
class QCritic(nn.Module):
    def __init__(self, 
                 observation_dim: int,
                 num_action: int, 
                 hidden_size: int = 64, 
                 activation_fn: Type[nn.Module] = nn.Tanh, 
                 net_arch: Optional[List[Any]] = None,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 3e-4}
                ):
        
        super(QCritic, self).__init__()
        
        self.observation_dim = observation_dim
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.net_arch = net_arch
        
        if net_arch is not None:
            in_features = observation_dim + num_action
            self.model = nn.ModuleList()
            for i, out_features in enumerate(net_arch):
                self.model.append(nn.Linear(in_features, out_features))
                if i != len(net_arch) - 1:
                    self.model.append(activation_fn())
                in_features = out_features
        
        else:
            self.fc_s = nn.Linear(observation_dim, hidden_size)
            self.fc_a = nn.Linear(num_action, hidden_size)
            self.fc_q1 = nn.Linear(hidden_size * 2, hidden_size)
            self.fc_q2 = nn.Linear(hidden_size, 1)
        
        self.optimizer = optim.Adam(self.parameters(), **optimizer_kwargs)

    def forward(self, x_s: torch.Tensor, x_a: torch.Tensor) -> torch.Tensor:
        if self.net_arch is not None:
            x = torch.cat([x_s, x_a], dim=1)
            for i, layer in enumerate(self.model):
                x = layer(x)
        else: 
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
                 net_arch: Optional[List[int]] = None,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3},
                ):
        
        super(VCritic, self).__init__()
        
        self.observation_dim = observation_dim
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.net_arch = net_arch
        
        if net_arch is not None:
            in_features = observation_dim
            self.model = nn.ModuleList()
            for i, out_features in enumerate(net_arch):
                self.model.append(nn.Linear(in_features, out_features))
                if i != len(net_arch) - 1:
                    self.model.append(activation_fn())
                in_features = out_features
        else:
            self.model = nn.Sequential(
                nn.Linear(observation_dim, hidden_size),
                activation_fn(),
                nn.Linear(hidden_size, hidden_size),
                activation_fn(),
                nn.Linear(hidden_size, 1),
            )
        
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net_arch is not None:
            for i, layer in enumerate(self.model):
                x = layer(x)
        else:
            x = self.model(x)
        return x