# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 13:50:39 2022

@author: lanaya
"""
import torch 
import torch.nn as nn
import torch.optim as optim

class ContinuousActor(nn.Module):
    def __init__(self, 
                 observation_dim, 
                 num_action, 
                 hidden_size=64, 
                 activation_fn=nn.Tanh,
                 net_arch=None,
                 optimizer=optim.Adam,
                 optimizer_kwargs={"lr":3e-4}
                ):
        
        super(ContinuousActor, self).__init__()
        
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
    
    def _orthogonal_init__(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.orthogonal_(layer.weight)
    
    def forward(self, x):
        if self.net_arch is not None:
            for i, layer in enumerate(self.model):
                x = layer(x)
            means = x
        else:
            means = self.model(x)
         
        stds = torch.clamp(self.logstds.exp(), 7e-4, 50)
        
        return torch.distributions.Normal(means, stds) 

class Critic(nn.Module):
    def __init__(self, 
                 observation_dim, 
                 hidden_size=64, 
                 activation_fn=nn.Tanh, 
                 net_arch=None
                ):
        
        super(Critic, self).__init__()
        
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
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        
    def forward(self, x):
        if self.net_arch is not None:
            for i, layer in enumerate(self.model):
                x = layer(x)
            return x
        else:
            return self.model(x)
        
class DeepQNetwork(nn.Module):
    def __init__(self, 
                 observation_dim, 
                 num_action, 
                 hidden_size=64, 
                 activation_fn=nn.Tanh,
                 net_arch=None,
                 optimizer=optim.Adam,
                 optimizer_kwargs={"lr":1e-3}
                ):
        super(DeepQNetwork, self).__init__()
        
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
        
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
    
    def forward(self, x):
        if self.net_arch is not None:
            for i, layer in enumerate(self.model):
                x = layer(x)
        else:
            x = self.model(x)
        return x