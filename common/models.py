# -*- coding: utf-8 -*-
import torch 
import torch.nn as nn
import torch.optim as optim
from common.kfac import AddBias, KFACOptimizer

class ACKTRActor(nn.Module):
    def __init__(self, 
                 observation_dim, 
                 num_action, 
                 hidden_size=64, 
                 activation_fn=nn.Tanh,
                 net_arch=None,
                 optimizer_kwargs={"lr":0.25},
                ):
        
        super(ACKTRActor, self).__init__()
        
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
        
        self.logstds = AddBias(torch.zeros(num_action))
        
        self.optimizer = KFACOptimizer(self, **optimizer_kwargs)
    
    def forward(self, x):
        if self.net_arch is not None:
            for i, layer in enumerate(self.model):
                x = layer(x)
            means = x
        else:
            means = self.model(x)
        
        zeros = torch.zeros(means.size())
        logstds = self.logstds(zeros)
         
        stds = torch.clamp(logstds.exp(), 7e-4, 50)
        return torch.distributions.Normal(means, stds) 

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

class DDPGActor(nn.Module):
    def __init__(self, 
                 observation_dim, 
                 num_action,
                 hidden_size=64, 
                 activation_fn=nn.Tanh,
                 net_arch=None,
                 optimizer=optim.Adam,
                 optimizer_kwargs={"lr":3e-4}
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
    
    def forward(self, x):
        if self.net_arch is not None:
            for i, layer in enumerate(self.model):
                x = layer(x)
        else:
            x = self.model(x)
        return x
         
class VPGActor(nn.Module):
    def __init__(self, 
                 observation_dim, 
                 num_action, 
                 hidden_size=64, 
                 activation_fn=nn.Tanh,
                 net_arch=None,
                 optimizer=optim.Adam,
                 optimizer_kwargs={"lr":3e-4}
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
    
    def forward(self, x):
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
                 observation_dim, 
                 hidden_size=64, 
                 activation_fn=nn.Tanh, 
                 net_arch=None,
                 optimizer_kwargs={"lr":0.25},
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
        
    def forward(self, x):
        if self.net_arch is not None:
            for i, layer in enumerate(self.model):
                x = layer(x)
        else:
            x = self.model(x)
        return x
    
class QCritic(nn.Module):
    def __init__(self, 
                 observation_dim,
                 num_action, 
                 hidden_size=64, 
                 activation_fn=nn.Tanh, 
                 net_arch=None,
                 optimizer=optim.Adam,
                 optimizer_kwargs={"lr":3e-4}
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

    def forward(self, x_s, x_a):
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
                 observation_dim, 
                 hidden_size=64, 
                 activation_fn=nn.Tanh, 
                 net_arch=None,
                 optimizer=optim.Adam,
                 optimizer_kwargs={"lr":1e-3},
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
        
    def forward(self, x):
        if self.net_arch is not None:
            for i, layer in enumerate(self.model):
                x = layer(x)
        else:
            x = self.model(x)
        return x