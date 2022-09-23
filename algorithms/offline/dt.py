# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
from typing import Any, Dict, Optional, Union
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from common.utils import DecisionTransformer, TrajectoryDataset

class DT():
    def __init__(
                 self,
                 dataset_dir: str,
                 seq_len: int,
                 batch_size: int = 64,
                 num_epochs: int = 10,
                 num_blocks: int = 4, 
                 num_heads: int = 4,
                 hidden_size: int = 256,
                 dropout_prob: float = 0.2,
                 max_timestep: int = 4096,
                 optimizer: optim.Optimizer = optim.AdamW,
                 optimizer_kwargs: Dict[str, Any] = {"lr":1e-3}
                ):
        self.num_epochs = num_epochs
        
        traj_dataset = TrajectoryDataset(dataset_dir, seq_len)
        
        self.observation_dim = traj_dataset.observation_dim
        self.num_actions = traj_dataset.num_actions
        
        self.traj_dataloader = DataLoader(traj_dataset, batch_size=batch_size)
        
        self.model = DecisionTransformer(
                                    self.observation_dim, 
                                    self.num_actions, 
                                    num_blocks, 
                                    hidden_size, 
                                    seq_len, 
                                    num_heads, 
                                    dropout_prob
                                   )
        
        self.optimizer = optimizer(self.model.parameters(), **optimizer_kwargs)
        
    def learn(self) -> None:
        for epoch in range(self.num_epochs):
            
            for trajs in self.traj_dataloader:
                
                 timesteps, observations, actions, returns_to_go, traj_masks = trajs
                 
                 returns_to_go = returns_to_go.unsqueeze(dim=2)
                 
                 target_actions = actions.detach()
                 
                 rtg_preds, obs_preds, act_preds = self.model(
                                                              timesteps, 
                                                              observations, 
                                                              actions, 
                                                              returns_to_go
                                                              )
                 
                 pred_actions = act_preds.view(-1, self.num_actions)[traj_masks.view(-1,) > 0]
                 target_actions = target_actions.view(-1, self.num_actions)[traj_masks.view(-1,) > 0]

                 loss = F.mse_loss(pred_actions, target_actions)
                 
                 self.optimizer.zero_grad()
                 loss.backward()
                 self.optimizer.step()  