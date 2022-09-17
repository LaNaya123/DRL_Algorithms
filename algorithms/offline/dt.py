# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from common.utils import DecisionTransformer, TrajectoryDataset

class DT():
    def __init__(
                 self,
                 dataset_dir,
                 seq_len,
                 batch_size=64,
                 num_epochs= 10,
                 num_blocks=4, 
                 num_heads=4,
                 hidden_size=256,
                 dropout_prob=0.2,
                 max_timestep=4096,
                 optimizer=optim.AdamW,
                 optimizer_kwargs={"lr":1e-3}
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
        
    def learn(self):
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
                 
    def inference():