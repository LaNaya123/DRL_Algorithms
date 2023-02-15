# -*- coding: utf-8 -*-
from typing import Union, Tuple
import os
import numpy as np
from envs import Monitor, VecEnv
from policies import OnPolicyAlgorithm, OffPolicyAlgorithm
from utils.functionality import evaluate_policy


class BaseCallback():
    def __init__(self, algor: Tuple[OnPolicyAlgorithm, OffPolicyAlgorithm], env: Union[Monitor, VecEnv]):
        self.algor = algor
        self.env = env
        self.num_calls = 0
        self.num_timesteps = 0

    def on_training_start(self) -> None:
        pass

    def on_rollout_start(self) -> None:
        pass

    def on_step(self) -> bool:
        self.num_calls += 1
        self.num_timesteps = self.algor.num_timesteps
        return True
    
    def on_training_end(self) -> None:
        pass
    
    def on_rollout_end(self) -> None:
        pass

class CheckpointCallback(BaseCallback):
    def __init__(self, model, env, save_freq: int, save_path: str, save_prefix: str = "rl_model"):
        super(CheckpointCallback).__init__(model, env)
        self.save_freq = save_freq
        self.save_path = save_path
        self.save_prefix = save_prefix
        
        os.makedirs(self.save_path, exist_ok=True)
        
    def _checkpoint_path(self):
        return os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
        
        
    def on_step(self):
        if self.num_calls % self.save_freq == 0:
            model_path = self._checkpoint_path()
            self.algor.save(model_path)
        return True
     
class EvalCallback(BaseCallback):
    def __init__(self, algor: Tuple[OnPolicyAlgorithm, OffPolicyAlgorithm], env: Union[Monitor, VecEnv],
                 best_model_save_path: str, eval_freq: int = 10000, num_eval_episodes: int = 10):
        super(EvalCallback).__init__(algor, env)
        self.best_model_save_path = best_model_save_path
        self.eval_freq = eval_freq
        self.num_eval_episodes = num_eval_episodes
        self.best_mean_rewards = -np.inf
        self.last_mean_rewards = -np.inf
        
        os.makedirs(self.best_model_save_path, exist_ok=True)
        
    def on_step(self):
        if self.num_calls > 0 and self.num_calls % self.eval_freq == 0:
            mean_rewards, std_rewards, mean_lengths, std_lengths = evaluate_policy(self.algor.policy_net, self.env, self.num_eval_episodes)
            
            if mean_rewards > self.best_mean_rewards:
                if self.algor.verbose >= 1:
                    print("New best mean reward")
                
                self.model.save(os.path.join(self.best_model_save_path, "best_model.pkl"))
                
                self.best_mean_rewards = mean_rewards

        return True
    


                
        
            
            
            
            
            
            
            
            
            
        
        
        
        
        
    
            
    
        
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    