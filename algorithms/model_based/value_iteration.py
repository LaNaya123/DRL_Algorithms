# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
import gym
from common.envs import ChainEnv

class VI():
    def __init__(
                 self, 
                 env: gym.Env, 
                 training_iterations: int = 10000, 
                 gamma: float = 0.9, 
                 delta: float = 1e-400
                ):
        
        self.env = env
        self.training_iterations = training_iterations
        self.gamma = gamma
        self.delta = delta
        
        self.v = [0] * self.env.observation_space.n
        self.pi = [None] * self.env.observation_space.n
        
    def learn(self) -> None:
        v = [0] * self.env.observation_space.n
        delta = float('-inf')
        
        for _ in range(self.training_iterations):
            for s in self.env.states:
                max_q = float("-inf")
                for a in self.env.actions:
                    q = 0
                    for s_ in self.env.states:
                        q += self.env.transitions[s][s_][a] * (self.env.rewards[s] + self.gamma * self.v[s_])
                    max_q = max(max_q, q)
                
                v[s] = max_q
            
                delta = max(delta, abs(self.v[s] - v[s]))
        
            self.v = v
        
            if delta <= self.delta:
                break
        
        for s in self.env.states:
            max_q = float("-inf")
            for a in self.env.actions:
                q = self.env.rewards[s]
                for s_ in self.env.states:
                    q += self.env.transitions[s][s_][a] * self.gamma * self.v[s_]
            
                if q > max_q:
                    max_q = q
                    self.pi[s] = a
                    

if __name__ == "__main__":
    env = ChainEnv()
    vi = VI(env, training_iterations=10)
    vi.learn()
    print(vi.v)