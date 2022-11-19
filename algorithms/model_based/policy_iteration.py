# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\lanaya\Desktop\DRLAlgorithms")
import gym
from common.envs import ChainEnv

class PI():
    def __init__(
                 self, 
                 env: gym.Env, 
                 training_iterations: int = 10000, 
                 gamma: float = 0.9
                ):
        
        self.env = env
        self.training_iterations = training_iterations
        self.gamma = gamma
        
        self.v = [0] * self.env.observation_space.n
        self.pi = [0] * self.env.observation_space.n
        
    def learn(self):
        v = [0] * self.env.observation_space.n
        
        for _ in range(self.training_iterations):
            for s in self.env.states:
                tmp_v = 0
                for s_ in self.env.states:
                    tmp_v += self.env.transitions[s][s_][self.pi[s]] * (self.env.rewards[s] + self.gamma * self.v[s_])
                
                v[s] = tmp_v
                
            self.v = v
            
            for s in self.env.states:
                max_q = float("-inf")
                for a in self.env.actions:
                    q = 0
                    for s_ in self.env.states:
                        q += self.env.transitions[s][s_][a] * (self.env.rewards[s] + self.gamma * self.v[s_])
                    if q > max_q:
                        self.pi[s] = a
                        max_q = q
                        policy_changed = True
            
            if not policy_changed:
                break
            
if __name__ == "__main__":
    env = ChainEnv()
    pi = PI(env, training_iterations=10)
    pi.learn()
    print(pi.v)