# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:12:51 2022

@author: lanaya
"""
import gym

env = gym.make("CartPole-v0")

a = [env] * 4
print(a)
