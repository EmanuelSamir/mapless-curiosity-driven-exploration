import sys
sys.path.insert(0, '../')

from algorithms.a2cilstmrndpn import A2CiLSTMRNDPNAgent
import gym_robot2d
#from robot2d import Robot2D
from torch import nn
import torch
import gym
import time
import numpy as np
from matplotlib import pyplot as plt
from src.sim2sim import *
from tqdm import tqdm


env = gym.make('robot2d-v0')

# Environments parameters
s = env.reset()



state_dim = s.shape[0]
actions =       [
                    np.array([3., 0., 0.]),
                    np.array([0., 0., -5.]),
                    np.array([0., 0., 5.]),
                ]
action_dim = 3

#lr_lst = [1e-3, 1e-4, 1e-5]
lr_lst = [1e-3]
gamma_lst = [0.999]

# Mejor caso fue para 0.99, deberia ser 0.999

trained = False

models = [
    '_0526_01-37-31/e=100_ri=61.3519287109375_re=-654.0_steps=402.pth',
    '_0526_01-37-31/e=250_ri=191.3413543701172_re=-531.0_steps=402.pth',
    '_0526_01-37-31/e=500_ri=53.332462310791016_re=-84.0_steps=402.pth',
    '_0526_01-37-31/e=750_ri=171.526611328125_re=-261.0_steps=402.pth',
    '_0526_01-37-31/best_e=876_ri=310.7063903808594_re=198.0_steps=402.pth'
]


for i, model in enumerate(models):
    print(model)
    cond = True
    while cond:
        agent = A2CiLSTMRNDPNAgent(env, state_dim, action_dim, actions, trial = True, load_model_fn = model)

        agent.test()

        print("Try again? (y/n)")
        z = input()
        if z == 'n':
            cond = False
