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

sets = [True, False]

# fn = '_0513_17-51-41/tmp_model.pth'
# fn = 'tmp_model.pth'#'e400_rtensor([43.9384]).pth'# 'best_model_e266_r102.66022491455078.pth'#  'best_model_e579_r200.0.pth
# '



# Baseline
#fn_base = '_0514_01-22-57/best_e=1899_ri=0_re=201.0_steps=402.pth'
fn_base = '_0527_17-22-18/tmp_model.pth'
#fn_base = '_0523_10-02-24/tmp_model.pth'
#fn_base = '_0519_01-05-52/e=1150_ri=25.208454132080078_re=177.0_steps=402.pth'

# Best case
fn_rnd = '_0526_01-37-31/best_e=876_ri=310.7063903808594_re=198.0_steps=402.pth'

# Random
fn_random = None

models = [fn_random, fn_rnd, fn_base]

n_experiments = 1000


for i, model in enumerate(models):
    print(model)
    cond = True
    while cond:
        if i == 0:
            agent = A2CiLSTMRNDPNAgent(env, state_dim, action_dim, actions, trial = True, load_model_fn = model)

            agent.test()

        else:
            agent.load_models(model)

            agent.test(going_back_start = True)

        print("Try again? (y/n)")
        z = input()
        if z == 'n':
            cond = False
