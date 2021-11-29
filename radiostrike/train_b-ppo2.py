'''
UFPA - LASSE - Telecommunications, Automation and Electronics Research and Development Center - www.lasse.ufpa.br
CAVIAR - Communication Networks and Artificial Intelligence Immersed in Virtual or Augmented Reality
Ailton Oliveira, Felipe Bastos, João Borges, Emerson Oliveira, Daniel Suzuki, Lucas Matni, Rebecca Aben-Athar, Aldebaro Klautau (UFPA): aldebaro@ufpa.br
CAVIAR: https://github.com/lasseufpa/ITU-Challenge-ML5G-PHY-RL.git

Script to train the baseline of reinforcement learning applied to Beam-selection
V1.0
'''

import os
import sys
import csv
import argparse
import numpy as np
import pandas as pd
import caviar_tools
from beamselect_env_customState import BeamSelectionEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2


# Create the folder
try:
    os.mkdir('./model')
except OSError as error:
    print(error)

'''
Trains an PPO2 network and stores it in a file.

Usage:

$ python3 train_b-ppo2.py -m <model_name> -ep <train_ep_id#first> <train_ep_id#last>

Example:

$ python3 train_b-ppo2.py -m baseline.ppo2 -ep 0 1
'''
parser = argparse.ArgumentParser()

parser.add_argument("--model", "-m", 
                    help="Pass RL model name",
                    action="store", 
                    dest="model", 
                    type=str)

parser.add_argument("--episode", "-ep",
                    nargs='+',
                    help="IDs of the first and " +
                         "last episodes to train", 
                    action="store", 
                    dest="episode", 
                    type=str)
                   
args = parser.parse_args()

e = BeamSelectionEnv(ep=args.episode)

# Get total number of steps based on the timestamps for a specific UE  
n_steps = caviar_tools.linecount(args.episode)

n_epochs = 1
train_steps = n_epochs * n_steps
# batch size(self.n_batch) is n_steps * n_envs 
# n_steps : the number of steps to run for each environment per update
# n_envs : number of environment copies running in parallel (今回はデフォルトで1)

# nminibatches: (int) Number of training minibatches per update. For recurrent policies,
#   the number of environments run in parallel should be a multiple of nminibatches
# batch_size = self.n_batch // self.nminibatches

model = PPO2(policy="MlpPolicy", 
            learning_rate=1e-3, 
            n_steps=32,    #n_minibatchの倍数ならエラーは出ない
            verbose=1,
            nminibatches=1,
            gamma=0.7, 
            env=e, 
            seed=0,
            tensorboard_log="./log_tensorboard/",
            # tensorboard_log="/data/radiostrike2021/result/log_tensorboard/"
            )

model.learn(total_timesteps=train_steps, tb_log_name="PPO2_test")
model_path = "./model/"+str(args.model)
model.save(model_path) 

e.save_rewards_log(args.model)

