'''
updated 20210917
UFPA - LASSE - Telecommunications, Automation and Electronics Research and Development Center - www.lasse.ufpa.br
CAVIAR - Communication Networks and Artificial Intelligence Immersed in Virtual or Augmented Reality
Ailton Oliveira, Felipe Bastos, João Borges, Emerson Oliveira, Daniel Suzuki, Lucas Matni, Rebecca Aben-Athar, Aldebaro Klautau (UFPA): aldebaro@ufpa.br
CAVIAR: https://github.com/lasseufpa/ITU-Challenge-ML5G-PHY-RL.git
​
Enviroment for reinforcement learning applied to Beam-selection
V1.0
'''
import numpy as np
import tensorflow as tf
from gym import Env
from gym.spaces import Box, MultiDiscrete
from communications.buffer import Buffer
from communications.base_station import BaseStation
from communications.ue import UE

from collections import deque
import csv


class BeamSelectionEnv(Env):
    def __init__(self, ep=[0]):
        # Which episode to take data from (Only used when use_airsim=False).
        self.eps = ep
        '''
        Defining simulation environment with one BS and three UEs
        '''
        self.ue1 = UE(name='uav1', obj_type='UAV', total_number_rbs=15, episode = self.eps, use_airsim=False)
        self.ue2 = UE(name='simulation_car1', obj_type='CAR', total_number_rbs=15, episode = self.eps, use_airsim=False)
        self.ue3 = UE(name='simulation_pedestrian1', obj_type='PED', total_number_rbs=15, episode = self.eps, use_airsim=False)
        self.caviar_bs = BaseStation(Elements=64, frequency=60e9,name='BS1',ep_lenght=20, traffic_type = 'dense', BS_type = 'UPA', change_type=True)

        #Append users
        self.caviar_bs.append(self.ue1)
        self.caviar_bs.append(self.ue2)
        self.caviar_bs.append(self.ue3)
        
        '''
        The observation space is composed by an array with 7 float numbers. 
        The first three represent the user position in XYZ, while the 
        remaining ones are respectively: dropped packages, sent packages, 
        buffered and bit rate.
        '''
        '''
        self.observation_space = Box(
            low=np.array([-5e2,-5e2,-5e2,0,0,0,0]), 
            high=np.array([5e2,5e2,5e2,1e3,1e3,2e4,1e9]),
            shape=(7,)
        )
        '''
        self.observation_space = Box(
            # low=np.array([-10, -10, -10, 0, 0, -10, -10, -10, 0, 0, -10, -10, -10, 0, 0, 0, 0, 0, 0, 0, 0]),
            # high=np.array([10, 10, 10, 1, 1, 10, 10, 10, 1, 1, 10, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1]),
            low=np.array([-10, -10, -10, -10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0, 0]),
            high=np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 1, 1, 1, 1, 1, 1]),
            
            # low=np.array([-10, -10, -10, 0, 0, 0, 0, 0, 0, 0]),
            # high=np.array([10, 10, 10, 1, 1, 1, 1, 1, 1, 1]),

            # low=np.array([-10, -10, -10, 0, 0, 0, 0]),
            # high=np.array([10, 10, 10, 1, 1, 1, 1]),
            shape=(15,)
        )
        '''
        The action space is composed by an array with two integers. The first one 
        represents the user that is currently being allocated and the second one, 
        the codebook index.
        '''
        self.action_space = MultiDiscrete([len(self.caviar_bs.UEs), self.caviar_bs._NTx])
        
        self.reset()

        ##############################################
        #for training 
        self.recent_obj_type = deque([0] * 10)
        self.last_uav = [0, 0, 0]
        self.last_car = [0, 0, 0]
        self.last_ped = [0, 0, 0]

        #for reward log
        self.cnt_steps = 0
        self.log_steps = []
        self.log_orig_reward = []
        self.log_weighted_reward = []
        self.log_ue = []
        self.log_ue_type = []
        self.log_bonus_reward = []
        self.log_bonus_reward2 = []
        self.log_beam_bonus_reward = []
        self.log_reward = []

        self.log_beam = []
        ##############################################



    def reset(self):
        # self._state = np.zeros(7)
        # self._state = np.zeros(10)
        self._state = np.zeros(15)
        return self._state
    
    '''
    The step function receives a user and the beam index to serve it. The user state 
    is updated at every step by checking the correspondent element inside the simulator.
     
    :param action: (array) is composed by the user ID and the codebook index
    '''
    def step(self, action):
        target, index = action
        bs_example_state, bs_example_reward, info, done = self.caviar_bs.step(target,index)
        #self.state = bs_example_state

        self.recent_obj_type.append(info["chosen_ue"])
        self.recent_obj_type.popleft()

        pos_x = round(float(info["pos_x"])/50, 3)
        pos_y = round(float(info["pos_y"])/50, 3)
        pos_z = round(float(info["pos_z"])/50, 3)
        bit_rate = round(float(info["bit_rate"]/1e+9), 3)


        if info["chosen_ue"]=="uav1":
            obj_type = [1,0,0]
            # last_index = self.last_uav[3]
            # self.last_uav = [pos_x, pos_y, pos_z, index/64, bit_rate]
            self.last_uav = [pos_x, pos_y, pos_z ]
        if info["chosen_ue"]=="simulation_car1":
            obj_type = [0,1,0] 
            # last_index = self.last_car[3]
            # self.last_car = [pos_x, pos_y, pos_z, index/64, bit_rate]
            self.last_car = [pos_x, pos_y, pos_z ]
        if info["chosen_ue"]=="simulation_pedestrian1":
            obj_type = [0,0,1]
            # last_index = self.last_ped[3]
            # self.last_ped = [pos_x, pos_y, pos_z, index/64, bit_rate]
            self.last_ped = [pos_x, pos_y, pos_z ]

        cnt_uav = self.recent_obj_type.count("uav1")
        cnt_car = self.recent_obj_type.count("simulation_car1")
        cnt_ped = self.recent_obj_type.count("simulation_pedestrian1")
        cnt_recent_obj = [cnt_uav/10, cnt_car/10, cnt_ped/10]

        self.state = np.concatenate((self.last_uav, self.last_car, self.last_ped, obj_type, cnt_recent_obj), axis=None)
        # self.state = np.concatenate((pos_x, pos_y, pos_z, bit_rsate, obj_type, cnt_recent_obj), axis=None)
        # self.state = np.concatenate((pos_x, pos_y, pos_z, bit_rate, obj_type), axis=None)


        # reward = bs_example_reward

        if info["chosen_ue"]=="uav1":
            weight = 5 / cnt_uav
        if info["chosen_ue"]=="simulation_car1":
            weight = 5 / cnt_car
        if info["chosen_ue"]=="simulation_pedestrian1":
            weight = 5 / cnt_ped
        weighted_reward = weight * (bs_example_reward + 2)

        bonus_reward = 0
        if bs_example_reward >= -0.1:
            # bonus_reward = 10 * bs_example_reward + 1
            bonus_reward = 20 * bs_example_reward + 2 
            # bonus_reward = 10 

        # bonus_reward2 = 0
        # if bs_example_reward < -0.1 and last_index != index:
        #     bonus_reward2 = 1

        beam_bonus = 5 * (1 - 1 / np.log10(info["pkts_transmitted"]/10 + 10))
        
        # reward = weighted_reward
        # reward = weighted_reward + bonus_reward + bonus_reward2
        # reward = (1 - 1 / np.log10(info["pkts_transmitted"]/10 + 10) )
        reward = weighted_reward + beam_bonus + bonus_reward

        ###########for logging############
        self.cnt_steps += 1
        if self.cnt_steps % 1 ==0:

            self.log_steps.append(self.cnt_steps)
            self.log_orig_reward.append(bs_example_reward)
            self.log_weighted_reward.append(weighted_reward)
            self.log_bonus_reward.append(bonus_reward)
            # self.log_bonus_reward2.append(bonus_reward2)
            self.log_beam_bonus_reward.append(beam_bonus)
            self.log_reward.append(reward)

            self.log_beam.append(index)
            self.log_ue.append(target)
            self.log_ue_type.append(info['chosen_ue'])
        
        return self.state, reward, done, info
    
    def best_beam_step(self, target):
        bs_example_state, bs_example_reward, info, done = self.caviar_bs.best_beam_step(target)
        self.state = bs_example_state
        reward = bs_example_reward
        return self.state, reward, done, info

    
    def save_rewards_log(self, name):
        output_file = './result/log_rewards-'+str(name)+'.csv'
        with open(output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['time_steps']+
                                ['orig_reward']+
                                ['weighted_reward']+
                                ['bonus_reward']+
                                # ['bonus_reward2'] +
                                ['beam_bonus_reward'] +
                                ['beam_index'] +
                                ['reward'] +
                                ['obj_type'] +
                                ['UE']
                                )
                for i in range(len(self.log_steps)):
                    writer.writerow([self.log_steps[i]]+
                                    [self.log_orig_reward[i]]+
                                    [self.log_weighted_reward[i]]+
                                    [self.log_bonus_reward[i]]+
                                    # [self.log_bonus_reward2[i]]+
                                    [self.log_beam_bonus_reward[i]] +
                                    [self.log_beam[i]]+
                                    [self.log_reward[i]] +
                                    [self.log_ue[i]] +
                                    [self.log_ue_type[i]]
                                    )
        