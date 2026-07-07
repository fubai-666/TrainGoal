import pandas as pd
import yaml
from model_vrlocomotion import YNet
import torch
import os
import re

# Instructions:
# 1. update waypoints in .yaml file
# 2. update dataloader for entire traj or w/waypoints on traj in dataloader_vrlocomotion.py
# 3. update test set path in this cell
# 4. update OBS_LEN, PRED_LEN, NUM_GOALS in this cell
# 5. update model path in the cells below

def get_best_epoch(model_dir):
    fname_list = sorted(os.listdir(model_dir))
    
    best_epoch = -1
    
    for fname in fname_list:
        if 'best' in fname:
        
            if best_epoch < int(re.sub(r'\D','',fname)):
                best_epoch = int(re.sub(r'\D','',fname))
    
    return best_epoch

CONFIG_FILE_PATH = 'config/vrlocomotion.yaml'  # yaml config file containing all the hyperparameters
DATASET_NAME = 'vrlocomotion'

#DATA_DIR = '/home/takeyama/work/trajectory-prediction-dataset/vr_locomotion_dataset/processed_data/'
DATA_DIR = '../../dataset/'
#DATA_DIR = '../real_dataset_all/processed_data/'
TEST_DATA_PATH = DATA_DIR + 'Loco3D_R/test'
TEST_IMAGE_PATH = DATA_DIR + 'map_real/binary_map/'

model_dir = './vrlocomotion_models_000/'

NUM_GOALS = 1  # K_e, the number of predicted goals
NUM_TRAJ = 1 # 1,2,3  # K_a, the number of predicted trajs given a goal
ROUNDS = 1  # Y-net is stochastic. How often to evaluate the whole dataset
BATCH_SIZE = 1

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1]

OBS_LEN = params['num_obs']     # in timesteps
PRED_LEN = params['num_pred']    # in timesteps

df_test = pd.read_pickle(TEST_DATA_PATH)#[:3000]

model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)

#model.load_pred_gm(model_dir+'model_pred_gm_49epoch.pt')
model.load_pred_goal(model_dir+'model_pred_goal_6epoch.pt')
params['save_name'] = 'calc_result'
#model.load_pred_traj_pos(model_dir+'model_pred_traj_pos1_5epoch.pt')

model.evaluate(df_test, params, image_path=TEST_IMAGE_PATH,
               batch_size=BATCH_SIZE, rounds=ROUNDS, 
               device=None, dataset_name=DATASET_NAME, 
               plot_traj=False, plot_map=False)