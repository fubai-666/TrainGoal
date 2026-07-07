import pandas as pd
import yaml
import os
from model_vrlocomotion import GoalNet

CONFIG_FILE_PATH = './config/vrlocomotion.yaml'  # yaml config file containing all the hyperparameters
DATA_DIR = './dataset/'

TRAIN_DATA_PATH = DATA_DIR + 'LocoVR/train'
TRAIN_IMAGE_PATH = DATA_DIR + 'map_vr/binary_map/'

VAL_DATA_PATH = DATA_DIR + 'LocoVR/val'
VAL_IMAGE_PATH = DATA_DIR + 'map_vr/binary_map/'

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1]
params

BATCH_SIZE = params['batch_size']
OBS_LEN = params['num_obs']     # in timesteps
PRED_LEN = params['num_pred']    # in timesteps

df_train = pd.read_pickle(TRAIN_DATA_PATH)
df_val = pd.read_pickle(VAL_DATA_PATH)

model = GoalNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)

if params['start_epoch']!=0:
    model_ckpt_path = os.path.join(
        params['model_dir'],
        'model_pred_goal_' + str(params['start_epoch']) + 'epoch.pt',
    )
    model.load_pred_goal(model_ckpt_path)

model.train(df_train, df_val, params, train_image_path=TRAIN_IMAGE_PATH, val_image_path=VAL_IMAGE_PATH, batch_size=BATCH_SIZE, device=None)
