import os
import cv2
import numpy as np
import copy
import torch
from model_transformer import PRED_GOAL_Transformer
from utils.image_utils import get_patch, get_patch2, sampling, image2world, get_mask, get_idx, create_gaussian_heatmap_template, conv_points2img, conv_heading2img, conv_heading2img2

class PRED:

    def __init__(self):
        self.obs_traj1 = []
        self.obs_traj2 = []
        self.goal_traj = []
        self.goal_obj = []
        self.scene_img = []
        self.goal_map_pred = []
        self.scene_id = -1
        self.id = -1
        self.time = []
        self.time_goal = -1

def spatial_cross_entropy(pred_logits, target_heatmap):
    pred_logits = pred_logits.flatten(1)
    target_heatmap = target_heatmap.flatten(1)
    target_prob = target_heatmap / target_heatmap.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return -(target_prob * torch.log_softmax(pred_logits, dim=1)).sum(dim=1).mean()

def viz_input(scene_image, observed_map, heading_map, goal_obj, goal_traj):
    
	save_dir = './viz_input_debug/'
                        
	if not os.path.exists(save_dir):
	    os.makedirs(save_dir)
    
	fname = os.listdir(save_dir)
        
	s_img = scene_image.detach().cpu().numpy()*255
	obs = observed_map.detach().cpu().numpy()*255
	head = heading_map.detach().cpu().numpy()*255
    
	for i in range(obs.shape[0]):
		for j in range(obs.shape[1]):
            
			img_out = cv2.UMat(np.swapaxes(np.swapaxes(np.concatenate([s_img[i], obs[i,j:j+1], head[i,j:j+1]], axis=0),0,2),0,1))
			cv2.circle(img_out, (int(goal_obj[i,0,0].detach().cpu().numpy()), int(goal_obj[i,0,1].detach().cpu().numpy())), 2, (0, 255, 255), thickness=-1)
			#cv2.circle(img_out, (int(goal_traj[i,0,0].detach().cpu().numpy()), int(goal_traj[i,0,1].detach().cpu().numpy())), 2, (0, 155, 155), thickness=-1)
			
			cv2.imwrite(save_dir + '{:0=6}'.format(len(fname)) + '_{:0=3}'.format(i) + '_{:0=3}'.format(j) +'.png', img_out)
        
def main_process(trajectory, trajectory2, img_template, scene_image, goal_obj, goal_traj, params, x, y, device, img_size, model, criterion, mode):

    pred = PRED()
    
    num_step = 3
    
    observed_map = conv_points2img([y.to(device),x.to(device)], traj=trajectory[:,::num_step,:]*params['img_size_r'], distribution=[0.5, 0.5], img_size=img_size)

    feature_input = torch.cat([scene_image, observed_map], dim=1).type(torch.float32)
    pred_map = model(feature_input)
    goal_map_gt1 = conv_points2img([y.to(device),x.to(device)], traj=goal_obj*params['img_size_r'], distribution=[0.5, 0.5], img_size=img_size)
    if isinstance(model, PRED_GOAL_Transformer):
        loss = spatial_cross_entropy(pred_map[:,0:1], goal_map_gt1)
    else:
        loss = criterion(pred_map[:,0:1], goal_map_gt1) * params['loss_scale']
    
    if mode=='test':
        pred.obs_traj1 = trajectory[:,::num_step,:].detach().cpu().numpy()
        pred.obs_traj2 = trajectory2[:,::num_step,:].detach().cpu().numpy()
        pred.scene_img = scene_image.detach().cpu().numpy()
        pred.goal_traj = goal_traj.detach().cpu().numpy()
        pred.goal_obj = goal_obj.detach().cpu().numpy()
        if isinstance(model, PRED_GOAL_Transformer):
            pred_map = torch.softmax(pred_map.flatten(2), dim=2).reshape_as(pred_map)
        else:
            pred_map = torch.sigmoid(pred_map)
        pred.goal_map_pred = pred_map.detach().cpu().numpy()
    
    return pred, loss
