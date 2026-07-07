import torch
import torch.nn as nn
import torchvision.transforms as T
from utils.image_utils import get_temp_img
from main_process import main_process

import numpy as np


output_idx = 0

VAL_ADE = 0
VAL_ADE_count = 0
resize = 0.25
mem_ade = []

def get_traj_aug(traj1, traj2, heading, time, step):
    
    len_data = traj1.shape[1]
    traj_ext1 = torch.cat([traj1[0,0,:].repeat(1,len_data,1),traj1],dim=1)
    traj_ext2 = torch.cat([traj2[0,0,:].repeat(1,len_data,1),traj2],dim=1)
    time_ext = torch.cat([time[0,0].repeat(1,len_data),time],dim=1)
    heading_ext = torch.cat([heading[0,0].repeat(1,len_data),heading],dim=1)
    v = np.array(torch.sqrt(torch.sum((traj1[:,1:,:]-traj1[:,:-1,:])**2,dim=2)).cpu())
	
    traj1_out = []
    traj2_out = []
    heading_out = []
    time_out = []
    dist2goal = []
    
    for i in range(v.shape[0]):
        len_traj = np.max(np.where(v[i]>0.3))
        
        for j in range(np.ceil(len_traj/step).astype(np.int32)+2):
            if j*step>=len_data:
                break
            traj1_out.append(traj_ext1[i:i+1,j*step:len_data+j*step,:])
            traj2_out.append(traj_ext2[i:i+1,j*step:len_data+j*step,:])
            heading_out.append(heading_ext[i:i+1,j*step:len_data+j*step])
            time_out.append(time_ext[i:i+1,j*step:len_data+j*step])
            dist2goal.append(np.sum(v[0,j*step:]))
    
    return traj1_out, traj2_out, heading_out, time_out, dist2goal	
        
def evaluate(model, val_loader, val_images, pred_len, obs_len, batch_size, device, 
				gt_template, waypoints, resize, temperature, normalize_map, 
				use_TTST=False, use_CWS=False, rel_thresh=0.002, CWS_params=None, 
				dataset_name=None, homo_mat=None, mode='val', 
				plot_traj=False, plot_map=False, epoch=0, params=[]):

	model.eval()

	criterion = nn.BCEWithLogitsLoss()
	val_loss_gm1 = torch.tensor(0.0).float().to(device)
	val_loss_traj2 = torch.tensor(0.0).float().to(device)
	val_loss_gt = torch.tensor(0).float().to(device)
	val_loss_gm = torch.tensor(0).float().to(device)
	val_loss_map = torch.tensor(0).float().to(device)
	val_loss_overlap = torch.tensor(0).float().to(device)
	val_loss_smooth = torch.tensor(0).float().to(device)
	val_loss = torch.tensor(0).float().to(device)
	count = 0
    
	with torch.no_grad():
		# outer loop, for loop over each scene as scenes have different image size and to calculate segmentation only once

		img_size = int(params['img_size_r']*256)
        
		x = torch.linspace(1, img_size, img_size)-0.5
		y = torch.linspace(1, img_size, img_size)-0.5
		x, y = torch.meshgrid(x, y)
    
		img_template = torch.tensor(get_temp_img(0.2, img_size)).unsqueeze(0).type(torch.float32)
    
		for trajectory, trajectory2, scene, goal, time_traj, time_goal in val_loader:
            
			if mode=='test' and int(scene[0])>3:
				break
            
			scene_image = []
			for s in scene:      
				if scene_image == []:
					scene_image = val_images[s]
				else:
					scene_image = torch.cat([scene_image,val_images[s]],dim=0)
        
			if params['img_size_r']!=1.0:
				transform = T.Resize(size = (img_size,img_size))
				scene_image = transform(scene_image).reshape(-1,1,img_size,img_size).to(device)
			else:
				scene_image = scene_image.reshape(-1,1,256,256).to(device)
        
			goal_obj = goal[:,0:1].to(device)
			goal_traj = goal[:,1:2].to(device)
			heading = trajectory[:,1,0,:,2].to(device)
			trajectory = trajectory[:,0,1,:,:2].to(device)
			trajectory2 = trajectory2[:,0,1,:,:2].to(device)
            
			if mode == 'test':
				traj1, traj2, hdg, time, d = get_traj_aug(trajectory, trajectory2, heading, time_traj, step=5)
            
				for idx in range(len(traj1)):
                    
					traj1_ = traj1[idx]
					traj2_ = traj2[idx]

					pred, loss = main_process(traj1_, traj2_, img_template, scene_image, goal_obj, goal_traj, params, x, y, device, img_size, model, criterion, mode)
					#disp_pred_goal(scene_image, trajectory*params['img_size_r'], pred.goal_map_pred[:,0:1], goal_obj*params['img_size_r'], count, idx*5)
					pred.scene_id = int(scene[0])
					pred.time = time[idx].detach().cpu().numpy()[0]
					pred.time_goal = time_goal.detach().cpu().numpy()
					pred.dist2goal = (d[idx] + torch.sqrt(torch.sum((trajectory[0,-1,:]-goal_obj[0,0])**2)).detach().cpu().numpy())*params['img_size_r']/img_size*10

				count += traj1_.shape[0]
				val_loss_traj2 += loss
                
			else:

				pred, loss = main_process(trajectory, trajectory2, img_template, scene_image, goal_obj, goal_traj, params, x, y, device, img_size, model, criterion, mode)
        	    
				count += trajectory.shape[0]
				val_loss_traj2 += loss
                    
		return torch.tensor(val_loss_gm1).item()/count, torch.tensor(val_loss_traj2).item()/count, [val_loss.item()/count, val_loss_gt.item()/count, val_loss_gm.item()/count, val_loss_map.item()/count, val_loss_overlap.item()/count, val_loss_smooth.item()/count]
