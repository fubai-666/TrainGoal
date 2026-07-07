from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch
import copy


class SceneDataset(Dataset):
	def __init__(self, data, resize, total_len, num_aug, flag_test=0):
		""" Dataset that contains the trajectories of one scene as one element in the list. It doesn't contain the
		images to save memory.
		:params data (pd.DataFrame): Contains all trajectories
		:params resize (float): image resize factor, to also resize the trajectories to fit image scale
		:params total_len (int): total time steps, i.e. obs_len + pred_len
		"""

		self.trajectories, self.trajectories2, self.scene_list, self.goal_list, self.time_traj, self.time_goal = self.split_trajectories_by_scene(data, total_len, num_aug, flag_test)
		self.trajectories[:,0,:,:,:2] = self.trajectories[:,0,:,:,:2] * resize
		self.trajectories2[:,0,:,:,:2] = self.trajectories2[:,0,:,:,:2] * resize
		self.goal_list = self.goal_list * resize

	def __len__(self):
		return len(self.trajectories)

	def __getitem__(self, idx):
		trajectory = self.trajectories[idx]
		trajectory2 = self.trajectories2[idx]
		#meta = self.meta[idx]
		scene = self.scene_list[idx]
		goal = self.goal_list[idx]
		time_traj = self.time_traj[idx]
		time_goal = self.time_goal[idx]
        
		return trajectory, trajectory2, scene, goal, time_traj, time_goal

	def split_trajectories_by_scene(self, data, total_len, num_aug, flag_test):

		trajectories1 = []
		trajectories2 = []    

		scene_list = []
		goal_obj = []
		goal_traj = []
		time_traj = []
		time_goal = []
        
		for d in data:
			for p in ['p1','p2']:
				org_len = d[p]['pos'].shape[1]

				array_pos = np.ones([d[p]['pos'].shape[0],total_len,d[p]['pos'].shape[2]])*d[p]['pos'][:,-1:,:]
				array_pose = np.ones([d[p]['pose'].shape[0],total_len,d[p]['pose'].shape[2]])*d[p]['pose'][:,-1:,:]
				array_time = np.ones(total_len)*d['time'][-1]
                
				if total_len > org_len:
					array_pos[:,:org_len,:] = d[p]['pos']
					array_pose[:,:org_len,:] = d[p]['pose']
					array_time[:org_len] = d['time']
				else:
					array_pos = d[p]['pos'][:,:total_len,:]
					array_pose = d[p]['pose'][:,:total_len,:]      
					array_time = d['time'][:total_len]
                
				if p=='p1':
					trajectories1.append([array_pos,array_pose])                    
				if p=='p2':
					trajectories2.append([array_pos,array_pose])
			
			time_traj.append(array_time)
			scene_list.append(d['scene_id'])
			goal_obj.append(d['goal'])
			goal_traj.append(d['p1']['pos'][1,-1,:2])
			time_goal.append(d['time'][-1])
            
		if flag_test==0:
			trajectories1, trajectories2, scene_list, goal_obj, goal_traj = aug_traj_len(trajectories1, trajectories2, scene_list, goal_obj, goal_traj, len_data=total_len)
			time_traj = np.zeros(len(scene_list))
			time_goal = np.zeros(len(scene_list))
            
		n_traj = len(trajectories1)
		n_con = 2 #number of contents (=2: pos and pose)
		n_body = 2    #number of observed body parts (ex. n_body=2 if head and waist observations are included)
		n_dim = trajectories1[0][0].shape[2]
        
		trajectories1 = np.array(trajectories1).reshape(n_traj,n_con,n_body,total_len,n_dim)
		trajectories2 = np.array(trajectories2).reshape(n_traj,n_con,n_body,total_len,n_dim)
		scene_list = np.array(scene_list).reshape(-1)
		goal_obj = np.array(goal_obj)[:,:2].reshape(-1,1,2)
		goal_traj = np.array(goal_traj)[:,:2].reshape(-1,1,2)
        
		return trajectories1, trajectories2, scene_list, np.concatenate([goal_obj, goal_traj],axis=1), time_traj, time_goal

def scene_collate(batch):
	trajectories = []
	trajectories2 = []
	#meta = []
	scene = []
	goal = []
	time_traj = []
	time_goal = []
    
	for _batch in batch:
		trajectories.append(_batch[0])
		trajectories2.append(_batch[1])
		#meta.append(_batch[2])
		scene.append(_batch[2])
		goal.append(_batch[3])
		time_traj.append(_batch[4])
		time_goal.append(_batch[5])
	
	return torch.Tensor(np.array(trajectories)), torch.Tensor(np.array(trajectories2)), scene, torch.Tensor(np.array(goal)), torch.Tensor(np.array(time_traj)), torch.Tensor(np.array(time_goal))

def aug_traj_len(traj1,traj2,scene_list,goal_obj,goal_traj,len_data):
    
    #traj_ext1 = np.concatenate([np.array(traj1),np.repeat(np.array(traj1)[:,:,-1:,:],len_data,axis=2)],axis=2)
    #traj_ext2 = np.concatenate([np.array(traj1),np.repeat(np.array(traj1)[:,:,-1:,:],len_data,axis=2)],axis=2)
    traj_ext1 = np.concatenate([np.repeat(np.array(traj1)[:,:,:,:1,:],len_data,axis=3),np.array(traj1),np.repeat(np.array(traj1)[:,:,:,-1:,:],len_data,axis=3)],axis=3)
    traj_ext2 = np.concatenate([np.repeat(np.array(traj2)[:,:,:,:1,:],len_data,axis=3),np.array(traj2),np.repeat(np.array(traj2)[:,:,:,-1:,:],len_data,axis=3)],axis=3)
    v = np.sqrt(np.sum((np.array(traj1)[:,0,1,1:,:]-np.array(traj1)[:,0,1,:-1,:])**2,axis=2))
	
    traj1_out = []
    traj2_out = []
    scene_list_out = []
    goal_obj_out = []
    goal_traj_out = []

    step = 10
    
    for i in range(v.shape[0]):
        len_traj = np.max(np.where(v[i]>0.3))
        for j in range(np.floor(len_traj/step).astype(np.int32)):
            #if j>5:
            #    continue
            #traj1_out.append(traj_ext1[i,:,j*10:j*10+len_data,:])
            #traj2_out.append(traj_ext2[i,:,j*10:j*10+len_data,:])
            traj1_out.append(traj_ext1[i,:,:,len_traj-j*step:len_data+len_traj-j*step,:])
            traj2_out.append(traj_ext2[i,:,:,len_traj-j*step:len_data+len_traj-j*step,:])
            scene_list_out.append(scene_list[i])
            goal_obj_out.append(goal_obj[i])
            goal_traj_out.append(goal_traj[i])
            
            traj1_out.append(traj_ext1[i,:,:,len_data+len_traj-j*step:2*len_data+len_traj-j*step,:])
            traj2_out.append(traj_ext2[i,:,:,len_data+len_traj-j*step:2*len_data+len_traj-j*step,:])
            scene_list_out.append(scene_list[i])
            goal_obj_out.append(goal_obj[i])
            goal_traj_out.append(goal_traj[i])
    
    return traj1_out, traj2_out, scene_list_out, goal_obj_out, goal_traj_out 