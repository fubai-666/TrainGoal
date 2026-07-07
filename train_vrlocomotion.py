import torch
import torchvision.transforms as T
from utils.image_utils import get_temp_img
from main_process import main_process
from tqdm.auto import tqdm

def train_pred_goal(model, train_loader, train_images, e, obs_len, pred_len, batch_size, params, gt_template, device, input_template, optimizer, criterion, dataset_name, homo_mat, mode):

	train_loss = 0
	count = 0
	model.train()
    
	img_size = int(params['img_size_r']*256)
    
	x = torch.linspace(0, img_size, img_size)
	y = torch.linspace(0, img_size, img_size)
	x, y = torch.meshgrid(x, y)
    
	img_template = torch.tensor(get_temp_img(0.2, img_size)).unsqueeze(0).type(torch.float32)
            
	# outer loop, for loop over each scene as scenes have different image size and to calculate segmentation only once
	progress = tqdm(train_loader, desc=f"Train epoch {e}", dynamic_ncols=True)
	for batch, (trajectory, trajectory2, scene, goal, time_traj, time_goal) in enumerate(progress):
		scene_image = []
		for s in scene:      
			if scene_image == []:
				scene_image = train_images[s]
			else:
				scene_image = torch.cat([scene_image,train_images[s]],dim=0)
        
		if params['img_size_r']!=1.0:
			transform = T.Resize(size = (img_size,img_size))
			scene_image = transform(scene_image).reshape(-1,1,img_size,img_size).to(device)
		else:
			scene_image = scene_image.reshape(-1,1,256,256).to(device)
        
		goal_obj = goal[:,0:1].to(device)
		goal_traj = goal[:,1:2].to(device)
		trajectory = trajectory[:,0,1,:,:2].to(device)
		trajectory2 = trajectory2[:,0,1,:,:2].to(device)
        
		pred, loss = main_process(trajectory, trajectory2, img_template, scene_image, goal_obj, goal_traj, params, x, y, device, img_size, model, criterion, mode)
        
        # Back propagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
            
		with torch.no_grad():
			train_loss += loss
			count += trajectory.shape[0]
                
	return train_loss.item()/count
