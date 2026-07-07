import numpy as np
import torch
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
import math
import copy

def rotate_img(img, angle):
    
    img_out = torch.zeros(angle.shape[0],angle.shape[1],1,img.shape[1],img.shape[2])
    
    for i in range(angle.shape[0]):
        
        rotation_matrix = torch.zeros(angle.shape[1], 2, 3)
        rotation_matrix[:, 0, 0] = torch.cos(angle[i])
        rotation_matrix[:, 1, 1] = rotation_matrix[:, 0, 0]
        rotation_matrix[:, 0, 1] = -torch.sin(angle[i])  # +/- sin(angle)
        rotation_matrix[:, 1, 0] = -rotation_matrix[:, 0, 1]

        rotation_grids = torch.nn.functional.affine_grid(rotation_matrix, (angle.shape[1], 1, img.shape[1], img.shape[2]))
        img_out[i] = torch.nn.functional.grid_sample(img.repeat(angle.shape[1],1,1,1), rotation_grids)
    
    return torch.swapaxes(img_out,3,4)

def gaussian_distribution(x, mu, sigma):
    # Compute the prefactor 1/(sqrt(2*pi*sigma^2))
    prefactor = 1 / (math.sqrt(2 * math.pi * sigma**2))
    # Compute the exponent factor of the Gaussian formula
    exponent = math.exp(-((x - mu) ** 2) / (2 * sigma**2))
    
    return prefactor * exponent

def get_temp_img(distribution, img_size):
    
    img_out = np.zeros([img_size*2, img_size*2])
    
    for x in range(-img_size, img_size):
        for y in range(img_size):
            
            theta = np.arctan2(y,x)
            img_out[img_size+y,img_size+x] = gaussian_distribution(theta, torch.pi/2, distribution)
    
    return img_out
    
    
def conv_heading2img(head, heading, distribution, img_size):
    
    gaze = head.detach().cpu().numpy().astype(np.int16) + np.swapaxes(np.swapaxes(np.array([np.cos(heading.detach().cpu().numpy())*1000,np.sin(heading.detach().cpu().numpy())*1000]).astype(np.int16), 0, 2), 0, 1)
    
    img_out = []
    
    for i in range(gaze.shape[0]):
        img2 = []
        for j in range(gaze.shape[1]):
            img1 = np.zeros([img_size, img_size, 1])
            img1 = cv2.line(img1, tuple(head[i,j].detach().cpu().numpy().astype(np.int16)), tuple(gaze[i,j]), (1, 1, 1), thickness=3, lineType=cv2.LINE_AA)
            img2.append(img1.reshape(img_size,img_size))
        img_out.append(img2)
    
    return torch.tensor(np.array(img_out))

def conv_heading2img2(pos, heading, template, img_size):
    
    img_r = rotate_img(template, heading)
    
    img_out = torch.zeros(pos.shape[0],pos.shape[1],img_size,img_size)
    
    for i in range(pos.shape[0]):
        for j in range(pos.shape[1]):
            
            x = img_size - int(pos[i,j,0])
            y = img_size - int(pos[i,j,1])
            
            if x<0:
                x = 0
            if y<0:
                y = 0
                
            img_out[i,j] = img_r[i,j,0,y:y+img_size,x:x+img_size]
            img_out[i,j] = img_out[i,j]/torch.max(img_out[i,j])
    
    return img_out

def conv_points2img(grid, traj, distribution, img_size):
    
    return 1 / (2*math.pi*distribution[0]*distribution[1])*torch.exp(-((grid[0].repeat(traj.shape[0],traj.shape[1],1,1) - torch.swapaxes(torch.swapaxes(traj[:,:,0].repeat(img_size,img_size,1,1),0,2),1,3))**2 / (2*distribution[0]**2) + (grid[1].repeat(traj.shape[0],traj.shape[1],1,1) - torch.swapaxes(torch.swapaxes(traj[:,:,1].repeat(img_size,img_size,1,1),0,2),1,3))**2 / (2*distribution[1]**2)))

def get_idx(gt):

    a = (torch.norm(gt[:,1:]-gt[:,:-1], dim=2)!=0)
    
    if torch.where(a[0]==0)[0].shape[0]==0:
        idx = torch.tensor(gt.shape[1])
    else:
        idx = torch.min(torch.where(a[0]==0)[0])
        
    return idx

def get_mask(gt, map_shape, device):

    idx_msk = (torch.norm(gt[:,1:]-gt[:,:-1], dim=2)!=0).to(device)
    mask = torch.ones(map_shape).to(device)
    
    for i in range(gt.shape[0]):
        for j in range(1,gt.shape[1]):
            mask[i,j,:] = mask[i,j,:]*idx_msk[i,j-1]
    
    return mask
    
def gkern(kernlen=31, nsig=4):
	"""	creates gaussian kernel with side length l and a sigma of sig """
	ax = np.linspace(-(kernlen - 1) / 2., (kernlen - 1) / 2., kernlen)
	xx, yy = np.meshgrid(ax, ax)
	kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(nsig))
	return kernel / np.sum(kernel)


def create_gaussian_heatmap_template(size, kernlen=81, nsig=4, normalize=True):
	""" Create a big gaussian heatmap template to later get patches out """
	template = np.zeros([size, size])
	kernel = gkern(kernlen=kernlen, nsig=nsig)
	m = kernel.shape[0]
	x_low = template.shape[1] // 2 - int(np.floor(m / 2))
	x_up = template.shape[1] // 2 + int(np.ceil(m / 2))
	y_low = template.shape[0] // 2 - int(np.floor(m / 2))
	y_up = template.shape[0] // 2 + int(np.ceil(m / 2))
	template[y_low:y_up, x_low:x_up] = kernel
	if normalize:
		template = template / template.max()
	return template


def create_dist_mat(size, normalize=True):
	""" Create a big distance matrix template to later get patches out """
	middle = size // 2
	dist_mat = np.linalg.norm(np.indices([size, size]) - np.array([middle, middle])[:,None,None], axis=0)
	if normalize:
		dist_mat = dist_mat / dist_mat.max() * 2
	return dist_mat

def create_determistic_template(size):

	template = np.zeros([size, size])
	x_low = template.shape[1] // 2 - 1
	x_up = template.shape[1] // 2 + 1
	y_low = template.shape[0] // 2 - 1
	y_up = template.shape[0] // 2 + 1
	template[y_low:y_up, x_low:x_up] = 1
	template = template / template.max()
	
	return template

def get_patch(template, traj, H, W):
	x = np.round(traj[:,0]).astype('int')
	y = np.round(traj[:,1]).astype('int')

	x_low = template.shape[1] // 2 - x
	x_up = template.shape[1] // 2 + W - x
	y_low = template.shape[0] // 2 - y
	y_up = template.shape[0] // 2 + H - y

	patch = [template[y_l:y_u, x_l:x_u] for x_l, x_u, y_l, y_u in zip(x_low, x_up, y_low, y_up)]

	return patch

def get_patch2(template, traj, H, W):
	x = np.round(traj[:,0]).astype('int')
	y = np.round(traj[:,1]).astype('int')

	n_t = template.shape[0]
	img = copy.deepcopy(template)
	
	for i in range(n_t):
		img[i,y[i],x[i]] = 1

	return img

def preprocess_image_for_segmentation(images, encoder='resnet101', encoder_weights='imagenet', seg_mask=False, classes=6):
	""" Preprocess image for pretrained semantic segmentation, input is dictionary containing images
	In case input is segmentation map, then it will create one-hot-encoding from discrete values"""
	
	# import segmentation_models_pytorch as smp

	# preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

	for key, im in images.items():
		# if seg_mask:
		# 	im = [(im == v) for v in range(classes)]
		# 	im = np.stack(im, axis=-1)  # .astype('int16')
		# else:
		# 	im = preprocessing_fn(im)
		
		im = im.transpose(2, 0, 1).astype('float32')
		im = torch.Tensor(im)
		images[key] = im


def resize(images, factor, seg_mask=False):
	for key, image in images.items():
		if seg_mask:
			images[key] = np.atleast_3d(cv2.resize(image, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST))
		else:
			images[key] = np.atleast_3d(cv2.resize(image, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_AREA))


def pad(images, division_factor=32):
	""" Pad image so that it can be divided by division_factor, as many architectures such as UNet needs a specific size
	at it's bottlenet layer"""
	for key, im in images.items():
		if im.ndim == 3:
			H, W, C = im.shape
		else:
			H, W = im.shape
		H_new = int(np.ceil(H / division_factor) * division_factor)
		W_new = int(np.ceil(W / division_factor) * division_factor)
		im = cv2.copyMakeBorder(im, 0, H_new - H, 0, W_new - W, cv2.BORDER_CONSTANT)
		images[key] = np.atleast_3d(im)


def sampling(probability_map, num_samples, rel_threshold=None, replacement=False):
	# new view that has shape=[batch*timestep, H*W]
	prob_map = probability_map.reshape(probability_map.size(0) * probability_map.size(1), -1)
	if rel_threshold is not None:
		thresh_values = prob_map.max(dim=1)[0].unsqueeze(1).expand(-1, prob_map.size(1))
		mask = prob_map < thresh_values * rel_threshold
		prob_map = prob_map * (~mask).int()
		prob_map = prob_map / prob_map.sum()

	# samples.shape=[batch*timestep, num_samples]
	samples = torch.multinomial(prob_map, num_samples=num_samples, replacement=replacement)
	# samples.shape=[batch, timestep, num_samples]

	# unravel sampled idx into coordinates of shape [batch, time, sample, 2]
	samples = samples.reshape(probability_map.size(0), probability_map.size(1), -1)
	idx = samples.unsqueeze(3)
	preds = idx.repeat(1, 1, 1, 2).float()
	preds[:, :, :, 0] = (preds[:, :, :, 0]) % probability_map.size(3)
	preds[:, :, :, 1] = torch.floor((preds[:, :, :, 1]) / probability_map.size(3))

	return preds


def image2world(image_coords, scene, homo_mat, resize):
	"""
	Transform trajectories of one scene from image_coordinates to world_coordinates
	:param image_coords: torch.Tensor, shape=[num_person, (optional: num_samples), timesteps, xy]
	:param scene: string indicating current scene, options=['eth', 'hotel', 'student01', 'student03', 'zara1', 'zara2']
	:param homo_mat: dict, key is scene, value is torch.Tensor containing homography matrix (data/eth_ucy/scene_name.H)
	:param resize: float, resize factor
	:return: trajectories in world_coordinates
	"""
	traj_image2world = image_coords.clone()
	if traj_image2world.dim() == 4:
		traj_image2world = traj_image2world.reshape(-1, image_coords.shape[2], 2)
	if scene in ['eth', 'hotel']:
		# eth and hotel have different coordinate system than ucy data
		traj_image2world[:, :, [0, 1]] = traj_image2world[:, :, [1, 0]]
	traj_image2world = traj_image2world / resize
	traj_image2world = F.pad(input=traj_image2world, pad=(0, 1, 0, 0), mode='constant', value=1)
	traj_image2world = traj_image2world.reshape(-1, 3)
	traj_image2world = torch.matmul(homo_mat[scene], traj_image2world.T).T
	traj_image2world = traj_image2world / traj_image2world[:, 2:]
	traj_image2world = traj_image2world[:, :2]
	traj_image2world = traj_image2world.view_as(image_coords)
	return traj_image2world
