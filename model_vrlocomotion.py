import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

from utils.preprocessing_vrlocomotion import augment_data, create_images_dict
from utils.image_utils import create_gaussian_heatmap_template, create_determistic_template, create_dist_mat, \
	preprocess_image_for_segmentation, pad, resize
from utils.dataloader_vrlocomotion import SceneDataset, scene_collate
from test_vrlocomotion import evaluate
from train_vrlocomotion import train_pred_goal


class Encoder(nn.Module):
	def __init__(self, in_channels, channels=(64, 128, 256, 512, 512)):
		"""
		Encoder model
		:param in_channels: int, semantic_classes + obs_len
		:param channels: list, hidden layer channels
		"""
		super(Encoder, self).__init__()
		self.stages = nn.ModuleList()

		# First block
		self.stages.append(nn.Sequential(
			nn.Conv2d(in_channels, channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True),
		))

		# Subsequent blocks, each starting with MaxPool
		for i in range(len(channels)-1):
			self.stages.append(nn.Sequential(
				nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
				nn.Conv2d(channels[i], channels[i+1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
				nn.ReLU(inplace=True),
				nn.Conv2d(channels[i+1], channels[i+1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
				nn.ReLU(inplace=True)))

		# Last MaxPool layer before passing the features into decoder
		self.stages.append(nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)))

	def forward(self, x):
		# Saves the feature maps Tensor of each layer into a list, as we will later need them again for the decoder
		features = []
		for stage in self.stages:
			x = stage(x)
			features.append(x)
		return features


class Decoder(nn.Module):
	def __init__(self, encoder_channels, decoder_channels, output_len, traj=False):

		super(Decoder, self).__init__()

		# The trajectory decoder takes in addition the conditioned goal and waypoints as an additional image channel
		if traj:
			encoder_channels = [channel+traj for channel in encoder_channels]
		encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder
		center_channels = encoder_channels[0]

		decoder_channels = decoder_channels

		# The center layer (the layer with the smallest feature map size)
		self.center = nn.Sequential(
			nn.Conv2d(center_channels, center_channels*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True),
			nn.Conv2d(center_channels*2, center_channels*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True)
		)

		# Determine the upsample channel dimensions
		upsample_channels_in = [center_channels*2] + decoder_channels[:-1]
		upsample_channels_out = [num_channel // 2 for num_channel in upsample_channels_in]

		# Upsampling consists of bilinear upsampling + 3x3 Conv, here the 3x3 Conv is defined
		self.upsample_conv = [
			nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			for in_channels_, out_channels_ in zip(upsample_channels_in, upsample_channels_out)]
		self.upsample_conv = nn.ModuleList(self.upsample_conv)

		# Determine the input and output channel dimensions of each layer in the decoder
		# As we concat the encoded feature and decoded features we have to sum both dims
		in_channels = [enc + dec for enc, dec in zip(encoder_channels, upsample_channels_out)]
		out_channels = decoder_channels

		self.decoder = [nn.Sequential(
			nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True))
			for in_channels_, out_channels_ in zip(in_channels, out_channels)]
		self.decoder = nn.ModuleList(self.decoder)


		# Final 1x1 Conv prediction to get our heatmap logits (before softmax)
		self.predictor = nn.Conv2d(in_channels=decoder_channels[-1], out_channels=output_len, kernel_size=1, stride=1, padding=0)

	def forward(self, features):
		# Takes in the list of feature maps from the encoder. Trajectory predictor in addition the goal and waypoint heatmaps
		features = features[::-1]  # reverse the order of encoded features, as the decoder starts from the smallest image
		center_feature = features[0]
		x = self.center(center_feature)
		t = x.dtype
		for i, (feature, module, upsample_conv) in enumerate(zip(features[1:], self.decoder, self.upsample_conv)):
			x = F.interpolate(x.float(), scale_factor=2, mode='bilinear', align_corners=False).to(t)  # bilinear interpolation for upsampling
			x = upsample_conv(x)  # 3x3 conv for upsampling
			x = torch.cat([x, feature], dim=1)  # concat encoder and decoder features
			x = module(x)  # Conv
		x = self.predictor(x)  # last predictor layer
		return x

class PRED_GOAL(nn.Module):
	def __init__(self, obs_len,	pred_len, map_channel, encoder_channels=[], decoder_channels=[]):

		super(PRED_GOAL, self).__init__()

        #goal=1 + past_traj=15
		self.encoder = Encoder(in_channels= map_channel + int((obs_len+pred_len)/3), channels=encoder_channels)
		self.decoder = Decoder(encoder_channels, decoder_channels, output_len=1)

	def dec(self, features):
		v = self.decoder(features)
		return v

	def enc(self, x):
		features = self.encoder(x)
		return features

	def forward(self,x):
        
		f = self.enc(x)
		v = self.dec(f)
        
		return v
        
class GoalNet:
	def __init__(self, obs_len, pred_len, params):

		self.obs_len = obs_len
		self.pred_len = pred_len
		self.division_factor = 2 ** len(params['encoder_channels'])
		self.bfloat16 = params['bfloat16']

		self.model = PRED_GOAL(obs_len=self.obs_len,
							   pred_len=self.pred_len,
							   map_channel=1,
							   encoder_channels=params['encoder_channels'],
							   decoder_channels=params['decoder_channels']
                               )
        
	def train(self, train_data, val_data, params, train_image_path, val_image_path, batch_size=8, device=None, dataset_name=None, test_scene=0):

		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		obs_len = self.obs_len
		pred_len = self.pred_len
		total_len = pred_len + obs_len

		print('Preprocess data')

		model_dir = params['model_dir']
		os.makedirs(model_dir, exist_ok=True)
			
		self.homo_mat = None
		seg_mask = True 
		normalize_map = True
		
		# Load train images and augment train data and images (by rotating and flipping)
		# df_train, train_images = augment_data(train_data, image_path=train_image_path, images={}, seg_mask=seg_mask, normalize_map=normalize_map)
		train_images = create_images_dict(train_data, image_path=train_image_path, seg_mask=seg_mask, normalize_map=normalize_map)

		# Load val scene images
		val_images = create_images_dict(val_data, image_path=val_image_path, seg_mask=seg_mask, normalize_map=normalize_map)

		# Initialize dataloaders
		# train_dataset = SceneDataset(df_train, resize=params['resize'], total_len=total_len, num_aug=8)
		train_dataset = SceneDataset(train_data, resize=params['resize'], total_len=total_len, num_aug=1)
		train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=scene_collate, shuffle=True)

		val_dataset = SceneDataset(val_data, resize=params['resize'], total_len=total_len, num_aug=1)
		val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=scene_collate)

		# Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
		resize(train_images, factor=params['resize'], seg_mask=seg_mask)
		pad(train_images, division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
		preprocess_image_for_segmentation(train_images, seg_mask=seg_mask)

		resize(val_images, factor=params['resize'], seg_mask=seg_mask)
		pad(val_images, division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
		preprocess_image_for_segmentation(val_images, seg_mask=seg_mask)

		model = self.model.to(device)

		# # Freeze segmentation model
		# for param in model.semantic_segmentation.parameters():
		# 	param.requires_grad = False

		optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
		criterion = nn.BCEWithLogitsLoss()

		# Create template
		size = int(4200 * params['resize'])

		if params['mode_train']==2:
			input_template = create_determistic_template(size=size)
			input_template = torch.Tensor(input_template).to(device)

			gt_template = create_determistic_template(size=size)
			gt_template = torch.Tensor(gt_template).to(device)
		else:
			input_template = create_dist_mat(size=size)
			input_template = torch.Tensor(input_template).to(device)

			gt_template = create_gaussian_heatmap_template(size=size, kernlen=params['kernlen'], nsig=params['nsig'], normalize=False)
			gt_template = torch.Tensor(gt_template).to(device)

		if self.bfloat16:
			input_template, gt_template = input_template.bfloat16(), gt_template.bfloat16()

		best_val_loss = 99999999999999

		loss_csv_path = os.path.join(model_dir, 'loss_train-val.csv')
		self.train_loss_mem = []
		self.val_loss_mem = []
		self.epoch_mem = []
		if os.path.exists(loss_csv_path):
			loss_hist = np.loadtxt(loss_csv_path, delimiter=',', skiprows=1)
			loss_hist = np.atleast_2d(loss_hist)
			if loss_hist.shape[1] != 3:
				raise ValueError(f"Expected 3 columns in {loss_csv_path}, got {loss_hist.shape[1]}")
			self.epoch_mem = loss_hist[:, 0].astype(int).tolist()
			self.train_loss_mem = loss_hist[:, 1].tolist()
			self.val_loss_mem = loss_hist[:, 2].tolist()
			if self.val_loss_mem:
				best_val_loss = min(self.val_loss_mem)

		print('Start training')
		start_global_epoch = 0 if params['start_epoch'] == 0 else params['start_epoch'] + 1
		epoch_progress = tqdm(range(start_global_epoch, params['num_epochs']), desc='Epoch', dynamic_ncols=True)
		for epoch_id in epoch_progress:
			epoch_progress.set_description(f'Epoch {epoch_id}')
            
			train_loss = train_pred_goal(model, train_loader, train_images, epoch_id, obs_len, pred_len,
									 batch_size, params, gt_template, device,
									 input_template, optimizer, criterion, dataset_name, self.homo_mat, mode='train')
			print(f'Train loss: {train_loss}')

			# For faster inference, we don't use TTST and CWS here, only for the test set evaluation
			val_loss1, val_loss2, val_loss3 = evaluate(model, val_loader, val_images, pred_len=pred_len,
										obs_len=obs_len, batch_size=batch_size,
										device=device, gt_template=gt_template,
										waypoints=params['waypoints'], resize=params['resize'],
										temperature=params['temperature'], normalize_map=normalize_map, 
										use_TTST=False, use_CWS=False, dataset_name=dataset_name,
										homo_mat=self.homo_mat, mode='val', 
										plot_traj=False, plot_map=False, epoch=epoch_id, params=params)
            
			val_loss = val_loss2
						
			print(f'Val loss: {val_loss}')

			# save the model weights with the lowest val ADE
			if val_loss < best_val_loss:
				print(f'Best Epoch {epoch_id}: \nVal loss: {val_loss}')
				best_val_loss = val_loss
			
			torch.save(
				{
					'epoch': epoch_id,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
				},
				os.path.join(model_dir, 'model_pred_goal_{}epoch.pt'.format(epoch_id)),
			)
				
			self.epoch_mem.append(epoch_id)
			self.train_loss_mem.append(train_loss)
			self.val_loss_mem.append(val_loss)
			np.savetxt(
				loss_csv_path,
				np.array([self.epoch_mem, self.train_loss_mem, self.val_loss_mem]).transpose(),
				delimiter=',',
				header='epoch,train_loss,val_loss',
				comments='',
			)


	def evaluate(self, data, params, image_path, batch_size=8, 
	      rounds=1, device=None, dataset_name=None, plot_traj=False, plot_map=False):

		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		obs_len = self.obs_len
		pred_len = self.pred_len
		total_len = pred_len + obs_len
		# total_len = 6000

		print('Preprocess data')

		self.homo_mat = None
		seg_mask = True
		normalize_map = True

		test_images = create_images_dict(data, image_path=image_path, seg_mask=seg_mask, normalize_map=normalize_map)

		test_dataset = SceneDataset(data, resize=params['resize'], total_len=total_len, num_aug=1, flag_test=1)
		test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=scene_collate)

		# Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
		resize(test_images, factor=params['resize'], seg_mask=seg_mask)
		pad(test_images, division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet architecture
		preprocess_image_for_segmentation(test_images, seg_mask=seg_mask)
		# test_images is a dict containing images

		model = self.model.to(device)
		if self.bfloat16:
			model = model.bfloat16()

		# Create template
		size = int(4200 * params['resize'])

		if params['mode_train']==2:
			gt_template = create_determistic_template(size=size)
			gt_template = torch.Tensor(gt_template).to(device)
		else:
			gt_template = create_gaussian_heatmap_template(size=size, kernlen=params['kernlen'], nsig=params['nsig'], normalize=False)
			gt_template = torch.Tensor(gt_template).to(device)

		print('Start testing')
		for e in tqdm(range(rounds), desc='Round'):
			val_loss = evaluate(model, test_loader, test_images, pred_len=pred_len,
										  obs_len=obs_len, batch_size=batch_size,
										  device=device, gt_template=gt_template,
										  waypoints=params['waypoints'], resize=params['resize'],
										  temperature=params['temperature'], normalize_map=normalize_map,
										  use_TTST=params['use_TTST'], rel_thresh=params['rel_threshold'],
										#   use_CWS=False,
										#   use_CWS=True if len(params['waypoints']) > 1 else False,
										  use_CWS=params['use_CWS'], CWS_params=params['CWS_params'],
										  dataset_name=dataset_name, homo_mat=self.homo_mat, mode='test', 
										  plot_traj=plot_traj, plot_map=plot_map, epoch=e, params=params)
			
		print(val_loss)

	def load(self, path):
		print(self.model.load_state_dict(torch.load(path)))
    
	def load_pred_goal(self, path, flag_freeze=0):
		checkpoint = torch.load(path)        
		print(self.model.load_state_dict(checkpoint['model_state_dict']))
		if flag_freeze==1:
		    for param in self.model.parameters():
    		 	param.requires_grad = False

	def save(self, path):
		torch.save(self.model.state_dict(), path)
