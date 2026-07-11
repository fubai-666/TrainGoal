import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import csv
import importlib
import runpy
import shutil
import sys
from pathlib import Path

MAIN_ROOT = Path(__file__).resolve().parent.parent
if str(MAIN_ROOT) not in sys.path:
	sys.path.insert(0, str(MAIN_ROOT))

from utils.preprocessing_vrlocomotion import augment_data, create_images_dict
from utils.image_utils import create_gaussian_heatmap_template, create_determistic_template, create_dist_mat, \
	preprocess_image_for_segmentation, pad, resize
from utils.dataloader_vrlocomotion import SceneDataset, scene_collate
from test_vrlocomotion import evaluate
from train_vrlocomotion import train_pred_goal
from model_transformer import PRED_GOAL_Transformer


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
		self.main_root = Path(__file__).resolve().parent.parent
		self.best_traj_score = -float("inf")
		self.best_traj_model_path = None

		# self.model = PRED_GOAL(obs_len=self.obs_len,
		# 					   pred_len=self.pred_len,
		# 					   map_channel=1,
		# 					   encoder_channels=params['encoder_channels'],
		# 					   decoder_channels=params['decoder_channels']
        #                        )
		self.model = PRED_GOAL_Transformer(obs_len=self.obs_len,
									pred_len=self.pred_len,
									map_channel=1,
									decoder_channels=[512, 256, 128],
									embed_dim=512,
									patch_size=8,
									num_layers=6,
									num_heads=8,
								)
	def _get_main_setting(self):
		if str(self.main_root) not in sys.path:
			sys.path.insert(0, str(self.main_root))
		return importlib.import_module("setting")

	def _run_main_script(self, script_name, argv):
		old_argv = sys.argv[:]
		old_cwd = os.getcwd()
		old_sys_path = sys.path[:]
		modules_to_reset = [
			"model_vrlocomotion",
			"utility",
			"path_generator",
			"prompt_manager",
			"eval_traj",
		]
		module_backup = {name: sys.modules.get(name) for name in modules_to_reset}
		try:
			sys.path = [str(self.main_root)] + [p for p in sys.path if p != str(self.main_root)]
			sys.argv = argv
			os.chdir(self.main_root)
			for name in modules_to_reset:
				sys.modules.pop(name, None)
			runpy.run_path(str(self.main_root / script_name), run_name="__main__")
		finally:
			for name in modules_to_reset:
				sys.modules.pop(name, None)
			for name, module in module_backup.items():
				if module is not None:
					sys.modules[name] = module
			sys.path = old_sys_path
			sys.argv = old_argv
			os.chdir(old_cwd)

	def _read_traj_eval_metrics(self, csv_path):
		with open(csv_path, "r", encoding="utf-8") as f:
			rows = list(csv.reader(f))
		if len(rows) < 2:
			raise RuntimeError(f"Invalid traj eval csv: {csv_path}")
		header = rows[0]
		values = [float(v) for v in rows[1]]
		return header, values

	def _save_best_checkpoint(self, checkpoint_path, eval_csv_path, epoch_id, model_dir):
		best_dir = os.path.join(model_dir, "best")
		os.makedirs(best_dir, exist_ok=True)
		for fname in os.listdir(best_dir):
			if fname.endswith(".pt") or fname.endswith(".csv"):
				os.remove(os.path.join(best_dir, fname))
		best_name = f"model_pred_goal_best_epoch{epoch_id}.pt"
		shutil.copy2(checkpoint_path, os.path.join(best_dir, best_name))
		shutil.copy2(eval_csv_path, os.path.join(best_dir, f"eval_traj_best_epoch{epoch_id}.csv"))

	def eval_traj_checkpoint(self, checkpoint_path, epoch_id, model_dir):
		st_main = self._get_main_setting()
		run_name = os.path.basename(os.path.normpath(model_dir))
		path_output = f"{run_name}_epoch{epoch_id}"
		batch_eval_dir = self.main_root / "Eval-traj" / run_name
		batch_eval_dir.mkdir(parents=True, exist_ok=True)
		result_dir = self.main_root / "Result" / path_output
		eval_dir = self.main_root / "Eval-traj" / path_output

		old_goal_model_path = st_main.goal_model_path
		old_path_output = st_main.path_output
		old_scene_id = st_main.scene_id
		old_act_id = st_main.act_id

		try:
			st_main.goal_model_path = checkpoint_path
			st_main.path_output = path_output
			st_main.scene_id = [444]
			st_main.act_id = [0]

			self._run_main_script("TR-LLM.py", ["TR-LLM.py", "", "1", "0", "0", "0"])
			self._run_main_script("eval_traj.py", ["eval_traj.py"])

			eval_csv_path = eval_dir / "eval_traj.csv"
			header, values = self._read_traj_eval_metrics(eval_csv_path)
			score = float(np.mean(values))
			export_eval_csv_path = batch_eval_dir / f"eval_traj_epoch{epoch_id}.csv"
			shutil.copy2(eval_csv_path, export_eval_csv_path)

			summary_csv_path = os.path.join(model_dir, "traj_eval_summary.csv")
			write_header = not os.path.exists(summary_csv_path)
			with open(summary_csv_path, "a", encoding="utf-8", newline="") as f:
				writer = csv.writer(f)
				if write_header:
					writer.writerow(["epoch", "checkpoint", "mean_score"] + header)
				writer.writerow([epoch_id, os.path.basename(checkpoint_path), score] + values)

			if score > self.best_traj_score:
				self.best_traj_score = score
				self.best_traj_model_path = checkpoint_path
				self._save_best_checkpoint(checkpoint_path, export_eval_csv_path, epoch_id, model_dir)

			return score
		finally:
			if result_dir.exists():
				shutil.rmtree(result_dir)
			if eval_dir.exists():
				shutil.rmtree(eval_dir)
			st_main.goal_model_path = old_goal_model_path
			st_main.path_output = old_path_output
			st_main.scene_id = old_scene_id
			st_main.act_id = old_act_id
        
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
		traj_eval_csv_path = os.path.join(model_dir, 'traj_eval_summary.csv')
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
		if os.path.exists(traj_eval_csv_path):
			with open(traj_eval_csv_path, "r", encoding="utf-8") as f:
				rows = list(csv.reader(f))
			if len(rows) > 1:
				best_row = max(rows[1:], key=lambda row: float(row[2]))
				self.best_traj_score = float(best_row[2])
				self.best_traj_model_path = os.path.join(model_dir, best_row[1])

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
			
			checkpoint_path = os.path.abspath(os.path.join(model_dir, 'model_pred_goal_{}epoch.pt'.format(epoch_id)))
			torch.save(
				{
					'epoch': epoch_id,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
				},
				checkpoint_path,
			)
			traj_score = self.eval_traj_checkpoint(checkpoint_path, epoch_id, model_dir)
			print(f'Traj eval mean score: {traj_score}')
			print(f'Best traj score so far: {self.best_traj_score}')
				
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
