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
from datetime import datetime
from pathlib import Path

MAIN_ROOT = Path(__file__).resolve().parent.parent
if str(MAIN_ROOT) not in sys.path:
	sys.path.insert(0, str(MAIN_ROOT))

from utils.preprocessing_vrlocomotion import augment_data_old, augment_data, create_images_dict
from utils.image_utils import create_gaussian_heatmap_template, create_determistic_template, create_dist_mat, \
	preprocess_image_for_segmentation, pad, resize
from utils.dataloader_vrlocomotion import SceneDataset, scene_collate
from test_vrlocomotion import evaluate
from train_vrlocomotion import train_pred_goal
from model_factory import build_goal_model, goal_model_config_from_params


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
		self.bfloat16 = params['bfloat16']
		self.main_root = Path(__file__).resolve().parent.parent
		self.best_traj_score = -float("inf")
		self.best_traj_model_path = None
		self.num_epochs = int(params["num_epochs"])
		self.mini_eval_samples = [
			tuple(int(value) for value in sample)
			for sample in params.get("mini_eval_samples", [])
		]
		if not self.mini_eval_samples:
			raise ValueError("mini_eval_samples must not be empty")
		if len(set(self.mini_eval_samples)) != len(self.mini_eval_samples):
			raise ValueError("mini_eval_samples must contain unique triples")
		self.mini_eval_top_k = int(params.get("mini_eval_top_k", 2))
		if self.mini_eval_top_k <= 0:
			raise ValueError("mini_eval_top_k must be positive")
		self.full_eval_top_k = int(params.get("full_eval_top_k", 2))
		if self.full_eval_top_k <= 0:
			raise ValueError("full_eval_top_k must be positive")
		self.run_full_eval_after_training = bool(
			params.get("run_full_eval_after_training", True)
		)
		self.full_eval_scene_ids = [
			int(value) for value in params.get("full_eval_scene_ids", [])
		]
		self.full_eval_action_ids = [
			int(value) for value in params.get("full_eval_action_ids", [])
		]
		if self.run_full_eval_after_training and (
			not self.full_eval_scene_ids or not self.full_eval_action_ids
		):
			raise ValueError(
				"full_eval_scene_ids and full_eval_action_ids are required for full evaluation"
			)

		self.model_config = goal_model_config_from_params(
			params,
			obs_len=self.obs_len,
			pred_len=self.pred_len,
			map_channel=1,
		)
		self.model = build_goal_model(self.model_config)
		self.division_factor = self.model.division_factor
	def _get_main_setting(self):
		if str(self.main_root) not in sys.path:
			sys.path.insert(0, str(self.main_root))
		return importlib.import_module("setting")

	def _get_full_eval_dir(self, model_dir):
		"""Mirror a model's path below models/ into Eval-traj/."""
		model_path = Path(model_dir).resolve()
		models_root = (Path(__file__).resolve().parent / "models").resolve()
		try:
			relative_run_path = model_path.relative_to(models_root)
		except ValueError:
			relative_run_path = Path(model_path.name)
		return self.main_root / "Eval-traj" / relative_run_path

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
		if len(header) != len(values):
			raise RuntimeError(
				f"Header/value length mismatch in traj eval csv: {csv_path}"
			)

		metric_header = []
		metric_values = []
		mean_score = None
		for name, value in zip(header, values):
			if name == "mean_score":
				if mean_score is None:
					mean_score = value
				continue
			metric_header.append(name)
			metric_values.append(value)

		if mean_score is None:
			mean_score = float(np.mean(metric_values))
		return mean_score, metric_header, metric_values

	def _append_traj_eval_summary(
		self,
		summary_csv_path,
		epoch_id,
		checkpoint_path,
		mean_score,
		metric_header,
		metric_values,
	):
		"""Append one evaluation row, including the mean of its three Top-5 metrics."""
		top5_names = [name for name in metric_header if name.startswith("traj_top5(")]
		if len(top5_names) != 3:
			raise RuntimeError(
				f"Expected exactly three Top-5 metrics, got {top5_names}"
			)
		metric_by_name = dict(zip(metric_header, metric_values))
		top5_mean_score = float(np.mean([metric_by_name[name] for name in top5_names]))
		desired_header = [
			"epoch",
			"checkpoint",
			"top5_mean_score",
			"mean_score",
			*metric_header,
		]
		existing_rows = []

		if os.path.exists(summary_csv_path):
			with open(summary_csv_path, "r", encoding="utf-8", newline="") as f:
				rows = list(csv.reader(f))
			if rows:
				existing_header = rows[0]
				column_index = {}
				for index, name in enumerate(existing_header):
					column_index.setdefault(name, index)
				missing = [
					name for name in desired_header
					if name != "top5_mean_score" and name not in column_index
				]
				if missing:
					raise RuntimeError(
						f"Unsupported traj eval summary schema in {summary_csv_path}; "
						f"missing columns: {missing}"
					)
				for row in rows[1:]:
					if not row:
						continue
					try:
						migrated = []
						for name in desired_header:
							if name == "top5_mean_score" and name not in column_index:
								migrated.append(str(float(np.mean([
									float(row[column_index[top5_name]])
									for top5_name in top5_names
								]))))
							else:
								migrated.append(row[column_index[name]])
						existing_rows.append(migrated)
					except IndexError as exc:
						raise RuntimeError(
							f"Invalid row in traj eval summary: {summary_csv_path}"
						) from exc

		new_row = [
			epoch_id,
			os.path.basename(checkpoint_path),
			top5_mean_score,
			mean_score,
			*metric_values,
		]
		existing_rows = [
			row
			for row in existing_rows
			if str(row[0]) != str(epoch_id)
		]
		with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
			writer = csv.writer(f)
			writer.writerow(desired_header)
			writer.writerows(existing_rows)
			writer.writerow(new_row)

	def _write_full102_top_epochs(self, summary_csv_path, top_csv_path):
		"""Write the best Full102 epochs ranked by the three-metric Top-5 mean."""
		with open(summary_csv_path, "r", encoding="utf-8", newline="") as f:
			reader = csv.DictReader(f)
			fieldnames = reader.fieldnames or []
			rows = list(reader)

		if "top5_mean_score" not in fieldnames:
			raise RuntimeError(
				f"Missing top5_mean_score in Full102 summary: {summary_csv_path}"
			)
		rows.sort(
			key=lambda row: (-float(row["top5_mean_score"]), int(row["epoch"]))
		)
		top_rows = rows[:self.full_eval_top_k]

		with open(top_csv_path, "w", encoding="utf-8", newline="") as f:
			writer = csv.DictWriter(f, fieldnames=["rank", *fieldnames])
			writer.writeheader()
			for rank, row in enumerate(top_rows, start=1):
				writer.writerow({"rank": rank, **row})

		if top_rows:
			self.best_traj_score = float(top_rows[0]["top5_mean_score"])
			self.best_traj_model_path = top_rows[0]["checkpoint"]
		return top_rows

	def _run_traj_eval(
		self,
		checkpoint_path,
		epoch_id,
		model_dir,
		eval_kind,
		eval_samples=None,
		scene_ids=None,
		action_ids=None,
	):
		checkpoint_path = os.path.abspath(os.fspath(checkpoint_path))
		st_main = self._get_main_setting()
		run_name = os.path.basename(os.path.normpath(model_dir))
		path_output = f"{run_name}_{eval_kind}_epoch{epoch_id}"
		batch_eval_dir = self._get_full_eval_dir(model_dir)
		batch_eval_dir.mkdir(parents=True, exist_ok=True)
		result_dir = self.main_root / "Result" / path_output
		eval_dir = self.main_root / "Eval-traj" / path_output

		old_goal_model_path = st_main.goal_model_path
		old_path_output = st_main.path_output
		old_scene_id = st_main.scene_id
		old_act_id = st_main.act_id
		had_eval_samples = hasattr(st_main, "eval_samples")
		old_eval_samples = getattr(st_main, "eval_samples", None)
		had_write_epoch_tables = hasattr(st_main, "write_epoch_evaluation_tables")
		old_write_epoch_tables = getattr(
			st_main, "write_epoch_evaluation_tables", True
		)

		try:
			st_main.goal_model_path = checkpoint_path
			st_main.path_output = path_output
			st_main.eval_samples = eval_samples
			st_main.write_epoch_evaluation_tables = False
			if scene_ids is not None:
				st_main.scene_id = list(scene_ids)
			if action_ids is not None:
				st_main.act_id = list(action_ids)

			self._run_main_script("TR-LLM.py", ["TR-LLM.py", "", "1", "0", "0", "0"])
			self._run_main_script("eval_traj.py", ["eval_traj.py"])

			eval_csv_path = eval_dir / "eval_traj.csv"
			score, metric_header, metric_values = self._read_traj_eval_metrics(
				eval_csv_path
			)
			if eval_kind == "full102":
				export_eval_csv_path = batch_eval_dir / f"full102_epoch{epoch_id}.csv"
				shutil.copy2(eval_csv_path, export_eval_csv_path)
			return score, metric_header, metric_values
		finally:
			if result_dir.exists():
				shutil.rmtree(result_dir)
			if eval_dir.exists():
				shutil.rmtree(eval_dir)
			st_main.goal_model_path = old_goal_model_path
			st_main.path_output = old_path_output
			st_main.scene_id = old_scene_id
			st_main.act_id = old_act_id
			if had_eval_samples:
				st_main.eval_samples = old_eval_samples
			else:
				delattr(st_main, "eval_samples")
			if had_write_epoch_tables:
				st_main.write_epoch_evaluation_tables = old_write_epoch_tables
			else:
				delattr(st_main, "write_epoch_evaluation_tables")

	def _rank_mini_eval_checkpoints(self, model_dir):
		summary_csv_path = os.path.join(model_dir, "mini_eval_summary.csv")
		if not os.path.exists(summary_csv_path):
			return []

		with open(summary_csv_path, "r", encoding="utf-8", newline="") as f:
			reader = csv.DictReader(f)
			fieldnames = reader.fieldnames or []
			rows = [row for row in reader if row.get("checkpoint")]
		if "top5_mean_score" not in fieldnames:
			raise RuntimeError(
				f"Missing top5_mean_score in Mini-Eval summary: {summary_csv_path}"
			)
		rows.sort(
			key=lambda row: (-float(row["top5_mean_score"]), int(row["epoch"]))
		)
		top_rows = rows[:self.mini_eval_top_k]

		top_path = os.path.join(model_dir, f"mini_eval_top{self.mini_eval_top_k}.csv")
		with open(top_path, "w", encoding="utf-8", newline="") as f:
			writer = csv.DictWriter(f, fieldnames=["rank", *fieldnames])
			writer.writeheader()
			for rank, row in enumerate(top_rows, start=1):
				writer.writerow({"rank": rank, **row})

		if top_rows:
			self.best_traj_score = float(top_rows[0]["top5_mean_score"])
			self.best_traj_model_path = os.path.join(
				model_dir, top_rows[0]["checkpoint"]
			)
		return top_rows

	def eval_mini_checkpoint(self, checkpoint_path, epoch_id, model_dir):
		score, metric_header, metric_values = self._run_traj_eval(
			checkpoint_path,
			epoch_id,
			model_dir,
			eval_kind="mini_eval",
			eval_samples=self.mini_eval_samples,
		)
		self._append_traj_eval_summary(
			os.path.join(model_dir, "mini_eval_summary.csv"),
			epoch_id,
			checkpoint_path,
			score,
			metric_header,
			metric_values,
		)
		top5_values = [
			float(value)
			for name, value in zip(metric_header, metric_values)
			if str(name).startswith("traj_top5(")
		]
		if len(top5_values) != 3:
			raise RuntimeError(
				f"Expected 3 traj_top5 metrics for Mini-Eval, got {len(top5_values)}"
			)
		top5_mean_score = float(np.mean(top5_values))
		self._rank_mini_eval_checkpoints(model_dir)
		return top5_mean_score

	def evaluate_missing_mini_checkpoints(self, model_dir):
		"""Backfill Mini-Eval when resuming checkpoints created before this workflow."""
		completed_epochs = set()
		summary_path = os.path.join(model_dir, "mini_eval_summary.csv")
		if os.path.exists(summary_path):
			with open(summary_path, "r", encoding="utf-8", newline="") as f:
				for row in csv.DictReader(f):
					try:
						completed_epochs.add(int(row["epoch"]))
					except (KeyError, TypeError, ValueError):
						continue

		prefix = "model_pred_goal_"
		suffix = "epoch.pt"
		checkpoints = []
		for name in os.listdir(model_dir):
			if not name.startswith(prefix) or not name.endswith(suffix):
				continue
			epoch_text = name[len(prefix):-len(suffix)]
			if epoch_text.isdigit():
				checkpoints.append((int(epoch_text), os.path.join(model_dir, name)))

		for epoch_id, checkpoint_path in sorted(checkpoints):
			if epoch_id >= self.num_epochs or epoch_id in completed_epochs:
				continue
			print(f"Backfill Mini-Eval epoch {epoch_id}")
			self.eval_mini_checkpoint(checkpoint_path, epoch_id, model_dir)

		return self._rank_mini_eval_checkpoints(model_dir)

	def evaluate_top_checkpoints_full(self, model_dir):
		top_rows = self._rank_mini_eval_checkpoints(model_dir)
		if not top_rows:
			raise RuntimeError("No Mini-Eval checkpoints are available for Full102 evaluation")

		print(
			f"Running Full102 evaluation for Top {len(top_rows)} Mini-Eval checkpoints"
		)
		final_eval_dir = self._get_full_eval_dir(model_dir)
		final_eval_dir.mkdir(parents=True, exist_ok=True)
		full_summary_path = final_eval_dir / "full102_summary.csv"
		top_epochs_path = final_eval_dir / f"full102_top{self.full_eval_top_k}.csv"

		# Rebuild both tables from the per-epoch CSV files. This prevents stale
		# rows from an older run while retaining resumability of expensive evals.
		if full_summary_path.exists():
			full_summary_path.unlink()
		if top_epochs_path.exists():
			top_epochs_path.unlink()

		for row in top_rows:
			epoch_id = int(row["epoch"])
			checkpoint_path = os.path.abspath(
				os.path.join(model_dir, row["checkpoint"])
			)
			if not os.path.isfile(checkpoint_path):
				raise FileNotFoundError(checkpoint_path)

			completed_eval_path = final_eval_dir / f"full102_epoch{epoch_id}.csv"
			if completed_eval_path.is_file():
				score, metric_header, metric_values = self._read_traj_eval_metrics(
					completed_eval_path
				)
				self._append_traj_eval_summary(
					full_summary_path,
					epoch_id,
					checkpoint_path,
					score,
					metric_header,
					metric_values,
				)
				print(f"Skip completed Full102 epoch {epoch_id}: {completed_eval_path}")
			else:
				score, metric_header, metric_values = self._run_traj_eval(
					checkpoint_path,
					epoch_id,
					model_dir,
					eval_kind="full102",
					eval_samples=None,
					scene_ids=self.full_eval_scene_ids,
					action_ids=self.full_eval_action_ids,
				)
				self._append_traj_eval_summary(
					full_summary_path,
					epoch_id,
					checkpoint_path,
					score,
					metric_header,
					metric_values,
				)
				print(f"Full102 epoch {epoch_id} mean score: {score}")

			self._write_full102_top_epochs(full_summary_path, top_epochs_path)

		print(f"Full102 summary for Mini-Eval Top {len(top_rows)}: {full_summary_path}")
		print(f"Full102 ranking by Top-5 mean: {top_epochs_path}")
        
	def train(self, train_data, val_data, params, train_image_path, val_image_path, batch_size=8, device=None, dataset_name=None, test_scene=0):

		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		obs_len = self.obs_len
		pred_len = self.pred_len
		total_len = pred_len + obs_len

		print('Preprocess data')

		model_dir = os.path.abspath(os.fspath(params['model_dir']))
		params['model_dir'] = model_dir
		resume_epoch = None
		if params['use_latest_epoch']:
			if os.path.isdir(model_dir):
				prefix = 'model_pred_goal_'
				suffix = 'epoch.pt'
				epochs = [
					int(name[len(prefix):-len(suffix)])
					for name in os.listdir(model_dir)
					if name.startswith(prefix)
					and name.endswith(suffix)
					and name[len(prefix):-len(suffix)].isdigit()
				]
				if epochs:
					resume_epoch = max(epochs)
		elif os.path.exists(model_dir):
			model_dir = '{}_{}'.format(model_dir, datetime.now().strftime('%m%d%H'))
			params['model_dir'] = model_dir
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

		optimizer_name = str(params.get("optimizer", "adam")).strip().lower()
		weight_decay = float(params.get("weight_decay", 0.0))
		if weight_decay < 0:
			raise ValueError("weight_decay must be non-negative")
		optimizer_classes = {
			"adam": torch.optim.Adam,
			"adamw": torch.optim.AdamW,
		}
		if optimizer_name not in optimizer_classes:
			raise ValueError(
				f"Unsupported optimizer: {optimizer_name}. "
				f"Choose one of: {', '.join(sorted(optimizer_classes))}"
			)
		optimizer = optimizer_classes[optimizer_name](
			model.parameters(),
			lr=params["learning_rate"],
			weight_decay=weight_decay,
		)
		if resume_epoch is not None:
			checkpoint_path = os.path.join(model_dir, 'model_pred_goal_{}epoch.pt'.format(resume_epoch))
			checkpoint = torch.load(checkpoint_path, map_location=device)
			checkpoint_config = checkpoint.get('model_config')
			if checkpoint_config is not None and checkpoint_config != self.model_config:
				raise ValueError(
					f"Checkpoint model_config does not match the active config: "
					f"{checkpoint_config} != {self.model_config}"
				)
			if (
				hasattr(model, "goal_query_reasoner")
				and model.goal_query_reasoner.use_goal_query_gates
			) and not all(
				key in checkpoint["model_state_dict"]
				for key in (
					"goal_query_reasoner.feedback_gate",
					"goal_query_reasoner.prior_gate",
				)
			):
				raise ValueError(
					"This checkpoint predates the gated GoalQueryFormer and cannot "
					"resume its optimizer state. Use a new model_dir to train the "
					"gated model from epoch 0."
				)
			checkpoint_optimizer_name = str(
				checkpoint.get("optimizer_name", "adam")
			).strip().lower()
			if checkpoint_optimizer_name != optimizer_name:
				raise ValueError(
					"Checkpoint optimizer does not match the active config: "
					f"{checkpoint_optimizer_name} != {optimizer_name}. "
					"Use a new model_dir for a different optimizer."
				)
			model.load_state_dict(checkpoint['model_state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			# Optimizer state loading also restores the old parameter-group
			# hyperparameters. Reapply the active experiment settings on resume.
			for param_group in optimizer.param_groups:
				param_group['lr'] = params["learning_rate"]
				param_group['weight_decay'] = weight_decay
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
		start_global_epoch = 0 if resume_epoch is None else resume_epoch + 1
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
					'architecture': self.model_config['model_name'],
					'model_config': self.model_config,
					'optimizer_name': optimizer_name,
					'weight_decay': weight_decay,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
				},
				checkpoint_path,
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

			mini_eval_score = self.eval_mini_checkpoint(
				checkpoint_path,
				epoch_id,
				model_dir,
			)
			print(f'Mini-Eval Top-5 mean score: {mini_eval_score}')
			print(f'Best Mini-Eval Top-5 mean so far: {self.best_traj_score}')

		self.evaluate_missing_mini_checkpoints(model_dir)
		if self.run_full_eval_after_training:
			self.evaluate_top_checkpoints_full(model_dir)


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
