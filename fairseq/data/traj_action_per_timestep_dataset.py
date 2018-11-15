import numpy as np
import torch
import os
from . import FairseqDataset
import collections

class TrajectoryActionTimestepDataset(FairseqDataset):
	def __init__(self, root_dir, window_size, shuffle, test_action=None, action_file=None, use_benchmark_data=True):
		self.root_dir = root_dir
		self.window_size = window_size
		self.shuffle = shuffle
		
		print("Root dir ", self.root_dir)
		split_type = self.root_dir.split("/")[-1]

		print("Split type ", split_type)

		filepaths = [a for (a, b, c) in os.walk(self.root_dir) if len(b) == 0]
		
		self.all_filepaths = []
		self.lengths = []

		max_len = 0
		gid = 0
		fid = 0

		self.action_labels = self.load_config()
		
		# maps global identifier to file idx and window
		self.id2file = {}

		self.smaller = set()
		self.smaller_actions = collections.defaultdict(int)
		self.total_actions = {}
		self.action_lens = {}

		permitted_files = set()

		if use_benchmark_data:
			if split_type == "train":
				permitted_files = self.load_benchmark_train()
			elif split_type == "valid":
				permitted_files = self.load_benchmark_test()

		## Delete empty paths
		for idx in range(len(filepaths)):
			filepath = os.path.join(filepaths[idx], "skeleton.txt")
			
			if action_file is not None:
				print("Loading data from ", action_file, "Action ", test_action)
				filepath = action_file

			if use_benchmark_data:
				data = filepath.split("/")
				ff = data[-4] + "/" + data[-3] + "/" + data[-2]
				
				if ff not in permitted_files :
					continue

			action =  self.action_labels[filepath.split("/")[-3]]
			
			with open(filepath) as file:

				file_contents = file.readlines()
				if len(file_contents) > 0 and (test_action is None or action == test_action):

					if test_action is not None or action_file is not None:
						print("Action being tested ", filepath.split("/")[-3], " ", len(file_contents))

					self.all_filepaths.append(filepath)
					self.lengths.append(len(file_contents))

					if len(file_contents) > max_len:
						max_len = len(file_contents)

					length = len(file_contents) - window_size+1
					
					if length <= 0:
						self.id2file[gid] = (fid, 0)
						gid +=1

					for l in range(0, length):
						self.id2file[gid] = (fid, l)
						gid +=1

					fid +=1

			if action_file is not None:
				break

		self.max_len = max_len
		self.total_len = gid

		print("Total length ", self.total_len)
		
		self.num_hand_points = 63

	def __len__(self):
		return self.total_len

	def __getitem__(self, idx):

		filepath_idx, w_idx = self.id2file[idx]

		filepath = self.all_filepaths[filepath_idx]
		target = [45]*self.window_size ## Keeping a default target + 45 actions which makes 46 total actions
					# 45 is an unknown action that model outputs when it doesn't know what to do
		with open(filepath) as file:
			file_contents = file.readlines()
			
			traj_array_len = min(self.window_size, len(file_contents))
			traj_array = np.zeros((self.window_size, self.num_hand_points))
			
			# if filepath_idx in self.smaller:
			# 	print("Fetching object from smaller file", self.window_size, " ", filepath_idx, " ",len(file_contents), " ", self.lengths[filepath_idx])
			# 	print("Found a smaller file ", traj_array_len)

			seq_len = 0
			for k, i in enumerate(range(w_idx, w_idx + traj_array_len)):
				contents = file_contents[i].split()
				for idx in range(1, len(contents)):
					traj_array[k, idx-1] = contents[idx]
				seq_len+=1
				target[k] = self.action_labels[filepath.split("/")[-3]]

			item = {
				'id': filepath_idx,
				'source': traj_array,
				'target': target,
				'length': traj_array_len 
				}
		return item

	def collater(self, samples):
		ids = [sample['id'] for sample in samples if sample['id'] is not None]
		src_tokens = [sample['source'] for sample in samples if len(sample['source']) > 0]
		target = [sample['target'] for sample in samples if sample['target'] is not None]
		seq_lengths = [sample['length'] for sample in samples]
		
		ordered = sorted(src_tokens, key=len, reverse=True)
		seq_lengths = sorted(seq_lengths, reverse=True)

		padded = [
			np.pad(li, pad_width=((0, self.window_size-len(li)), (0, 0)), mode='constant')
			for li in ordered
		]
		
		net_input = {}
		net_input["src_tokens"] = torch.FloatTensor(padded)
		net_input["src_lengths"] = torch.LongTensor(seq_lengths)
		
		return {"id": torch.LongTensor(ids),
			"ntokens": self.window_size * len(samples),
			"net_input":net_input,
			"target": torch.LongTensor(target)
		}

	def load_config(self):
		import json
		script_dir = os.path.dirname(__file__)
		filepath = os.path.join(script_dir, 'config/action_config.json')
		with open(filepath) as f:
			action_labels = json.load(f)
		return action_labels

	def load_benchmark_train(self):
		script_dir = os.path.dirname(__file__)
		filepath = os.path.join(script_dir, 'config/benchmark_train.txt')
		train_files = set()
		with open(filepath) as f:
			all_files = f.readlines()
			for f in all_files:
				train_files.add(f.split(" ")[0])
		return train_files

	def load_benchmark_test(self):
		script_dir = os.path.dirname(__file__)
		filepath = os.path.join(script_dir, 'config/benchmark_test.txt')
		test_files = set()
		with open(filepath) as f:
			all_files = f.readlines()
			for f in all_files:
				test_files.add(f.split(" ")[0])
		return test_files

	def num_tokens(self, idx):
		return self.window_size

	def num_classes(self):
		return len(self.action_labels.keys())
		
	def get_dummy_batch(self, num_tokens, max_positions):
		bsz = num_tokens // self.window_size
		print('bsz', bsz)
		print('num_tokens', num_tokens)

		return self.collater([
				{
					'id': i,
					'source': self.__getitem__(i)['source'],
					'target': self.__getitem__(i)['target'],
					'length': self.__getitem__(i)['length']
				} for i in range(bsz)
			])  
			

	def size(self, idx):
		return (self.window_size, 1)

	def ordered_indices(self):
		if self.shuffle:
			order = [np.random.permutation(len(self))]
		else:
			order = [np.arange(len(self))]
		# order.append(self.sizes)
		return np.lexsort(order)
