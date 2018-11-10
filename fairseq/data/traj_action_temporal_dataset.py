import numpy as np
import torch
import os
from . import FairseqDataset

class TrajectoryActionTemporalDataset(FairseqDataset):
	def __init__(self, root_dir, window_size, shuffle):
		self.root_dir = root_dir
		self.window_size = window_size
		self.shuffle = shuffle
		
		print("Root dir ", self.root_dir)
		
		filepaths = [a for (a, b, c) in os.walk(self.root_dir) if len(b) == 0]
		
		self.all_filepaths = []
		self.lengths = []

		max_len = 0
		gid = 0
		fid = 0

		# maps global identifier to file idx and window
		self.id2file = {}
		## Delete empty paths
		for idx in range(len(filepaths)):
			filepath = os.path.join(filepaths[idx], "skeleton.txt")
			with open(filepath) as file:
				
				file_contents = file.readlines()
				if len(file_contents) > 0:
					
					self.all_filepaths.append(filepaths[idx])
					self.lengths.append(len(file_contents))

					if len(file_contents) > max_len:
						max_len = len(file_contents)

					length = len(file_contents) - window_size+1
					if length < 0:
						self.id2file[gid] = (fid, 0)

					for l in range(0, length):
						self.id2file[gid] = (fid, l)
						gid += 1
					
					fid+=1

					# if len(file_contents) < 50: # Window size is 50
					# 	print(filepath)

		self.max_len = max_len
		self.total_len = gid

		print("Total length ", self.total_len)

		self.action_labels = self.load_config()
		
		self.num_hand_points = 63
	
	def __len__(self):
		return self.total_len

	def __getitem__(self, idx):

		filepath_idx, w_idx = self.id2file[idx]

		#print(idx)

		filepath = os.path.join(self.all_filepaths[filepath_idx], "skeleton.txt")
		target = 45 ## Keeping a default target + 45 actions which makes 46 total actions
					# 45 is an unknown action that model outputs when it doesn't know what to do
		with open(filepath) as file:
			file_contents = file.readlines()
			
			traj_array = np.zeros((self.window_size, self.num_hand_points))
			
			seq_len = 0
			for k, i in enumerate(range(w_idx, min(w_idx + self.window_size, len(file_contents)))):
				contents = file_contents[i].split()
				for idx in range(1, len(contents)):
					traj_array[k, idx-1] = contents[idx]
				seq_len+=1

			target = self.action_labels[filepath.split("/")[-3]]
			item = {
				'id': filepath_idx,
				'source': traj_array,
				'target': target,
				'length': seq_len 
				}
		return item

	def collater(self, samples):
		ids = [sample['id'] for sample in samples if sample['id'] is not None]
		src_tokens = [sample['source'] for sample in samples if len(sample['source']) > 0]
		target = [sample['target'] for sample in samples if sample['target'] is not None]
		seq_lengths = [sample['length'] for sample in samples]
		
		net_input = {}
		net_input["src_tokens"] = torch.FloatTensor(src_tokens)
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
