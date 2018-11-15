import numpy as np
import torch
import os
from . import FairseqDataset

class TrajectoryActionDataset(FairseqDataset):
	def __init__(self, root_dir, num_input_points, shuffle):
		self.root_dir = root_dir
		self.num_input_points = num_input_points
		self.shuffle = shuffle
		
		filepaths = [a for (a, b, c) in os.walk(self.root_dir) if len(b) == 0]
		
		self.all_filepaths = []
		self.lengths = []
		max_len = 0
		min_len = 1000000
		avg_len = 0
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
					if len(file_contents) < min_len:
						min_len = len(file_contents)
					avg_len += len(file_contents)
		self.max_len = max_len

		print("length of video is ", self.max_len, " ", min_len, " ", avg_len/float(len(self.all_filepaths)))
	
		self.valid_actions = set()
		for path in self.all_filepaths:
			self.valid_actions.add(path.split("/")[-2])
		
		self.action_labels = {}
		self.num_hand_points = 63
		for idx, action in enumerate(self.valid_actions):
			if action not in self.action_labels:
				self.action_labels[action] = idx
	
	def __len__(self):
		return len(self.all_filepaths)

	def __getitem__(self, filepath_idx):
		filepath = os.path.join(self.all_filepaths[filepath_idx], "skeleton.txt")
		target = 45 ## Keeping a default target + 45 actions which makes 46 total actions
					# 45 is an unknown action that model outputs when it doesn't know what to do
		#traj_array = np.zeros((self.num_input_points, self.num_hand_points))
		with open(filepath) as file:
			file_contents = file.readlines()
			
			traj_array_len = min(len(file_contents), self.num_input_points)
			traj_array = np.zeros((traj_array_len, self.num_hand_points))
			for i in range(traj_array_len):
				contents = file_contents[i].split()
				for idx in range(1, len(contents)):
					traj_array[i, idx-1] = contents[idx]
			target = self.action_labels[filepath.split("/")[-3]]

		return {
			'id': filepath_idx,
			'source': traj_array,
			'target': target,
			'length': min(self.lengths[filepath_idx], self.num_input_points) 
			}

	def collater(self, samples):
		#print(samples)
		ids = [sample['id'] for sample in samples if sample['id'] is not None]
		src_tokens = [sample['source'] for sample in samples if len(sample['source']) > 0]
		#prev_output_tokens = [sample['target'] for sample in samples if sample['target'] is not None]
		target = [sample['target'] for sample in samples if sample['target'] is not None]
		seq_lengths = [sample['length'] for sample in samples]
		#import pdb; pdb.set_trace()
		
		ordered = sorted(src_tokens, key=len, reverse=True)
		seq_lengths = sorted(seq_lengths, reverse=True)

		padded = [
			np.pad(li, pad_width=((0, self.num_input_points-len(li)), (0, 0)), mode='constant')
			for li in ordered
		]

		net_input = {}
		net_input["src_tokens"] = torch.FloatTensor(padded)
		net_input["src_lengths"] = torch.LongTensor(seq_lengths)
		
		return {"id": torch.LongTensor(ids),
			"ntokens": self.num_input_points * len(samples),
			"net_input":net_input,
			"target": torch.LongTensor(target)
		}

	def num_tokens(self, idx):
		return self.num_input_points

	def num_classes(self):
		return len(self.valid_actions)
		
	def get_dummy_batch(self, num_tokens, max_positions):
		bsz = num_tokens // self.num_input_points
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
		return (self.num_input_points, 1)

	def ordered_indices(self):
		if self.shuffle:
			order = [np.random.permutation(len(self))]
		else:
			order = [np.arange(len(self))]
		# order.append(self.sizes)
		return np.lexsort(order)
