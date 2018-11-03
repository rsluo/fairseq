import numpy as np
import torch
import os
from . import FairseqDataset

class TrajectoryActionDataset(FairseqDataset):
	def __init__(self, root_dir, num_input_points, shuffle):
		self.root_dir = root_dir
		self.num_input_points = num_input_points
		self.shuffle = shuffle
		self.all_filepaths = [a for (a, b, c) in os.walk(self.root_dir) if len(b) == 0]
		self.valid_actions = set()
		for path in self.all_filepaths:
			self.valid_actions.add(path.split("/")[-2])

		self.action_labels = {}
		self.num_hand_points = 64
		for idx, action in enumerate(self.valid_actions):
			if action not in self.action_labels:
				self.action_labels[action] = idx
	
	def __len__(self):
		return len(self.all_filepaths)

	def __getitem__(self, filepath_idx):
		filepath = os.path.join(self.all_filepaths[filepath_idx], "skeleton.txt")
		target = 45 ## Keeping a default target + 45 actions which makes 46 total actions
					# 45 is an unknown action that model outputs when it doesn't know what to do
		traj_array = np.zeros((self.num_input_points, self.num_hand_points))
		with open(filepath) as file:
			file_contents = file.readlines()
			#traj_array_len = min(len(file_contents), self.num_input_points)
			#traj_array = np.zeros((traj_array_len, self.num_hand_points))
			for i in range(min(len(file_contents), self.num_input_points)):
				contents = file_contents[i].split()
				#print(len(contents))
				#print("Index ", i, " ", idx)
				for idx in range(len(contents)):
				#	print("Index ", i, " ", idx)
					traj_array[i, idx] = contents[idx]
			target = self.action_labels[filepath.split("/")[-3]]
		return {
			'id': filepath_idx,
			'source': traj_array,
			'target': target
		}

	def collater(self, samples):
		#print(samples)
		ids = [sample['id'] for sample in samples if sample['id'] is not None]
		src_tokens = [sample['source'] for sample in samples if len(sample['source']) > 0]
		#prev_output_tokens = [sample['target'] for sample in samples if sample['target'] is not None]
		target = [sample['target'] for sample in samples if sample['target'] is not None]
		#import pdb; pdb.set_trace()
		net_input = {}
		net_input["src_tokens"] = torch.FloatTensor(src_tokens)
		net_input["src_lengths"] = torch.LongTensor(np.ones(len(samples))*self.num_input_points)
		
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
		# return self.collater(np.arange(bsz))

		return self.collater([
				{
					'id': i,
					'source': self.__getitem__(i)['source'],
					'target': self.__getitem__(i)['target'],
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
