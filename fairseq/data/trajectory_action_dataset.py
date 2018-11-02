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
		self.num_hand_points = 63
		for idx, action in enumerate(self.valid_actions):
			if action not in self.action_labels:
				self.action_labels[action] = idx
	
	def __len__(self):
		return len(self.all_filepaths)

	def __getitem__(self, filepath_idx):
		filepath = os.path.join(self.all_filepaths[filepath_idx], "skeleton.txt")
		target = None
		with open(filepath) as file:
			file_contents = file.readlines()
			traj_array = np.zeros(min(len(file_contents), self.num_input_points))
			for i in range(min(len(file_contents), self.num_input_points)):
				for idx in range(self.num_hand_points):
					traj_array[i, idx] = file_contents[i].split()[idx]
			target = self.action_labels[filepath.split("/")[-3]]

		return {
			'id': filepath_idx,
			'source': traj_array,
			'target': target,
		}

	def collater(self, samples):
		ids = [sample['id'] for sample in samples]
		src_tokens = [sample['source'] for sample in samples]
		prev_output_tokens = [sample['target'] for sample in samples]
		target = [sample['target'] for sample in samples]
		return {"id": torch.LongTensor(ids),
				"ntokens": self.num_input_points * len(samples),
				"net_input": {"src_tokens": torch.LongTensor(src_tokens),
								"src_lengths": torch.LongTensor(np.ones(len(samples)) * self.num_input_points),
								"prev_output_tokens": torch.LongTensor(prev_output_tokens)},
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
