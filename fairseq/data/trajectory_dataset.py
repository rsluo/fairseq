import numpy as np
import torch
import os
from . import FairseqDataset

class TrajectoryDataset(FairseqDataset):
	def __init__(self, root_dir, num_input_points, shuffle):
		self.root_dir = root_dir
		self.num_input_points = num_input_points
		self.shuffle = shuffle
		self.all_filepaths = [a for (a, b, c) in os.walk(self.root_dir) if len(b) == 0]

	def __len__(self):
		return len(self.all_filepaths)

	def __getitem__(self, filepath_idx):
		traj_array = np.zeros((self.num_input_points, 3))
		target_array = np.zeros((self.num_input_points, 3))
		filepath = os.path.join(self.all_filepaths[filepath_idx], "skeleton.txt")
		print(filepath)
		with open(filepath) as file:
			file_contents = file.readlines()
			if len(file_contents) >= self.num_input_points + 1:
				for i in range(self.num_input_points):
					traj_array[i, 0] = file_contents[i].split()[1]
					traj_array[i, 1] = file_contents[i].split()[2]
					traj_array[i, 2] = file_contents[i].split()[3]
					target_array[i, 0] = file_contents[i+1].split()[1]
					target_array[i, 1] = file_contents[i+1].split()[2]
					target_array[i, 2] = file_contents[i+1].split()[3]

		# return (target_array, traj_array)
		return {
			'id': filepath_idx,
			'source': traj_array,
			'target': target_array,
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
