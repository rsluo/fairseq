import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np

class pLSTMMemUnit(nn.Module):
	def __init__(self, input_size, hidden_size):
		"""
		input_size : size of input for each part
		hidden_size : size of hidden_layer for each part

		Takes in x_t of shape (batch, input_size)
				 prev h_t of shape (batch, hidden_size)
				 prev c_t of shape (batch, hidden_size)
		
		Returns context vector for that part
		
		"""
		super(pLSTMMemUnit, self).__init__()
		self.hidden_size = hidden_size
		self.input_size = input_size

		### input weights for each of the gates
		self.Wpii = nn.Linear(input_size, hidden_size)
		init.xavier_normal(self.Wpii.state_dict()['weight'])

		self.Wpfi = nn.Linear(input_size, hidden_size)
		init.xavier_normal(self.Wpfi.state_dict()['weight'])

		self.Wpgi = nn.Linear(input_size, hidden_size)
		init.xavier_normal(self.Wpgi.state_dict()['weight'])

		### hidden weights for each of the gates
		self.Wpih = nn.Linear(hidden_size*5, hidden_size)
		init.xavier_normal(self.Wpih.state_dict()['weight'])

		self.Wpfh = nn.Linear(hidden_size*5, hidden_size)
		init.xavier_normal(self.Wpfh.state_dict()['weight'])
		
		self.Wpgh = nn.Linear(hidden_size*5, hidden_size)
		init.xavier_normal(self.Wpgh.state_dict()['weight'])
	
	def forward(self, xt, prev_ht, prev_ct):
		"""
		Implements the following equations for a single part
		
		i_p = sigmoid(x_t W_ix + h_{t-1} W_ih)
		f_p = sigmoid(x_t W_fx + h_{t-1} W_fh)
		g_p = tanh(x_t W_gx + h_{t-1} W_gh)

		c_t = f_p * c_(t-1) + i_p * g_p
		
		Note: bias vectors are included by default in nn.Linear
		Therefore skipping explicitly creating bias vectors for 
		each of the individual gates.

		"""
		# print("Input size ", self.input_size)
		# print(self.Wpii(xt).size())
		# print(self.Wpih(prev_ht).size())
		# print(prev_ht.size())

		ip = F.sigmoid(self.Wpii(xt) + self.Wpih(prev_ht))
		fp = F.sigmoid(self.Wpfi(xt) + self.Wpfh(prev_ht))
		gp = F.tanh(self.Wpgi(xt) + self.Wpgh(prev_ht))

		ct = fp * prev_ct + ip * gp
		return ct

class pLSTMCell(nn.Module):
	"""
	Implements Part-LSTM cell to process inputs at time t

	"""
	def __init__(self, input_size, hidden_size, part_idx_vals, num_parts=5, part_input_size=15):
		super(pLSTMCell, self).__init__()

		self.hidden_size=hidden_size

		### We have 21 total 3D points 
		###	- 4 points per finger 
		### - 1 point for wrist
		### - 5 points per part = 15 dimensional vector 
		### create pLSTMMemUnit for each of the fingers
		### This code assumes that these points are 3D

		self.input_size = input_size

		# if reorder_input:
		# 	self.part_indices = {
		# 							"1" : [0,1,6,11,16],
		# 							"2" : [0,2,7,12,17],
		# 							"3" : [0,3,8,13,18],
		# 							"4" : [0,4,9,14,19],
		# 							"5" : [0,5,10,15,20]
		# 						}
		# self.part_idx_vals = {}
		# for k in self.part_indices.keys():
		# 	self.part_idx_vals[k] = [3*n+i for n in self.part_indices[k] for i in range(3)]

		self.part_idx_vals = part_idx_vals

		self.p1 = pLSTMMemUnit(part_input_size, hidden_size)
		self.p2 = pLSTMMemUnit(part_input_size, hidden_size)
		self.p3 = pLSTMMemUnit(part_input_size, hidden_size)
		self.p4 = pLSTMMemUnit(part_input_size, hidden_size)
		self.p5 = pLSTMMemUnit(part_input_size, hidden_size)

		self.Woi = nn.Linear(part_input_size*5, hidden_size*5)
		self.Woh = nn.Linear(hidden_size*5, hidden_size*5)

	def forward(self, xt, prev_ht, prev_ct):
		"""
		Implements following equations

		o = sigmoid(W_oi x_t + W_oh h_{t-1})
		c_t = cat(c_1, c2, ..., c_p)
		ht = o * tanh (c_t)
		"""
		# print("xt ", xt.size())
		# print(torch.tensor(self.part_idx_vals["1"]).cuda().long().size())
		# print(prev_ct.size())

		x1t = xt.gather(1, torch.tensor(np.stack([self.part_idx_vals["1"]]*xt.size(0))).cuda().long())
		x2t = xt.gather(1, torch.tensor(np.stack([self.part_idx_vals["2"]]*xt.size(0))).cuda().long())
		x3t = xt.gather(1, torch.tensor(np.stack([self.part_idx_vals["3"]]*xt.size(0))).cuda().long())
		x4t = xt.gather(1, torch.tensor(np.stack([self.part_idx_vals["4"]]*xt.size(0))).cuda().long())
		x5t = xt.gather(1, torch.tensor(np.stack([self.part_idx_vals["5"]]*xt.size(0))).cuda().long())
		c1 = self.p1(x1t, prev_ht, prev_ct[0])
		c2 = self.p2(x2t, prev_ht, prev_ct[1])
		c3 = self.p3(x3t, prev_ht, prev_ct[2])
		c4 = self.p4(x4t, prev_ht, prev_ct[3])
		c5 = self.p5(x5t, prev_ht, prev_ct[4])

		#import pdb
		#pdb.set_trace()

		ct = torch.cat((c1, c2, c3, c4, c5), dim=1)
		xt_cat = torch.cat((x1t, x2t, x3t, x4t, x5t), dim=1)

		o = F.sigmoid(self.Woi(xt_cat) + self.Woh(prev_ht))
		ht = o * F.tanh(ct)
		return ht, (c1, c2, c3, c4, c5)

class pLSTM(nn.Module):
	def __init__(self, input_size, hidden_size, part_idx_vals, batch_first=True, num_parts=5, part_input_size=15):
		super(pLSTM, self).__init__()
		
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = 1
		self.batch_first = batch_first

		self.cell = pLSTMCell(input_size, hidden_size, part_idx_vals, num_parts, part_input_size)
	
	def init_hidden(self):
		h_0 = torch.zeros(1, self.hidden_size*5, requires_grad=True).cuda()
		c_0 = torch.zeros(1, self.hidden_size, requires_grad=True).cuda()
		return h_0, (c_0, c_0, c_0, c_0, c_0)

	def forward(self, input):

		def recurrence(xt, hidden):
			ht, ct = hidden
			hy, cy = self.cell(xt, ht, ct)
			return hy, cy

		if self.batch_first:
			input = input.transpose(0, 1)

		output = []
		steps = range(input.size(0))
		hidden = self.init_hidden()

		for i in steps:
			hidden = recurrence(input[i], hidden)
			if isinstance(hidden, tuple):
				output.append(hidden[0])
			else:
				output.append(hidden)

		output = torch.cat(output, 0).view(input.size(0), *output[0].size())

		if self.batch_first:
			output = output.transpose(0, 1)
		# print("Size of output ", output.size())
		return output, hidden
