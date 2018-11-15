import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import FairseqEncoder
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models import register_model_architecture
from .plstm import pLSTM

import numpy as np

@register_model('plstm_dense_action_model')
class pLSTMDenseActionModel(BaseFairseqModel):

	def __init__(self, encoder):
			super().__init__()

			self.encoder = encoder
			assert isinstance(self.encoder, FairseqEncoder)

	def forward(self, src_tokens, src_lengths):
			"""
			Run the forward pass for an encoder-dense model.

			First feed a batch of source tokens through the encoder. Then, feed the
			encoder output and previous decoder outputs (i.e., input feeding/teacher
			forcing) to the decoder to produce the next outputs::

				encoder_out = self.encoder(src_tokens, src_lengths)

			Args:
				src_tokens (LongTensor): tokens in the source language of shape
					`(batch, src_len)`
				src_lengths (LongTensor): source sentence lengths of shape `(batch)`

			Returns:
				the decoder's output, typically of shape `(batch, tgt_len, vocab)`
			"""
			encoder_out = self.encoder(src_tokens, src_lengths)['final_hidden']
			return encoder_out
	
	def get_normalized_probs(self, net_output, log_probs, sample=None):
		"""Get normalized probabilities (or log probs) from a net's output."""
		if torch.is_tensor(net_output):
			logits, _ = net_output.float()
			if log_probs:
				return F.log_softmax(logits, dim=-1)
			else:
				return F.softmax(logits, dim=-1)
		raise NotImplementedError

	def max_positions(self):
			"""Maximum length supported by the model."""
			return (self.encoder.max_positions(), 1)
	
	def max_decoder_positions(self):
		"""Maximum length supported by the decoder."""
		raise NotImplementedError

	@staticmethod
	def add_args(parser):
		# Models can override this method to add new command-line arguments.
		# Here we'll add some new command-line arguments to configure dropout
		# and the dimensionality of the embeddings and hidden states.
		parser.add_argument(
			'--encoder-hidden-dim', type=int, metavar='N',
			help='dimensionality of the encoder hidden state',
		)
		parser.add_argument(
			'--encoder-dropout', type=float, default=0.1,
			help='encoder dropout probability',
		)

	@classmethod
	def build_model(cls, args, task):
		# Fairseq initializes models by calling the ''build_model()''
		# function. This provides more flexibility, since the returned model
		# instance can be of a different type than the one that was called.
		# In this case we'll just return a SimpleLSTMModel instance.

		# Initialize our Encoder and Decoder.
		encoder = SimpleLSTMEncoder(
			args=args,
			hidden_dim=args.encoder_hidden_dim,
			dropout=args.encoder_dropout
		)
		
		model = pLSTMDenseActionModel(encoder)

		# Print the model architecture.
		print(model)

		return model

class SimpleLSTMEncoder(FairseqEncoder):
	def __init__(self, 
				args,
				dictionary={},
				hidden_dim=32*5, 
				input_dim=63, 
				num_layers=1, 
				dropout=0.0, 
				use_bidirection=False, 
				use_attention=False, 
				cell_type='pLSTM', 
				use_cuda=True, 
				max_length=784,
				out_classses = 46   
				# Hardcoding; number of classes present in the current dataset
				# 0 represents an unknown action - \
				# model should output this when it doesn't know what to do
			):
		super().__init__(dictionary)
		self.args = args
		self.use_cuda = use_cuda
		self.input_dim = input_dim
		self.num_layers = num_layers
		self.hidden_dim = hidden_dim
		self.use_attention = use_attention
		self.use_bidirection = use_bidirection
		self.cell_type = cell_type

		self.num_parts = 5
		### We have 21 total 3D points 
		###	- 4 points per finger 
		### - 1 point for wrist
		### - 5 points per part = 15 dimensional vector 
		### create pLSTMMemUnit for each of the fingers
		### This code assumes that these points are 3D

		self.part_indices = {
								"1" : [0,1,6,7,8],
								"2" : [0,2,9,10,11],
								"3" : [0,3,12,13,14],
								"4" : [0,4,15,16,17],
								"5" : [0,5,18,19,20]
							}
		self.part_idx_vals = {}
		for k in self.part_indices.keys():
			self.part_idx_vals[k] = [3*n+i for n in self.part_indices[k] for i in range(3)]

		self.part_indices_hidden = {}
		for idx, i in enumerate(range(0, hidden_dim*5, hidden_dim)):
			self.part_indices_hidden[str(idx+1)] = np.arange(i, i+hidden_dim) 

		# print(self.part_indices_hidden)

		if cell_type == 'pLSTM':
			self.cell = pLSTM(input_size=input_dim,
							   hidden_size=hidden_dim,
							   batch_first=True, part_idx_vals=self.part_idx_vals, part_input_size=15)
			# self.cell2 = pLSTM(input_size= hidden_dim,
			# 				   hidden_size=hidden_dim,
			# 				   batch_first=True, part_idx_vals=self.part_indices_hidden, part_input_size=hidden_dim)
		else:
			self.baseline = nn.Linear(input_dim, hidden_dim)

		# self.dense = nn.Linear(hidden_dim*5, 128)
		# self.dense2 = nn.Linear(128, out_classses)

		self.dense = nn.Linear(hidden_dim*5, out_classses)

		# ### #
		# use_attention = False ###### TODO Setting this to false to obtain a good baseline #####
		# if use_attention:
		# 	self.attn = nn.Linear((2 if use_bidirection else 1) * hidden_dim, 1)
		# 	self.attn_softmax = nn.Softmax(dim=1)

		

	def forward(self, inputs, lengths=None, return_attn=False):
		#print("Forward pass ", inputs.size())
		_outputs, (final_hidden, _final_cell) = self.cell(inputs.float())
		# _outputs, (final_hidden, _final_cell) = self.cell2(_outputs)
		encoded = self.dense(final_hidden)
		return {'final_hidden': encoded}

	def reorder_encoder_out(self, encoder_out, new_order):
		"""
		Reorder encoder output according to 'new_order'.

		Args:
			encoder_out: output from the ''forward()'' method
			new_order (LongTensor): desired order

		Returns:
			'encoder_out' rearranged according to 'new_order'
		"""
		print("Reordering encoder_out")
		final_hidden = encoder_out['final_hidden']
		return {
			'final_hidden': final_hidden.index_select(0, new_order),
		}

@register_model_architecture('plstm_dense_action_model', 'plstm_dense_am')
def tutorial_simple_lstm(args):
	args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 32)


