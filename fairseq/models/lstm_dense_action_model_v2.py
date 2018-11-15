import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import FairseqEncoder
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models import register_model_architecture


@register_model('lstm_dense_action_model_v2')
class LSTMDenseActionModel(BaseFairseqModel):

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
			# print(src_tokens.size())
			# print(src_lengths)
			encoder_out = self.encoder(src_tokens, src_lengths)['final_hidden']
			
			#probs = F.softmax(encoder_out, dim=-1)
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
		
		model = LSTMDenseActionModel(encoder)

		# Print the model architecture.
		print(model)

		return model

class SimpleLSTMEncoder(FairseqEncoder):
	def __init__(self, 
				args,
				dictionary={},
				hidden_dim=100, 
				input_dim=63, 
				num_layers=2, 
				dropout=0.0, 
				use_bidirection=False, 
				use_attention=False, 
				cell_type='LSTM', 
				use_cuda=True,
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
		if cell_type == 'GRU':
			self.cell = nn.GRU(input_dim,
							   hidden_dim,
							   num_layers,
							   batch_first=True,
							   dropout=dropout,
							   bidirectional=use_bidirection)
		elif cell_type == 'LSTM':
			self.cell = nn.LSTM(input_size=input_dim,
							   hidden_size=hidden_dim,
							   num_layers=num_layers,
							   batch_first=True,
							   dropout=dropout,
							   bidirectional=use_bidirection)
		elif cell_type == 'RNN':
			self.cell = nn.RNN(input_dim,
							   hidden_dim,
							   num_layers,
							   batch_first=True,
							   dropout=dropout,
							   bidirectional=use_bidirection)
		else:
			self.baseline = nn.Linear(input_dim, hidden_dim)

		self.dense1 = nn.Linear(hidden_dim, out_classses)

	def forward(self, inputs, lengths=None, return_attn=False):
		#print("Forward pass ", inputs.size())
		packed = nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True)
		_outputs, (final_hidden, _final_cell) = self.cell(packed.float())
		_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(_outputs, batch_first=True)
		# print(_outputs.size())
		# print(_outputs[:,-1,:].size())
		encoded = self.dense1(final_hidden[-1].squeeze(0))
		# encoded = self.dense2(encoded)
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

@register_model_architecture('lstm_dense_action_model_v2', 'bi_lstm_dense_am')
def tutorial_simple_lstm(args):
	args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 256)


