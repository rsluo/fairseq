import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import FairseqEncoder
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models import register_model_architecture


@register_model('lstm_dense_action_model')
class LSTMDenseActionModel(BaseFairseqModel):

	def __init__(self, encoder):
			super().__init__()

			self.encoder = encoder
			self.dense = nn.Linear(self.encoder)
			assert isinstance(self.encoder, FairseqEncoder)

	def forward(self, src_tokens, src_lengths, prev_output_tokens):
			"""
			Run the forward pass for an encoder-decoder model.

			First feed a batch of source tokens through the encoder. Then, feed the
			encoder output and previous decoder outputs (i.e., input feeding/teacher
			forcing) to the decoder to produce the next outputs::

				encoder_out = self.encoder(src_tokens, src_lengths)
				return self.decoder(prev_output_tokens, encoder_out)

			Args:
				src_tokens (LongTensor): tokens in the source language of shape
					`(batch, src_len)`
				src_lengths (LongTensor): source sentence lengths of shape `(batch)`
				prev_output_tokens (LongTensor): previous decoder outputs of shape
					`(batch, tgt_len)`, for input feeding/teacher forcing

			Returns:
				the decoder's output, typically of shape `(batch, tgt_len, vocab)`
			"""
			encoder_out = self.encoder(src_tokens, src_lengths)
			#out = nn.Linear
			#return out, prob
			return encoder_out

	def max_positions(self):
			"""Maximum length supported by the model."""
			return (self.encoder.max_positions(), self.decoder.max_positions())
			
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
			dropout=args.encoder_dropout,
		)
		
		model = LSTMDenseActionModel(encoder)

		# Print the model architecture.
		print(model)

		return model

class SimpleLSTMEncoder(FairseqEncoder):
	def __init__(self, 
				args,
				dictionary={},
				hidden_dim=128, 
				input_dim=3, 
				num_layers=1, 
				dropout=0.0, 
				use_bidirection=False, 
				use_attention=False, 
				cell_type='LSTM', 
				use_cuda=True, 
				max_length=784
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

		### #
		use_attention = False ###### TODO Setting this to false to obtain a good baseline #####
		if use_attention:
			self.attn = nn.Linear((2 if use_bidirection else 1) * hidden_dim, 1)
			self.attn_softmax = nn.Softmax(dim=1)
		

	def forward(self, inputs, lengths=None, return_attn=False):
		# # inputs = Variable(inputs)
		# # batch_size, seq_length, input_dim = inputs.shape

		# if self.use_cuda:
		# 	inputs = inputs.cuda()
		# if self.cell_type:
		# 	cell_outputs, _ = self.cell(inputs)
		# else:
		# 	cell_outputs = self.baseline(inputs)
		# if self.use_attention:
		# 	logits = self.attn(cell_outputs)
		# 	softmax = self.attn_softmax(logits)
		# 	mean = torch.sum(softmax * cell_outputs, dim=1)
		# # else:
		# # 	mean = torch.mean(cell_outputs, dim=1)
		# # model_outputs = self.output_layer(cell_outputs)
		_outputs, (final_hidden, _final_cell) = self.cell(inputs.float())
		if return_attn:
			return {'final_hidden': final_hidden.squeeze(0)}, softmax
		else:
			return {'final_hidden': final_hidden.squeeze(0)}

	def reorder_encoder_out(self, encoder_out, new_order):
		"""
		Reorder encoder output according to 'new_order'.

		Args:
			encoder_out: output from the ''forward()'' method
			new_order (LongTensor): desired order

		Returns:
			'encoder_out' rearranged according to 'new_order'
		"""
		final_hidden = encoder_out['final_hidden']
		return {
			'final_hidden': final_hidden.index_select(0, new_order),
		}

@register_model_architecture('lstm_dense_action_model', 'lstm_dense_am')
def tutorial_simple_lstm(args):
	args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 256)


