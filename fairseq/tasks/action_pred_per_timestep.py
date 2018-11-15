
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import numpy as np
import os

from fairseq.data import (
    Dictionary, IndexedInMemoryDataset, IndexedRawTextDataset,
    MonolingualDataset, TokenBlockDataset, TruncatedDictionary, TrajectoryActionTimestepDataset
)
from fairseq.tasks import FairseqTask, register_task


@register_task('action_prediction_per_time')
class ActionPredictionPerTimestepTask(FairseqTask):
    """
    Train an action prediction model.

    Args:
        root_dir (str): the root directory for the data; holds the 'train,' 'val,' 'test' folders 

        num_input_points (int): the number of past trajectory points; we will predict future trajectory points from num_input_points past points

        targets (List[str]): list of the target types that the language model should predict.
        Can be one of "self", "future", and "past". Defaults to "future".

    .. note::

        The language modeling task is compatible with :mod:`train.py <train>`,
        :mod:`generate.py <generate>`, :mod:`interactive.py <interactive>` and
        :mod:`eval_lm.py <eval_lm>`.

    The language modeling task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.language_modeling_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='path to root data directory')
        parser.add_argument('--num_input_points', default=10, type=int, 
                            help='number of past trajectory points')
        parser.add_argument('--sample-break-mode',
                            choices=['none', 'complete', 'eos'],
                            help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 'If set to "eos", includes only one sentence per sample.')
        parser.add_argument('--tokens-per-sample', default=1024, type=int,
                            help='max number of tokens per sample for LM dataset')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--output-dictionary-size', default=-1, type=int,
                            help='limit the size of output dictionary')

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return cls(args)


    def __init__(self, args):
        super().__init__(args)


    def build_model(self, args):
        model = super().build_model(args)
        return model


    def load_dataset(self, split):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        print("Loading dataset split ", split)
        shuffle = False
        root_dir = self.args.data

        split_dir = os.path.join(root_dir, split)
        num_input_points = self.args.num_input_points

        loaded_datasets = []
        loaded_datasets.append(TrajectoryActionTimestepDataset(split_dir, num_input_points, shuffle))
        
        self.datasets[split] = TrajectoryActionTimestepDataset(split_dir, num_input_points, shuffle)
        self.num_classes = self.datasets[split].num_classes()


    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return {}

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return {}
