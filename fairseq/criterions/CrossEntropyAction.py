# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('cross_entropy_action')
class CrossEntropyActionCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        out, probs = model(**sample['net_input'])
        target = model.get_targets(sample)

        _, actions = torch.max(probs, dim=1)

        loss = F.cross_entropy(out, target)
        accuracy = sum(actions == target)
        #print(" Actions ", actions, " Target ", target)
        sample_size = sample['target'].size(0)
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'accuracy': accuracy
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        accuracy = sum(log.get('accuracy', 0) for log in logging_outputs)
        agg_output = {
            'ntokens': ntokens,
            'nsentences': nsentences,
            'loss': loss_sum / sample_size / math.log(2),
            'accuracy': accuracy / sample_size
        }
        return agg_output
