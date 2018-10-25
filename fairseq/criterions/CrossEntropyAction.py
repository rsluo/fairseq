# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
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
        actions, probs = model(**sample['net_input'])
        target = model.get_targets(sample)
        #target = target.view(target.size()[0]*target.size()[1], -1)

        # loss = F.nll_loss(lprobs, target, size_average=False, ignore_index=self.padding_idx,
        #                   reduce=reduce)
        loss = F.cross_entropy(actions, target)
        sample_size = sample['target'].size(0)
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        agg_output = {
            'ntokens': ntokens,
            'nsentences': nsentences,
            'loss': loss_sum / sample_size / math.log(2)
        }
        return agg_output
