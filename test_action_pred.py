import collections
import itertools
import os
import math
import torch
import numpy as np
import seaborn as sn
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from fairseq import distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter

from torch.utils.data import DataLoader

label_counts = collections.defaultdict(int)
acc_counts = collections.defaultdict(int)
conf_mat = []

def main(args):
    if args.max_tokens is None:
        args.max_tokens = 6000
    print(args)

    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')

    torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load dataset splits
    load_dataset_splits(task, ['valid'])

    # Build model and criterion
    model = task.build_model(args)

    model = load_checkpoint(args, model)

    print('| num. model params: {}'.format(sum(p.numel() for p in model.parameters())))

    loader = DataLoader(dataset=task.datasets['valid'], shuffle=False, batch_size=120)

    action_labels = task.datasets['valid'].load_config()

    for i in range(45):
        conf_mat.append(np.zeros(45))

    accuracy = 0
    for t, x in enumerate(loader):
        model.cuda()

        out = model(x['source'].cuda(), x['length'].cuda())
        _, actions = torch.max(out, dim=1)

        accuracy += torch.sum(torch.eq(actions, x['target'].cuda()))

        compute_per_class_accuracy(actions.cpu().numpy(), x['target'].cpu().numpy(), action_labels)

    print(accuracy)
    print(accuracy/len(loader))

    id2labels = { v: k for k,v in action_labels.items()}

    print("---Per class accuracy---")

    for k in range(45): # Total actions are 45
        denom = label_counts[k]
        if denom == 0:
            denom = 1
        acc = acc_counts[k]/float(denom)
        print("%s  - accuracy %.4f" % (id2labels[k], acc))
    
    for t in range(45):
        for p in range(45):
            print("%d " % conf_mat[t][p], end="")
        print("\n") 

    plt.matshow(conf_mat)
    plt.title("Confusion matrix - Test data")
    plt.colorbar()
    plt.ylabel("Ground Truth Action")
    plt.xlabel("Predicted Action")
    plt.savefig('confusion_matrix_test.jpg')
    
def compute_per_class_accuracy(actions, target, action_labels):

    for s,t in zip(actions, target):
        conf_mat[t][s] += 1

        if t in label_counts:
            label_counts[t] += 1
        else:
            label_counts[t] = 1

        if s == t:
            if t in acc_counts:
                acc_counts[t] += 1
            else:
                acc_counts[t] = 1

    # id2labels = { v: k for k,v in action_labels.items()}

    # print("---Per class accuracy---")

    # for k in range(45): # Total actions are 45
    #     denom = label_counts[k]
    #     if denom == 0:
    #         denom = 1
    #     acc = acc_counts[k]/float(denom)
    #     print("Class %s Accuracy %.4f" % (id2labels[k], acc))

def load_checkpoint(args, model):
    """Load a checkpoint and replay dataloader to match."""
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    print("Loading checkpoint from path ", checkpoint_path)
    
    extra_state, _, _ = \
            utils.load_model_state(checkpoint_path, model)
    model.eval()
    return model


def load_dataset_splits(task, splits):
    for split in splits:
        if split == 'test':
            task.load_dataset(split)
        elif split == "valid" or split == "train":
            task.load_dataset(split)
        else:
            raise Exception("Unknown or invalid data split")


if __name__ == '__main__':
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    if args.distributed_port > 0 or args.distributed_init_method is not None:
        from distributed_train import main as distributed_main

        distributed_main(args)
    elif args.distributed_world_size > 1:
        from multiprocessing_train import main as multiprocessing_main

        multiprocessing_main(args)
    else:
        main(args)
