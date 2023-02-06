import os
import shutil
import sys
from argparse import Namespace

import torch


def saveCheckpoint(state, is_best, logdir, run_id=None ,filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    #directory = "%s/%s/" % (logdir,run_name)

    # if logdir in os.getcwd():
    #     logdir = './'

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    filename = os.path.join(logdir, filename)
    print('saving', filename)
    torch.save(state, filename)
    if is_best:
        print('saved checkpoint with top accuracy')
        if run_id is not None:
            shutil.copyfile(filename, os.path.join(logdir, 'model_best_{}.pth.tar'.format(run_id)))
        else:
            shutil.copyfile(filename, os.path.join(logdir, 'model_best.pth.tar'))


def loadForEval(checkpoint_filepath):
    if os.path.isfile(checkpoint_filepath):
        print("=> loading checkpoint '{}'".format(checkpoint_filepath))
        checkpoint = torch.load(checkpoint_filepath)
        #start_epoch = checkpoint['epoch']
        #best_precision = checkpoint['best_precision']
        hyperparameters = checkpoint['arch']
        model = DenseNet(growth_rate=hyperparameters.growth, block_config=(6, 12, 24, 16),
            num_init_features=hyperparameters.input_size[0]*hyperparameters.input_size[1], bn_size=4, 
            drop_rate=hyperparameters.droprate, num_classes=4)
        model = model.cuda().eval()
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_filepath, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_filepath))
        import sys
        sys.exit(0)


def resume_from_checkpoint(checkpoint_path: str) -> Namespace:
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        #best_precision = checkpoint['best_precision']
        print('replacing hyperparameters')
        epochs = args.epochs
        args = checkpoint['arch']
        args.epochs = epochs
        #args.start_epoch = checkpoint['epoch']
    else:
        print('=> no checkpoint found at', checkpoint_path)
        sys.exit(0)

        return args, checkpoint
