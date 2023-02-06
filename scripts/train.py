'''
Train a regression network to predict quality labels, e.g. from the motion tracker.
Ground truth is given by csv and images are read from the rhineland dataset.
'''

import context


import os
import random

import torch
import pandas as pd
#from torch.utils.data import DataLoader

import monai
from monai.transforms import Compose
#from monai.data import DataLoader # PersistentDataset, CacheDataset, Dataset, 
from monai.data.grid_dataset import PatchDataset, GridPatchDataset, iter_patch
#from monai.data.grid_dataset import iter_patch as PatchIter
from monai.transforms import RandSpatialCropSamplesd
#from monai.transforms import *

from modenet.models import densenet3d, densenet2d
from modenet.models.FastSurferCNN import FastSurferCNN
from modenet.models.FCN import SFCN, FCN2D, FCN3D, custom_SFCN

from modenet.logging import tensorboard
from modenet.utils.misc import load_parameters
from modenet.transforms import utility, augmentation, regularization
from modenet.utils.training import training_loop
from modenet.dataset import hdf5_dataset, patching
from modenet.losses.regression import L1L2Loss, my_KLDivLoss, Frequency_w_KLDivLoss
from modenet.dataset.dataloader import DataLoader


def main():

    print('Starting training')
    print('Pytorch Version: ', torch.__version__)
    

    # change working directory to the files directory
    from_file_dir = lambda x : os.path.join(os.path.dirname(__file__),x)
    os.chdir(from_file_dir('../..'))

    args = load_parameters(from_file_dir('parameters.json'))
    args.ground_truth = from_file_dir(args.ground_truth.replace('t1', args.sequence.lower().replace('background', 't1')))
    print(f'Sequence {args.sequence}, ground truth {args.ground_truth}')

    #data_dir = from_file_dir('data_2022')
    data_dir = from_file_dir(args.data_dir)

    # cache the paths to the files here
    train_files = os.path.join(data_dir, args.sequence + '_train.csv')
    val_files = os.path.join(data_dir, args.sequence + '_val.csv')
    # directory to store logfiles
    LOG_DIR = 'tensorboard_logs'

    # softlabel constants
    #softlabel_max = pd.read_csv(args.ground_truth).max().values[1] + pd.read_csv(args.ground_truth).max().values[1] * 0.05
    softlabel_max = 0.39
    num_bins = 40
    step_per_bin = softlabel_max/(num_bins+1)
    
    print('Softlabel max: ', softlabel_max)

    # augmentation workers
    num_workers = 16
    

    if args.tensorboard:
        writer, tensorboard_logdir = tensorboard.setup_tensorboard(args.name if not args.testrun else 'TESTRUN_'+args.name, logdir=LOG_DIR)
        tensorboard.create_backup(tensorboard_logdir, exclude=[LOG_DIR,'motiontracker','trash', 'env'], verbose=True)
        writer.add_text('hyperparameters', str(args))
    else:
        writer = None
        tensorboard_logdir = None
        print('WARNING: checkpointing and logging disabled')


    if args.resume:
        args, checkpoint = checkpointing.resume_from_checkpoint(args.resume)

    # seed random generators and apply cuda settings to enforce determinism
    random.seed(4)
    monai.utils.set_determinism(seed=4, additional_settings=None)

    # dtype and shape
    utility_transforms = Compose([

        utility.ApplyFunction(['image'],lambda x: x%256),
        utility.ValueToTensor(['image'], dtype=torch.float32),


        utility.TensorToFloatTensor(['image','label']),
        utility.TensorUnsqueeze(['image'], 0)
    ])


    # softlabels flag converts labels into soft class labels, for classifier architectures
    if args.soft_labels:
        softlabel_helper = utility.ToSoftLabel('label', 'hard_label',(0,num_bins*step_per_bin),step_per_bin)
        utility_transforms = Compose([
            softlabel_helper,
            utility_transforms
        ])
    else:
        softlabel_helper = None

    # reoragnize axes only for 2d
    post_aug_transforms = Compose([
        utility.TensorSqueeze(['image'], 0),
        utility.TensorPermute(keys=['image'],dim=(2,0,1))
    ]) if args.twoD else utility.Identity()

    

    fft_transform = Compose([
        utility.TensorSqueeze(['image'], 0),
        regularization.FourierTransform3d(['image']),
        utility.TensorPermute(keys=['image'],dim=(3,0,1,2)),
    ])
            
    # ------------ data augmentation happens here !!! ----------------------------
    transforms_train = Compose([
        utility_transforms,
        # ---- data augmentation --------------
        utility.TensorSqueeze(['image'], 0),
        augmentation.RandomFlip(axes=[0,1,2], p=.3),
        utility.TensorUnsqueeze(['image'], 0),
        augmentation.RandomIntensityScaling((0.9,1.1)),
        # ----- utility after augmentation ----
        post_aug_transforms
    ])

    transforms_test = Compose([utility_transforms,post_aug_transforms])
    if args.tensorboard:
        #torch.save(transforms_train, os.path.join(tensorboard_logdir, 'transforms_train.pt'))
        torch.save(transforms_test, os.path.join(tensorboard_logdir, 'transforms_test.pt'))
        if args.soft_labels:
            torch.save(softlabel_helper, os.path.join(tensorboard_logdir, 'softlabel_helper.pt'))

    print('loading dataset ...')
   

    train_dataset = hdf5_dataset.loadDataset(transforms_train, path_csv=train_files, no_files=5 if args.testrun else None,
        ground_truth_csv=args.ground_truth, dataset_name='training_files_' + args.sequence, dataset_folder=data_dir, training=True)
    val_dataset   = hdf5_dataset.loadDataset(transforms_test,  path_csv=val_files,  no_files=2 if args.testrun else None,
        ground_truth_csv=args.ground_truth, dataset_name='validation_files_' + args.sequence, dataset_folder=data_dir, training=False)

    if num_workers > 0:
        print('WARNING: Using multiple workers in data loader - this may decrease augmentation variance')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    try:
        print('training files', len(train_dataset))
        print('validation files', len(val_dataset))
    except TypeError:
        print('val/test: ???')

    sample_input = next(iter(train_loader))


    input_shape  = sample_input['image'].shape
    output_shape = sample_input['label'].shape
    
    print('training 3D DenseNet with input size', input_shape)
    channels = input_shape[1]
    output_classes = 1 if not args.soft_labels else output_shape[-1]
    print('channels', channels, '    number of outputs', output_classes)


    if args.twoD:
        model = densenet2d.DenseNet(growth_rate=args.growth, 
                block_config=(6, 12, 6), channels=channels, 
                num_init_features=32,
                bn_size=4, drop_rate=args.droprate, num_classes=output_classes)
    else:
        # model = densenet3d.DenseNet(n_input_channels=channels,
        #          conv1_t_size=7,
        #          conv1_t_stride=1, # 2 ?
        #          no_max_pool=True,
        #          growth_rate=args.growth,
        #          block_config=(6, 12, 16),
        #          num_init_features=32,
        #          bn_size=4,
        #          drop_rate=args.droprate,
        #          num_classes=output_classes,
        #          softmax=args.soft_labels)
        # model = monai.networks.nets.DenseNet(3, 1, output_classes, 
        #         init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), bn_size=4, dropout_prob=args.droprate)
        #model = FCN3D(input_shape)
        model = custom_SFCN(sequence=args.sequence, dropout=args.droprate, output_dim=output_classes)


    model = torch.nn.DataParallel(model)


    model = model.cuda()

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))


    # -------------------- configure loss functions and optimizers -----------------------

    #loss_function = torch.nn.MSELoss()
    #loss_function = torch.nn.L1Loss()
    #loss_function = L1L2Loss()
    loss_function = my_KLDivLoss()

    #optimizer =  monai.optimizers.Novograd(model.parameters(), args.lr, weight_decay=args.weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    # optionally resume from a checkpoint
    if args.resume:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    for i in range(args.reruns):
        print('======================> starting run', i, 'of', args.reruns)
        args.run_id = i
        training_loop(args, train_loader, val_loader, model, loss_function, optimizer, writer, softlabel_helper, tensorboard_logdir) # send it

        # recreate model and optimizer
        model_class = model.module.__class__ # module unpacks the DataParallel wrapper
        del model
        model = model_class()
        model = torch.nn.DataParallel(model)
        model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


        # keep logging in same folder with incrasing epochs
        args.start_epoch += args.epochs



if __name__ == '__main__':
    main()
