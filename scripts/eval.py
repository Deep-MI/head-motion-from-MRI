import context


import os
from itertools import product
import re
#import argparse
from argparse import Namespace
import os
#import shutil
#import time
from datetime import datetime
#import json
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid







import cv2 # this has to come before pytorch import to avoid collision
import torch
from torch import nn
#from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import Compose

from modenet.logging import tensorboard
from modenet.transforms import utility, augmentation, regularization
from modenet.utils.training import training_loop
from modenet.dataset import hdf5_dataset, patching
from modenet.dataset.dataloader import DataLoader
from modenet.logging import metrics, checkpointing, tensorboard
from modenet.models.FCN import SFCN, FCN2D, FCN3D, custom_SFCN
from modenet.models import densenet3d

from medcam import medcam
import captum.attr as capt


def run_model_on_dataset(model, val_loader, denormalizer, IS_3D, PATCHING, softlabel_helper):
    """
    takes input model and data loader and 
    returns all model inputs and outputs in
    two dictionaries
    """
    initialized = False
    score_keeper = metrics.ScoreKeeper()

    if softlabel_helper is not None:
        SOFTLABEL = True
    else:
        SOFTLABEL = False

    if PATCHING:
        prep_image = lambda x: x['image']['data'].float()#.cuda()
    else:
        prep_image = lambda x: x['image']#.cuda()

    model = model.cuda().eval()

    with torch.no_grad():


        for i_batch, sample in enumerate(val_loader):
            if not initialized:
                images = prep_image(sample).clone() #['image'].float().clone()
                

                labels = sample['label'].float()
                if len(labels.shape) == 1:
                    labels = labels.unsqueeze(1)
                outputs_raw = model(images.cuda()).cpu()
                if SOFTLABEL:
                    outputs = softlabel_helper.softLabelToHardLabel(outputs_raw).unsqueeze(0)
                outputs = outputs.T

                #outputs = denormalizer(outputs)

                print('outputs shape: ', outputs.shape)
                print('labels shape: ', labels.shape)
                score_keeper.update(outputs, labels, sample['subject'])
                
                initialized = True
            else:
                images = torch.cat((images, prep_image(sample)))#.clone()

                if len(sample['label'].shape) == 1:
                    sample['label'] = sample['label'].unsqueeze(1)
                # elif len(labels.shape) == 0:
                #     labels = labels.unsqueeze(0).unsqueeze(0)

                


                labels = torch.cat((labels,  sample['label'].float()))
                
                output_raw = model(prep_image(sample).cuda()).cpu()

                if SOFTLABEL:
                    output = softlabel_helper.softLabelToHardLabel(output_raw).unsqueeze(0)
                #output = denormalizer(output)
                output = output.T
                score_keeper.update(output, sample['label'], sample['subject'])
                outputs = torch.cat((outputs, output))
                outputs_raw = torch.cat((outputs_raw, output_raw))

            print('evaluated ', i_batch)

        input_dict = {'image':images, 'label':labels}
        output_dict = {'label':outputs, 'raw_out': outputs_raw}
    return input_dict, output_dict, score_keeper


def run_analysis(run_dir, DATA_DIR, FILE_PATHS_CSV, DENSE, IS_3D, PATCHING, PATCH_SIZE, SOFTLABEL, TESTSET=False):
    
    checkpoint_filepath = run_dir

    if os.path.isfile(checkpoint_filepath):
        print("=> loading checkpoint '{}'".format(checkpoint_filepath))
        checkpoint = torch.load(checkpoint_filepath)

        hyperparameters = checkpoint['arch']

        print('=> was saved at epoch',checkpoint['epoch'])

        if not os.getcwd().startswith('/workspace/') and hyperparameters.ground_truth.startswith('/workspace/'):
            hyperparameters.ground_truth = hyperparameters.ground_truth.replace('/workspace/','/home/pollakc/MoDe-Net/')
            print('warning, loading data from private workspace')

        print('=> ground truth',hyperparameters.ground_truth)
        transform_test = torch.load(os.path.join('/'.join(checkpoint_filepath.split('/')[:-1]), 'transforms_test.pt'))
        softlabel_helper = torch.load(os.path.join('/'.join(checkpoint_filepath.split('/')[:-1]), 'softlabel_helper.pt'))

        
        if PATCHING:
            val_dataset = hdf5_dataset.loadDataset(transform_test, csv_file=hyperparameters.ground_truth, dataset_name=DATA_DIR, torchio=True, training=False, patch_size=PATCH_SIZE)
        else:
            # train_dataset   = hdf5_dataset.loadDataset(transform_test,  path_csv=FILE_PATHS_CSV.replace('val','train'),no_files=None,
            #     ground_truth_csv=hyperparameters.ground_truth, dataset_name='training_files_' + hyperparameters.sequence, 
            #     dataset_folder=DATA_DIR, training=False)
            if not TESTSET:
                val_dataset   = hdf5_dataset.loadDataset(transform_test,  path_csv=FILE_PATHS_CSV,no_files=None,
                    ground_truth_csv=hyperparameters.ground_truth, dataset_name='validation_files_' + hyperparameters.sequence, 
                    dataset_folder=DATA_DIR, training=False)
            else:
                val_dataset  = hdf5_dataset.loadDataset(transform_test,  path_csv=FILE_PATHS_CSV.replace('val', 'test'),no_files=None,
                    ground_truth_csv=hyperparameters.ground_truth, dataset_name='test_files_' + hyperparameters.sequence, 
                    dataset_folder=DATA_DIR, training=False)


        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=8, drop_last=False)
        sample = iter(val_loader).next()

        
        if PATCHING:
            sample_input = sample['image']['data'].float()

        sample_input = sample['image']
        sample_output = sample['label']

        input_shape = sample_input.shape
        

        channels = sample_input.shape[1]
        output_classes = 40

        print('input image shape', input_shape)
        print('channels', channels)
        print('classes', output_classes)

        if DENSE:
            if IS_3D:
                model = DenseNet3D.DenseNet(n_input_channels=channels,
                     conv1_t_size=7,
                     conv1_t_stride=1,
                     no_max_pool=True,
                     growth_rate=hyperparameters.growth,
                     block_config=(6, 12, 16),
                     num_init_features=32,
                     bn_size=4,
                     drop_rate=0,
                     num_classes=1)
            else:
                model = DenseNet.DenseNet(growth_rate=hyperparameters.growth, 
                         block_config=(6, 12, 24, 16), channels=channels,
                         num_init_features=128,
                         bn_size=4, drop_rate=hyperparameters.droprate, num_classes=output_classes)
        else:
            model = custom_SFCN(sequence=hyperparameters.sequence, dropout=hyperparameters.droprate, output_dim=output_classes)

        model = torch.nn.DataParallel(model)

        # get the number of model parameters
        print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])))

        model.load_state_dict(checkpoint['state_dict'])

        print("=> loaded checkpoint '{}' (epoch {}) from model {}"
              .format(checkpoint_filepath, checkpoint['epoch'], hyperparameters.name))

        model = model.eval()

    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_filepath))
        import sys
        sys.exit(0)



    print('checking ',len(val_dataset),'samples')
    input_dict, output_dict, score_keeper = run_model_on_dataset(model, val_loader, None, IS_3D, PATCHING, softlabel_helper)


    #plot_output(input_dict, output_dict)    
    print('best measured accuracy (R2) ', checkpoint['best_precision'])
    print('measured accuracy      (R2) ', score_keeper.calculateR2())
    print('measured accuracy      (spearmanR) ', score_keeper.calculateSpearmanR())
    print('measured accuracy      (pearsonR) ', score_keeper.calculateSpearmanR())

    input_dict['image'] = input_dict['image'].squeeze()

    img1, img2 = tensorboard.saveInputOutput(0, input_dict['image'], output_dict['label'], input_dict['label'], prefix='val')

    plt.imshow(np.invert(img1[0,:,:]), cmap='gray')
    plt.imshow(np.swapaxes(np.swapaxes(img2,0,2),0,1))

    plt.figure()
    if len(score_keeper.output.squeeze().shape) > 1:
        plt.scatter(np.mean(score_keeper.output, axis=1), np.mean(score_keeper.target, axis=1))
    else:
        plt.scatter(score_keeper.output, score_keeper.target)
    plt.ylabel('ground truth')
    plt.xlabel('prediction')
    plt.show()

    SAVE_CSV = False
    if SAVE_CSV:
        #with open('/workspace/network_output_testset_2.pickle', 'wb') as f:
        #    pickle.dump({'output' : score_keeper.output, 'target' : score_keeper.target},f)

        pd.DataFrame(score_keeper.output,index=score_keeper.subjects, columns=['network_output']).to_csv('/workspace/cfcn_t1_output_valset.csv')



if __name__ == '__main__':
    # change working directory to the files directory
    if len(os.path.dirname(__file__)) > 0:
        os.chdir(os.path.dirname(__file__))


    # best on validation;   testset- 0.4329353048642992
    run_dir = '../../tensorboard_logs/2022-10-14-14:49:40_sfcn_kld_t1ns_csfcn_test/model_best.pth.tar'
    

    DENSE = False
    IS_3D = True
    PATCHING = False
    PATCH_SIZE = None
    SOFTLABEL = True
    data_dir = './data_2022_nostrat'
    val_files = './data_2022_nostrat/T1_val.csv'


    print('Pytorch Version', torch.__version__)

    run_analysis(run_dir, data_dir, val_files, DENSE, IS_3D, PATCHING, PATCH_SIZE, SOFTLABEL, TESTSET=True)
