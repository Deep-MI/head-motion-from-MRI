import os
import os
from datetime import datetime

import argparse
import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import torch
from torchvision.transforms import Compose


from modenet.transforms import utility
from modenet.logging import tensorboard
from modenet.dataset import hdf5_dataset
from modenet.dataset.dataloader import DataLoader
from modenet.logging import metrics, tensorboard
from modenet.models.FCN import custom_SFCN




def run_model_on_dataset(model, val_loader, PATCHING, softlabel_helper):
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


def run_analysis(run_dir, output_csv, DATA_DIR, FILE_PATHS_CSV, DENSE, IS_3D, PATCHING, PATCH_SIZE, SOFTLABEL, TESTSET=False, GT_PATH=None):
    
    checkpoint_filepath = run_dir

    if os.path.isfile(checkpoint_filepath):
        print("=> loading checkpoint '{}'".format(checkpoint_filepath))
        checkpoint = torch.load(checkpoint_filepath)

        hyperparameters = checkpoint['arch']

        print('=> was saved at epoch',checkpoint['epoch'])

        if GT_PATH is not None and os.path.isfile(GT_PATH):
            print(f'using {GT_PATH} instead of {hyperparameters.ground_truth}')
            hyperparameters.ground_truth = GT_PATH
        else:
            hyperparameters.ground_truth = None

        if not os.getcwd().startswith('/workspace/') and hyperparameters.ground_truth.startswith('/workspace/'):
            print('warning, loading data from private workspace')

        print('=> ground truth',hyperparameters.ground_truth)
        # transform_test = torch.load(os.path.join('/'.join(checkpoint_filepath.split('/')[:-1]), 'transforms_test.pt'))
        # softlabel_helper = torch.load(os.path.join('/'.join(checkpoint_filepath.split('/')[:-1]), 'softlabel_helper.pt'))

        transform_test = Compose([
            utility.ValueToTensor(['image'], dtype=torch.float32),
            utility.TensorToFloatTensor(['image','label']),
            utility.TensorUnsqueeze(['image'], 0)
        ])

        # softlabel constants
        softlabel_max = 0.39
        #softlabel_max = pd.read_csv(hyperparameters.ground_truth).max().values[1] + pd.read_csv(hyperparameters.ground_truth).max().values[1] * 0.005
        num_bins = 40
        step_per_bin = softlabel_max/(num_bins+1)
        softlabel_helper = utility.ToSoftLabel('label', 'hard_label',(0,num_bins*step_per_bin),step_per_bin, require_grad=True)

        
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


        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
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

        #model = torch.nn.DataParallel(model)

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
    input_dict, output_dict, score_keeper = run_model_on_dataset(model, val_loader, PATCHING, softlabel_helper)


    # #plot_output(input_dict, output_dict)    
    # print('best measured accuracy (R2) ', checkpoint['best_precision'])
    # print('measured accuracy      (R2) ', score_keeper.calculateR2())
    # print('measured accuracy      (spearmanR) ', score_keeper.calculateSpearmanR())
    # print('measured accuracy      (pearsonR) ', score_keeper.calculateSpearmanR())

    # input_dict['image'] = input_dict['image'].squeeze()

    # img1, img2 = tensorboard.saveInputOutput(0, input_dict['image'], output_dict['label'], input_dict['label'], prefix='val')

    # plt.imshow(np.invert(img1[0,:,:]), cmap='gray')
    # plt.imshow(np.swapaxes(np.swapaxes(img2,0,2),0,1))

    # plt.figure()
    # if len(score_keeper.output.squeeze().shape) > 1:
    #     plt.scatter(np.mean(score_keeper.output, axis=1), np.mean(score_keeper.target, axis=1))
    # else:
    #     plt.scatter(score_keeper.output, score_keeper.target)
    # plt.ylabel('ground truth')
    # plt.xlabel('prediction')
    # plt.show()


    output_df = pd.DataFrame(score_keeper.output,index=score_keeper.subjects, columns=['network_output'])
    output_df.index.name = 'subjectID'
    output_df.to_csv(output_csv)


def parse_args():
    parser = argparse.ArgumentParser(description='Image Analysis')
    parser.add_argument('-type', choices=['T1', 'T2', 'FLAIR'], help='Type of image (T1, T2, FLAIR)', required=True)
    parser.add_argument('-i', '--input', help='Path to a txt file with image paths or a path to a single image', required=True)
    parser.add_argument('-o', '--output', help='Path to the csv', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    image_type = args.type.upper()
    image_path = args.input

    if args.input.endswith('.nii.gz') or args.input.endswith('.nii'):
        # write text file with the path to the image
        with open('input.txt', 'w') as f:
            f.write(args.input)
        args.input = '../input.txt'

    # change working directory to the files directory
    if len(os.path.dirname(__file__)) > 0:
        os.chdir(os.path.dirname(__file__))


    # best on validation
    if image_type == 'T1':
        run_dir = os.path.join(os.getcwd(), '..', 'weights', 'best_t1.pth.tar')
    elif image_type == 'T2':
        run_dir = os.path.join(os.getcwd(), '..', 'weights', 'best_t2.pth.tar')
    elif image_type == 'FLAIR':
        run_dir = os.path.join(os.getcwd(), '..', 'weights', 'best_flair.pth.tar')

    DENSENET = False
    IS_3D = True
    PATCHING = False
    PATCH_SIZE = None
    SOFTLABEL = True
    #data_dir = '../data_new'
    #val_files = '../data/T1_test.csv'

    #gt_path = '../data/fMRI_optim_motion_avg_t1_drop.csv'


    print('Pytorch Version', torch.__version__)

    data_dir = '/tmp/tmp_hdf5s'
    os.makedirs(data_dir, exist_ok=True)
    run_analysis(run_dir=run_dir, output_csv=args.output, DATA_DIR=data_dir, FILE_PATHS_CSV=args.input, DENSE=DENSENET, IS_3D=IS_3D, PATCHING=PATCHING, PATCH_SIZE=PATCH_SIZE, SOFTLABEL=SOFTLABEL, TESTSET=False, GT_PATH=None)
    # remove the temporary directory with the hdf5 files
    print(f'removing temporary directory with hdf5 files {data_dir}')
    shutil.rmtree(data_dir)

