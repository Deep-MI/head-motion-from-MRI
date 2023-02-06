"""
functions specific to logging with tensorboard
"""

import math
import os
from datetime import datetime
import fnmatch
from shutil import copyfile


import cv2 # needs to be before pytorch?
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

def setup_tensorboard(run_name: str = 'traninig_run', logdir: str = 'tensorboard_logs') -> torch.utils.tensorboard.SummaryWriter:
    # make the name unique by adding the date and time
    if not os.path.isdir(logdir):
        print('creating logdir', os.path.abspath(logdir))
        os.mkdir(logdir)
    run_name = datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + '_' + run_name
    writer_dir = os.path.join(logdir, run_name)
    os.mkdir(writer_dir)
    print('created', writer_dir)
    writer = SummaryWriter(writer_dir)
    return writer, writer_dir


def create_backup(writer_dir, exclude, verbose=False):
    """
    copies of all python files in the root directory
    to the logging directory for future reference
    writer_dir      log directory (as returned by setup tensorboard)
    exclude         directories and files to exclude from backup
    """
    if not os.path.isdir(writer_dir):
        raise FileNotFoundError('logdir not found, make sure to run setup_tensorboard() before creating a backup')

    
    # determine relevant python code
    py_files = []
    exclude = set(exclude)
    for path, dirs, files in os.walk(os.path.abspath('.')):
        dirs[:] = [d for d in dirs if d not in exclude]
        for filename in fnmatch.filter(files, '*.py'):
            py_files.append(os.path.join(path, filename))

    #[f if not f.startsWith('/workspace') for f in py_files]
    os.mkdir(os.path.join(writer_dir, 'src'))

    # copy file strutcture from delveopement directory to log directory
    for f in py_files:
        target = os.path.join(writer_dir,'src','/'.join(f.split('/')[3:]))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if verbose:
            print('copying', target)
        copyfile(f, target)

    print('copied python files to', writer_dir)


# helper function to show an image
def matplotlib_imshow(ax, img, one_channel=False):
    #if one_channel:
    #    img = img.mean(dim=0)
    #img = (img + img.min()) / img.max()     # unnormalize
    img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))  # scale to range (0,1)
    npimg = img.numpy()
    if one_channel:
        return ax.imshow(npimg, cmap='gray')#, vmin=img.min(), vmax=img.max())
    else:
        return ax.imshow(np.transpose(npimg, (1, 2, 0)))


def saveInputOutput(epoch, images, output, target, summary_writer=None, prefix='train', mode='opencv', denormalizer=lambda x:x):
    #matplotlib.rcParams['savefig.pad_inches'] = 0
    # pick one slice
    SLICE   = 150
    SLICING_AXIS = 2 # still need to change code if you change this
    NROW    = 8
    PADDING = 2

    

    if len(images.shape) == 4:
        if SLICE > images.shape[SLICING_AXIS+1]: # probably operating in patch mode
            SLICE = int(round(images.shape[SLICING_AXIS+1]/2)) #np.random.randint(0,images.shape[SLICING_AXIS+1])
        grid_images = images[:,:,:,SLICE].unsqueeze(1)
    elif len(images.shape) == 3:
        grid_images = images.unsqueeze(1)
    else:
        raise NotImplementedError('unsupported image dimension', images.shape)



    if grid_images.shape[2] > 150:
        pretext_1 = 'Prediction: '
        pretext_2 = 'Ground Truth: '
        rounding_dec = 5
    elif grid_images.shape[2] > 100:
        pretext_1 = 'P: '
        pretext_2 = 'G: '
        rounding_dec = 5
    else:
        pretext_1 = ''
        pretext_2 = ''
        rounding_dec = 5

    grid = torchvision.utils.make_grid(grid_images, nrow = NROW, padding = PADDING, 
        normalize = False, value_range = (0,255), scale_each = True, pad_value = 0)

    
    grid = (grid - torch.min(grid)) / (torch.max(grid) - torch.min(grid))  # scale to range (0,1)
    grid = (grid * 255).int()  # convert to (0, 255) int and invert

    if summary_writer != None:
        summary_writer.add_image('%s/input_images' % prefix, np.uint8(grid), epoch)
    else:
        print('not writing to tensorboard - no summary_writer provided')


    if target != None and output != None:

        assert(len(output) == len(target))

        target = denormalizer(target)
        output = denormalizer(output)

        if mode == 'matplotlib':
            import matplotlib
            import matplotlib.pyplot as plt
            from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid

            images_batch = images[:,SLICE,:,:].cpu()
            batch_size = len(images_batch)
            im_height = images_batch.shape[-2]
            im_width = images_batch.shape[-2]

            cols = 8
            rows = math.ceil(batch_size/cols)

            dpi = matplotlib.rcParams['figure.dpi']
            imsize_inch = im_width / float(dpi), im_height / float(dpi)

            fig = plt.figure(figsize=(imsize_inch[0]*cols, imsize_inch[1]*rows), frameon=False, dpi=200)
            image_grid = ImageGrid(fig, 111,  # similar to subplot(111)
                             nrows_ncols=(rows, cols),
                             axes_pad=0.01,  # pad between axes in inch.
                             )
            # plot the images in the batch, along with predicted and true labels
            for ax, idx in zip(image_grid, range(batch_size)): # iterating over the grid returns the Axes.        
                matplotlib_imshow(ax, images_batch[idx], one_channel=True)
                # show a text, green for correct 
                ax.text(3,im_height-10,"output {0:.2f}, ground truth {1:.2f}".format(output[idx].item(), target[idx]),
                        color="green", fontdict={'fontsize': 6})
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.autoscale(enable=False, axis='both', tight=True)
            
            # remove white borders around plot
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                        hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            if summary_writer != None:
                summary_writer.add_figure('%s/predictions matplotlib' % prefix, fig, global_step=epoch)

        elif mode == 'opencv':


            image_size = np.array((images.shape[-2:][1], images.shape[-2:][0]))+PADDING
            grid_size = np.array((grid.shape[2], grid.shape[1]))
            #no_images = images.shape[0]
            cols, rows = np.round(grid_size/image_size).astype(int)

            #if cols < NROW:
            

            font                   = cv2.FONT_HERSHEY_SIMPLEX
            fontScale              = 0.5
            #fontColor              = (255,255,255)
            lineType               = 2
            thickness              = 2
            offset                 = np.array((10,25))

            grid = grid.numpy()
            #grid /= np.max()
            annotated_img = cv2.UMat(np.uint8(np.transpose(grid,(1,2,0))))

            for r in range(rows):
                for c in range(cols):
                    idx = r*cols+c

                    annotated_img = cv2.putText(annotated_img, pretext_1 + str(np.round(torch.mean(output[idx]).item(),rounding_dec)), 
                        org=(image_size[0]*c+offset[0], image_size[1]*r+offset[1]), color=(3, 248, 252), bottomLeftOrigin=False,
                        fontFace=font, fontScale=fontScale, thickness=thickness, lineType=lineType)
                    annotated_img = cv2.putText(annotated_img, pretext_2 + str(np.round(torch.mean(target[idx]).item(),rounding_dec)), 
                        org=(image_size[0]*c+offset[0], image_size[1]*r+offset[1] +15), color=(129, 245, 66), bottomLeftOrigin=False,
                        fontFace=font, fontScale=fontScale, thickness=thickness, lineType=lineType)

                    if idx >= len(output)-1:
                        break

            # shape (3, 454, 2578)
            annotated_img = np.transpose(annotated_img.get(), (2,0,1))

            

            if summary_writer != None:
                summary_writer.add_image('%s/predictions opencv' % prefix, annotated_img, epoch)

        return grid, annotated_img
    else:
        return grid