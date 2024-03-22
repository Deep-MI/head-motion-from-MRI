from glob import glob
import os
import time
import sys
from collections import OrderedDict

import torch
from torch.utils.data import Dataset
from torchvision import utils

import pandas as pd
import numpy as np
import h5py
import nibabel as nib
import random
import pickle


"""
loads labels from csv with

filepath  path to csv file
"""
def loadMotionData(filepath):
    #FILEPATH = './dataset/smoothed_motion_avgs.csv'

    if not os.path.isfile(filepath):
        #filepath = filepath
        if not os.path.isfile(filepath):
            raise FileNotFoundError('motion ground truth not found in %s' % os.path.abspath(filepath))

    if filepath.endswith('.csv'):
        motion_avgs = pd.read_csv(filepath)
    elif filepath.endswith('.json'):
        motion_avgs = pd.read_json(filepath)
        motion_avgs = motion_avgs.T
        motion_avgs = pd.DataFrame(motion_avgs.apply(lambda x: np.vstack(x.apply(lambda x: np.array(x))), axis=1))
        motion_avgs.index = motion_avgs.index.rename('subjects')
    else:
        print('unknown ground truth type')

    return motion_avgs



"""

loads desired dataset from static datadet folder
an hdf5 file is created as cache and used if it already exists

augments      data augmentation class instances
dataset_name  folder name in static dataset folder, can also be list of multiple folders
csv_file      path to csv with ground truth labels
training      (bool) whether this a training set
pattern       re

"""
#def loadDataset(augments, dataset_name, csv_file, training, pattern='*', image_name='T1_RMS.nii.gz', debug=False, torchio=False, external_normalizer=None, patch_size=None):
def loadDataset(augments, dataset_name, dataset_folder, ground_truth_csv, training, path_csv, debug=False, torchio=False, external_normalizer=None, no_files=None):
    '''
    loads dataset from hdf5 and creates it if it wasn't created for those input parameters
    '''
    #DATASET_FOLDER = './scripts/rhineland/data/'

    # handle single input
    if not isinstance(dataset_name, list):
        dataset_name = [dataset_name]


    # param_dict = OrderedDict([('dataset_name', dataset_name), ('pattern', pattern), 
    #                           ('image_name', image_name), ('debug', debug), ('dataset_folder', DATASET_FOLDER)])
    param_dict = OrderedDict([('dataset_name', dataset_name), ('debug', debug), ('path_csv', path_csv), ('dataset_folder', dataset_folder)])
    hdf5 = Hdf5Handler(param_dict)

    if hdf5.exists():
       image_data = hdf5.load_hdf5_dataset(no_files)
    else:
        hdf5.write_hdf5_dataset(is_small=False)
        image_data = hdf5.load_hdf5_dataset(no_files)

    if ground_truth_csv is not None:
        motion_data = loadMotionData(ground_truth_csv)
    else:
        motion_data = None

    if torchio:
        #return TorchIOMotionDataset(image_data, motion_data, normalize=False, transforms=augments, external_normalizer=external_normalizer, training=training, patch_size=patch_size)
        raise NotImplementedError('')
    else:
        return MRIMotionDataset(image_data, motion_data, normalize=False, transforms=augments)


"""
    Creates dataset from MRI images and value vector, by matching the subject keys.
    The dataset can be automatically normalized.
"""
class MRIMotionDataset(Dataset):
    

    def __init__(self, images, labels, transforms, normalize=True, add_synthetic_data=None):
        """
        loads and normalizes images (usually from hdf5 file) 
        and values (from csv as pandas data frame)

        images: dictionary-like with keys 'images' and 'subject'
                should contain lists of mri images and subject names
        labels: either pandas data frame with subjects as index or 
                a numpy array of labels
        transforms: pytorch style transformations
        normalize: whether to automatically normalize the labels
        add_synthetic data: float that indicates the ratio of synthetic to real world data in the training dataset
        """
        self.images = images['images']
        self.subjects = images['subjects'].astype('str')
        self.count = self.images.shape[0]
        self.transforms = transforms

        if type(labels) == pd.core.frame.DataFrame:
            if not labels.index.name == 'subjects':
                labels = labels.set_index('subjects')
            # if labels.shape[0] > labels.shape[1] or (
            #         (labels.shape[0] > 1000 or labels.shape[1] > 1000) and 
            #         labels.shape[0] < labels.shape[1]):
            labels = labels.T

            
            if normalize:
                self.stddev = labels.values.std(ddof=1)
                self.mean = labels.values.mean()
                self.is_normalized = True

                self.normalizer = lambda x: (x - self.mean) / self.stddev
                self.denormalizer = lambda x: (x * self.stddev) + self.mean

                labels = self.normalizer(labels)
            else:
                self.stddev = 0
                self.mean = 0
                self.is_normalized = False
                self.normalizer = None
                self.denormalizer = None

            keys = labels.keys().to_list()
            labels = labels.to_numpy().squeeze().T
            
            not_found = []
            self.labels = []

            

            for i in range(self.count):
                try:
                    self.labels.append(labels[keys.index(self.subjects[i])])
                except ValueError:
                    not_found.append(i)

            print('no motion trace for', len(not_found), 'subjects')
            # delete subjects that are not found to sync indices
            self.images = np.delete(self.images,not_found, axis=0)
            self.subjects = np.delete(self.subjects,not_found, axis=0)

            


            self.count = self.images.shape[0]
            self.labels = torch.from_numpy(np.array(self.labels))

            self.shape = (self.images[0].shape, self.labels[0].shape)

        elif isinstance(labels, np.ndarray):

            # assert(isinstance(labels[1], np.ndarray))
            # assert(isinstance(labels[0], list))

            # subjects = labels[0]
            # labels = labels[1]

            # if labels.shape[0] > labels.shape[1]:
            #     labels = labels.T


            assert(self.images.shape[0] == labels.shape[0])
            assert(self.images.shape[0] == len(self.subjects))

            self.stddev = labels.std(ddof=1)
            self.mean = labels.mean()
            if normalize:
                self.is_normalized = True

                self.normalizer = lambda x: (x - self.mean) / self.stddev
                self.denormalizer = lambda x: (x * self.stddev) + self.mean

                labels = self.normalizer(labels)
            else:
                self.is_normalized = False
                self.normalizer = None
                self.denormalizer = None

            #self.images = images
            self.labels = labels.squeeze()
            #self.subjects = subjects
        elif labels is None:
            assert(self.images.shape[0] == len(self.subjects))
            self.is_normalized = False
            self.normalizer = None
            self.denormalizer = None
            self.labels = torch.zeros(self.images.shape[0])

        else:
            raise ValueError('unknown input type for parameter "labels"', type(labels))


    def normalize(self):
        assert(not self.is_normalized)
        self.labels = self.normalizer(self.labels)
        self.is_normalized = True

    def get_subject_names(self):
        return self.subjects

    def set_normalizer(self, normalizer, denormalizer):
        self.normalizer = normalizer
        self.denormalizer = denormalizer
        self.normalize()

    def calc_normalizer(self):
        self.normalizer   = lambda x: (x - self.mean) / self.stddev
        self.denormalizer = lambda x: (x * self.stddev) + self.mean
        self.normalize()

    def get_normalizer(self):
        if not self.is_normalized:
            print('no normalizer')
            return lambda x: x

        return self.normalizer

    def get_denormalizer(self):
        if not self.is_normalized:
            print('no denormalizer')
            return lambda x: x

        return self.denormalizer

    def __getitem__(self, index):
        img = self.images[index].copy()
        label = self.labels[index]
        subject = self.subjects[index]

        if self.transforms is not None:
            tx_sample = self.transforms({'image': img, 'label': label})
            #img = tx_sample['image']
            #label = tx_sample['label']
        else:
            tx_sample = {'image': img, 'label': label}

        return {**tx_sample, 'subject': subject}


    # equal to __getitem__, but it allows to get not normalized values
    def get(self, index, normalized=False):

        img = self.images[index].copy()
        label = self.labels[index]
        subject = self.subjects[index]

        if self.transforms is not None:
            tx_sample = self.transforms({'image': img, 'label': label})
            img = tx_sample['image']
            label = tx_sample['label']

        if not normalized:
            label = self.denormalizer(label)

        return {'image': img, 'label': label, 'subject': subject, 'subject_index': index}

    def __len__(self):
        return self.count



class Hdf5Handler:
    """
    Class to load all images in a directory into a hdf5-file and load it up again.
    """

    def __init__(self, params, scale_img_minmax=(-1., 1.)):
        # self.scale_img_min = scale_img_minmax[0]
        # self.scale_img_max = scale_img_minmax[1]

        self.dataset_name = params['dataset_name']
        self.dataset_folder = params['dataset_folder']
        #self.data_path = params["data_path"]
        # self.orig_name = params["image_name"]
        self._filename = os.path.join(self.dataset_folder, '_'.join(self.dataset_name) + '.hdf5')

        # self.search_pattern = [os.path.join(params['dataset_folder'], i, params["pattern"]) for i in self.dataset_name]
        # self.subject_dirs = [glob(i) for i in self.search_pattern]

        self.subject_dirs = np.genfromtxt(params['path_csv'],dtype='str')
        # check if subjects dirs is scalar
        if self.subject_dirs.shape == ():
            self.subject_dirs = np.array([self.subject_dirs])

        #flatten = lambda l: [item for sublist in l for item in sublist]
        #self.subject_dirs = flatten(self.subject_dirs)
            
        print(self.subject_dirs.shape)

        self.data_set_size = len(self.subject_dirs)


    def load_hdf5_dataset(self, no_files=None):
        # Open file in reading mode
        with h5py.File(self._filename, "r") as hf:
            if isinstance(no_files, int):
                self.images = hf.get('images')[:no_files+1]
                self.subjects = hf.get('subject')[:no_files+1]
            else:
                self.images = hf.get('images')[:]
                self.subjects = hf.get('subject')[:]



        print("Successfully loaded {}".format(self._filename))
        return {'images': self.images, 'subjects': self.subjects}

    def exists(self):
        return os.path.isfile(self._filename)

    @property
    def filename(self):
        return self._filename


    def write_hdf5_dataset(self, is_small=False):
        """
        Function to store all images in a given directory (or pattern) in a hdf5-file.
        """
        start_d = time.time()

        # load motion data from pickles first
        #motion_avgs = loadMotionData(self.motion_csv)

        # Prepare arrays to hold the data
        try:
            nib_img = nib.load(self.subject_dirs[0])
        except IndexError:
            raise FileNotFoundError('couldnt find any files in ' + self.search_pattern)

        if is_small:
            self.subject_dirs = self.subject_dirs[:3]

        orig_dataset = np.ndarray(shape=(len(self.subject_dirs), *nib_img.shape), dtype=nib_img.get_data_dtype())
        #motion_dataset = np.ndarray(shape=(len(self.subject_dirs)), dtype=np.float32)
        subjects = []

        #motion_subjects = list(motion_avgs.keys())

        # Loop over all subjects and load orig, aseg and create the weights
        for idx, current_subject in enumerate(self.subject_dirs):

            # if current_subject.split('/')[-1] not in motion_subjects:
            #     print('skipping', current_subject, '- no motion data found')
            #     continue


            start = time.time()

            print("Volume Nr: {} Processing MRI Data from {}".format(idx, current_subject))

            # Load image
            try:
                orig = np.asanyarray(nib.load(current_subject).dataobj)
            except Exception as e:
                print(f"Volume: {idx} Failed Reading Data. Error: {e}")
                print(f"this could be due to incorrect permissions for the file, try running the container as root")
                continue

            if orig.shape == (240, 320, 320):  # T2 version 2.0 with larger grid
                print(f"cropping subj {current_subject} to 224x320x320 (original size: {orig.shape})")
                D, H, W = orig.shape
                d_D = D - 224
                d_H = H - 320
                d_W = W - 320
                orig = orig[d_D // 2 : 224 + d_D // 2, d_H // 2 : 320 + d_H // 2, d_W // 2 : 320 + d_W // 2]
            
            
            # orig = nib.load(os.path.join(current_subject, self.orig_name))
            #src_min, scale = conform.getscale(orig.get_fdata(), dst_min=0, dst_max=1, f_low=0.0, f_high=0.999)
            #orig = conform.scalecrop(orig.get_fdata(), dst_min=0, dst_max=1, src_min=src_min, scale=scale)

            #map_image = map_image(orig.get_data(), out_affine, out_shape)
            #print(orig.shape)
            # Append finally processed images to arrays
            orig = np.expand_dims(orig, axis=0)
            #orig_dataset = np.append(orig_dataset, orig, axis=0)
            orig_dataset[idx] = orig


            sub_name = current_subject.split("/")[-2]


            subjects.append(sub_name.encode("ascii", "ignore"))
            #subjects[idx] = sub_name.encode("ascii", "ignore")

            #motion_dataset = np.append(motion_dataset, motion_avgs[sub_name])

            end = time.time() - start

            print("Volume: {} Finished Data Reading and Appending in {:.3f} seconds.".format(idx, end))

            if is_small and idx == 2:
                break

            #except Exception as e:
            #    print("Volume: {} Failed Reading Data. Error: {}".format(idx, e))
            #    continue

        # Write the hdf5 file
        with h5py.File(self._filename, "w") as hf:
            dt = h5py.special_dtype(vlen=str)
            print('writing hdf5 data to', self._filename)
            hf.create_dataset("subject", data=subjects, dtype=dt, compression='lzf')
            hf.create_dataset('images', data=orig_dataset, shape=orig_dataset.shape, dtype=np.float32, compression='lzf', chunks=True)
            #hf.create_dataset('motion', data=motion_dataset, compression='gzip')
            

        end_d = time.time() - start_d
        print("Successfully written {} in {:.3f} seconds.".format(self.dataset_name, end_d))


