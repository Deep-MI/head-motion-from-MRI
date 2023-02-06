"""
image operations that standardize the input
"""

from modenet.utils import fourier
from monai.transforms.compose import Transform, MapTransform
import skimage
import torch
import torchvision
import numpy as np

from modenet.transforms.conform import rescale

class FourierTransform3d(MapTransform):
    """
    Applies Fourier transform to 3d image
    """

    def __init__(self, convert_tensor=False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        #super().__init__(keys)
        self.convert_tensor = convert_tensor

    def __call__(self, data: dict) -> dict:
        #d = dict(data)
        #for key in self.keys:
        if isinstance(data['image'], np.ndarray):
            data['image'] = torch.from_numpy(data['image'])
            was_numpy = True

        data['image'] = fourier.fourier_transform(data['image']).real

        if not self.convert_tensor and was_numpy:
            data['image'] = data['image'].numpy()

        return data


class Normalize(object):
    """Normalize images in sample with fixed mean and standard deviation"""
    def __init__(self, mean, std):
        # self.transform = transforms.Normalize(mean, std)
        def normalize_tensor(t):
            return (t - mean)/std
        self.transform = normalize_tensor
    
    def __call__(self, sample):
        sample['image'] = self.transform(sample['image']) 
        return sample

class Equalize(object):
    """Do histogram equalization on the image in the sample dict"""

    def __init__(self, batch_dim=None):
        self.batch_dim = batch_dim

    def __call__(self, sample):

        if self.batch_dim is not None:
            # Equalization
            if sample['image'].shape[0] == 1:
                sample['image'] = skimage.exposure.equalize_hist(sample['image'])
            else:
                for i in range(sample['image'].shape[0]):
                    sample['image'][i] = skimage.exposure.equalize_hist(sample['image'][i])

        else:
            sample['image'] = skimage.exposure.equalize_hist(sample['image'])
        #sample['image'] = exposure.equalize_adapthist(sample['image'], clip_limit=0.03)

        # adjust the mean to zero and sttdev to 1
        #sample['image'] = (sample['image'] - np.mean(sample['image']))/np.std(sample['image'])
        return sample


class RobustScale(object):
    """
    Do contrast stetching on the image in the sample dict

        Args:
            perc (tuple
            No): percentiles of intensities to be discarded
    """
    def __init__(self, perc, batch_dim=None):
        self.perc = perc
        self.batch_dim = batch_dim

    def __call__(self, sample):

        if self.batch_dim is not None:
            assert(self.batch_dim == 0)
            for i in range(sample['image'].shape[0]):
                sample['image'][i] = rescale(sample['image'][i], 0, 255, f_low=0.0, f_high=self.perc)
                
                # adjust the mean to zero and sttdev to 1
                #sample['image'][i] = (img - np.mean(img))/np.std(img)
        else:
            sample['image'] = rescale(sample['image'], 0, 255, f_low=0.0, f_high=self.perc)
        return sample

class HighPassFilterGauss():
    """
    Apply high pass filter to the image in the sample dict

        Args:
            sigma (float): sigma of the gaussian filter
    """
    def __init__(self, sigma, batch_dim=None):
        self.sigma = sigma
        self.batch_dim = batch_dim

    def __call__(self, sample):
        if self.batch_dim is not None:
            assert(self.batch_dim == 0)
            for i in range(sample['image'].shape[0]):
                sample['image'][i] = skimage.filters.gaussian(sample['image'][i], sigma=self.sigma)
        else:
            sample['image'] = skimage.filters.gaussian(sample['image'], sigma=self.sigma)

        #for i in range(sample['image'].shape[0]):
        #    sample['image'][i] = skimage.filters.gaussian(sample['image'][i], sigma=self.sigma)
        return sample


class ButterWorthFilter():
    """
    Apply butterworth filter to the image in the sample dict

        Args:
            freq (float): cutoff frequency of the filter
            highpass (bool): if True, high pass filter is applied
            order (int): order of the filter
    
    """
    def __init__(self, freq, highpass=True, order=4, batch_dim=None):
        self.order = order
        self.freq = freq
        self.highpass = highpass
        self.batch_dim = batch_dim

    def __call__(self, sample):

        if self.batch_dim is not None:
            assert(self.batch_dim == 0)
            for i in range(sample['image'].shape[0]):
                #img = sample['image'][i]
                sample['image'][i] = skimage.filters.butterworth(sample['image'][i], cutoff_frequency_ratio=self.freq, high_pass=self.highpass, order=self.order)
                # adjust the mean to zero and sttdev to 1
                #sample['image'][i] = (img - np.mean(img))/np.std(img)
        else:
            sample['image'] = skimage.filters.butterworth(sample['image'], cutoff_frequency_ratio=self.freq, high_pass=self.highpass, order=self.order)

        #sample['image'] = skimage.filters.butterworth(sample['image'], cutoff_frequency_ratio=self.freq, high_pass=self.highpass, order=self.order)
        return sample

class TresholdImage():
    """
    Apply tresholding to the image in the sample dict

        Args:
            treshold (float): treshold value
    """
    def __init__(self, threshold, absolute=True):
        self.threshold = threshold
        self.absolute = absolute

    def __call__(self, sample):
        #sample['image'] = sample['image'] > self.treshold

        if self.absolute:
            sample['image'] = np.clip(sample['image'], a_min=None, a_max=self.threshold)
        else:
            for i in range(sample['image'].shape[0]):
                sample['image'][i] = np.clip(sample['image'][i], a_min=None, a_max=self.threshold*np.max(sample['image'][i]))
        # np.clip
        return sample


class TorchEqualize():
    """Do histogram equalization on the image in the sample dict"""

    def __init__(self, return_tensor=False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        #super().__init__(keys)
        self.return_tensor = return_tensor

    def __call__(self, data):
        if isinstance(data['image'], np.ndarray):
            data['image'] = torch.from_numpy(data['image'])
            was_numpy = True

        data['image'] = torchvision.transforms.functional.equalize(data['image'])

        if not self.return_tensor and was_numpy:
            data['image'] = data['image'].numpy()

        return data
