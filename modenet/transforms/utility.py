"""
This file is for transformations, that do not provide new functionality,
but are provided for convienience.
"""

from typing import *

import torch

from monai.transforms.compose import Transform, MapTransform
from monai.config import KeysCollection


class ApplyFunction(MapTransform):

    def __init__(self, keys: KeysCollection, function) -> None:
        """
            Args:
                keys: keys of the corresponding items to be transformed.
                    See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)
        self.function = function

    def __call__(self, data: Mapping[Hashable, Union[torch.Tensor]]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.function(d[key])
        return d

class Identity(Transform):
    """
    Does not perform any operation. 
    Useful as a placeholder for constructing architecture-dependent augmentation.
    """

    def __call__(self, data):
        return data


class ValueToTensor(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`dlmi.transforms.ValueToTensor`.
    """

    def __init__(self, keys: KeysCollection, dtype=torch.float32) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)
        self.dtype = dtype
        #self.converter = ValueToTensor()

    def __call__(self, data: Mapping[Hashable, Union[int, float]]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = torch.tensor(d[key],dtype=self.dtype)
        return d

class SelectKey(MapTransform):

    def __init__(self, keys: KeysCollection) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)

    def __call__(self, data: Mapping[Hashable, Union[int, float]]) -> Dict[Hashable, torch.Tensor]:
        return data[self.keys]


class TensorToFloatTensor(MapTransform):

    def __init__(self, keys: KeysCollection) -> None:
        """
            Args:
                keys: keys of the corresponding items to be transformed.
                    See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)

    def __call__(self, data: Mapping[Hashable, Union[torch.Tensor]]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = d[key].float()
        return d


class TensorToDoubleTensor(MapTransform):

    def __init__(self, keys: KeysCollection) -> None:
        """
            Args:
                keys: keys of the corresponding items to be transformed.
                    See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)

    def __call__(self, data: Mapping[Hashable, Union[torch.Tensor]]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = d[key].double()
        return d


class TensorUnsqueeze(MapTransform):

    def __init__(self, keys: KeysCollection, dim: int) -> None:
        """
            Args:
                keys: keys of the corresponding items to be transformed.
                    See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)
        self.dim = dim

    def __call__(self, data: Mapping[Hashable, Union[torch.Tensor]]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = d[key].unsqueeze(self.dim)
        return d

class TensorSqueeze(MapTransform):

    def __init__(self, keys: KeysCollection, dim: int) -> None:
        """
            Args:
                keys: keys of the corresponding items to be transformed.
                    See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)
        self.dim = dim

    def __call__(self, data: Mapping[Hashable, Union[torch.Tensor]]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = d[key].squeeze(self.dim)
        return d



class TensorPermute(MapTransform):

    def __init__(self, keys: KeysCollection, dim: Union[int]) -> None:
        """
            Args:
                keys: keys of the corresponding items to be transformed.
                    See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)
        self.dim = dim

    def __call__(self, data: Mapping[Hashable, Union[torch.Tensor]]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = d[key].permute(*self.dim)
        return d


class Slice(object):
    """
    Retrieve a slice from an MRI image
    """

    def __init__(self, axes=0, slice_range=None):
        if isinstance(axes, tuple):
            def get_axis():
                yield axes[random.randint(0,len(axes))]
        else:
            assert(isinstance(axes, int))
            def get_axis():
                return axes

        self.get_axis = get_axis
        self.slice_range = slice_range

    def __call__(self, sample):
        ax = self.get_axis()

        if self.slice_range == None:
            slice_range = (0,sample['image'].shape[ax])
        else:
            slice_range = self.slice_range

        slice_no = random.randint(*slice_range)

        if ax == 0:
            sample['image'] = sample['image'][slice_no,:,:]
        elif ax == 1:
            sample['image'] = sample['image'][:,slice_no,:]
        elif ax ==2:
            sample['image'] = sample['image'][:,:,slice_no]
        else:
            raise ValueError('invalid slicing axis')


        return {**sample}


import numpy as np
from scipy.stats import norm

class ToSoftLabel(MapTransform):
    
    def __init__(self, keys: KeysCollection, backup_keys: KeysCollection, bin_range: tuple, bin_step: float, soft_label: bool=True, require_grad=False):
        """
        adapted from https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain/blob/master/dp_model/dp_utils.py

        v,bin_centers = number2vector(x,bin_range,bin_step,sigma)
        bin_range: (start, end), size-2 tuple
        bin_step: should be a divisor of |end-start|
        soft_label:True 'soft label', v is vector else 'hard label', v is index
        debug: True for error messages.
        """
        super().__init__(keys)
        if isinstance(backup_keys, tuple):
            self.backup_keys = backup_keys
        else:
            self.backup_keys = (backup_keys,)

        self.bin_start = bin_range[0]
        self.bin_end = bin_range[1]
        self.bin_length = self.bin_end - self.bin_start
        if not round(self.bin_length / bin_step,5) % 1 == 0:
            raise ValueError("bin's range should be divisible by bin_step!")
        
        #self.bin_range = bin_range
        self.bin_step  = bin_step
        self.soft_label = soft_label
        self.bin_number = int(round(self.bin_length / bin_step))
        self.bin_centers = self.bin_start + float(bin_step) / 2 + bin_step * np.arange(self.bin_number)

        if require_grad:
            self.bin_centers = torch.tensor(self.bin_centers, dtype=torch.float32)

    def __call__(self, data: Mapping[Hashable, Union[torch.Tensor]]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key, backup in zip(self.keys, self.backup_keys):
            d[backup] = d[key].clone()
            d[key] = self.valueToSoftlabel(d[key])

        return d

    def valueToSoftlabel(self, x):

        if torch.is_tensor(x):
            was_tensor = True
            x = x.squeeze().numpy()
            assert(len(x.shape) == 1 or len(x.shape) == 0)
            x = x.tolist()
        else:
            was_tensor = False

        if not self.soft_label:
            x = np.array(x)
            i = np.floor((x - self.bin_start) / self.bin_step)
            i = i.astype(int)
            return i if not was_tensor else torch.tensor(i)
        else:
            if np.isscalar(x):
                v = np.zeros((self.bin_number,))
                for i in range(self.bin_number):
                    x1 = self.bin_centers[i] - float(self.bin_step) / 2
                    x2 = self.bin_centers[i] + float(self.bin_step) / 2
                    cdfs = norm.cdf([x1, x2], loc=x, scale=self.bin_length*0.03) # TODO: test effects of sigma
                    v[i] = cdfs[1] - cdfs[0]
            else:
                v = np.zeros((len(x), self.bin_number))
                for j in range(len(x)):
                    for i in range(self.bin_number):
                        x1 = self.bin_centers[i] - float(self.bin_step) / 2
                        x2 = self.bin_centers[i] + float(self.bin_step) / 2
                        cdfs = norm.cdf([x1, x2], loc=x[j], scale=self.bin_length*0.03)
                        v[j, i] = cdfs[1] - cdfs[0]
        
            return v if not was_tensor else torch.tensor(v)
        
    def softLabelToHardLabel(self, x):
        
        if torch.sum(x) > 1.0:
            x = torch.nn.functional.log_softmax(x.squeeze())

        prob = torch.exp(x)
        pred = prob @ self.bin_centers

        return pred