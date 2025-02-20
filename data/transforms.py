"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
import torch.fft
import matplotlib.pyplot as plt
import h5py
from .subsample import create_mask_for_mask_type


def rss(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS).

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform

    Returns:
        torch.Tensor: The RSS value.
    """
    return torch.sqrt((data ** 2).sum(dim))


def apply_mask(data_shape, mask_func, seed=None, padding=None):
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data_shape : The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.
        padding

    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """
    shape = np.array(data_shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed)
    if padding is not None:
        mask[:, :, : padding[0]] = 0
        mask[:, :, padding[1]:] = 0  # padding value inclusive on right of zeros

    # masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return mask


def normalize(data, mean, stddev, eps=0.0):
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data (torch.Tensor): Input data to be normalized.
        mean (float): Mean value.
        stddev (float): Standard deviation.
        eps (float, default=0.0): Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_zero_to_one(data, eps=0.):
    data_min = float(data.min())
    data_max = float(data.max())
    return (data - data_min) / (data_max - data_min + eps)


def normalize_instance(data, eps=0.0):  # 1e-11
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data (torch.Tensor): Input data to be normalized
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std


class ReconstructionTransform(object):
    """
       Data Transformer for training Rec models.
       """

    def __init__(self, which_challenge, mask_func=None, use_seed=True, mask_path=None):
        """
        Args:
            which_challenge (str): Either "singlecoil" or "multicoil" denoting
                the dataset.
            mask_func (fastmri.data.subsample.MaskFunc): A function that can
                create a mask of appropriate shape.
            use_seed (bool): If true, this class computes a pseudo random
                number generator seed from the filename. This ensures that the
                same mask is used for all the slices of a given volume every
                time.
            mask_path: test_mask
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.mask_path = mask_path

    def __call__(self, target, fname, slice_num, is_complement=False):
        """
        Args:
            target (numpy.array): Target image.
            fname (str): File name.
            slice_num (int): Serial number of the slice.
            is_complement: is complement image

        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch
                    Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                fname (str): File name.
                slice_num (int): Serial number of the slice.
        """
        target = torch.from_numpy(target).float()  # (640, 372, 2)
        if is_complement:
            return {'target': target}
        kspace = torch.fft.fftn(target)
        kspace_shape = [kspace.shape[0], kspace.shape[1], 2]

        # apply mask  train-use_seed=false
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            mask = apply_mask(kspace_shape, self.mask_func, seed)
            mask = mask.squeeze(-1)

        else:
            with h5py.File(self.mask_path, "r") as hf:
                mask = hf['mask'][0]
                mask = torch.from_numpy(mask)

        # inverse Fourier transform to get zero filled solution
        kspace = torch.fft.fftshift(kspace)
        kspace = kspace * mask + 0.0
        kspace = torch.fft.ifftshift(kspace)
        image = torch.fft.ifftn(kspace)
        image = torch.abs(image)

        image = normalize_zero_to_one(image, eps=1e-6)
        return {'under_img': image, 'target': target, 'mask': mask, 'file_name': fname, 'slice_num': slice_num}


def build_transforms(args, mode='train'):
    if mode == 'train':
        mask = create_mask_for_mask_type(
            args.TRANSFORMS.MASKTYPE, args.TRANSFORMS.CENTER_FRACTIONS, args.TRANSFORMS.ACCELERATIONS,
        )
        return ReconstructionTransform(args.DATASET.CHALLENGE, mask, use_seed=False)
    elif mode == 'val':
        mask = create_mask_for_mask_type(
            args.TRANSFORMS.MASKTYPE, args.TRANSFORMS.CENTER_FRACTIONS, args.TRANSFORMS.ACCELERATIONS,
        )
        return ReconstructionTransform(args.DATASET.CHALLENGE, mask)
    else:
        return ReconstructionTransform(args.DATASET.CHALLENGE, mask_path=args.TRANSFORMS.MASK_PATH)

