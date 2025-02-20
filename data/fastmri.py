import csv
import os
import cv2
import random
import h5py
from torch.utils.data import Dataset
from .transforms import build_transforms, normalize_zero_to_one


def center_crop(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. It
            should have at least 3 dimensions and the cropping is applied along
            dimensions -3 and -2 and the last dimensions should have a size of
            2.
        shape (tuple): The output shape. The shape should be smaller than
            the corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """

    w_from = (data.shape[0] - shape[0]) // 2   # 80
    h_from = (data.shape[1] - shape[1]) // 2   # 80
    w_to = w_from + shape[0]  # 240
    h_to = h_from + shape[1]  # 240

    return data[w_from:w_to, h_from:h_to]


class SliceDataset(Dataset):
    def __init__(self, root, transform, challenge, sample_rate=1, mode='train'):
        self.mode = mode

        # challenge
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        # transform
        self.transform = transform

        self.examples = []
        self.cur_path = root
        self.csv_file = os.path.join("./dataset/fastmri/singlecoil_" + self.mode + "_split_less.csv")

        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)

            id = 0
            start_id, end_id = 0, 25

            for row in reader:
                for slice_id in range(start_id, end_id):
                    self.examples.append(
                        (os.path.join(self.cur_path, row[0] + '.h5'), os.path.join(self.cur_path, row[1] + '.h5')
                         , slice_id, id))
                id += 1

        if sample_rate < 1:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)

            self.examples = self.examples[0:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        pd_fname, pdfs_fname, slice, id = self.examples[i]

        with h5py.File(pd_fname, "r") as hf:
            pd_target = hf[self.recons_key][slice] if self.recons_key in hf else None  # reconstruction_esc

            img_height, img_width = pd_target.shape
            img_matRotate = cv2.getRotationMatrix2D((img_height * 0.5, img_width * 0.5), 0, 0.6)
            pd_target = cv2.warpAffine(pd_target, img_matRotate, (img_height, img_width))
            pd_target = center_crop(pd_target, (192, 192))

            pd_target = normalize_zero_to_one(pd_target, eps=1e-6)

        pd_sample = self.transform(pd_target, pd_fname, slice, is_complement=True)

        with h5py.File(pdfs_fname, "r") as hf:
            pdfs_target = hf[self.recons_key][slice] if self.recons_key in hf else None

            img_height, img_width = pdfs_target.shape
            img_matRotate = cv2.getRotationMatrix2D((img_height * 0.5, img_width * 0.5), 0, 0.6)
            pdfs_target = cv2.warpAffine(pdfs_target, img_matRotate, (img_height, img_width))
            pdfs_target = center_crop(pdfs_target, (192, 192))

            pdfs_target = normalize_zero_to_one(pdfs_target, eps=1e-6)

        pdfs_sample = self.transform(pdfs_target, pdfs_fname, slice, is_complement=False)

        return pd_sample, pdfs_sample, id


def build_dataset(args, mode='train'):
    assert mode in ['train', 'val', 'test'], 'unknown mode'
    transforms = build_transforms(args, mode)  # mask
    return SliceDataset(os.path.join(args.DATASET.ROOT, 'fastmri_' + mode), transforms, args.DATASET.CHALLENGE,  # singlecoil_ fastmri_
                        sample_rate=args.DATASET.SAMPLE_RATE, mode=mode)
