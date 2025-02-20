import os
import csv
import random
import nibabel as nib
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

    w_from = (data.shape[0] - shape[0]) // 2
    h_from = (data.shape[1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[w_from:w_to, h_from:h_to]


class BraTS_Data(Dataset):
    def __init__(self, data_path, transform, sample_rate, mode):
        super(BraTS_Data, self).__init__()
        self.mode = mode
        self.data_path = data_path
        self.transform = transform
        self.sample_rate = sample_rate

        if self.mode == 'train':
            self.data_mode_path = os.path.join(self.data_path, 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData')
        else:
            self.data_mode_path = os.path.join(self.data_path, 'BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData')

        self.csv_file = os.path.join("./dataset/brats20/singlecoil_" + self.mode + "_split_less.csv")

        self.examples = []
        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)

            id = 0
            start_id, end_id = 65, 95

            for row in reader:
                for slice_id in range(start_id, end_id):
                    self.examples.append((os.path.join(self.data_mode_path, row[0], row[0] + '_t1.nii'),
                                          os.path.join(self.data_mode_path, row[0], row[0] + '_t2.nii'), slice_id, id))
                id += 1

        if self.examples is None:
            raise Exception("Dataset is empty!!!")

        if self.sample_rate < 1:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * self.sample_rate)
            self.examples = self.examples[:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # T1
        file_t1, file_t2, slice_id, id = self.examples[item]

        data_t1 = nib.load(file_t1)  # (240, 240, 146)
        label_t1 = data_t1.dataobj[..., slice_id]  # (240, 240)
        label_t1 = center_crop(label_t1, (192, 192))
        label_t1 = normalize_zero_to_one(label_t1, eps=1e-6)

        t1_sample = self.transform(label_t1, file_t1, slice_id, is_complement=True)

        # T2
        data_t2 = nib.load(file_t2)  # (240, 240, 146)
        label_t2 = data_t2.dataobj[..., slice_id]  # (240, 240)
        label_t2 = center_crop(label_t2, (192, 192))
        label_t2 = normalize_zero_to_one(label_t2, eps=1e-6)

        t2_sample = self.transform(label_t2, file_t2, slice_id, is_complement=False)

        return t1_sample, t2_sample, id


def build_dataset(args, mode='train'):
    assert mode in ['train', 'val', 'test'], 'unknown mode'
    transforms = build_transforms(args, mode)  # mask
    return BraTS_Data(args.DATASET.ROOT, transforms, sample_rate=args.DATASET.SAMPLE_RATE, mode=mode)

