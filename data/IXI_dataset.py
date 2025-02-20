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

    w_from = (data.shape[0] - shape[0]) // 2   # 80
    h_from = (data.shape[1] - shape[1]) // 2   # 80
    w_to = w_from + shape[0]  # 240
    h_to = h_from + shape[1]  # 240

    return data[w_from:w_to, h_from:h_to]


class IXIData(Dataset):
    def __init__(self, data_path, transform, sample_rate=1, mode='train'):
        super(IXIData, self).__init__()
        self.mode = mode
        self.data_path = data_path
        self.transform = transform
        self.sample_rate = sample_rate  # 0.06

        self.csv_file = os.path.join('./dataset/IXI', "IXI_" + self.mode + "_long.csv")

        self.examples = []

        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)

            id = 0

            start_id, end_id = 45, 90
            for row in reader:
                for slice_id in range(start_id, end_id):
                    self.examples.append((os.path.join(self.data_path, 'IXI-PD', row[0]),
                                          os.path.join(self.data_path, 'IXI-T2', row[1]), slice_id, id))
                id += 1

        if self.sample_rate < 1:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * self.sample_rate)
            self.examples = self.examples[:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        file_pd, file_t2, slice_id, id = self.examples[item]

        data_pd = nib.load(file_pd)  # (256, 256, 146)
        label_pd = data_pd.dataobj[..., slice_id]  # (256, 256)
        label_pd = center_crop(label_pd, (192, 256))
        label_pd = normalize_zero_to_one(label_pd, eps=1e-6)

        pd_sample = self.transform(label_pd, file_pd, slice_id, is_complement=True)

        data_t2 = nib.load(file_t2)  # (256, 256, 146)
        label_t2 = data_t2.dataobj[..., slice_id]  # (256, 256)
        label_t2 = center_crop(label_t2, (192, 256))
        label_t2 = normalize_zero_to_one(label_t2, eps=1e-6)

        t2_sample = self.transform(label_t2, file_t2, slice_id, is_complement=False)

        return pd_sample, t2_sample, id


def build_dataset(args, mode='train'):
    assert mode in ['train', 'val', 'test'], 'unknown mode'
    transforms = build_transforms(args, mode)  # mask
    return IXIData(args.DATASET.ROOT, transforms, sample_rate=args.DATASET.SAMPLE_RATE, mode=mode)

