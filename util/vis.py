import os
import torch
import h5py
import torch.nn as nn


def init_weights(net, init_type='xavier', gain=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('Initialization method {} is not implemented.'.format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def save_reconstructions(reconstructions, out_dir, writer, step, error_map=None, errzf_map=None, under_img=None):
    """
    Save reconstruction images.

    This function writes to h5 files that are appropriate for submission to the
    leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input
            filenames to corresponding reconstructions (of shape num_slices x
            height x width).
        out_dir (pathlib.Path): Path to the output directory where the
            reconstructions should be saved.
        writer
        step
        error_map
        errzf_map
        under_img
    """
    os.makedirs(str(out_dir), exist_ok=True)
    for fname in reconstructions.keys():
        f_output = torch.stack([v for _, v in reconstructions[fname].items()])
        if error_map:
            f_error = torch.stack([v for _, v in error_map[fname].items()])
            f_errzf = torch.stack([v for _, v in errzf_map[fname].items()])
            f_input = torch.stack([v for _, v in under_img[fname].items()])

        basename = os.path.basename(fname)
        basename, _ = os.path.splitext(basename)
        with h5py.File(str(out_dir) + '/' + str(basename) + '.hdf5', "w") as f:
            f.create_dataset("reconstruction", data=f_output.cpu())
            if error_map:
                f.create_dataset("error_map", data=f_error.cpu())
                f.create_dataset("errzf_map", data=f_errzf.cpu())
                f.create_dataset("under_img", data=f_input.cpu())

        if basename == '':  # img_file_name, e.g., file1001191
            writer.add_image('rec_img', f_output[20].unsqueeze(0).cpu(), step)
        elif basename == '':  # IXI226-HH-1618-T2.nii
            writer.add_image('rec_img_complex', f_output[0].unsqueeze(0).cpu(), step)
            writer.add_image('rec_img_easy', f_output[20].unsqueeze(0).cpu(), step)
        elif basename == '':  # BraTS19_MDA_907_1_t2
            writer.add_image('rec_img', f_output[12].unsqueeze(0).cpu(), step)
        elif basename == '':  # BraTS20_Validation_024_t2
            writer.add_image('rec_img', f_output[12].unsqueeze(0).cpu(), step)
