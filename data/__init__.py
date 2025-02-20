from .BraTS_20_dataset import build_dataset as b20
from .fastmri import build_dataset as fast
from .IXI_dataset import build_dataset as ixi

config_factory = {
    'IXI_stability_mhwt_cat': ixi,
    'IXI_stability_mhwt_cnn': ixi,
    'IXI_stability_mhwt_cross': ixi,
    'IXI_stability_mhwt_single': ixi,

    # dataset
    'fast_stability_mhwt_cnn': fast,
    'brats_stability_mhwt_cnn': b20,
    'fast_stability_mhwt_single': fast,
    'brats_stability_mhwt_single': b20,
}


def build_dataset(dataset, args, mode):
    assert dataset in config_factory.keys(), 'unknown config'
    return config_factory[dataset](args, mode)
