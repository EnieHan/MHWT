from .mhwt_cnn_fastmri import _C as CCF
from .mhwt_cnn_BraTS20 import _C as CCB
from .mhwt_cat_IXI import _C as CCI
from .mhwt_cnn_IXI import _C as CNO

from .mhwt_single_IXI import _C as CSI
from .mhwt_single_BraTS20 import _C as CSBF


config_factory = {
    # fusion
    'IXI_stability_mhwt_cat': CCI,
    'IXI_stability_mhwt_cnn': CNO,
    'IXI_stability_mhwt_cross': CNO,
    'IXI_stability_mhwt_single': CSI,

    # dataset
    'fast_stability_mhwt_cnn': CCF,
    'brats_stability_mhwt_cnn': CCB,
    'fast_stability_mhwt_single': CSBF,
    'brats_stability_mhwt_single': CSBF,
}


def build_config(mode):
    assert mode in config_factory.keys(), 'unknown config'
    return config_factory[mode]
