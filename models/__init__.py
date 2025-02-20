from .swin2_stability_mhwt_cat import define_G as SSC
from .swin2_stability_mhwt_cnn import define_G as SSN
from .swin2_stability_mhwt_cross import define_G as SSO
from .swin2_stability_mhwt_single import define_G as SSS


model_factory = {
    'IXI_stability_mhwt_cat': SSC,
    'IXI_stability_mhwt_cnn': SSN,
    'IXI_stability_mhwt_cross': SSO,
    'IXI_stability_mhwt_single': SSS,

    # dataset
    'fast_stability_mhwt_cnn': SSN,
    'brats_stability_mhwt_cnn': SSN,
    'fast_stability_mhwt_single': SSS,
    'brats_stability_mhwt_single': SSS,
}


def build_model_from_name(args, model_name):
    assert model_name in model_factory.keys(), 'unknown model name'
    return model_factory[model_name](args)
