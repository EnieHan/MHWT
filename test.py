import os
import torch
import time
import argparse

from models import build_model_from_name
from models.loss import build_criterion
from data import build_dataset
from torch.utils.data import DataLoader
from engine import evaluate_test
from config import build_config
from util.logger import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(args, work):
    # build criterion and model first
    model = build_model_from_name(args, work)
    criterion = build_criterion(args)

    device = torch.device(args.SOLVER.DEVICE)

    model.to(device)
    criterion.to(device)

    dataset_val = build_dataset(work, args, mode='test')

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    dataloader_val = DataLoader(dataset_val, batch_size=args.SOLVER.BATCH_SIZE,
                                sampler=sampler_val, num_workers=args.SOLVER.NUM_WORKERS)

    logger_name = work + '_test'
    logger_info(logger_name, os.path.join(args.LOGDIR, logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(dict2str(args))

    for i in range(0, 1):  # Need to modify
        model_path = os.path.join(args.RESUME, 'checkpoint0{}.pth'.format(i))
        logger.info('resume from %s' % model_path)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        run_epoch = checkpoint['epoch']
        run_step = checkpoint['run_step']
        checkpoint = checkpoint['model']
        checkpoint = {key.replace("module.", ""): val for key, val in checkpoint.items()}
        model.load_state_dict(checkpoint, strict=True)
        logger.info('running epoch is:{}'.format(run_epoch))
        logger.info('running step is:{}'.format(run_step))

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params: %.2f M' % (n_parameters / 1024 / 1024))

        start_time = time.time()
        with torch.no_grad():
            evaluate_test(args, model, criterion, dataloader_val, device, args.TEST_OUTPUTDIR, logger, step=0)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('evaluate time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="a unit Cross Multi modity transformer")
    parser.add_argument(
        "--experiment", default="IXI_stability_mhwt_cnn", help="choose a experiment to do")
    args = parser.parse_args()
    print('doing ', args.experiment)

    cfg = build_config(args.experiment)
    main(cfg, args.experiment)

"""
    'IXI_stability_mhwt_cat'
    'IXI_stability_mhwt_cnn'
    'IXI_stability_mhwt_cross'
    'IXI_stability_mhwt_single'
    
    # dataset
    'fast_stability_mhwt_cnn'
    'brats_stability_mhwt_cnn'
    'fast_stability_mhwt_single'
    'brats_stability_mhwt_single'
"""
