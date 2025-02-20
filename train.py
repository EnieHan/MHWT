import torch
import time
import os
import random
import numpy as np
import argparse

from models import build_model_from_name
from models.loss import build_criterion
from data import build_dataset
from torch.utils.data import DataLoader

from engine import train_one_epoch
from config import build_config
from util.logger import *
from util.vis import init_weights
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(args, work):
    # build criterion and model
    model = build_model_from_name(args, work)
    init_weights(model)
    criterion = build_criterion(args)

    start_epoch = 0
    run_step = 0
    best_status = {'NMSE': 10000000, 'PSNR': 0, 'SSIM': 0}

    seed = args.SEED

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(args.SOLVER.DEVICE)
    model.cuda()
    criterion.cuda()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params: %.2f M' % (n_parameters / 1024 / 1024))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.SOLVER.LR, weight_decay=args.SOLVER.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.SOLVER.LR_DROP, gamma=0.5)

    # build dataset
    dataset_train = build_dataset(work, args, 'train')
    dataset_val = build_dataset(work, args, 'val')

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.SOLVER.BATCH_SIZE, drop_last=True)

    dataloader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                  num_workers=args.SOLVER.NUM_WORKERS)
    dataloader_val = DataLoader(dataset_val, batch_size=args.SOLVER.BATCH_SIZE,
                                sampler=sampler_val, num_workers=args.SOLVER.NUM_WORKERS)

    # load
    if args.RESUME != '':
        checkpoint = torch.load(args.RESUME)
        run_step = checkpoint['run_step']
        start_epoch = checkpoint['epoch']
        best_status['NMSE'] = checkpoint['status']['NMSE']
        best_status['PSNR'] = checkpoint['status']['PSNR']
        best_status['SSIM'] = checkpoint['status']['SSIM']
        checkpoint = checkpoint['model']
        checkpoint = {key.replace("module.", ""): val for key, val in checkpoint.items()}
        print('resume from %s' % args.RESUME)
        model.load_state_dict(checkpoint)

    # configure logger
    logger_name = work + "_train"
    logger_info(logger_name, os.path.join(args.LOGDIR, logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(dict2str(args))

    start_time = time.time()

    for epoch in range(start_epoch, args.TRAIN.EPOCHS):
        run_step,  best_status = train_one_epoch(args, model, criterion, dataloader_train, optimizer, epoch,
                                                 args.SOLVER.PRINT_FREQ, device, logger, run_step, dataloader_val,
                                                 best_status, lr_scheduler)

    logger.info('The best status is: NMSE: {:.4}; PSNR: {:.4}; SSIM: {:.4}'
                .format(best_status['NMSE'], best_status['PSNR'], best_status['SSIM']))
    print("------------------")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="a unit Cross Multi modality transformer")
    parser.add_argument("--experiment", default="IXI_stability_mhwt_cnn", help="choose a experiment to do")
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
