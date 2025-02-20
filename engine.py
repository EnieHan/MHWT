import os
import torch
import time
import datetime
from pathlib import Path
from typing import Iterable
import util.misc as utils
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from util.vis import save_reconstructions
from util.metric import nmse, psnr, ssim, AverageMeter

writer = SummaryWriter('../tf-logs')


def train_one_epoch(args, model, criterion, data_loader: Iterable, optimizer, epoch, print_freq,
                    device, logger, run_step, dataloader_val, best_status, lr_scheduler):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for data in metric_logger.log_every(data_loader, print_freq, logger, header):

        pd, pdfs, _ = data
        target = pdfs['target'].unsqueeze(1)
        mask = pdfs['mask'].unsqueeze(1)
        pdfs_img = pdfs['under_img'].unsqueeze(1)
        pd_img = pd['target'].unsqueeze(1)

        pd_img = pd_img.to(device)
        pdfs_img = pdfs_img.to(device)
        target = target.to(device)
        mask = mask.to(device)

        if args.USE_MULTI_MODEL and args.LOSS.USE_MM_LOSS:
            outputs, complement = model(pdfs_img, pd_img, mask)
            loss = criterion(outputs, target, complement, pd_img)
        elif args.USE_MULTI_MODEL:
            outputs = model(pdfs_img, pd_img, mask)
            loss = criterion(outputs, target)
        else:
            outputs = model(pdfs_img, mask)
            loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()

        metric_logger.update(loss=loss['loss'].item())
        if args.LOSS.TYPE == 'L1':
            metric_logger.update(l1_loss=loss['l1_loss'].item())
            if args.LOSS.USE_MM_LOSS:
                metric_logger.update(cl1_loss=loss['cl1_loss'].item())
        else:
            metric_logger.update(loss_rec_image=loss['loss_rec_image'].item())
            metric_logger.update(loss_rec_freq=loss['loss_rec_freq'].item())
            metric_logger.update(loss_rec_perc=loss['loss_rec_perc'].item())
            metric_logger.update(loss_rec=loss['rec_loss'].item())
            if args.LOSS.USE_MM_LOSS:
                metric_logger.update(loss_com_image=loss['loss_com_image'].item())
                metric_logger.update(loss_com_freq=loss['loss_com_freq'].item())
                metric_logger.update(loss_com_perc=loss['loss_com_perc'].item())
                metric_logger.update(loss_com=loss['com_loss'].item())

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        run_step += 1

        if run_step % args.SOLVER.PRINT_LOSS == 0:
            writer.add_scalar('loss', metric_logger.meters['loss'].global_avg, run_step // args.SOLVER.PRINT_LOSS)
            writer.add_scalar('loss_rec', metric_logger.meters['loss_rec'].global_avg, run_step // args.SOLVER.PRINT_LOSS)
            writer.add_scalar('loss_rec_image', metric_logger.meters['loss_rec_image'].global_avg, run_step // args.SOLVER.PRINT_LOSS)

        if run_step % args.SOLVER.SAVE_MODAL == 0:
            lr_scheduler.step()
            with torch.no_grad():
                eval_status = evaluate(args, model, criterion, dataloader_val, device, args.OUTPUTFILEDIR,
                                       logger, run_step // args.SOLVER.SAVE_MODAL)
            writer.add_scalar('Evaluate_NMSE', eval_status['NMSE'], run_step // args.SOLVER.SAVE_MODAL)
            writer.add_scalar('Evaluate_PSNR', eval_status['PSNR'], run_step // args.SOLVER.SAVE_MODAL)
            writer.add_scalar('Evaluate_SSIM', eval_status['SSIM'], run_step // args.SOLVER.SAVE_MODAL)

            if eval_status['PSNR'] > best_status['PSNR']:
                best_status = eval_status
                best_checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'run_step': run_step,
                    'status': best_status,
                    'args': args,
                }
                best_path = os.path.join(args.OUTPUTDIR, 'best.pth')
                torch.save(best_checkpoint, best_path)

            # save model
            if args.OUTPUTDIR:
                Path(args.OUTPUTDIR).mkdir(parents=True, exist_ok=True)
                checkpoint_path = os.path.join(args.OUTPUTDIR, f'checkpoint{(run_step // args.SOLVER.SAVE_MODAL):04}.pth')
                print(checkpoint_path)

                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'run_step': run_step,
                    'status': eval_status,
                    'args': args,
                }, checkpoint_path)
    return run_step, best_status


@torch.no_grad()
def evaluate(args, model, criterion, data_loader, device, output_dir, logger, step):
    model.eval()
    criterion.eval()

    nmse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    output_dic = defaultdict(dict)
    target_dic = defaultdict(dict)
    start_time = time.time()

    for data in data_loader:
        pd, pdfs, _ = data
        target = pdfs['target']

        fname = pdfs['file_name']
        slice_num = pdfs['slice_num']

        pd_img = pd['target'].unsqueeze(1)
        pdfs_img = pdfs['under_img'].unsqueeze(1)
        mask = pdfs['mask'].unsqueeze(1)

        pd_img = pd_img.to(device)
        pdfs_img = pdfs_img.to(device)
        target = target.to(device)
        mask = mask.to(device)

        if args.USE_MULTI_MODEL and args.LOSS.USE_MM_LOSS:
            outputs, _ = model(pdfs_img, pd_img, mask)
        elif args.USE_MULTI_MODEL:
            outputs = model(pdfs_img, pd_img, mask)
        else:
            outputs = model(pdfs_img, mask)
        outputs = outputs.squeeze(1)

        for i, f in enumerate(fname):
            output_dic[f][slice_num[i]] = outputs[i]
            target_dic[f][slice_num[i]] = target[i]

    for name in output_dic.keys():
        f_output = torch.stack([v for _, v in output_dic[name].items()])
        f_target = torch.stack([v for _, v in target_dic[name].items()])
        our_nmse = nmse(f_target.cpu().numpy(), f_output.cpu().numpy())
        our_psnr = psnr(f_target.cpu().numpy(), f_output.cpu().numpy())
        our_ssim = ssim(f_target.cpu().numpy(), f_output.cpu().numpy())

        nmse_meter.update(our_nmse, 1)
        psnr_meter.update(our_psnr, 1)
        ssim_meter.update(our_ssim, 1)

    save_reconstructions(output_dic, output_dir, writer, step)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    logger.info('Evaluate Metric — Evaluate time: {}, NMSE: {:.4}; PSNR: {:.4}; SSIM: {:.4}'
                .format(total_time_str, nmse_meter.avg, psnr_meter.avg, ssim_meter.avg))
    return {'NMSE': nmse_meter.avg, 'PSNR': psnr_meter.avg, 'SSIM': ssim_meter.avg}


@torch.no_grad()
def evaluate_test(args, model, criterion, data_loader, device, output_dir, logger, step):
    model.eval()
    criterion.eval()

    nmse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    output_dic = defaultdict(dict)
    target_dic = defaultdict(dict)

    zfnmse_meter = AverageMeter()
    zfpsnr_meter = AverageMeter()
    zfssim_meter = AverageMeter()
    input_dic = defaultdict(dict)
    error_dic = defaultdict(dict)
    errzf_dic = defaultdict(dict)
    start_time = time.time()

    for data in data_loader:
        pd, pdfs, _ = data
        target = pdfs['target']

        fname = pdfs['file_name']
        slice_num = pdfs['slice_num']

        pd_img = pd['target'].unsqueeze(1)
        pdfs_img = pdfs['under_img'].unsqueeze(1)
        mask = pdfs['mask'].unsqueeze(1)

        pd_img = pd_img.to(device)
        pdfs_img = pdfs_img.to(device)
        target = target.to(device)
        mask = mask.to(device)

        if args.USE_MULTI_MODEL and args.LOSS.USE_MM_LOSS:
            outputs, _ = model(pdfs_img, pd_img, mask)
        elif args.USE_MULTI_MODEL:
            outputs = model(pdfs_img, pd_img, mask)
        else:
            outputs = model(pdfs_img, mask)
        outputs = outputs.squeeze(1)
        inputs = pdfs_img.squeeze(1)

        for i, f in enumerate(fname):
            output_dic[f][slice_num[i]] = outputs[i]
            target_dic[f][slice_num[i]] = target[i]
            input_dic[f][slice_num[i]] = inputs[i]
            error_dic[f][slice_num[i]] = torch.abs(target[i] - outputs[i])
            errzf_dic[f][slice_num[i]] = torch.abs(target[i] - inputs[i])

    for name in output_dic.keys():
        f_output = torch.stack([v for _, v in output_dic[name].items()])
        f_target = torch.stack([v for _, v in target_dic[name].items()])
        our_nmse = nmse(f_target.cpu().numpy(), f_output.cpu().numpy())
        our_psnr = psnr(f_target.cpu().numpy(), f_output.cpu().numpy())
        our_ssim = ssim(f_target.cpu().numpy(), f_output.cpu().numpy())

        f_input = torch.stack([v for _, v in input_dic[name].items()])
        zf_nmse = nmse(f_target.cpu().numpy(), f_input.cpu().numpy())
        zf_psnr = psnr(f_target.cpu().numpy(), f_input.cpu().numpy())
        zf_ssim = ssim(f_target.cpu().numpy(), f_input.cpu().numpy())

        nmse_meter.update(our_nmse, 1)
        psnr_meter.update(our_psnr, 1)
        ssim_meter.update(our_ssim, 1)
        zfnmse_meter.update(zf_nmse, 1)
        zfpsnr_meter.update(zf_psnr, 1)
        zfssim_meter.update(zf_ssim, 1)

    save_reconstructions(output_dic, output_dir, writer, step, error_dic, errzf_dic, input_dic)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    logger.info('Evaluate Metric — Evaluate time: {}, NMSE: {:.4}; PSNR: {:.4}; SSIM: {:.4}'
                .format(total_time_str, nmse_meter.avg, psnr_meter.avg, ssim_meter.avg))

    logger.info('ZF Evaluate Metric — ZFNMSE: {:.4}; ZFPSNR: {:.4}; ZFSSIM: {:.4}'
                .format(zfnmse_meter.avg, zfpsnr_meter.avg, zfssim_meter.avg))

    return {'NMSE': nmse_meter.avg, 'PSNR': psnr_meter.avg, 'SSIM': ssim_meter.avg}

