import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
import cv2
import numpy as np
import os
import copy
import time

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)
    
def train_one_epoch(model, vae,
                    model_params, ema_params,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ", log_file=args.log_dir)
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            with vae.ema_scope():
                posterior = vae.encode(samples)
            
            # normalize the std of latent to be 1. Change it if you use a different tokenizer
            x = posterior.sample().mul_(args.vae_norm)
        
        # forward
        if args.float32:
            loss = model(x, labels)
        else:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = model(x, labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            
        loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()
        torch.cuda.synchronize()

        update_ema(ema_params, model_params, rate=args.ema_rate)

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        loss_value_reduce = misc.all_reduce_mean(loss_value)
                
        if log_writer is not None and data_iter_step % 100 == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss/loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
def evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=16, log_writer=None, cfg=1.0,
             use_ema=True, data_loader=None):
    model_without_ddp.eval()
    num_steps = args.num_images // (batch_size * misc.get_world_size()) + 1
    save_folder = os.path.join(args.output_dir, "ariter{}-diffsteps{}-temp{}-{}-{}cfg{}-image{}".format(args.num_iter,
                                                                                                     args.num_sampling_steps,
                                                                                                     args.temperature,
                                                                                                     args.ckpt,
                                                                                                     args.cfg_schedule,
                                                                                                     cfg,
                                                                                                     args.num_images))
    if use_ema:
        save_folder = save_folder + "_ema"
    if args.evaluate:
        save_folder = save_folder + "_evaluate"
    print("Save to:", save_folder)
    if misc.get_rank() == 0:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    # switch to ema params
    if use_ema:
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)
            
    class_num = args.class_num
    assert args.num_images % class_num == 0  # number of images per class must be the same
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    used_time = 0
    gen_img_cnt = 0

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        labels_gen = class_label_gen_world[world_size * batch_size * i + local_rank * batch_size:
                                                world_size * batch_size * i + (local_rank + 1) * batch_size]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float):
                sampled_tokens = model_without_ddp.sample_tokens(bsz=batch_size, num_iter=args.num_iter, cfg=cfg,
                                                                    cfg_schedule=args.cfg_schedule, labels=labels_gen,
                                                                    temperature=args.temperature, gt_latents=None)
                sampled_tokens = sampled_tokens.squeeze().permute(0, 2, 1)      
                with vae.ema_scope():
                    sampled_images, _ = vae.decode(sampled_tokens / args.vae_norm)

        # measure speed after the first generation batch
        if i >= 1:
            torch.cuda.synchronize()
            used_time += time.time() - start_time
            gen_img_cnt += batch_size
            print("Generating {} images takes {:.5f} seconds, {:.5f} sec per image".format(gen_img_cnt, used_time, used_time / gen_img_cnt))

        torch.distributed.barrier()
        sampled_images = sampled_images.detach().cpu()
        sampled_images = (sampled_images + 1.) / 2

        # distributed save
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)
            
    torch.distributed.barrier()
    time.sleep(10)

    # back to no ema
    if use_ema:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)
    
    if log_writer is not None and args.metrics == True:
        if args.img_size == 256:
            input2 = None
            fid_statistics_file = 'fid_stats/adm_in256_stats.npz'
        else:
            raise NotImplementedError
        # calculate_vgg(save_folder)
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=input2,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=True,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = ""
        if use_ema:
           postfix = postfix + "_ema"
        if not cfg == 1.0:
           postfix = postfix + "_cfg{}".format(cfg)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))

    torch.distributed.barrier()
    time.sleep(10)
