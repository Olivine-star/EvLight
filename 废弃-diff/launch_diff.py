"""
Training and evaluation launcher for EvWDiff (Event-guided Wavelet Diffusion).
Handles diffusion-specific training: noise loss + content loss + HF loss + EMA.
Does NOT modify the original EvLight launch.py.
"""
import os
import copy
import random
import shutil
import time
from collections import OrderedDict
from os.path import isfile, join

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl.logging import debug, flags, info
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from egllie.datasets import get_dataset
from egllie.losses import AverageMeter
from egllie.losses.image_loss import EglliePSNR, EgllieSSIM, EglliePSNR_star, SSIM
from egllie.models.evwdiff import EvWDiff

FLAGS = flags.FLAGS


class EMAHelper:
    """Exponential Moving Average for model parameters."""
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if name in self.shadow:
                self.shadow[name].data = (
                    (1.0 - self.mu) * param.data + self.mu * self.shadow[name].data
                )

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].data)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b = x.size(0)
        h_x, w_x = x.size(2), x.size(3)
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum()
        count_h = (x.size(1) * (h_x - 1) * w_x)
        count_w = (x.size(1) * h_x * (w_x - 1))
        return 2 * (h_tv / count_h + w_tv / count_w) / b


def rot_aug(batch):
    """Data augmentation with random rotation."""
    rot_times = np.random.randint(0, 4)
    batch['lowligt_image'] = torch.rot90(batch['lowligt_image'], k=rot_times, dims=[2, 3])
    batch['normalligt_image'] = torch.rot90(batch['normalligt_image'], k=rot_times, dims=[2, 3])
    batch['event_free'] = torch.rot90(batch['event_free'], k=rot_times, dims=[2, 3])
    batch['lowlight_image_blur'] = torch.rot90(batch['lowlight_image_blur'], k=rot_times, dims=[2, 3])
    batch['ill_list'] = [torch.rot90(batch['ill_list'][i], k=rot_times, dims=[2, 3])
                         for i in range(len(batch['ill_list']))]
    return batch


def move_tensors_to_cuda(d):
    if isinstance(d, dict):
        return {k: move_tensors_to_cuda(v) for k, v in d.items()}
    if isinstance(d, list):
        return [move_tensors_to_cuda(v) for v in d]
    if isinstance(d, torch.Tensor):
        return d.cuda(non_blocking=True)
    return d


class DiffusionLaunch:
    """Training launcher for EvWDiff."""

    def __init__(self, config):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6544"
        self.config = config

        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)
        random.seed(config.SEED)
        np.random.seed(config.SEED)

        self.tb_recoder = SummaryWriter(FLAGS.log_dir)
        self.visualizer = None
        if config.VISUALIZE:
            self.visualizer = Visualization(config.VISUALIZATION)

    def run(self):
        # Build dataset (reuses EvLight's data pipeline)
        train_dataset, val_dataset = get_dataset(self.config.DATASET)

        # Build model
        model = EvWDiff(self.config.EVWDIFF)

        # Load pretrained IlluminationNet weights from EvLight if specified
        pretrain_cfg = self.config.EVWDIFF.get('pretrained_illum', {})
        pretrain_path = pretrain_cfg.get('path', '') if pretrain_cfg else ''
        if pretrain_path and isfile(pretrain_path):
            info(f"Loading pretrained IlluminationNet from {pretrain_path}")
            ckpt = torch.load(pretrain_path, map_location='cpu')
            sd = ckpt.get('state_dict', ckpt)
            # Map EvLight keys: module.IllumiinationNet.xxx -> illum_net.xxx
            illum_state = {}
            for k, v in sd.items():
                if 'IllumiinationNet' in k:
                    new_k = k.replace('module.IllumiinationNet.', '')
                    illum_state[new_k] = v
            model.illum_net.load_state_dict(illum_state, strict=True)
            info(f"  Loaded {len(illum_state)} IlluminationNet params")

            # Optionally freeze IlluminationNet
            freeze_epochs = int(pretrain_cfg.get('freeze_epochs', 0))
            if freeze_epochs > 0:
                for p in model.illum_net.parameters():
                    p.requires_grad = False
                info(f"  IlluminationNet frozen for first {freeze_epochs} epochs")

        if self.config.IS_CUDA:
            model = nn.DataParallel(model)
            model = model.cuda()

        # EMA
        ema_helper = EMAHelper(mu=float(self.config.EVWDIFF.get('ema_rate', 0.999)))
        ema_helper.register(model)

        # Losses
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        tv_loss = TVLoss()
        ssim_fn = SSIM()

        # Metrics
        psnr_fn = EglliePSNR()
        ssim_metric = EgllieSSIM()
        psnr_star_fn = EglliePSNR_star()

        # Optimizer
        opt_cfg = self.config.OPTIMIZER
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(opt_cfg.LR),
            weight_decay=float(opt_cfg.get('weight_decay', 0)),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.END_EPOCH, eta_min=1e-7
        )

        # Loss weights
        lw = self.config.EVWDIFF.get('loss_weights', {})
        w_noise = float(lw.get('noise', 1.0))
        w_content = float(lw.get('content', 1.0))
        w_hf = float(lw.get('hf', 0.1))
        w_tv = float(lw.get('tv', 0.01))

        # Resume
        start_epoch = self.config.START_EPOCH
        if self.config.RESUME.PATH:
            ckpt_path = self.config.RESUME.PATH
            if not isfile(ckpt_path):
                raise ValueError(f"File not found: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location='cuda:0')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            if 'ema_helper' in checkpoint:
                ema_helper.load_state_dict(checkpoint['ema_helper'])
            if self.config.RESUME.SET_EPOCH:
                start_epoch = checkpoint.get('epoch', 0)
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                if 'scheduler' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler'])
            info(f"Resumed from {ckpt_path}")

        # Dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.TRAIN_BATCH_SIZE,
            shuffle=True, num_workers=self.config.JOBS,
            pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.VAL_BATCH_SIZE,
            shuffle=False, num_workers=self.config.JOBS,
            pin_memory=True, drop_last=True,
        )

        # Test only
        if self.config.TEST_ONLY:
            # Use EMA weights for evaluation
            ema_helper.ema(model)
            self._validate(val_loader, model, psnr_fn, ssim_metric, psnr_star_fn, 0)
            return

        # Training loop
        min_loss = float('inf')
        freeze_epochs = int(pretrain_cfg.get('freeze_epochs', 0)) if pretrain_cfg else 0
        for epoch in range(start_epoch, self.config.END_EPOCH):
            # Unfreeze IlluminationNet after freeze_epochs with reduced LR
            if freeze_epochs > 0 and epoch == freeze_epochs:
                illum_mod = model.module.illum_net if isinstance(model, nn.DataParallel) else model.illum_net
                for p in illum_mod.parameters():
                    p.requires_grad = True
                # Add IllumNet params to optimizer with 10x smaller LR
                illum_lr = float(opt_cfg.LR) * 0.1
                optimizer.add_param_group({'params': list(illum_mod.parameters()), 'lr': illum_lr})
                info(f"IlluminationNet unfrozen at epoch {epoch} with lr={illum_lr}")
            model.train()
            losses_meter = {k: AverageMeter(k) for k in
                           ['total', 'noise', 'content', 'hf', 'ssim']}
            metric_meter = {k: AverageMeter(k) for k in ['PSNR', 'SSIM']}

            info(f"Train Epoch [{epoch}/{self.config.END_EPOCH}]: len({len(train_loader)})")

            for idx, batch in enumerate(train_loader):
                if self.config.IS_CUDA:
                    batch = move_tensors_to_cuda(batch)
                batch = rot_aug(batch)

                with torch.cuda.amp.autocast(enabled=self.config.get('MIX_PRECISION', False)):
                    outputs = model(batch)

                # Compute losses
                noise_loss = l2_loss(outputs['noise_pred'], outputs['noise_gt'])

                # Content loss with alpha_bar floor to ensure structural supervision
                alpha_bar = outputs['alpha_bar']  # [B,1,1,1]
                content_weight_t = torch.clamp(alpha_bar.squeeze().mean(), min=0.3)
                content_loss = l1_loss(outputs['pred'], outputs['gt'])
                ssim_loss = 1.0 - ssim_fn(outputs['pred'], outputs['gt'])

                # Direct LL reconstruction loss
                ll_loss = l1_loss(outputs['x0_pred_LL'], outputs['gt_LL'])

                hf_loss = l2_loss(outputs['restored_HF'], outputs['gt_HF'])
                hf_tv = tv_loss(outputs['restored_HF'])

                total_loss = (w_noise * noise_loss +
                              w_content * content_weight_t * (content_loss + ssim_loss) +
                              w_content * ll_loss +
                              w_hf * hf_loss +
                              w_tv * hf_tv)

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ema_helper.update(model)

                # Metrics
                with torch.no_grad():
                    psnr_val = psnr_fn(outputs)
                    ssim_val = ssim_metric(outputs)

                losses_meter['total'].update(total_loss.item())
                losses_meter['noise'].update(noise_loss.item())
                losses_meter['content'].update(content_loss.item())
                losses_meter['hf'].update(hf_loss.item())
                losses_meter['ssim'].update(ssim_loss.item())
                metric_meter['PSNR'].update(psnr_val.item())
                metric_meter['SSIM'].update(ssim_val.item())

                if idx % self.config.LOG_INTERVAL == 0:
                    info(f"  [{idx}/{len(train_loader)}] "
                         f"loss={total_loss.item():.4f} "
                         f"noise={noise_loss.item():.4f} "
                         f"content={content_loss.item():.4f} "
                         f"hf={hf_loss.item():.4f} "
                         f"PSNR={psnr_val.item():.2f}")

            scheduler.step()

            # Log epoch
            for name, meter in losses_meter.items():
                self.tb_recoder.add_scalar(f"Train/{name}", meter.avg, epoch)
                info(f"  Train {name}: {meter.avg:.4f}")
            for name, meter in metric_meter.items():
                self.tb_recoder.add_scalar(f"Train/{name}", meter.avg, epoch)
                info(f"  Train {name}: {meter.avg:.4f}")
            self.tb_recoder.add_scalar("Train/LR", optimizer.param_groups[0]['lr'], epoch)

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'ema_helper': ema_helper.state_dict(),
            }
            path = join(self.config.SAVE_DIR, "checkpoint.pth.tar")
            torch.save(checkpoint, path)

            # Validate
            if epoch % self.config.VAL_INTERVAL == 0:
                # Backup current weights, apply EMA for validation
                original_state = copy.deepcopy(model.state_dict())
                ema_helper.ema(model)

                val_loss = self._validate(
                    val_loader, model, psnr_fn, ssim_metric, psnr_star_fn, epoch
                )
                if val_loss < min_loss:
                    min_loss = val_loss
                    copy_path = join(self.config.SAVE_DIR, "model_best.pth.tar")
                    # Save EMA weights as best model
                    torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'ema_helper': ema_helper.state_dict(),
                    }, copy_path)

                # Restore original weights
                model.load_state_dict(original_state)

            if epoch % self.config.MODEL_SANING_INTERVAL == 0:
                path = join(self.config.SAVE_DIR, f"checkpoint-{str(epoch).zfill(3)}.pth.tar")
                torch.save(checkpoint, path)

    def _validate(self, val_loader, model, psnr_fn, ssim_fn, psnr_star_fn, epoch):
        model.eval()
        psnr_meter = AverageMeter('PSNR')
        ssim_meter = AverageMeter('SSIM')
        psnr_star_meter = AverageMeter('PSNR_star')
        loss_meter = AverageMeter('loss')

        info(f"Valid Epoch [{epoch}/{self.config.END_EPOCH}]: len({len(val_loader)})")
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                if self.config.IS_CUDA:
                    batch = move_tensors_to_cuda(batch)

                outputs = model(batch)

                l1 = F.l1_loss(outputs['pred'], outputs['gt'])
                loss_meter.update(l1.item())

                psnr_val = psnr_fn(outputs)
                ssim_val = ssim_fn(outputs)
                psnr_star_val = psnr_star_fn(outputs)

                psnr_meter.update(psnr_val.item())
                ssim_meter.update(ssim_val.item())
                psnr_star_meter.update(psnr_star_val.item())

                if self.visualizer:
                    self.visualizer.visualize(batch, outputs)

                if idx % self.config.LOG_INTERVAL == 0:
                    info(f"  Val [{idx}/{len(val_loader)}] "
                         f"PSNR={psnr_val.item():.2f} "
                         f"SSIM={ssim_val.item():.4f} "
                         f"PSNR*={psnr_star_val.item():.2f}")

                del batch, outputs
                torch.cuda.empty_cache()

        info(f"Valid Epoch [{epoch}]: "
             f"PSNR={psnr_meter.avg:.2f} "
             f"SSIM={ssim_meter.avg:.4f} "
             f"PSNR*={psnr_star_meter.avg:.2f}")
        self.tb_recoder.add_scalar("Valid/PSNR", psnr_meter.avg, epoch)
        self.tb_recoder.add_scalar("Valid/SSIM", ssim_meter.avg, epoch)
        self.tb_recoder.add_scalar("Valid/PSNR_star", psnr_star_meter.avg, epoch)
        self.tb_recoder.add_scalar("Valid/Loss", loss_meter.avg, epoch)

        return loss_meter.avg


class Visualization:
    def __init__(self, vis_config):
        self.saving_folder = join(FLAGS.log_dir, vis_config.folder)
        os.makedirs(self.saving_folder, exist_ok=True)

    def visualize(self, inputs, outputs):
        def _save(image, path):
            if not isinstance(image, torch.Tensor):
                return
            image = image.detach().permute(1, 2, 0).cpu().numpy()
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, image)

        B = len(inputs['seq_name'])
        for b in range(B):
            video_name = inputs['seq_name'][b]
            frame_name = inputs['frame_id'][b]
            testfolder = join(self.saving_folder, video_name)
            os.makedirs(testfolder, exist_ok=True)
            _save(outputs['gt'][b], join(testfolder, f"{frame_name}_gt.png"))
            _save(outputs['pred'][b], join(testfolder, f"{frame_name}_pred.png"))
