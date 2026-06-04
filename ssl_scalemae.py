from typing import Any

import re
import math
import numpy as np

import torch

from pytorch_lightning import LightningModule
from utils import NativeScalerWithGradNormCount


class ResolutionScheduler:
    def __init__(self, *args, **kwargs):
        pass

    def get_target_size(self, epoch):
        raise NotImplementedError


class ConstantResolutionScheduler(ResolutionScheduler):
    def __init__(self, target_size):
        self.target_size = target_size

    def get_target_size(self, epoch):
        return self.target_size


class RandomResolutionScheduler(ResolutionScheduler):
    def __init__(self, target_size, n=1):
        self.target_size = target_size
        self.n = n

    def get_target_size(self, epoch):
        return sorted(np.random.choice(self.target_size, self.n).tolist(), reverse=True)


def get_target_size_scheduler(target_size_scheduler, target_size):
    if target_size_scheduler == "random":
        target_size_scheduler = RandomResolutionScheduler(target_size)
    elif target_size_scheduler == "constant":
        target_size_scheduler = ConstantResolutionScheduler(target_size)
    else:
        match = re.compile("random:([0-9])").findall(target_size_scheduler)
        if match:
            target_size_scheduler = RandomResolutionScheduler(
                target_size, int(match[0])
            )
        else:
            raise NotImplementedError

    return target_size_scheduler


def get_source_size_scheduler(source_size_scheduler, source_size):
    if source_size_scheduler == "random":
        source_size_scheduler = RandomResolutionScheduler(source_size)
    elif source_size_scheduler == "constant":
        source_size_scheduler = ConstantResolutionScheduler(source_size)
    else:
        raise NotImplementedError

    return source_size_scheduler


def get_output_size_scheduler(fixed_output_size_min, fixed_output_size_max):
    if fixed_output_size_min or fixed_output_size_max:
        assert (
            fixed_output_size_min > 0 and fixed_output_size_max > 0 and fixed_output_size_max >= fixed_output_size_min
        )
        output_size_scheduler = RandomResolutionScheduler(
            target_size=np.arange(
                fixed_output_size_min, fixed_output_size_max + 1, 16
            )
        )
    else:
        output_size_scheduler = ConstantResolutionScheduler(target_size=0)

    return output_size_scheduler


class ScaleMAE(LightningModule):
    # def __init__(self, cfg, dm_train, network, scheduler, source_size_scheduler, fix_resolution_scheduler):
    def __init__(self, cfg: Any, datamodule: Any, network: Any) -> None:
        """
        Args:
            cfg (dict or OmegaConf): Configuration dictionary with keys for model and optimizer.
            dm_train: Training datamodule.
            network (nn.Module): Your scalemae network.
            scheduler: A resolution scheduler for the target size.
            source_size_scheduler: A scheduler for the source size.
            fix_resolution_scheduler: A scheduler for fixing the decoding size.
        """
        super().__init__()
        self.save_hyperparameters('cfg')
        self.cfg = cfg
        self.datamodule = datamodule
        self.network = network  # Your scalemae model.

        # hyperparameter hard-coded
        # self.target_size_scheduler = 'constant'
        # self.target_size           = [56]
        # self.source_size_scheduler = 'constant'
        # self.source_size           = [112]
        # self.fixed_output_size_min = 56
        # self.fixed_output_size_max = 112
        self.target_size_scheduler = 'constant'
        self.target_size           = [224]
        self.source_size_scheduler = 'constant'
        self.source_size           = [112]
        self.fixed_output_size_min = 112
        self.fixed_output_size_max = 168

        self.weight_decay = 0.05
        self.lr = 0.00015

        self.min_lr = 0
        self.warmup_epochs = 10

        self.scheduler = get_target_size_scheduler(self.target_size_scheduler, self.target_size)
        self.source_size_scheduler = get_source_size_scheduler(self.source_size_scheduler, self.source_size)
        self.fix_resolution_scheduler = get_output_size_scheduler(self.fixed_output_size_min, self.fixed_output_size_max)

        # Use mask_ratio from configuration.
        self.mask_ratio = 0.75
        self.accum_iter = 16
        # Instantiate the loss scaler for AMP.
        self.loss_scaler = NativeScalerWithGradNormCount()
        # Disable Lightning's automatic optimization for manual control.
        self.automatic_optimization = False

        for module in self.network.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.float()

        self.network = self.network.to(torch.float32)

    def forward(self, samples, input_res, targets, target_res, mask_ratio, source_size):
        """
        Forward pass wrapper.
        """
        return self.network(
            samples,
            input_res=input_res,
            targets=targets,
            target_res=target_res,
            mask_ratio=mask_ratio,
            source_size=source_size,
        )

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()  # Manually get the optimizer.

        if batch_idx % self.accum_iter == 0 and self.cfg.ssl.mae_use_lr_scheduling:
            self.adjust_learning_rate(optimizer, batch_idx / self.trainer.num_training_batches + self.trainer.current_epoch)

        # Unpack the batch. Here we assume the batch is a tuple: ((samples, res, targets, target_res), metadata)
        samples, res, targets, target_res = batch[0]

        # Move to the correct device.
        samples = samples.to(self.device)
        targets = targets.to(self.device)

        # Obtain dynamic sizes from the schedulers.
        target_size = self.scheduler.get_target_size(self.current_epoch)
        source_size = self.source_size_scheduler.get_target_size(self.current_epoch)[0]
        fix_decoding_size = self.fix_resolution_scheduler.get_target_size(self.current_epoch)

        # Set these sizes on the model.
        self.network.set_target_size(target_size)
        self.network.set_fix_decoding_size(fix_decoding_size)

        # Forward pass under AMP.
        with torch.amp.autocast('cuda'):
            loss, y, mask, mean, var, pos_emb, pos_emb_decoder, samples_out = self.forward(
                samples,
                input_res=res,
                targets=targets,
                target_res=target_res,
                mask_ratio=self.mask_ratio,
                source_size=source_size,
            )
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        # Apply gradient accumulation.
        loss = loss / self.accum_iter
        update_grad = ((batch_idx + 1) % self.accum_iter == 0)

        self.loss_scaler(
            loss,
            optimizer,
            parameters=self.network.parameters(),
            update_grad=update_grad,
        )
        if update_grad:
            optimizer.zero_grad()

        # Optionally, log training loss (here we use Lightning's self.log for minimal reporting)
        self.log("train_loss", loss_value, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def add_weight_decay(self, model, weight_decay):
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": weight_decay},
        ]

    def adjust_learning_rate(self, optimizer, epoch):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < self.warmup_epochs:
            lr = self.lr * epoch / self.warmup_epochs
        else:
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * (epoch - self.warmup_epochs)
                    / (self.cfg.params.max_epochs - self.warmup_epochs)
                )
            )
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr

    def configure_optimizers(self):
        # Create parameter groups with proper weight decay.
        param_groups = self.add_weight_decay(self.network, self.weight_decay)
        optimizer = torch.optim.AdamW(
            param_groups, lr=self.lr, betas=(0.9, 0.95)
        )
        return optimizer

    # Common dataloader hooks.
    def train_dataloader(self):
        return self.datamodule.train_dataloader(drop_last=True)

    def train_te_dataloader(self):
        return self.datamodule.train_te_dataloader()

    def val_dataloader(self):
        return self.datamodule.test_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()
