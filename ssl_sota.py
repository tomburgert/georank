from typing import Any

import math

import torch
from pytorch_lightning import LightningModule

from utils import NativeScalerWithGradNormCount


class SatMAE(LightningModule):
    """
    PyTorch Lightning module for training a SatMAE self-supervised model.
    
    This module wraps the SatMAE backbone (a Masked Autoencoder with a ViT backbone)
    and defines the forward pass as well as the training and validation steps.
    
    Attributes:
        cfg (dict): Configuration dictionary containing model and optimizer hyperparameters.
        dm_train: The datamodule used for training (provides properties such as num_cls if needed).
        network (nn.Module): An instance of the SatMAE backbone (e.g. MaskedAutoencoderViT).
        mask_ratio (float): Proportion of patches to mask during training.
    """
    def __init__(self, cfg: Any, datamodule: Any, network: Any) -> None:
        """
        Initialize the SatMAE Lightning module.
        
        Args:
            cfg (dict): Configuration parameters. Expected to have entries for 'model' and 'optimizer'.
            dm_train: The training datamodule (used here for potential auxiliary info, e.g., num_cls).
            network (nn.Module): The instantiated MAE network.
        """
        super().__init__()
        self.save_hyperparameters('cfg')
        self.cfg = cfg
        self.datamodule = datamodule
        self.network = network

        # Use mask ratio from configuration, defaulting to 0.75 if not specified.

        self.loss_scaler = NativeScalerWithGradNormCount()
        # Use gradient accumulation as in the original training loop.

        self.mask_ratio = cfg.model.get("mask_ratio", 0.75)
        self.accum_iter = 16
        self.weight_decay = 0.0
        self.lr = 0.0001

        self.min_lr = 0
        self.warmup_epochs = 10

        # Disable Lightning’s automatic optimization so we can manually update gradients.
        self.automatic_optimization = False
    
    def forward(self, imgs, imgs_up):
        """
        Forward pass.
        
        Args:
            imgs (Tensor): Input images.
            imgs_up (Tensor): Upsampled images used for computing the multiscale loss.
            
        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: Returns the reconstruction loss,
            the multiscale (L1) loss, the decoder predictions, and the applied mask.
        """
        return self.network(imgs, imgs_up, mask_ratio=self.mask_ratio)
    
    def training_step(self, batch, batch_idx):
        # Retrieve the optimizer manually.
        optimizer = self.optimizers()

        if batch_idx % self.accum_iter == 0 and self.cfg.ssl.mae_use_lr_scheduling:
            self.adjust_learning_rate(optimizer, batch_idx / self.trainer.num_training_batches + self.trainer.current_epoch)

        # Unpack the batch. Here we assume the batch is a dict with keys:
        # 'img', 'img_up_2x', and 'img_up_4x'. Adjust as needed.
        imgs_up_1x = batch[0][0]
        imgs_up_2x = batch[0][1]
        imgs_up_4x = batch[0][2]

        # Prepare the upsampled images as a list.
        imgs_up = []
        if imgs_up_2x is not None:
            imgs_up.append(imgs_up_2x)
        if imgs_up_4x is not None:
            imgs_up.append(imgs_up_4x)

        # (Lightning already moves tensors to the correct device, but you can ensure it here.)
        imgs = imgs_up_1x.to(self.device)
        imgs_up = [im.to(self.device) for im in imgs_up]

        # Use autocast for mixed-precision forward pass.
        # with torch.cuda.amp.autocast():
        with torch.amp.autocast('cuda'):
            mse_loss, l1_loss, pred, mask = self.forward(imgs, imgs_up)
            # Combine losses using your weighting.
            loss = 0.6 * mse_loss + 0.4 * l1_loss

        loss_value = loss.item()

        # Normalize the loss for gradient accumulation.
        loss = loss / self.accum_iter

        # Determine if this is the final sub-iteration.
        update_grad = ((batch_idx + 1) % self.accum_iter == 0)

        # Use the loss scaler to:
        #   - Scale the loss,
        #   - Perform backward,
        #   - Unscale and (if applicable) clip gradients,
        #   - And update parameters when update_grad is True.
        self.loss_scaler(
            loss, optimizer, parameters=self.network.parameters(),
            update_grad=update_grad
        )

        # If we've just updated gradients, zero them out.
        if update_grad:
            optimizer.zero_grad()

        # Log losses for monitoring.
        self.log("train_loss", loss_value, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_mse_loss", mse_loss.item(), on_step=True, on_epoch=True)
        self.log("train_l1_loss", l1_loss.item(), on_step=True, on_epoch=True)

        return loss

    def add_weight_decay(self, model, weight_decay):
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # skip frozen parameters
            # Typically, we don't apply weight decay to biases and 1D parameters (often LayerNorm weights)
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": weight_decay}
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
        """
        Configure the optimizer for training.
        
        Returns:
            Optimizer: The AdamW optimizer configured with parameters from the configuration.
        """
        param_groups = self.add_weight_decay(self.network, self.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=self.lr, betas=(0.9, 0.95))
        # Optionally, add a scheduler here if needed.
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
