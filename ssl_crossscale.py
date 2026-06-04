from typing import Any

import math

import torch

from pytorch_lightning import LightningModule
from utils import NativeScalerWithGradNormCount


class CrossScaleMAE(LightningModule):
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

        # parameter
        self.mask_ratio = 0.75
        self.weight_decay = 0.05
        self.lr = 0.00005
        self.accum_iter = 16

        self.min_lr = 0
        self.warmup_epochs = 10

        self.automatic_optimization = False

        self.loss_scaler = NativeScalerWithGradNormCount()
    
    def forward(self, x, mask_ratio):
        # Call the model's forward pass.
        return self.network(x, mask_ratio=mask_ratio)

    def training_step(self, batch, batch_idx):
        # Unpack your batch (e.g., samples, labels)
        optimizer = self.optimizers()

        if batch_idx % self.accum_iter == 0 and self.cfg.ssl.mae_use_lr_scheduling:
            self.adjust_learning_rate(optimizer, batch_idx / self.trainer.num_training_batches + self.trainer.current_epoch)
    
        # Clear gradients at the start of accumulation cycle
        if batch_idx % self.accum_iter == 0:
            optimizer.zero_grad()

        samples = batch[0]
        samples = samples.to(self.device)
        # Use AMP context if needed
        with torch.amp.autocast('cuda'):
            loss, _, _ = self.forward(samples, mask_ratio=self.mask_ratio)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        # # If accumulation is handled by Lightning, simply return the loss
        # self.loss_scaler(loss, self.optimizer, parameters=self.network.parameters(), update_grad=True)
        # self.log('train_loss', loss_value, on_step=True, on_epoch=True)
        # return loss

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
        param_groups = self.add_weight_decay(self.network, self.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=self.lr, weight_decay=self.weight_decay)
        # Optionally, configure a scheduler here.
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
