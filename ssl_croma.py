from typing import Any
from typing import Optional
from typing import Tuple

import pandas as pd

import torch
from torch import nn

from pytorch_lightning import LightningModule

from utils import LinearWarmupCosineAnnealingLR
from utils import SoftRankLoss


class SSLCROMA(LightningModule):
    def __init__(self, cfg: Any, datamodule: Any, network: nn.Module) -> None:
        super().__init__()
        self.save_hyperparameters('cfg')
        self.cfg = cfg
        self.datamodule = datamodule
        self.network = network

        # Loss weights with safe defaults
        self.contrastive_weight = 1.0
        self.mae_weight = 1.0
        self.mask_ratio = 0.75

        # Optimizer settings with safe defaults
        self.lr = 0.000004  # 1e-4
        self.weight_decay = 0.01

        # Scheduler settings with safe defaults
        self.use_warmup_cosine = True
        self.warmup_steps = 1000

        if self.cfg.ssl.use_geography_loss:
            self.df_geo_metadata = self._load_geography_df()
            self.criterion_geo = SoftRankLoss(
                temperature=self.cfg.ssl.moco_temperature,
                reg_strength=self.cfg.ssl.gcl_softrank_reg_strength,
                min_dist_km=self.cfg.ssl.gcl_min_dist_km,
                max_dist_km=self.cfg.ssl.gcl_max_dist_km,
                soft_margin=self.cfg.ssl.gcl_rank_soft_margin,
                loss_type=self.cfg.ssl.gcl_rank_loss_type,
                distance_measure=self.cfg.ssl.gcl_geo_distance_measure,
            )
        else:
            self.df_geo_metadata = None
            self.criterion_geo = None

    def _load_geography_df(self) -> Optional[pd.DataFrame]:
        ds = getattr(self.cfg.params, "dataset", None)
        if ds == "SSL4EO":
            df = pd.read_parquet("/data/tomburgert/data/additional_data/ssl4eo_patch_id_to_kmeans_v3.parquet")
            if getattr(self.cfg.dataset, "temporal_views_path", None) is not None:
                groups = list(map(int, self.datamodule.trainset_tr.groups))
                df["prefix_int"] = df["patch_id"].str.split("_").str[0].astype(int)
                df = df[df["prefix_int"].isin(groups)].drop(columns=["prefix_int"])
            return df
        if ds == "SSL4EOEurope":
            return pd.read_parquet("/data/tomburgert/data/additional_data/ssl4eo_europe_patch_id_to_kmeans_v3.parquet")
        if ds == "BigEarthNetV2":
            return pd.read_parquet("/data/tomburgert/bigearthnet_stats/benv2_patch_id_to_country_kmeans_v14.parquet")
        return None

    def forward(self, x: torch.Tensor, rank=0, world_size=1):
        return self.network(
            imgs=x,
            radar_mask_info=None,
            optical_mask_info=None,
            rank=rank,
            world_size=world_size,
        )

    def _build_masks(self, batch_size: int, seq_len: int, device: torch.device):
        radar_mask_info = get_mask(bsz=batch_size, seq_len=seq_len, device=device, mask_ratio=self.mask_ratio)
        optical_mask_info = get_mask(bsz=batch_size, seq_len=seq_len, device=device, mask_ratio=self.mask_ratio)
        return radar_mask_info, optical_mask_info

    @staticmethod
    def _project_for_geo(network: nn.Module, radar_gap: torch.Tensor, optical_gap: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        proj = network.global_contrast_loss
        q = proj.radar_proj(radar_gap)
        k = proj.optical_proj(optical_gap)
        q = q / q.norm(dim=1, keepdim=True)
        k = k / k.norm(dim=1, keepdim=True)
        return q, k

    def training_step(self, batch, batch_idx):
        imgs = batch[0]
        indices = batch[2].cpu().detach()

        batch_size = imgs.shape[0]
        device = imgs.device
        seq_len = int(self.network.num_patches)

        radar_mask_info, optical_mask_info = self._build_masks(batch_size=batch_size, seq_len=seq_len, device=device)
        
        contrastive_loss, mae_loss, radar_gap, optical_gap = self.network.forward(
            imgs=imgs,
            radar_mask_info=radar_mask_info,
            optical_mask_info=optical_mask_info,
        )

        loss_geo = torch.tensor(0.0, device=device)
        if self.cfg.ssl.use_geography_loss:
            q, k = self._project_for_geo(self.network, radar_gap, optical_gap)
            df = self.df_geo_metadata.iloc[indices]
            gps = torch.tensor([*zip(df.latitude.values, df.longitude.values)])
            loss_geo = self.criterion_geo(q, k, gps)

        total_loss = self.contrastive_weight * contrastive_loss + self.mae_weight * mae_loss + self.cfg.ssl.gcl_alpha * loss_geo

        self.log("train_contrastive", contrastive_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train_mae", mae_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        if self.cfg.ssl.use_geography_loss:
            self.log("train_georank", loss_geo, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        return total_loss

    # def forward(self, x: torch.Tensor, rank=0, world_size=1):
    #     return self.network(imgs=x,
    #                         radar_mask_info=None,
    #                         optical_mask_info=None,
    #                         rank=rank,
    #                         world_size=world_size)

    # def _build_masks(self, batch_size: int, seq_len: int, device: torch.device):
    #     radar_mask_info = get_mask(bsz=batch_size, seq_len=seq_len, device=device, mask_ratio=self.mask_ratio)
    #     optical_mask_info = get_mask(bsz=batch_size, seq_len=seq_len, device=device, mask_ratio=self.mask_ratio)
    #     return radar_mask_info, optical_mask_info

    # def training_step(self, batch, batch_idx):
    #     # Expect datamodule to yield stacked tensors with shape [B, 12, H, W] if total_channels equals 12 plus 2 for SAR
    #     # In your backbone you split into 10 optical and 2 radar channels
    #     imgs = batch[0]

    #     batch_size = imgs.shape[0]
    #     device = imgs.device
    #     seq_len = int(self.network.num_patches)

    #     radar_mask_info, optical_mask_info = self._build_masks(batch_size=batch_size, seq_len=seq_len, device=device)

    #     contrastive_loss, mae_loss = self.network(imgs=imgs,
    #                                               radar_mask_info=radar_mask_info,
    #                                               optical_mask_info=optical_mask_info)

    #     total_loss = self.contrastive_weight * contrastive_loss + self.mae_weight * mae_loss

    #     self.log("train_contrastive", contrastive_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
    #     self.log("train_mae", mae_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
    #     self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
    #     return total_loss

    def configure_optimizers(self):
        param_groups = self.add_weight_decay(self.network, self.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=self.lr, weight_decay=self.weight_decay)

        if self.use_warmup_cosine:
            batches_per_epoch = len(self.datamodule.trainset_tr) / self.cfg.dataset.batch_size
            max_steps = int(self.trainer.max_epochs * batches_per_epoch)
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.warmup_steps,
                max_epochs=max_steps,
                warmup_start_lr=self.lr / 10.0,
                eta_min=self.lr / 10.0
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "learning_rate"
                },
            }
        return optimizer

    @staticmethod
    def add_weight_decay(model: nn.Module, weight_decay: float, skip_list: Optional[set] = None):
        if skip_list is None:
            skip_list = set()
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)
            else:
                no_decay_flag = False
                for module_name, module in model.named_modules():
                    if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                        if name.startswith(module_name):
                            no_decay_flag = True
                            break
                if no_decay_flag:
                    no_decay.append(param)
                else:
                    decay.append(param)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    # Common dataloader hooks
    def train_dataloader(self):
        return self.datamodule.train_dataloader(drop_last=True)

    def train_te_dataloader(self):
        return self.datamodule.train_te_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()


def get_mask(bsz, seq_len, device, mask_ratio):
    len_keep = int(seq_len * (1 - mask_ratio))
    noise = torch.rand(bsz, seq_len, device=device)  # noise in [0, 1]
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    mask = torch.ones([bsz, seq_len], device=device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    mask_info = {
        'ids_restore': ids_restore,
        'ids_keep': ids_keep,
        'len_keep': len_keep,
        'mask_for_mae': mask
    }
    return mask_info
