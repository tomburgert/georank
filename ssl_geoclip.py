from typing import Any

import torch

from pytorch_lightning import LightningModule

import pandas as pd


class SSLGeoCLIP(LightningModule):
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
        self.network = network  # model.

        self.criterion = torch.nn.CrossEntropyLoss()

        self._load_geography_data()

    def _load_geography_data(self) -> None:
        if self.cfg.params.dataset == 'SSL4EO':
            self.df_geo_metadata = pd.read_parquet(
                '/data/tomburgert/data/additional_data/ssl4eo_patch_id_to_kmeans_v3.parquet'
            )
        elif self.cfg.params.dataset == 'SSL4EOEurope':
            self.df_geo_metadata = pd.read_parquet(
                '/data/tomburgert/data/additional_data/ssl4eo_europe_patch_id_to_kmeans_v3.parquet'
            )
        elif self.cfg.params.dataset == 'BigEarthNetV2':
            self.df_geo_metadata = pd.read_parquet(
                '/data/tomburgert/bigearthnet_stats/benv2_patch_id_to_country_kmeans_v14.parquet'
            )

    def training_step(self, batch, batch_idx):
        # Unpack your batch (e.g., samples, labels)
        gps_queue = self.network.get_gps_queue()

        imgs = batch[0]
        indices = batch[2].cpu().detach()
        df = self.df_geo_metadata.iloc[indices]
        gps_locations = torch.tensor([*zip(df.latitude.values, df.longitude.values)], dtype=torch.float32).to(self.device)

        # Append GPS Queue & Queue Update
        gps_all = torch.cat([gps_locations, gps_queue], dim=0)
        self.network.dequeue_and_enqueue(gps_locations)

        # Forward pass
        logits_img_gps = self.network(imgs, gps_all)

        targets = torch.arange(imgs.size(0), device=self.device)  # shape [B]

        # Contrastive classification loss
        loss = self.criterion(logits_img_gps, targets)

        # Optionally, log training loss (here we use Lightning's self.log for minimal reporting)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.00003, weight_decay=0.000001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.87)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # decay every epoch
                "frequency": 1
            }
        }

        # Common dataloader hooks.
    def train_dataloader(self):
        return self.datamodule.train_dataloader(drop_last=True)

    def train_te_dataloader(self):
        return self.datamodule.train_te_dataloader()

    def val_dataloader(self):
        return self.datamodule.test_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()
