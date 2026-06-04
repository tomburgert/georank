import copy
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn import Identity

from lightly.loss import NTXentLoss, DINOLoss, NegativeCosineSimilarity
from lightly.models.modules import (
    MoCoProjectionHead,
    DINOProjectionHead,
    BYOLProjectionHead,
    BYOLPredictionHead,
    SimSiamProjectionHead,
    SimSiamPredictionHead,
    SimCLRProjectionHead
)
from lightly.utils.scheduler import (
    cosine_schedule,
)
from lightly.models.utils import (
    deactivate_requires_grad,
    update_momentum
)
from lightly.utils.lars import LARS

from utils import (
    GeographyMSELoss,
    SoftRankLoss,
    BYOLLoss
)

from haversine import haversine_vector

from ssl_world import DINOWorld, MoCoV3World, MAEWorld
from ssl_sota import SatMAE
from ssl_scalemae import ScaleMAE
from ssl_crossscale import CrossScaleMAE
from ssl_geoclip import SSLGeoCLIP
from ssl_croma import SSLCROMA


class BaseSSLModel(LightningModule):
    def __init__(self, cfg: Any, datamodule: Any, network: Any) -> None:
        super().__init__()
        self.save_hyperparameters('cfg')
        self.cfg = cfg
        self.datamodule = datamodule

        # Load geography-related data if enabled.
        self.df_geo_metadata = None
        if self.cfg.ssl.use_geography_loss:
            self._load_geography_data()

        # Define common parameters.
        self.alpha = self.cfg.ssl.gcl_alpha
        self.geo_k = self.cfg.ssl.gcl_geo_k

        # Initialize backbone and feature dimension.
        self.backbone, self.feature_dim = self._initialize_backbone(network)

    def _load_geography_data(self) -> None:
        if self.cfg.params.dataset == 'SSL4EO':
            self.df_geo_metadata = pd.read_parquet(
                '/data/tomburgert/data/additional_data/ssl4eo_patch_id_to_kmeans_v3.parquet'
            )
            if self.cfg.dataset.temporal_views_path is not None:
                groups = list(map(int, self.datamodule.trainset_tr.groups))
                self.df_geo_metadata["prefix_int"] = self.df_geo_metadata["patch_id"].str.split("_").str[0].astype(int)
                filtered_df = self.df_geo_metadata[self.df_geo_metadata["prefix_int"].isin(groups)]
                self.df_geo_metadata = filtered_df.drop(columns=["prefix_int"])

        elif self.cfg.params.dataset == 'SSL4EOEurope':
            self.df_geo_metadata = pd.read_parquet(
                '/data/tomburgert/data/additional_data/ssl4eo_europe_patch_id_to_kmeans_v3.parquet'
            )
        elif self.cfg.params.dataset == 'BigEarthNetV2':
            self.df_geo_metadata = pd.read_parquet(
                '/data/tomburgert/bigearthnet_stats/benv2_patch_id_to_country_kmeans_v14.parquet'
            )

    def _initialize_backbone(self, network: Any) -> Tuple[Any, int]:
        if not isinstance(network.resnet.fc, Identity):
            feature_dim = network.resnet.fc.in_features
            network.resnet.fc = Identity(feature_dim, feature_dim)
        else:
            if self.cfg.model.name in ['resnet18', 'resnet34']:
                feature_dim = 512
            elif self.cfg.model.name in ['resnet50', 'resnet101']:
                feature_dim = 2048
            else:
                raise ValueError("Unsupported model type")
        return network, feature_dim

    def _setup_geography_head(self) -> nn.Module:
        # The input dimension for the geography head is fixed to 128.
        return nn.Linear(128, self.geo_k)

    def _setup_loss_criteria(self, ssl_loss: str = 'moco') -> Tuple[Any, Any]:
        # Setup the self-supervised loss
        if ssl_loss == 'simclr':
            criterion_ssl = NTXentLoss()
        elif ssl_loss == 'moco':
            criterion_ssl = NTXentLoss(
                temperature=self.cfg.ssl.moco_temperature,
                memory_bank_size=self.cfg.ssl.moco_memory_bank_size
            )
        elif ssl_loss == 'dino':
            criterion_ssl = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)
        elif ssl_loss == 'simsiam':
            criterion_ssl = NegativeCosineSimilarity()
        elif ssl_loss == 'byol':
            criterion_ssl = NegativeCosineSimilarity()
            # criterion_ssl = BYOLLoss()
        else:
            raise ValueError("Unsupported ssl_loss criterion type")
        
        # Setup the geography loss
        if self.cfg.ssl.gcl_geo_loss == 'ayush_geo_aware':
            criterion_geo = nn.CrossEntropyLoss(reduction="mean")
        elif self.cfg.ssl.gcl_geo_loss == 'geo_mse':
            criterion_geo = GeographyMSELoss(
                temperature=self.cfg.ssl.moco_temperature,
                spacing=self.cfg.ssl.gcl_mse_spacing,
                min_dist_with_weight=self.cfg.ssl.gcl_mse_min_dist_with_weight,
                max_dist_with_weight=self.cfg.ssl.gcl_mse_max_dist_with_weight,
                soft_margin=self.cfg.ssl.gcl_mse_soft_margin,
            )
        elif self.cfg.ssl.gcl_geo_loss == 'georank':
            criterion_geo = SoftRankLoss(
                temperature=self.cfg.ssl.moco_temperature,
                reg_strength=self.cfg.ssl.gcl_softrank_reg_strength,
                min_dist_km=self.cfg.ssl.gcl_min_dist_km,
                max_dist_km=self.cfg.ssl.gcl_max_dist_km,
                soft_margin=self.cfg.ssl.gcl_rank_soft_margin,
                loss_type=self.cfg.ssl.gcl_rank_loss_type,
                distance_measure=self.cfg.ssl.gcl_geo_distance_measure
            )
        else:
            criterion_geo = None
        return criterion_ssl, criterion_geo

    # Common dataloader hooks.
    def train_dataloader(self):
        return self.datamodule.train_dataloader(drop_last=True)

    def train_te_dataloader(self):
        return self.datamodule.train_te_dataloader()

    def val_dataloader(self):
        return self.datamodule.test_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()


class MoCoV2(BaseSSLModel):
    def __init__(self, cfg: Any, datamodule: Any, network: Any) -> None:
        super().__init__(cfg, datamodule, network)
        self.projection_head = MoCoProjectionHead(self.feature_dim, 2048, 128)

        # Create the geography head only for ayush_geo_aware.
        if self.cfg.ssl.use_geography_loss and self.cfg.ssl.gcl_geo_loss == 'ayush_geo_aware':
            self.geography_aware_head = self._setup_geography_head()

        # Initialize momentum counterparts.
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion_ssl, self.criterion_geo = self._setup_loss_criteria(ssl_loss='moco')

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        country_logit = None
        # Compute the backbone features and flatten.
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        # For ayush_geo_aware, obtain additional geography prediction.
        if self.cfg.ssl.use_geography_loss and self.cfg.ssl.gcl_geo_loss == 'ayush_geo_aware':
            country_logit = self.geography_aware_head(query)
        return query, country_logit

    def forward_momentum(self, x: torch.Tensor) -> torch.Tensor:
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        momentum = cosine_schedule(self.current_epoch, self.trainer.max_epochs, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)

        x_query, x_key = batch[0]
        query, country_logit = self.forward(x_query)
        key = self.forward_momentum(x_key)
        loss_ssl = self.criterion_ssl(query, key) if not self.cfg.ssl.gcl_test else 0

        if self.cfg.ssl.use_geography_loss:
            indices = batch[2].cpu().detach()
            # For ayush_geo_aware, use the additional geography head output.
            if self.cfg.ssl.gcl_geo_loss == 'ayush_geo_aware':
                # Example for 'country'-type labels.
                df = self.df_geo_metadata.iloc[indices]
                key = 'k={}'.format(self.geo_k)
                labels = torch.tensor(df[key].to_list()).to(query.device)
                loss_geo = self.criterion_geo(country_logit, labels)
            else:  # For geo_mse and georank cases.
                df = self.df_geo_metadata.iloc[indices]
                gps_locations = torch.tensor([*zip(df.latitude.values, df.longitude.values)])
                loss_geo = self.criterion_geo(query, key, gps_locations)
            
            # deactivate one loss for test reasons
            if self.cfg.ssl.disable_geo_loss:
                loss_geo = 0
            if self.cfg.ssl.disable_ssl_loss:
                loss_ssl = 0

            loss = (1 - self.alpha) * loss_ssl + self.alpha * loss_geo if not self.cfg.ssl.gcl_test else loss_geo
            self.log_dict(
                {"train_loss": loss, "loss_ssl": loss_ssl, "loss_geo": loss_geo, "ema_momentum": momentum, "epoch": self.current_epoch},
                prog_bar=True, sync_dist=True, batch_size=len(x_query),
                on_step=not self.cfg.params.log_on_epoch, on_epoch=self.cfg.params.log_on_epoch
            )
        else:
            loss = loss_ssl
            self.log_dict(
                {"train_loss": loss, "ema_momentum": momentum, "epoch": self.current_epoch},
                prog_bar=True, sync_dist=True, batch_size=len(x_query),
                on_step=not self.cfg.params.log_on_epoch, on_epoch=self.cfg.params.log_on_epoch
            )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return LARS(self.parameters(), lr=self.cfg.ssl.moco_lr, momentum=0.9, weight_decay=1e-6)


class DINO(BaseSSLModel):
    def __init__(self, cfg: Any, datamodule: Any, network: Any) -> None:
        super().__init__(cfg, datamodule, network)
        # For DINO we hard-code input_dim to 512.
        input_dim = 512
        self.student_backbone = self.backbone
        self.student_head = DINOProjectionHead(input_dim, 512, 64, 2048, freeze_last_layer=1)
        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        if self.cfg.ssl.use_geography_loss and self.cfg.ssl.gcl_geo_loss == 'ayush_geo_aware':
            self.geography_aware_head = self._setup_geography_head()

        self.criterion_ssl, self.criterion_geo = self._setup_loss_criteria(ssl_loss='dino')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass for the student network.
        features = self.student_backbone(x).flatten(start_dim=1)
        features = self.student_head(features)
        return features

    def forward_teacher(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        country_logit = None
        features = self.teacher_backbone(x).flatten(start_dim=1)
        out = self.teacher_head(features)
        # Only for ayush_geo_aware do we compute an additional geography prediction.
        if self.cfg.ssl.use_geography_loss and self.cfg.ssl.gcl_geo_loss == 'ayush_geo_aware':
            country_logit = self.geography_aware_head(out)
        return out, country_logit

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # Update teacher parameters with momentum.
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)

        views = batch[0]
        views = [view.to(self.device) for view in views]
        # Use the first two views as "global" views.
        global_views = views[:2]
        teacher_outputs = [self.forward_teacher(view) for view in global_views]
        teacher_output_tuple = [self.forward_teacher(view) for view in global_views]
        teacher_outputs = [x[0] for x in teacher_output_tuple]
        teacher_query, teacher_country_logit = teacher_output_tuple[0]
        student_outputs = [self.forward(view) for view in views]
        student_query = student_outputs[0]

        if not self.cfg.ssl.gcl_test:
            loss_ssl = self.criterion_ssl(teacher_outputs, student_outputs, epoch=self.current_epoch)

        if self.cfg.ssl.use_geography_loss:
            indices = batch[2].cpu().detach()
            if self.cfg.ssl.gcl_geo_loss == 'ayush_geo_aware':
                # For label-based geography loss.
                df = self.df_geo_metadata.iloc[indices]
                key = 'k={}'.format(self.geo_k)
                labels = torch.tensor(df[key].to_list()).to(teacher_query.device)
                loss_geo = self.criterion_geo(teacher_country_logit, labels)
            else:
                # For geo_mse or georank we use GPS locations.
                df = self.df_geo_metadata.iloc[indices]
                gps_locations = torch.tensor([*zip(df.latitude.values, df.longitude.values)])

                loss_geo_1 = self.criterion_geo(teacher_query, teacher_query, gps_locations)
                loss_geo_2 = self.criterion_geo(student_query, student_query, gps_locations)
                loss_geo = 0.5 * loss_geo_1 + 0.5 * loss_geo_2

            # potentially disable one loss
            if self.cfg.ssl.disable_geo_loss:
                loss_geo = 0
            if self.cfg.ssl.disable_ssl_loss:
                loss_ssl = 0

            loss = (1 - self.alpha) * loss_ssl + self.alpha * loss_geo if not self.cfg.ssl.gcl_test else loss_geo

            self.log_dict(
                {"train_loss": loss, "loss_ssl": loss_ssl, "loss_geo": loss_geo, "ema_momentum": momentum, "epoch": self.current_epoch},
                prog_bar=True, sync_dist=True, batch_size=len(views[0]),
                on_step=not self.cfg.params.log_on_epoch, on_epoch=self.cfg.params.log_on_epoch,
            )
        else:
            loss = loss_ssl
            self.log_dict(
                {"train_loss": loss, "ema_momentum": momentum, "epoch": self.current_epoch},
                prog_bar=True, sync_dist=True, batch_size=len(views[0]),
                on_step=not self.cfg.params.log_on_epoch, on_epoch=self.cfg.params.log_on_epoch,
            )
        return loss

    def on_after_backward(self) -> None:
        # Cancel gradients on the last layer if needed.
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.001)


class BYOL(BaseSSLModel):
    def __init__(self, cfg: Any, datamodule: Any, network: Any) -> None:
        super().__init__(cfg, datamodule, network)
        self.projection_head = BYOLProjectionHead(512, 1024, 256)
        self.prediction_head = BYOLPredictionHead(256, 1024, 256)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        if self.cfg.ssl.use_geography_loss and self.cfg.ssl.gcl_geo_loss == 'ayush_geo_aware':
            self.geography_aware_head = self._setup_geography_head()

        self.criterion_ssl, self.criterion_geo = self._setup_loss_criteria(ssl_loss='byol')

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        country_logit = None
        features = self.backbone(x).flatten(start_dim=1)
        out = self.projection_head(features)
        out = self.prediction_head(out)
        if self.cfg.ssl.use_geography_loss and self.cfg.ssl.gcl_geo_loss == 'ayush_geo_aware':
            country_logit = self.geography_aware_head(out)
        return out, country_logit

    def forward_momentum(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone_momentum(x).flatten(start_dim=1)
        out = self.projection_head_momentum(features)
        return out.detach()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)

        (x0, x1) = batch[0]
        query1, country_logit1 = self.forward(x0)
        key1 = self.forward_momentum(x0)
        query2, _ = self.forward(x1)
        key2 = self.forward_momentum(x1)

        if not self.cfg.ssl.gcl_test:
            loss_ssl = 0.5 * (self.criterion_ssl(query1, key2) + self.criterion_ssl(query2, key1))
            # loss_ssl = self.criterion_ssl(query1, key2, query2, key1)

        if self.cfg.ssl.use_geography_loss:
            indices = batch[2].cpu().detach()
            if self.cfg.ssl.gcl_geo_loss == 'ayush_geo_aware':
                df = self.df_geo_metadata.iloc[indices]
                key = 'k={}'.format(self.geo_k)
                labels = torch.tensor(df[key].to_list()).to(query1.device)
                loss_geo = self.criterion_geo(country_logit1, labels)
            else:
                df = self.df_geo_metadata.iloc[indices]
                gps_locations = torch.tensor([*zip(df.latitude.values, df.longitude.values)])
                loss_geo = self.criterion_geo(query1, key1, gps_locations)

            # potentially disable loss
            if self.cfg.ssl.disable_geo_loss:
                loss_geo = 0
            if self.cfg.ssl.disable_ssl_loss:
                loss_ssl = 0

            loss = (1 - self.alpha) * loss_ssl + self.alpha * loss_geo if not self.cfg.ssl.gcl_test else loss_geo
            self.log_dict(
                {"train_loss": loss, "loss_ssl": loss_ssl, "loss_geo": loss_geo, "epoch": self.current_epoch},
                prog_bar=True, sync_dist=True, batch_size=len(x0),
                on_step=not self.cfg.params.log_on_epoch, on_epoch=self.cfg.params.log_on_epoch,
            )
        else:
            loss = loss_ssl
            self.log_dict(
                {"train_loss": loss, "epoch": self.current_epoch},
                prog_bar=True, sync_dist=True, batch_size=len(x0),
                on_step=not self.cfg.params.log_on_epoch, on_epoch=self.cfg.params.log_on_epoch,
            )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.06)
        # return LARS(self.parameters(), lr=self.cfg.ssl.moco_lr, momentum=0.9, weight_decay=1e-6)


class SimSiam(BaseSSLModel):
    def __init__(self, cfg: Any, datamodule: Any, network: Any) -> None:
        super().__init__(cfg, datamodule, network)
        # Initialize SimSiam-specific heads.
        self.projection_head = SimSiamProjectionHead(512, 512, 128)
        self.prediction_head = SimSiamPredictionHead(128, 64, 128)
        if self.cfg.ssl.use_geography_loss and self.cfg.ssl.gcl_geo_loss == 'ayush_geo_aware':
            self.geography_aware_head = self._setup_geography_head()

        self.criterion_ssl, self.criterion_geo = self._setup_loss_criteria(ssl_loss='simsiam')

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute backbone features.
        features = self.backbone(x).flatten(start_dim=1)
        # Projection and prediction.
        query_proj = self.projection_head(features)
        query_pred = self.prediction_head(query_proj)
        country_logit = None
        # For ayush_geo_aware, compute the additional geography prediction.
        if self.cfg.ssl.use_geography_loss and self.cfg.ssl.gcl_geo_loss == 'ayush_geo_aware':
            country_logit = self.geography_aware_head(query_proj)
        return query_proj, query_pred, country_logit

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        (x0, x1) = batch[0]
        query1, key1, country_logit1 = self.forward(x0)
        query2, key2, country_logit2 = self.forward(x1)
        
        if not self.cfg.ssl.gcl_test:
            loss_ssl = 0.5 * (self.criterion_ssl(query1, key2) + self.criterion_ssl(query2, key1))

        if self.cfg.ssl.use_geography_loss:
            indices = batch[2].cpu().detach()
            if self.cfg.ssl.gcl_geo_loss == 'ayush_geo_aware':
                # Use label-based geography loss.
                df = self.df_geo_metadata.iloc[indices]
                key = 'k={}'.format(self.geo_k)
                labels = torch.tensor(df[key].to_list()).to(query1.device)
                loss_geo = self.criterion_geo(country_logit1, labels)
            else:
                # For geo_mse or softrank, use GPS locations.
                df = self.df_geo_metadata.iloc[indices]
                gps_locations = torch.tensor([*zip(df.latitude.values, df.longitude.values)])
                loss_geo = self.criterion_geo(query1, key1, gps_locations)

            # potentially disable loss
            if self.cfg.ssl.disable_geo_loss:
                loss_geo = 0
            if self.cfg.ssl.disable_ssl_loss:
                loss_ssl = 0

            loss = (1 - self.alpha) * loss_ssl + self.alpha * loss_geo if not self.cfg.ssl.gcl_test else loss_geo
            self.log_dict(
                {"train_loss": loss, "loss_ssl": loss_ssl, "loss_geo": loss_geo, "epoch": self.current_epoch},
                prog_bar=True, sync_dist=True, batch_size=len(x0),
                on_step=not self.cfg.params.log_on_epoch, on_epoch=self.cfg.params.log_on_epoch,
            )
        else:
            loss = loss_ssl
            self.log_dict(
                {"train_loss": loss, "epoch": self.current_epoch},
                prog_bar=True, sync_dist=True, batch_size=len(x0),
                on_step=not self.cfg.params.log_on_epoch, on_epoch=self.cfg.params.log_on_epoch,
            )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.06)


class SimCLR(BaseSSLModel):
    def __init__(self, cfg: Any, datamodule: Any, network: Any) -> None:
        super().__init__(cfg, datamodule, network)
        self.projection_head = SimCLRProjectionHead(512, 2048, 2048)
        if self.cfg.ssl.use_geography_loss and self.cfg.ssl.gcl_geo_loss == 'ayush_geo_aware':
            self.geography_aware_head = self._setup_geography_head()

        self.criterion_ssl, self.criterion_geo = self._setup_loss_criteria(ssl_loss='simclr')

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        country_logit = None
        if self.cfg.ssl.use_geography_loss and self.cfg.ssl.gcl_geo_loss == 'ayush_geo_aware':
            country_logit = self.geography_aware_head(query)
        return query, country_logit

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        (x0, x1) = batch[0]
        query1, country_logit1 = self.forward(x0)
        query2, country_logit2 = self.forward(x1)
        
        if not self.cfg.ssl.gcl_test:
            loss_ssl = self.criterion_ssl(query1, query2)

        if self.cfg.ssl.use_geography_loss:
            indices = batch[2].cpu().detach()
            if self.cfg.ssl.gcl_geo_loss == 'ayush_geo_aware':
                df = self.df_geo_metadata.iloc[indices]
                labels = torch.tensor([self.country2id[x] for x in df.country]).to(query1.device)
                loss_geo = self.criterion_geo(country_logit1, labels)
            else:
                df = self.df_geo_metadata.iloc[indices]
                gps_locations = torch.tensor([*zip(df.latitude.values, df.longitude.values)])
                loss_geo = self.criterion_geo(query1, query2, gps_locations)

            # potentially disable losses
            if self.cfg.ssl.disable_geo_loss:
                loss_geo = 0
            if self.cfg.ssl.disable_ssl_loss:
                loss_ssl = 0

            loss = (1 - self.alpha) * loss_ssl + self.alpha * loss_geo if not self.cfg.ssl.gcl_test else loss_geo
            self.log_dict(
                {"train_loss": loss, "loss_ssl": loss_ssl, "loss_geo": loss_geo, "epoch": self.current_epoch},
                prog_bar=True, sync_dist=True, batch_size=len(x0),
                on_step=not self.cfg.params.log_on_epoch, on_epoch=self.cfg.params.log_on_epoch,
            )
        else:
            loss = loss_ssl
            self.log_dict(
                {"train_loss": loss, "epoch": self.current_epoch},
                prog_bar=True, sync_dist=True, batch_size=len(x0),
                on_step=not self.cfg.params.log_on_epoch, on_epoch=self.cfg.params.log_on_epoch,
            )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.06)


class Tile2Vec(BaseSSLModel):
    def __init__(self, cfg: Any, datamodule: Any, network: Any) -> None:
        super().__init__(cfg, datamodule, network)
        self.projection_head = SimCLRProjectionHead(self.feature_dim, 2048, 512)
        self.criterion_geo = nn.TripletMarginLoss(margin=self.cfg.ssl.tile2vec_margin, p=2)
        self._load_geography_data()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(z)
        return z

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, _, indices = batch
        embeddings = self.forward(x)

        # Get geographic coordinates
        df = self.df_geo_metadata.iloc[indices.cpu().numpy()]
        # coords = torch.tensor(df[["latitude", "longitude"]].values, device=self.device)
        gps_locations_np = torch.tensor([*zip(df.latitude.values, df.longitude.values)]).numpy()

        # Compute pairwise distances
        dists = torch.tensor(haversine_vector(gps_locations_np, gps_locations_np, comb=True))

        pos_thresh = self.cfg.ssl.tile2vec_pos_radius
        neg_thresh = self.cfg.ssl.tile2vec_neg_radius

        # Create masks for valid positives and negatives
        pos_mask = (dists > 0) & (dists <= pos_thresh)
        neg_mask = dists >= neg_thresh

        # Triplet index collection
        anchor_idx, pos_idx, neg_idx = [], [], []
        for i in range(dists.size(0)):
            pos_candidates = pos_mask[i].nonzero(as_tuple=False).view(-1)
            neg_candidates = neg_mask[i].nonzero(as_tuple=False).view(-1)

            if pos_candidates.numel() == 0 or neg_candidates.numel() == 0:
                continue

            pos_i = pos_candidates[torch.randint(pos_candidates.size(0), (1,))].item()
            neg_i = neg_candidates[torch.randint(neg_candidates.size(0), (1,))].item()

            anchor_idx.append(i)
            pos_idx.append(pos_i)
            neg_idx.append(neg_i)

        if len(anchor_idx) == 0:
            return torch.tensor(0.0, requires_grad=True, device=self.device)

        anchor_batch = embeddings[anchor_idx]
        positive_batch = embeddings[pos_idx]
        negative_batch = embeddings[neg_idx]
        
        loss = self.criterion_geo(anchor_batch, positive_batch, negative_batch)
        self.log_dict(
            {"train_loss": loss, "epoch": self.current_epoch},
            prog_bar=True, sync_dist=True,
            on_step=not self.cfg.params.log_on_epoch,
            on_epoch=self.cfg.params.log_on_epoch,
            batch_size=len(x),
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.cfg.ssl.tile2vec_lr, weight_decay=1e-6)


def get_ssl_model(cfg, dm, network):
    if cfg.ssl.algorithm == 'MoCoV2':
        return MoCoV2(cfg, dm, network)
    if cfg.ssl.algorithm == 'DINO':
        return DINO(cfg, dm, network)
    if cfg.ssl.algorithm == 'BYOL':
        return BYOL(cfg, dm, network)
    if cfg.ssl.algorithm == 'SimSiam':
        return SimSiam(cfg, dm, network)
    if cfg.ssl.algorithm == 'Tile2Vec':
        return Tile2Vec(cfg, dm, network)
    if cfg.ssl.algorithm == 'SimCLR':
        return SimCLR(cfg, dm, network)
    if cfg.ssl.algorithm == 'SatMAE':
        return SatMAE(cfg, dm, network)
    if cfg.ssl.algorithm == 'ScaleMAE':
        return ScaleMAE(cfg, dm, network)
    if cfg.ssl.algorithm == 'CrossScaleMAE':
        return CrossScaleMAE(cfg, dm, network)
    if cfg.ssl.algorithm == 'GeoCLIP':
        return SSLGeoCLIP(cfg, dm, network)
    if cfg.ssl.algorithm == 'DINOWorld':
        return DINOWorld(cfg, dm, network)
    if cfg.ssl.algorithm == 'MoCoV3World':
        return MoCoV3World(cfg, dm, network)
    if cfg.ssl.algorithm == 'MAEWorld':
        return MAEWorld(cfg, dm, network)
    if cfg.ssl.algorithm == 'CROMA':
        return SSLCROMA(cfg, dm, network)
