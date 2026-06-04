from omegaconf import OmegaConf

from typing import Tuple, Union

import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Linear, Conv2d
from torch.optim import SGD, AdamW

from lightly.models.utils import activate_requires_grad, deactivate_requires_grad
from lightly.utils.scheduler import CosineWarmupScheduler

from base import BaseModel
from models.segmentation import UperNetHead, UperNetFCNHead
from utils import LinearWarmupCosineAnnealingLR


class KNNClassifier(BaseModel):
    def __init__(
        self,
        cfg: OmegaConf,
        datamodule: pl.LightningDataModule,
        network: Module,
        num_classes: int,
        knn_k: int = 200,
        knn_t: float = 0.1,
        # topk: Tuple[int, ...] = (1, 5),
        feature_dtype: torch.dtype = torch.float32,
        save_features: bool = False,
    ):
        super().__init__(cfg, datamodule, network)
        print(network)
        self.save_hyperparameters(
            {
                "num_classes": num_classes,
                "knn_k": knn_k,
                "knn_t": knn_t,
                # "topk": topk,
                "feature_dtype": str(feature_dtype),
            }
        )
        self.num_classes = num_classes
        self.knn_k = knn_k
        self.knn_t = knn_t
        # self.topk = topk
        self.feature_dtype = feature_dtype

        self._train_features = []
        self._train_targets = []
        self._train_indices = []
        self._train_features_tensor: Union[Tensor, None] = None
        self._train_targets_tensor: Union[Tensor, None] = None

        self._test_features = []
        self._test_targets = []

        self.save_features = save_features

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features for kNN.
        If cfg.dataset.s1_only is True, use the radar branch (S1, 2 channels).
        Otherwise use the optical branch (S2, 10 channels).
        No masking, no decoder, just encoder tokens then pooled features.
        """
        m = self.model  # BaseModel stores the backbone in self.model

        # Fast path for CROMA backbones
        if hasattr(m, "optical_encoder") and hasattr(m, "radar_encoder") and hasattr(m, "attn_bias"):
            alibi = m.attn_bias.to(images.device)

            if self.cfg.dataset.s1_only:
                # Expect images shaped [B, 2, H, W]
                tokens = m.radar_encoder(imgs=images, attn_bias=alibi, mask_info=None)  # [B, L, D]
                pooled = tokens.mean(dim=1)                                             # [B, D]
                pooled = m.GAP_FFN_radar(pooled)                                        # [B, D]
            else:
                if self.cfg.dataset.s1_mm:
                    # Expect images shaped [B, 10, H, W]
                    tokens1 = m.optical_encoder(imgs=images[:, :10], attn_bias=alibi, mask_info=None)  # [B, L, D]
                    pooled1 = tokens1.mean(dim=1)                                               # [B, D]
                    pooled1 = m.GAP_FFN_optical(pooled1)                                        # [B, D]
                    # Expect images shaped [B, 2, H, W]
                    tokens2 = m.radar_encoder(imgs=images[:, 10:], attn_bias=alibi, mask_info=None)  # [B, L, D]
                    pooled2 = tokens2.mean(dim=1)                                                 # [B, D]
                    pooled2 = m.GAP_FFN_radar(pooled2)                                            # [B, D]
                    pooled = torch.cat([pooled1, pooled2], dim=1)                                 # [B, 2 * D]
                else:
                    # Expect images shaped [B, 10, H, W]
                    tokens = m.optical_encoder(imgs=images, attn_bias=alibi, mask_info=None)  # [B, L, D]
                    pooled = tokens.mean(dim=1)                                               # [B, D]
                    pooled = m.GAP_FFN_optical(pooled)                                        # [B, D]

            # Optionally honor your existing cfg.ssl.mae_knn_eval flag
            mode = getattr(getattr(self.cfg, "ssl", object()), "mae_knn_eval", "use_all_tokens")
            if mode == "use_all_tokens":
                feats = tokens.mean(dim=1)
            elif mode == "use_other_tokens":
                feats = tokens.mean(dim=1)  # no CLS token in this backbone, same as all tokens
            elif mode == "use_cls_token":
                feats = pooled             # fall back to pooled representation
            else:
                feats = pooled

            feats = F.normalize(feats, dim=1).to(self.feature_dtype)
            return feats

        # Fallbacks for non CROMA models, keep your original logic
        if hasattr(m, 'forward_encoder'):
            if self.cfg.ssl.algorithm == 'ScaleMAE':
                B = images.size(0)
                input_res = torch.tensor([10.0] * B, device=images.device)
                enc_out, _, _, _ = m.forward_encoder(images, mask_ratio=0.0, input_res=input_res)
            else:
                enc_out, _, _ = m.forward_encoder(images, mask_ratio=0.0)
            mode = getattr(self.cfg.ssl, 'mae_knn_eval', 'use_all_tokens')
            if mode == 'use_cls_token':
                feats = enc_out[:, 0, :]
            elif mode == 'use_other_tokens':
                feats = enc_out[:, 1:, :].mean(dim=1)
            else:
                feats = enc_out.mean(dim=1)
        elif hasattr(m, 'image_encoder'):
            feats = m.image_encoder(images)
        elif hasattr(m, 'encode'):
            feats = m.forward(images)
            mode = getattr(self.cfg.ssl, 'mae_knn_eval', 'use_all_tokens')
            if mode == 'use_cls_token':
                feats = feats[:, 0, :]
            elif mode == 'use_other_tokens':
                feats = feats[:, 1:, :].mean(dim=1)
            else:
                feats = feats.mean(dim=1)
        else:
            feats = m.forward(images).flatten(start_dim=1)

        feats = F.normalize(feats, dim=1).to(self.feature_dtype)
        return feats

    # def extract_features(self, images: torch.Tensor) -> torch.Tensor:
    #     """
    #     Extract features from the backbone for kNN evaluation.

    #     For a MAE backbone, use the encoder's output (with mask_ratio=0)
    #     and extract the [CLS] token (first token). For a ResNet (or similar CNN),
    #     use the regular forward pass.
    #     """
    #     if hasattr(self.model, 'forward_encoder'):
    #         if self.cfg.ssl.algorithm == 'ScaleMAE':
    #             B = images.size(0)
    #             input_res = torch.tensor([10.0] * B, device=images.device)
    #             encoder_out, _, _, _ = self.model.forward_encoder(images, mask_ratio=0.0, input_res=input_res)
    #         else:
    #             encoder_out, _, _ = self.model.forward_encoder(images, mask_ratio=0.0)
    #         if self.cfg.ssl.mae_knn_eval == 'use_cls_token':
    #             features = encoder_out[:, 0, :]
    #         elif self.cfg.ssl.mae_knn_eval == 'use_other_tokens':
    #             features = encoder_out[:, 1:, :].mean(dim=1)
    #         elif self.cfg.ssl.mae_knn_eval == 'use_all_tokens':
    #             features = encoder_out.mean(dim=1)
    #     elif hasattr(self.model, 'image_encoder'):
    #         features = self.model.image_encoder(images)
    #     elif hasattr(self.model, 'encode'):
    #         features = self.model.forward(images)
    #         if self.cfg.ssl.mae_knn_eval == 'use_cls_token':
    #             features = features[:, 0, :]
    #         elif self.cfg.ssl.mae_knn_eval == 'use_other_tokens':
    #             features = features[:, 1:, :].mean(dim=1)
    #         elif self.cfg.ssl.mae_knn_eval == 'use_all_tokens':
    #             features = features.mean(dim=1)
    #     else:
    #         # Flatten from the first non-batch dimension
    #         features = self.model.forward(images)
    #         features = features.flatten(start_dim=1)
        
    #     # Normalize the features
    #     features = F.normalize(features, dim=1).to(self.feature_dtype)
    #     return features

    def training_step(self, batch, batch_idx) -> None:
        images, targets, idx = batch[0], batch[1], batch[2]
        features = self.extract_features(images)
        # print(torch.var(features, dim=0))
        self._train_features.append(features.cpu())
        self._train_targets.append(targets.cpu())
        self._train_indices.append(idx.cpu())

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if self._train_features_tensor is None or self._train_targets_tensor is None:
            return

        images, targets, idx = batch[0], batch[1], batch[2]
        features = self.extract_features(images)
        probs = self.knn_predict(feature=features)
        preds = torch.argmax(probs, dim=1)

        output = dict(y=targets, idx=idx, loss=torch.Tensor([0]), probs=probs, preds=preds)
        self.validation_step_outputs.append(output)

    def on_validation_epoch_start(self) -> None:
        if self._train_features and self._train_targets:
            # Concatenate stored features and targets
            features = torch.cat(self._train_features, dim=0)  # shape: (N, D)
            if not self.save_features:
                self._train_features = []
            targets = self.all_gather(torch.cat(self._train_targets, dim=0))
            self._train_targets = []

            features = features.flatten(end_dim=-2).t().contiguous()
            self._train_features_tensor = features.to(self.device)
            self._train_targets_tensor = targets.contiguous().to(self.device)

    def on_train_epoch_end(self):
        if self.save_features:
            print("Saving features")
            features_path = os.path.join(self.trainer.logger.log_dir, 'feature_analysis/features.pt')
            indices_path = os.path.join(self.trainer.logger.log_dir, 'feature_analysis/indices.pt')
            torch.save(self._train_features, features_path)
            torch.save(self._train_indices, indices_path)

    @torch.no_grad()
    def knn_predict(self, feature: Tensor) -> Tensor:
        # Everything is already on GPU
        feature = feature.to(self.device)
        feature_bank = self._train_features_tensor  # (D, N), on GPU
        target_bank = self._train_targets_tensor    # (N,) or (N, C), on GPU

        sim_matrix = torch.mm(feature, feature_bank)  # (B, N)
        sim_weight, sim_indices = sim_matrix.topk(k=self.knn_k, dim=-1)

        if self.cfg.dataset.task == 'single_label':
            sim_labels = target_bank[sim_indices]
            sim_weight = (sim_weight / self.knn_t).exp()

            one_hot = torch.zeros(
                feature.size(0) * self.knn_k, self.num_classes,
                device=sim_labels.device, dtype=sim_weight.dtype
            )
            one_hot.scatter_(
                dim=-1,
                index=sim_labels.view(-1, 1),
                value=1.0
            )
            pred_scores = torch.sum(
                one_hot.view(feature.size(0), self.knn_k, self.num_classes) * sim_weight.unsqueeze(-1),
                dim=1
            )
            return pred_scores

        elif self.cfg.dataset.task == 'multi_label':
            sim_labels = target_bank[sim_indices]
            sim_weight = (sim_weight / self.knn_t).exp()
            pred_probs = (sim_labels * sim_weight.unsqueeze(-1)).sum(dim=1)
            return pred_probs

    # @torch.no_grad()
    # def knn_predict(self, feature: Tensor) -> Tensor:
    #     feature_bank = self._train_features_tensor  # (D, N)
    #     target_bank = self._train_targets_tensor    # (N,) or (N, C)

    #     # Ensure everything is on the same device
    #     feature = feature.to(self.device)
    #     feature_bank = feature_bank.to(self.device)
    #     target_bank = target_bank.to(self.device)

    #     # Compute cosine similarity: (B, D) @ (D, N) → (B, N)
    #     sim_matrix = torch.mm(feature, feature_bank)

    #     # Top-k retrieval
    #     sim_weight, sim_indices = sim_matrix.topk(k=self.knn_k, dim=-1)

    #     if self.cfg.dataset.task == 'single_label':
    #         sim_labels = target_bank[sim_indices]  # shape: (B, K)
    #         sim_weight = (sim_weight / self.knn_t).exp()

    #         one_hot = torch.zeros(
    #             feature.size(0) * self.knn_k, self.num_classes,
    #             device=sim_labels.device, dtype=sim_weight.dtype
    #         )
    #         one_hot.scatter_(
    #             dim=-1,
    #             index=sim_labels.view(-1, 1),
    #             value=1.0
    #         )

    #         pred_scores = torch.sum(
    #             one_hot.view(feature.size(0), self.knn_k, self.num_classes) * sim_weight.unsqueeze(-1),
    #             dim=1
    #         )
    #         return pred_scores

    #     elif self.cfg.dataset.task == 'multi_label':
    #         sim_labels = target_bank[sim_indices]  # shape: (B, K, C)
    #         sim_weight = (sim_weight / self.knn_t).exp()
    #         pred_probs = (sim_labels * sim_weight.unsqueeze(-1)).sum(dim=1)
    #         return pred_probs

    def on_train_epoch_start(self) -> None:
        # Set model to eval mode to disable norm layer updates.
        self.model.eval()

        # Reset features and targets.
        self._train_features = []
        self._train_targets = []
        self._train_features_tensor = None
        self._train_targets_tensor = None

        if self.save_features:
            features_save_dir = os.path.join(self.trainer.logger.log_dir, 'feature_analysis')
            print(features_save_dir)
            if not os.path.exists(features_save_dir):
                os.mkdir(features_save_dir)

    def on_fit_start(self) -> None:
        # Freeze model weights.
        deactivate_requires_grad(model=self.model)

    def on_fit_end(self) -> None:
        # Unfreeze model weights.
        activate_requires_grad(model=self.model)

    def configure_optimizers(self) -> None:
        # configure_optimizers must be implemented for PyTorch Lightning. Returning None
        # means that no optimization is performed.
        pass


class LinearClassifier(BaseModel):
    def __init__(
        self,
        cfg: OmegaConf,
        datamodule: pl.LightningDataModule,
        network: Module,
        batch_size_per_device: int,
        feature_dim: int = 2048,
        num_classes: int = 1000,
        freeze_model: bool = False,
    ) -> None:
        super().__init__(cfg, datamodule, network)
        # self.save_hyperparameters(ignore="model")

        self.batch_size_per_device = batch_size_per_device
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.freeze_model = freeze_model

        self.classification_head = Linear(feature_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, 'forward_encoder'):
            if self.cfg.ssl.algorithm == 'ScaleMAE':
                B = images.size(0)
                input_res = torch.tensor([10.0] * B, device=images.device)
                encoder_out, _, _, _ = self.model.forward_encoder(images, mask_ratio=0.0, input_res=input_res)
            else:
                encoder_out, _, _ = self.model.forward_encoder(images, mask_ratio=0.0)
            if self.cfg.ssl.mae_knn_eval == 'use_cls_token':
                features = encoder_out[:, 0, :]
            elif self.cfg.ssl.mae_knn_eval == 'use_other_tokens':
                features = encoder_out[:, 1:, :].mean(dim=1)
            elif self.cfg.ssl.mae_knn_eval == 'use_all_tokens':
                features = encoder_out.mean(dim=1)
        else:
            # CNN backbone: use the normal forward pass.
            features = self.model.forward(images)
            features = features.flatten(start_dim=1)
        
        return self.classification_head(features)

    def configure_optimizers(self):
        parameters = list(self.classification_head.parameters())
        if not self.freeze_model:
            parameters += self.model.parameters()
        optimizer = AdamW(
            parameters,
            lr=self.cfg.optim.min_lr,
            weight_decay=self.cfg.optim.weight_decay
        )

        max_intervals = int(self.trainer.max_epochs * len(self.datamodule.trainset_tr) / self.cfg.dataset.batch_size)
        warmup = 10000 if max_intervals > 10000 else 100 if max_intervals > 100 else 0

        lr_scheduler = {'scheduler': LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup,
            max_epochs=max_intervals,
            warmup_start_lr=self.cfg.optim.min_lr / 10,
            eta_min=self.cfg.optim.min_lr / 10
        ), 'name': 'learning_rate', 'interval': "step", 'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    # def configure_optimizers(self):
    #     parameters = list(self.classification_head.parameters())
    #     if not self.freeze_model:
    #         parameters += self.model.parameters()
    #     optimizer = SGD(
    #         parameters,
    #         lr=0.1 * self.batch_size_per_device * self.trainer.world_size / 256,
    #         momentum=0.9,
    #         weight_decay=0.0,
    #     )
    #     scheduler = {
    #         "scheduler": CosineWarmupScheduler(
    #             optimizer=optimizer,
    #             warmup_epochs=0,
    #             max_epochs=self.trainer.estimated_stepping_batches,
    #         ),
    #         "interval": "step",
    #     }
    #     return [optimizer], [scheduler]

    def on_fit_start(self) -> None:
        # Freeze model weights.
        if self.freeze_model:
            deactivate_requires_grad(model=self.model)

    def on_fit_end(self) -> None:
        # Unfreeze model weights.
        if self.freeze_model:
            activate_requires_grad(model=self.model)


class FinetuneLinearClassifier(LinearClassifier):
    # def configure_optimizers(self):
    #     parameters = list(self.classification_head.parameters())
    #     parameters += self.model.parameters()
    #     optimizer = SGD(
    #         parameters,
    #         lr=0.05 * self.batch_size_per_device * self.trainer.world_size / 256,
    #         momentum=0.9,
    #         weight_decay=0.0,
    #     )
    #     scheduler = {
    #         "scheduler": CosineWarmupScheduler(
    #             optimizer=optimizer,
    #             warmup_epochs=0,
    #             max_epochs=self.trainer.estimated_stepping_batches,
    #         ),
    #         "interval": "step",
    #     }
    #     return [optimizer], [scheduler]

    def configure_optimizers(self):
        parameters = list(self.classification_head.parameters())
        parameters += self.model.parameters()
        optimizer = AdamW(
            parameters,
            lr=self.cfg.optim.min_lr,
            weight_decay=self.cfg.optim.weight_decay
        )

        max_intervals = int(self.trainer.max_epochs * len(self.datamodule.trainset_tr) / self.cfg.dataset.batch_size)
        warmup = 10000 if max_intervals > 10000 else 100 if max_intervals > 100 else 0

        lr_scheduler = {'scheduler': LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup,
            max_epochs=max_intervals,
            warmup_start_lr=self.cfg.optim.min_lr / 10,
            eta_min=self.cfg.optim.min_lr / 10
        ), 'name': 'learning_rate', 'interval': "step", 'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


class UPerSegmentationModel(BaseModel):
    def __init__(
            self, 
            cfg: OmegaConf,
            datamodule: pl.LightningDataModule,
            network: Module,
            num_classes: int
    ) -> None:
        super().__init__(cfg, datamodule, network)

        self.network = network
        self.feature_encoder = network.resnet
        # disable gradient computation for the feature encoder
        for param in self.feature_encoder.parameters():
            param.requires_grad = False
        self.feature_encoder.eval()
        # the feature fusion and upsampling part of the model
        self.upernet = UperNetHead(in_channels=[64, 128, 256, 512])
        # the "classification head" of the model
        self.head = UperNetFCNHead(num_labels=num_classes, in_index=0)

    def forward_down(self, x):
        # directly taken from the original model implementation in torchvision, but we need the intermediate feature maps
        x = self.feature_encoder.conv1(x)
        x = self.feature_encoder.bn1(x)
        x = self.feature_encoder.relu(x)
        x = self.feature_encoder.maxpool(x)

        feat1 = self.feature_encoder.layer1(x)
        feat2 = self.feature_encoder.layer2(feat1)
        feat3 = self.feature_encoder.layer3(feat2)
        feat4 = self.feature_encoder.layer4(feat3)

        x = self.feature_encoder.avgpool(feat4)
        x = torch.flatten(x, 1)
        x = self.feature_encoder.fc(x)
        return [feat1, feat2, feat3, feat4], x

    def forward_up(self, feats, original_shape=None):
        assert len(feats) == 4
        fused_feats = self.upernet(feats)
        # bilinear interpolation if original_shape is provided
        # this is a modification to the original implementation, which just returns the feature maps of 1/4 of the input size
        if original_shape is not None:
            fused_feats = torch.nn.functional.interpolate(fused_feats, size=original_shape, mode='bilinear',
                                                          align_corners=False)
        out = self.head([fused_feats])
        return out

    def forward(self, x):
        original_shape = x.shape[-2:]
        feats, x = self.forward_down(x)
        x = self.forward_up(feats, original_shape)
        return x

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-4)
        return optim


class KNN_Segmentation(BaseModel):
    def __init__(
        self,
        cfg: OmegaConf,
        datamodule: pl.LightningDataModule,
        network: Module,
        num_classes: int,
        knn_k: int = 10,
        knn_t: float = 0.1,
        feature_dtype: torch.dtype = torch.float32,
        save_features: bool = False,
    ):
        super().__init__(cfg, datamodule, network)
        # self.save_hyperparameters()

        self.num_classes = num_classes
        self.knn_k = knn_k
        self.knn_t = knn_t
        self.feature_dtype = feature_dtype
        self.save_features = save_features

        self._train_features = []  # List of feature tensors, shape: (B, C, H, W)
        self._train_masks = []     # List of corresponding segmentation masks, shape: (B, H, W)

        self._train_features_tensor: Union[Tensor, None] = None  # (N_total, C)
        self._train_masks_tensor: Union[Tensor, None] = None     # (N_total,)

    def forward_semseg(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for semantic segmentation.

        Supports both MAE-style models (e.g., ScaleMAE with forward_encoder) and CNNs (e.g., ResNet).
        """
        if hasattr(self.model, 'forward_encoder'):
            if self.cfg.ssl.algorithm == 'ScaleMAE':
                B = x.size(0)
                # input_res = torch.tensor([10.0, 10.0] * B, device=x.device)
                input_res = torch.tensor([10.0], device=x.device)
                encoder_out, _, _, _ = self.model.forward_encoder(x, mask_ratio=0.0, input_res=input_res)
            else:
                encoder_out, _, _ = self.model.forward_encoder(x, mask_ratio=0.0)

            # Return token-wise features for semseg (without CLS)
            features = encoder_out[:, 1:, :]  # [B, num_patches, D]
            B, N, D = features.shape
            H = W = int(N**0.5)  # Only valid if square grid
            features = features.permute(0, 2, 1).reshape(B, D, H, W)  # (B, D, H, W)

            return features
        else:
            # CNN-based forward (e.g., ResNet backbone)
            x = self.model.resnet.conv1(x)
            x = self.model.resnet.bn1(x)
            x = self.model.resnet.relu(x)
            x = self.model.resnet.maxpool(x)

            feat1 = self.model.resnet.layer1(x)
            feat2 = self.model.resnet.layer2(feat1)
            feat3 = self.model.resnet.layer3(feat2)
            feat4 = self.model.resnet.layer4(feat3)

            return feat4  # typically [B, C, H/32, W/32]

    def training_step(self, batch, batch_idx) -> None:
        images, masks, idx = batch  # images: (B, ...), masks: (B, H, W)
        features = self.forward_semseg(images)  # expected shape: (B, C, H, W)
        features = F.interpolate(features, size=(12, 12), mode='bilinear', align_corners=False)  # 🔼 Upsample
        features = F.normalize(features, dim=1).to(self.feature_dtype)
        masks = F.interpolate(
            masks.unsqueeze(1).float(),  # (B, 1, H, W)
            size=(12, 12),               # (H_feat, W_feat)
            mode='nearest'               # avoid class interpolation
        ).squeeze(1).long()              # (B, H_feat, W_feat)
        self._train_features.append(features.detach().cpu())
        self._train_masks.append(masks.detach().cpu())

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_start(self) -> None:
        if self._train_features and self._train_masks:
            train_features = torch.cat(self._train_features, dim=0)  # (B_total, C, H, W)
            train_masks = torch.cat(self._train_masks, dim=0)        # (B_total, H, W)
            B_total, C, H, W = train_features.shape

            flat_feats = train_features.view(B_total, C, -1).permute(0, 2, 1).reshape(-1, C)
            flat_masks = train_masks.view(B_total, -1).reshape(-1)

            # Move to device
            flat_feats = flat_feats.to(self.device)
            flat_masks = flat_masks.to(self.device)

            # Subsample
            max_train_pixels = 1000000
            perm = torch.randperm(flat_feats.size(0))[:max_train_pixels]
            self._train_features_tensor = flat_feats[perm]
            self._train_masks_tensor = flat_masks[perm]

            # Cleanup
            del flat_feats, flat_masks, train_features, train_masks
            self._train_features = []
            self._train_masks = []
            torch.cuda.empty_cache()

    # def on_validation_epoch_start(self) -> None:
    #     # Concatenate all training features and masks.
    #     if self._train_features and self._train_masks:
    #         train_features = torch.cat(self._train_features, dim=0)  # (B_total, C, H, W)
    #         train_masks = torch.cat(self._train_masks, dim=0)          # (B_total, H, W)
    #         B_total, C, H, W = train_features.shape
    #         # Flatten spatial dimensions and combine all training examples.
    #         self._train_features_tensor = train_features.view(B_total, C, -1).permute(0, 2, 1).reshape(-1, C).to(self.device)
    #         self._train_masks_tensor = train_masks.view(B_total, -1).reshape(-1).to(self.device)

    def validation_step(self, batch, batch_idx) -> None:
        images, masks, idx = batch  # images: (B, ...), masks: (B, H, W)
        features = self.forward_semseg(images)  # (B, C, H, W)
        features = F.interpolate(features, size=(12, 12), mode='bilinear', align_corners=False)
        features = F.normalize(features, dim=1).to(self.feature_dtype)
        masks = F.interpolate(
            masks.unsqueeze(1).float(), size=(12, 12), mode='nearest'
        ).squeeze(1).long()  # (B, H, W)

        B, C, H, W = features.shape
        features = features.view(B, C, -1).permute(0, 2, 1)  # (B, N_pix, C)
        masks = masks.view(B, -1)  # (B, N_pix)

        predictions = []
        train_feats = self._train_features_tensor  # (N_train, C)
        train_labels = self._train_masks_tensor    # (N_train,)
        
        # Chunked distance computation
        chunk_size = 1024  # adjust based on available VRAM
        for i in range(B):
            feat = features[i]  # (N_pix, C)
            pred = []
            for start in range(0, feat.size(0), chunk_size):
                end = min(start + chunk_size, feat.size(0))
                feat_chunk = feat[start:end].to(self.device)  # (chunk, C)

                # Compute distances: (chunk, N_train)
                dists = torch.cdist(feat_chunk, train_feats, p=2)  # Euclidean
                sim_weight, sim_indices = dists.topk(k=self.knn_k, largest=False, dim=-1)

                sim_labels = train_labels[sim_indices]  # (chunk, knn_k)
                sim_weight = (-sim_weight / self.knn_t).exp()  # softmax-style weighting

                one_hot = F.one_hot(sim_labels, num_classes=self.num_classes).float()  # (chunk, knn_k, C)
                scores = (one_hot * sim_weight.unsqueeze(-1)).sum(dim=1)  # (chunk, C)
                pred_chunk = scores.argmax(dim=-1)  # (chunk,)
                pred.append(pred_chunk)
            
            pred = torch.cat(pred).view(H, W)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)  # (B, H, W)
        masks = masks.view(B, H, W)

        output = dict(y=masks, idx=idx, preds=predictions, probs=predictions, loss=torch.tensor(0.0, device=self.device))
        self.validation_step_outputs.append(output)
        return output

    # def validation_step(self, batch, batch_idx) -> None:
    #     images, masks, idx = batch  # images: (B, ...), masks: (B, H, W)
    #     features = self.forward_semseg(images)  # (B, C, H, W)
    #     features = F.interpolate(features, size=(12, 12), mode='bilinear', align_corners=False)  # 🔼 Upsample
    #     features = F.normalize(features, dim=1).to(self.feature_dtype)
    #     masks = F.interpolate(
    #         masks.unsqueeze(1).float(),  # (B, 1, H, W)
    #         size=(12, 12),               # (H_feat, W_feat)
    #         mode='nearest'               # avoid class interpolation
    #     ).squeeze(1).long()              # (B, H_feat, W_feat)
    #     B, C, H, W = features.shape

    #     # Flatten spatial dimensions per image: (B, N_pixels, C)
    #     features = features.view(B, C, -1).permute(0, 2, 1)
    #     predictions = []
    #     # Loop over each image in the batch (if memory allows, you can try to batch this)
    #     for i in range(B):
    #         feat = features[i]  # (N_pixels, C)
    #         # Compute cosine similarity with training features.
    #         # Since features are normalized, dot-product equals cosine similarity.
    #         sim_matrix = torch.mm(feat, self._train_features_tensor.t())  # (N_pixels, N_train)
    #         # Retrieve top-k similar training features per pixel.
    #         sim_weight, sim_indices = sim_matrix.topk(k=self.knn_k, dim=-1)
    #         # Gather corresponding training mask labels.
    #         sim_labels = self._train_masks_tensor[sim_indices]  # (N_pixels, knn_k)
    #         # Reweight similarities.
    #         sim_weight = (sim_weight / self.knn_t).exp()
    #         # One-hot encode the neighbor labels.
    #         one_hot = torch.zeros(sim_labels.size(0), sim_labels.size(1), self.num_classes, device=sim_labels.device)
    #         one_hot.scatter_(2, sim_labels.unsqueeze(-1), 1.0)
    #         # Weighted vote per class.
    #         scores = (one_hot * sim_weight.unsqueeze(-1)).sum(dim=1)  # (N_pixels, num_classes)
    #         pred = scores.argmax(dim=-1)  # (N_pixels,)
    #         predictions.append(pred.view(H, W))
    #     predictions = torch.stack(predictions, dim=0)  # (B, H, W)

    #     output = dict(y=masks, idx=idx, preds=predictions, probs=predictions, loss=torch.tensor(0.0, device=self.device))
    #     self.validation_step_outputs.append(output)
    #     return output

    def on_train_epoch_start(self) -> None:
        # Set model to eval mode and reset feature banks.
        self.model.eval()
        self._train_features = []
        self._train_masks = []
        self._train_features_tensor = None
        self._train_masks_tensor = None
        if self.save_features:
            features_save_dir = os.path.join(self.trainer.logger.log_dir, 'feature_analysis')
            if not os.path.exists(features_save_dir):
                os.makedirs(features_save_dir)


class LinearSegmentation(BaseModel):
    def __init__(
        self,
        cfg: OmegaConf,
        datamodule: pl.LightningDataModule,
        network: Module,
        feature_dim: int = 2048,
        num_classes: int = 21,  # Adjust based on dataset
        freeze_backbone: bool = True,
    ):
        super().__init__(cfg, datamodule, network)

        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        self.segmentation_head = Conv2d(feature_dim, num_classes, kernel_size=1)

    def forward(self, images: Tensor) -> Tensor:
        features = self.model.forward(images)
        logits = self.segmentation_head(features)
        logits = F.interpolate(logits, size=images.shape[-2:], mode='bilinear', align_corners=False)
        return logits

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = F.cross_entropy(logits, masks)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = F.cross_entropy(logits, masks)

        pred_masks = torch.argmax(logits, dim=1)
        output = dict(y=masks, pred=pred_masks, loss=loss)
        self.validation_step_outputs.append(output)

    def configure_optimizers(self):
        optimizer = AdamW(self.segmentation_head.parameters(), lr=self.cfg.optim.min_lr, weight_decay=self.cfg.optim.weight_decay)

        max_intervals = int(self.trainer.max_epochs * len(self.datamodule.trainset_tr) / self.cfg.dataset.batch_size)
        warmup = 10000 if max_intervals > 10000 else 100 if max_intervals > 100 else 0

        lr_scheduler = {'scheduler': LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup,
            max_epochs=max_intervals,
            warmup_start_lr=self.cfg.optim.min_lr / 10,
            eta_min=self.cfg.optim.min_lr / 10
        ), 'name': 'learning_rate', 'interval': "step", 'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def on_fit_start(self) -> None:
        if self.freeze_backbone:
            deactivate_requires_grad(model=self.model)

    def on_fit_end(self) -> None:
        if self.freeze_backbone:
            activate_requires_grad(model=self.model)


class FinetuneSegmentation(BaseModel):
    def __init__(
        self,
        cfg: OmegaConf,
        datamodule: pl.LightningDataModule,
        network: Module,
        feature_dim: int = 2048,
        num_classes: int = 21,  # Adjust based on your dataset
    ):
        super().__init__(cfg, datamodule, network)
        self.num_classes = num_classes
        # Segmentation head: a 1x1 convolution mapping feature_dim → num_classes.
        self.segmentation_head = Conv2d(feature_dim, num_classes, kernel_size=1)

    def forward(self, images: Tensor) -> Tensor:
        # Extract features using the pre-trained backbone.
        features = self.model.forward(images)  # Expected shape: (B, C, H, W)
        logits = self.segmentation_head(features)  # (B, num_classes, H, W)
        # Upsample logits to match the input image resolution.
        logits = F.interpolate(logits, size=images.shape[-2:], mode='bilinear', align_corners=False)
        return logits

    def training_step(self, batch, batch_idx):
        images, masks = batch  # masks: (B, H, W) with integer class labels
        logits = self(images)
        loss = F.cross_entropy(logits, masks)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = F.cross_entropy(logits, masks)
        pred_masks = torch.argmax(logits, dim=1)
        output = dict(y=masks, pred=pred_masks, loss=loss)
        self.validation_step_outputs.append(output)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        # Fine-tuning: update both the segmentation head and the backbone.
        parameters = list(self.segmentation_head.parameters()) + list(self.model.parameters())
        optimizer = AdamW(
            parameters,
            lr=self.cfg.optim.min_lr,
            weight_decay=self.cfg.optim.weight_decay
        )
        max_intervals = int(self.trainer.max_epochs * len(self.datamodule.trainset_tr) / self.cfg.dataset.batch_size)
        warmup = 10000 if max_intervals > 10000 else 100 if max_intervals > 100 else 0

        lr_scheduler = {
            'scheduler': LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=warmup,
                max_epochs=max_intervals,
                warmup_start_lr=self.cfg.optim.min_lr / 10,
                eta_min=self.cfg.optim.min_lr / 10
            ),
            'name': 'learning_rate',
            'interval': "step",
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


def get_knn_classifier(task):
    if task == 'segmentation':
        return KNN_Segmentation
    elif task == 'multi_label' or 'single_label':
        return KNNClassifier


def get_linear_classifier(task):
    if task == 'segmentation':
        return LinearSegmentation
    elif task == 'multi_label' or 'single_label':
        return LinearClassifier


def get_finetune_classifier(task):
    if task == 'segmentation':
        return FinetuneSegmentation
    elif task == 'multi_label' or 'single_label':
        return FinetuneLinearClassifier
