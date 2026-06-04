import copy
from typing import Any

import pandas as pd

import torch
import torch.nn as nn

from pytorch_lightning import LightningModule
from lightly.loss import DINOLoss, NTXentLoss
from lightly.models.modules import DINOProjectionHead, MAEDecoderTIMM
from lightly.models import utils
from lightly.utils.scheduler import cosine_schedule
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.models.modules import MaskedVisionTransformerTIMM
from lightly.utils.lars import LARS


# ---------------------------------------------
# WorldEncoder module
# ---------------------------------------------
class WorldEncoder(nn.Module):
    def __init__(self, input_mode='spherical', proj_type='linear', out_dim=768, freeze=False):
        super().__init__()
        self.input_mode = input_mode

        if input_mode == 'spherical':
            in_dim = 3
            self.encoder_fn = self._spherical
        elif input_mode == 'fourier':
            self.freqs = 4
            in_dim = self.freqs * 4
            self.encoder_fn = self._fourier
        elif input_mode == 'raw':
            in_dim = 2
            self.encoder_fn = self._raw
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")

        if proj_type == 'linear':
            self.proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                # nn.LayerNorm(out_dim)
            )
        elif proj_type == 'mlp':
            self.proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
                # nn.LayerNorm(out_dim),
            )
        elif proj_type == 'frozen_linear':
            self.proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                # nn.LayerNorm(out_dim)
            )
            for p in self.proj[0].parameters():
                p.requires_grad = False
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")

        if isinstance(self.proj, nn.Linear):
            self.out_dim = self.proj.out_features
        elif isinstance(self.proj, nn.Sequential):
            # Assumes the last layer is a Linear layer
            # if isinstance(self.proj[-2], nn.Linear):
            #     self.out_dim = self.proj[-2].out_features
            if isinstance(self.proj[-1], nn.Linear):
                self.out_dim = self.proj[-1].out_features
            else:
                raise TypeError("Last layer of MLP projection is not nn.Linear")
        else:
            raise TypeError("Unknown projection module type")

    def forward(self, latlon):
        encoded = self.encoder_fn(latlon)
        projected = self.proj(encoded)
        return projected

    def _raw(self, latlon):
        # Normalize latitude [-90, 90] → [-1, 1], longitude [-180, 180] → [-1, 1]
        lat = latlon[:, 0] / 90.0
        lon = latlon[:, 1] / 180.0
        return torch.stack([lat, lon], dim=-1)

    def _spherical(self, latlon):
        lat, lon = torch.deg2rad(latlon[:, 0]), torch.deg2rad(latlon[:, 1])
        x = torch.cos(lat) * torch.cos(lon)
        y = torch.cos(lat) * torch.sin(lon)
        z = torch.sin(lat)
        return torch.stack([x, y, z], dim=-1)

    def _fourier(self, latlon):
        latlon = torch.deg2rad(latlon)
        freqs = 2 ** torch.arange(self.freqs, device=latlon.device).float()
        f_lat = torch.cat([torch.sin(freqs * latlon[:, 0:1]), torch.cos(freqs * latlon[:, 0:1])], dim=-1)
        f_lon = torch.cat([torch.sin(freqs * latlon[:, 1:2]), torch.cos(freqs * latlon[:, 1:2])], dim=-1)
        return torch.cat([f_lat, f_lon], dim=-1)


# ---------------------------------------------
# ViTWorldWrapper module
# ---------------------------------------------
class ViTWorldWrapper(nn.Module):
    def __init__(self, vit, world_encoder, injection_mode="add_cls"):
        super().__init__()
        self.vit = vit
        self.world_encoder = world_encoder
        self.injection_mode = injection_mode

        assert injection_mode in {"add_cls", "add", "concat", "concat_cls", "late_add", "late_concat", "none"}

        self.embed_dim = vit.embed_dim
        self.cls_token = hasattr(vit, 'cls_token')

        if injection_mode in {"concat", "concat_cls"}:
            world_dim = world_encoder.out_dim
            self.concat_proj = nn.Sequential(
                nn.Linear(self.embed_dim + world_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim)
            )

        if injection_mode in {"late_concat"}:
            world_dim = world_encoder.out_dim
            self.concat_proj_late = nn.Sequential(
                nn.Linear(self.embed_dim + world_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.GELU()
            )

        self.out_dim = self.embed_dim

    def forward(self, x, world_info=None):
        B = x.shape[0]

        tokens = self.vit.patch_embed(x)
        if self.cls_token:
            cls_token = self.vit.cls_token.expand(B, -1, -1)
            tokens = torch.cat((cls_token, tokens), dim=1)

        pos_embed = self.vit.pos_embed[:, :tokens.size(1), :]
        tokens = tokens + pos_embed

        # === Early injection block ===
        if self.injection_mode in {"add_cls", "add", "cls_concat", "concat"} and world_info is not None:
            world_emb = self.world_encoder(world_info)

            if self.injection_mode == "add_cls":
                tokens[:, 0:1, :] += world_emb.unsqueeze(1)
            elif self.injection_mode == "add":
                tokens += world_emb.unsqueeze(1)
            elif self.injection_mode == "cls_concat":
                cls_token = torch.cat([tokens[:, 0:1, :], world_emb.unsqueeze(1)], dim=-1)
                cls_token = self.concat_proj(cls_token)
                tokens[:, 0:1, :] = cls_token
            elif self.injection_mode == "concat":
                world_emb_exp = world_emb.unsqueeze(1).repeat(1, tokens.size(1), 1)
                tokens = torch.cat([tokens, world_emb_exp], dim=-1)
                tokens = self.concat_proj(tokens)

        # Transformer
        tokens = self.vit.pos_drop(tokens)
        tokens = self.vit.blocks(tokens)
        tokens = self.vit.norm(tokens)

        cls_token_out = tokens[:, 0]

        # === Late injection block ===
        if self.injection_mode in {"late_add", "late_concat"} and world_info is not None:
            world_emb = self.world_encoder(world_info)

            if self.injection_mode == "late_add":
                cls_token_out = cls_token_out + world_emb
            elif self.injection_mode == "late_concat":
                cls_token_out = self.concat_proj_late(torch.cat([cls_token_out, world_emb], dim=-1))

        return cls_token_out

    # def forward(self, x, world_info=None):
    #     B = x.shape[0]

    #     tokens = self.vit.patch_embed(x)
    #     if self.cls_token:
    #         cls_token = self.vit.cls_token.expand(B, -1, -1)
    #         tokens = torch.cat((cls_token, tokens), dim=1)

    #     pos_embed = self.vit.pos_embed[:, :tokens.size(1), :]
    #     tokens = tokens + pos_embed

    #     if self.injection_mode != "none" and world_info is not None:
    #         world_emb = self.world_encoder(world_info)

    #         if self.injection_mode == "add_cls":
    #             tokens[:, 0:1, :] += world_emb.unsqueeze(1)
    #         elif self.injection_mode == "add":
    #             tokens += world_emb.unsqueeze(1)
    #         elif self.injection_mode == "concat_cls":
    #             cls_token = torch.cat([tokens[:, 0:1, :], world_emb.unsqueeze(1)], dim=-1)
    #             cls_token = self.concat_proj(cls_token)
    #             tokens[:, 0:1, :] = cls_token
    #         elif self.injection_mode == "concat":
    #             world_emb_exp = world_emb.unsqueeze(1).repeat(1, tokens.size(1), 1)
    #             tokens = torch.cat([tokens, world_emb_exp], dim=-1)
    #             tokens = self.concat_proj(tokens)

    #     tokens = self.vit.pos_drop(tokens)
    #     tokens = self.vit.blocks(tokens)
    #     tokens = self.vit.norm(tokens)

    #     return tokens[:, 0]  # Return CLS token only

    # def encode(self, images, idx_keep=None, latlon=None):
    #     return self.forward(images, world_info=latlon)


def sample_mask_preserve_token(seq_len, mask_ratio, device):
    patch_idx = torch.arange(1, seq_len, device=device)              # index 0 is reserved
    num_mask = int((seq_len - 1) * mask_ratio)
    perm = patch_idx[torch.randperm(len(patch_idx))]
    idx_mask = perm[:num_mask]
    idx_keep = torch.cat([torch.zeros(1, device=device, dtype=torch.long),
                          perm[num_mask:]], dim=0)
    return idx_keep.unsqueeze(0), idx_mask.unsqueeze(0)


# class MaskedViTWorldWrapper(nn.Module):
#     def __init__(self, masked_vit_timm, world_encoder=None):
#         super().__init__()
#         self.masked_vit_timm = masked_vit_timm
#         self.world_encoder = world_encoder
#         self.embed_dim = masked_vit_timm.vit.embed_dim

#     def forward(self, images, idx_keep=None, latlon=None):
#         return self.encode(images, idx_keep=idx_keep, latlon=latlon)

#     def encode(self, images, idx_keep=None, latlon=None):
#         tokens = self.masked_vit_timm.encode(images, idx_keep=idx_keep)
#         if self.world_encoder is not None and latlon is not None:
#             world_embed = self.world_encoder(latlon).unsqueeze(1)
#             tokens = tokens + world_embed
#         return tokens

#     @property
#     def sequence_length(self):
#         return self.masked_vit_timm.sequence_length

#     @property
#     def vit(self):
#         return self.masked_vit_timm.vit


class MaskedViTWorldWrapper(nn.Module):
    def __init__(self, masked_vit_timm, world_encoder, injection_mode="add"):
        super().__init__()
        assert injection_mode in {"add", "world_token"}
        self.injection_mode = injection_mode
        self.masked_vit_timm = masked_vit_timm
        self.world_encoder = world_encoder
        self.embed_dim = masked_vit_timm.vit.embed_dim

        if self.injection_mode == "world_token":
            self.world_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            nn.init.trunc_normal_(self.world_token, std=0.02)
            old_pos = masked_vit_timm.vit.pos_embed         # shape (1, N, D)
            self.register_buffer("pos_embed",
                                 torch.cat([torch.zeros(1, 1, self.embed_dim), old_pos], dim=1))

    def forward(self, images, idx_keep=None, latlon=None):
        return self.encode(images, idx_keep=idx_keep, latlon=latlon)

    # sequence length rises by one only in token mode
    @property
    def sequence_length(self):
        if self.injection_mode == "world_token":
            return self.masked_vit_timm.sequence_length + 1
        return self.masked_vit_timm.sequence_length

    @property
    def vit(self):
        return self.masked_vit_timm.vit

    # encoder entry point used by MAEWorld.forward_encoder
    def encode(self, images, idx_keep=None, latlon=None):
        if self.injection_mode == "add":
            tokens = self.masked_vit_timm.encode(images, idx_keep=idx_keep)
            if self.world_encoder is not None and latlon is not None:
                world_embed = self.world_encoder(latlon).unsqueeze(1)
                tokens = tokens + world_embed
            return tokens

        elif self.injection_mode == "world_token":
            # token mode
            B = images.size(0)
            patch_tokens = self.vit.patch_embed(images)                 # (B, N, D)
            world_tok = self.world_token.expand(B, -1, -1)              # (B, 1, D)
            if self.world_encoder is not None and latlon is not None:
                world_tok = world_tok + self.world_encoder(latlon).unsqueeze(1)
            tokens = torch.cat([world_tok, patch_tokens], dim=1)        # (B, 1+N, D)
            tokens = tokens + self.pos_embed[:, : tokens.size(1)]
            tokens = self.vit.pos_drop(tokens)
            tokens = self.vit.blocks(tokens)
            tokens = self.vit.norm(tokens)
            return tokens


# ---------------------------------------------
# Helper: Conditional world-wrapper logic
# ---------------------------------------------
def maybe_apply_world_wrapper(cfg, network):

    if hasattr(network, 'vit'):
        vit_embed_dim = network.vit.embed_dim
    else:
        vit_embed_dim = network.embed_dim

    # Create world encoder only if enabled
    world_encoder = None
    if cfg.ssl.use_world_encoding:
        world_encoder = WorldEncoder(
            input_mode=cfg.ssl.world_input_mode,
            proj_type=cfg.ssl.world_proj_type,
            out_dim=vit_embed_dim,
            freeze=cfg.ssl.world_freeze,
        )

    # Use unified MaskedViTWorldWrapper for both cases
    if isinstance(network, MaskedVisionTransformerTIMM):
        return MaskedViTWorldWrapper(
            masked_vit_timm=network,
            world_encoder=world_encoder,
            injection_mode=cfg.ssl.injection_mode
        )
    else:
        # Same logic applies for ViTWorldWrapper if you want
        return ViTWorldWrapper(
            vit=network,
            world_encoder=world_encoder,
            injection_mode=cfg.ssl.injection_mode
        )


class DINOWorld(LightningModule):
    def __init__(self, cfg: Any, datamodule: Any, network: Any) -> None:
        super().__init__()
        self.save_hyperparameters('cfg')
        self.cfg = cfg
        self.datamodule = datamodule

        # Load geo metadata only if needed for world encoding
        self.df_geo_metadata = None
        if self.cfg.ssl.use_world_encoding:
            self._load_geography_data()

        # Wrap network if world encoding is enabled
        self.backbone = maybe_apply_world_wrapper(cfg, network)
        self.feature_dim = self.backbone.embed_dim

        # Setup student & teacher heads
        self.student_backbone = self.backbone
        self.student_head = DINOProjectionHead(self.feature_dim, 512, 64, 2048, freeze_last_layer=1)
        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_head = DINOProjectionHead(self.feature_dim, 512, 64, 2048)

        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion_ssl = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

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

    def forward(self, x: torch.Tensor, latlon: torch.Tensor = None) -> torch.Tensor:
        if self.cfg.ssl.use_world_encoding:
            features = self.student_backbone(x, latlon)
        else:
            features = self.student_backbone(x)
        features = self.student_head(features)
        return features

    def forward_teacher(self, x: torch.Tensor, latlon: torch.Tensor = None) -> torch.Tensor:
        if self.cfg.ssl.use_world_encoding:
            features = self.teacher_backbone(x, latlon)
        else:
            features = self.teacher_backbone(x)
        out = self.teacher_head(features)
        return out

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)

        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]

        # Prepare world encoding if enabled
        if self.cfg.ssl.use_world_encoding:
            indices = batch[2].cpu().detach()
            df = self.df_geo_metadata.iloc[indices]
            gps_locations = torch.tensor([*zip(df.latitude.values, df.longitude.values)]).float().to(self.device)
        else:
            gps_locations = None

        # Teacher gets no world info
        teacher_outputs = [self.forward_teacher(view, None) for view in global_views]
        # Student gets world info if enabled
        student_outputs = [self.forward(view, gps_locations) for view in views]

        loss_ssl = self.criterion_ssl(teacher_outputs, student_outputs, epoch=self.current_epoch)

        self.log_dict(
            {"train_loss": loss_ssl, "ema_momentum": momentum, "epoch": self.current_epoch},
            prog_bar=True, sync_dist=True, batch_size=len(views[0]),
            on_step=not self.cfg.params.log_on_epoch, on_epoch=self.cfg.params.log_on_epoch,
        )
        return loss_ssl

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        return self.datamodule.train_dataloader(drop_last=True)

    def val_dataloader(self):
        return self.datamodule.test_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()


class MoCoV3World(LightningModule):
    def __init__(self, cfg: Any, datamodule: Any, network: Any) -> None:
        super().__init__()
        self.save_hyperparameters('cfg')
        self.cfg = cfg
        self.datamodule = datamodule

        # Load geo metadata if world encoding is enabled
        self.df_geo_metadata = None
        if self.cfg.ssl.use_world_encoding:
            self._load_geography_data()

        # Wrap network with WorldWrapper if needed
        self.backbone = maybe_apply_world_wrapper(cfg, network)
        self.feature_dim = self.backbone.embed_dim

        # Projection head (MoCoV3: typically 2-layer MLP w/o BN)
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 256),
            # nn.LayerNorm(256)
        )

        # Momentum copies
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # Contrastive loss (standard NT-Xent)
        self.criterion_ssl = NTXentLoss(
            temperature=self.cfg.ssl.moco_temperature,
            memory_bank_size=self.cfg.ssl.moco_memory_bank_size
        )

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

    def forward(self, x: torch.Tensor, latlon: torch.Tensor = None) -> torch.Tensor:
        if self.cfg.ssl.use_world_encoding:
            features = self.backbone(x, latlon)
        else:
            features = self.backbone(x)
        features = self.projection_head(features)
        return features

    def forward_momentum(self, x: torch.Tensor, latlon: torch.Tensor = None) -> torch.Tensor:
        if self.cfg.ssl.use_world_encoding:
            features = self.backbone_momentum(x, latlon)
        else:
            features = self.backbone_momentum(x)
        features = self.projection_head_momentum(features)
        return features.detach()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        momentum = cosine_schedule(self.current_epoch, self.trainer.max_epochs, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)

        (x_query, x_key) = batch[0]

        # Prepare world encoding if enabled
        if self.cfg.ssl.use_world_encoding:
            indices = batch[2].cpu().detach()
            df = self.df_geo_metadata.iloc[indices]
            gps_locations = torch.tensor([*zip(df.latitude.values, df.longitude.values)]).float().to(self.device)
        else:
            gps_locations = None

        query = self.forward(x_query, gps_locations)
        key = self.forward_momentum(x_key, None)

        loss_ssl = self.criterion_ssl(query, key)

        self.log_dict(
            {"train_loss": loss_ssl, "ema_momentum": momentum, "epoch": self.current_epoch},
            prog_bar=True, sync_dist=True, batch_size=len(x_query),
            on_step=not self.cfg.params.log_on_epoch, on_epoch=self.cfg.params.log_on_epoch
        )
        return loss_ssl

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return LARS(self.parameters(), lr=self.cfg.ssl.moco_lr, momentum=0.9, weight_decay=1e-6)
        # return torch.optim.Adam(self.parameters(), lr=self.cfg.ssl.moco_lr)

    def train_dataloader(self):
        return self.datamodule.train_dataloader(drop_last=True)

    def val_dataloader(self):
        return self.datamodule.test_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()


class MAEWorld(LightningModule):
    def __init__(self, cfg, datamodule, network):
        super().__init__()
        self.cfg = cfg
        self.datamodule = datamodule
        self.df_geo_metadata = None
        self.backbone = maybe_apply_world_wrapper(cfg, network)
        self.network = network
        self.feature_dim = self.backbone.vit.embed_dim
        self.sequence_length = self.backbone.sequence_length
        self.mask_ratio = self.cfg.ssl.mae_mask_ratio
        self.patch_size = self.backbone.vit.patch_embed.patch_size[0]

        self.decoder = MAEDecoderTIMM(
            num_patches=self.backbone.vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=self.backbone.vit.embed_dim,
            decoder_embed_dim=self.cfg.ssl.mae_decoder_dim,
            decoder_depth=1,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        )

        # --- on-the-fly extension for world-token mode -------------------------
        if self.cfg.ssl.injection_mode == "world_token":
            with torch.no_grad():
                old_pos = self.decoder.decoder_pos_embed               # (1, 197, D)
                B, L_old, D = old_pos.shape                            # L_old = 197
                new_pos = nn.Parameter(torch.zeros(1, L_old + 1, D, device=old_pos.device, dtype=old_pos.dtype))
                new_pos.data[:, 1:, :] = old_pos                       # copy existing weights to slots 1 … end
                nn.init.trunc_normal_(new_pos.data[:, :1, :], std=0.02)  # init the new slot (index 0)
                self.decoder.decoder_pos_embed = new_pos               # replace buffer / parameter
        # -----------------------------------------------------------------------

        self.decoder.decoder_pred = nn.Linear(
            self.cfg.ssl.mae_decoder_dim,
            self.patch_size * self.patch_size * 10  # ← correct output size
        )

        self.criterion = nn.MSELoss()
        if self.cfg.ssl.use_world_encoding:
            self._load_geography_data()

    def _load_geography_data(self):
        if self.cfg.params.dataset == 'SSL4EO':
            self.df_geo_metadata = pd.read_parquet('/data/tomburgert/data/additional_data/ssl4eo_patch_id_to_kmeans_v3.parquet')
        elif self.cfg.params.dataset == 'SSL4EOEurope':
            self.df_geo_metadata = pd.read_parquet('/data/tomburgert/data/additional_data/ssl4eo_europe_patch_id_to_kmeans_v3.parquet')
        elif self.cfg.params.dataset == 'BigEarthNetV2':
            self.df_geo_metadata = pd.read_parquet('/data/tomburgert/bigearthnet_stats/benv2_patch_id_to_country_kmeans_v14.parquet')

    def forward_encoder(self, images, idx_keep=None, latlon=None):
        output = self.backbone.encode(images=images, idx_keep=idx_keep, latlon=latlon)
        # output = self.backbone.encode(images=images, idx_keep=idx_keep)
        return output

    # def forward_decoder(self, x_encoded, idx_keep, idx_mask):
    #     B = x_encoded.shape[0]

    #     if self.cfg.ssl.injection_mode == "world_token":
    #         # drop the token and adjust the index sets
    #         x_encoded = x_encoded[:, 1:, :]            # (B, L_vis, D)
    #         idx_keep_patch = idx_keep[:, 1:] - 1       # now ∈ [0 … N_patches−1]
    #         idx_mask_patch = idx_mask[:, 1:] - 1
    #         L_patches = self.sequence_length - 1
    #     else:
    #         idx_keep_patch = idx_keep
    #         idx_mask_patch = idx_mask
    #         L_patches = self.sequence_length

    #     # embed the visible tokens
    #     x_encoded_embed = self.decoder.embed(x_encoded)

    #     # build full token grid (mask + visible)
    #     x_masked = utils.repeat_token(self.decoder.mask_token, (B, L_patches))
    #     x_masked = x_masked.detach()
    #     x_masked = utils.set_at_index(
    #         x_masked, idx_keep_patch, x_encoded_embed.type_as(x_masked)
    #     )

    #     x_decoded = self.decoder.decode(x_masked)
    #     x_pred = utils.get_at_index(x_decoded, idx_mask_patch)
    #     x_pred = self.decoder.predict(x_pred)
    #     return x_pred, idx_mask_patch      # return mask indices for loss

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        batch_size = x_encoded.shape[0]

        # Apply embed directly to full encoded tokens
        x_encoded_embed = self.decoder.embed(x_encoded)

        x_masked = utils.repeat_token(self.decoder.mask_token, (batch_size, self.sequence_length))
        x_masked = x_masked.detach() 
        x_masked = utils.set_at_index(x_masked, idx_keep, x_encoded_embed.type_as(x_masked))

        x_decoded = self.decoder.decode(x_masked)
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def training_step(self, batch, batch_idx):
        images = batch[0]
        B = images.size(0)

        if self.cfg.ssl.injection_mode == "world_token":
            idx_keep, idx_mask = sample_mask_preserve_token(
                self.backbone.sequence_length, self.mask_ratio, images.device)
            idx_keep = idx_keep.repeat(B, 1)
            idx_mask = idx_mask.repeat(B, 1)
        else:
            idx_keep, idx_mask = utils.random_token_mask(
                size=(B, self.backbone.sequence_length),
                mask_ratio=self.mask_ratio,
                device=images.device)

        gps_locations = None
        if self.cfg.ssl.use_world_encoding:
            indices = batch[2].cpu().detach()
            df = self.df_geo_metadata.iloc[indices]
            gps_locations = torch.tensor([*zip(df.latitude.values, df.longitude.values)]
                                         ).float().to(self.device)

        x_encoded = self.forward_encoder(images, idx_keep=idx_keep, latlon=gps_locations)

        x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)
        patches = utils.patchify(images, self.patch_size)
        if self.cfg.ssl.injection_mode == "world_token":
            idx_mask = idx_mask[idx_mask != 0].view(B, -1)
        target = utils.get_at_index(patches, idx_mask)

        loss = self.criterion(x_pred, target)
        self.log('train_loss', loss)
        return loss

    # def training_step(self, batch, batch_idx):
    #     images = batch[0]
    #     batch_size = images.shape[0]
    #     idx_keep, idx_mask = utils.random_token_mask(
    #         size=(batch_size, self.sequence_length),
    #         mask_ratio=self.mask_ratio,
    #         device=images.device,
    #     )

    #     if self.cfg.ssl.use_world_encoding:
    #         indices = batch[2].cpu().detach()
    #         df = self.df_geo_metadata.iloc[indices]
    #         gps_locations = torch.tensor([*zip(df.latitude.values, df.longitude.values)]).float().to(self.device)
    #     else:
    #         gps_locations = None

    #     x_encoded = self.forward_encoder(images, idx_keep=idx_keep, latlon=gps_locations)

    #     x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)
    #     patches = utils.patchify(images, self.patch_size)
    #     target = utils.get_at_index(patches, idx_mask - 1)
    #     loss = self.criterion(x_pred, target)
    #     self.log('train_loss', loss)
    #     return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.ssl.mae_lr)

    def train_dataloader(self):
        return self.datamodule.train_dataloader(drop_last=True)

    def val_dataloader(self):
        return self.datamodule.test_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()
