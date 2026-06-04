from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from timm.models.vision_transformer import Block, PatchEmbed

from utils import get_2d_sincos_pos_embed

import abc
from pytorch_msssim import ms_ssim, ssim


def MLP(emd_dim, channel=64, hidden_size=1024):
    return nn.Sequential(
        nn.Linear(emd_dim, hidden_size),
        nn.BatchNorm1d(channel),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, emd_dim),
    )


def mask_type_transfer(mask):
    mask = mask.type(torch.bool)
    # mask = mask.type(torch.uint8)
    return mask


def get_pos_and_neg_mask(bs):
    """Org_NTXentLoss_mask"""
    zeros = torch.zeros((bs, bs), dtype=torch.uint8)
    eye = torch.eye(bs, dtype=torch.uint8)
    pos_mask = torch.cat(
        [
            torch.cat([zeros, eye], dim=0),
            torch.cat([eye, zeros], dim=0),
        ],
        dim=1,
    )
    neg_mask = _get_correlated_mask(bs)
    # (torch.ones(2*bs, 2*bs, dtype=torch.uint8) - torch.eye(2*bs, dtype=torch.uint8))
    pos_mask = mask_type_transfer(pos_mask)
    neg_mask = mask_type_transfer(neg_mask)
    return pos_mask, neg_mask


def _get_correlated_mask(batch_size):
    diag = np.eye(2 * batch_size)
    l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
    l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
    mask = torch.from_numpy((diag + l1 + l2))
    mask = 1 - mask  # .byte()#.type(torch)
    return mask  # .to(self.device)


class NTXentLoss(nn.Module):
    """NTXentLoss

    Args:
        tau: The temperature parameter.
    """

    def __init__(self, bs, tau=0.1, cos_sim=False, eps=1e-8, device=None):
        super().__init__()
        self.name = "NTXentLoss_Org"
        self.tau = tau
        self.use_cos_sim = cos_sim
        self.eps = eps
        self.bs = bs
        self.device = device

        if cos_sim:
            self.cosine_similarity = nn.CosineSimilarity(dim=-1)
            self.name += "_CosSim"

        # Get pos and neg mask
        self.pos_mask, self.neg_mask = get_pos_and_neg_mask(bs)

        if self.device is not None:
            self.pos_mask = self.pos_mask.to(self.device)
            self.neg_mask = self.neg_mask.to(self.device)

    def forward(self, zi, zj):
        """
        input: {'zi': out_feature_1, 'zj': out_feature_2}
        target: one_hot lbl_prob_mat
        """
        zi, zj = F.normalize(zi, dim=1), F.normalize(zj, dim=1)
        bs = zi.shape[0]

        z_all = torch.cat([zi, zj], dim=0)  # input1,input2: z_i,z_j
        # [2*bs, 2*bs] -  pairwise similarity
        if self.use_cos_sim:
            sim_mat = torch.exp(
                self.cosine_similarity(z_all.unsqueeze(1), z_all.unsqueeze(0)) / self.tau
            )  # s_(i,j)
        else:
            sim_mat = torch.exp(
                torch.mm(z_all, z_all.t().contiguous()) / self.tau
            )  # s_(i,j)

        # pos = torch.sum(sim_mat * self.pos_mask, 1)
        # neg = torch.sum(sim_mat * self.neg_mask, 1)
        # loss = -(torch.mean(torch.log(pos / (pos + neg))))
        sim_pos = sim_mat.masked_select(self.pos_mask).view(2 * bs).clone()

        # [2*bs, 2*bs-1]
        sim_neg = sim_mat.masked_select(self.neg_mask).view(2 * bs, -1)
        # Compute loss
        loss = (-torch.log(sim_pos / (sim_neg.sum(dim=-1) + self.eps))).mean()

        return loss


class MAE_ViT_Shared(nn.Module):
    def __init__(
        self,
        norm_pix_loss=False,
        loss="mse",
        **kwargs,
    ):
        super().__init__()
        self.loss = loss.lower()
        self.norm_pix_loss = norm_pix_loss
        # get the loss function from class based on string
        self.__forward_loss = getattr(self, f"forward_loss_{self.loss}")
        print(
            f"__forward_loss: {self.loss} -> {self.__forward_loss.__name__} (norm_pix_loss={self.norm_pix_loss})"
        )

    def patchify(self, imgs, p, c):
        """
        imgs: (N, C, H, W)
        p: Patch embed patch size
        c: Num channels
        x: (N, L, patch_size**2 *C)
        """
        # p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # c = self.in_c
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x

    def unpatchify(self, x, p, c):
        """
        x: (N, L, patch_size**2 *C)
        p: Patch embed patch size
        c: Num channels
        imgs: (N, C, H, W)
        """
        # c = self.in_c
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    @abc.abstractmethod
    def forward_encoder(self, x, mask_ratio):
        pass

    @abc.abstractmethod
    def forward_decoder(self, x, ids_restore):
        pass

    def scale_01(self, x):
        return (x - x.min()) / (x.max() - x.min() + 1.0e-6)

    def process_target(self, imgs, patch_embed_psize, input_channels):
        """
        imgs (before): [N, 3, H, W]
        target (after): [N, L, p*p*3]
        """
        target = self.patchify(imgs, patch_embed_psize, input_channels)
        # print("target", target.shape)
        # torch.Size([512, 64, 192])

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        return target

    def forward_loss_mse(self, target, pred, mask=None, **kwargs):
        loss = (pred - target) ** 2  # torch.Size([512, 64, 192])
        # loss per patch [N, L]
        loss = loss.mean(dim=-1)  # torch.Size([512, 64])

        # mask: [N, L], 0 is visible, 1 is reconstructed,
        loss = (loss * mask).sum() / mask.sum() if mask is not None else loss.mean()
        return loss

    def forward_loss_l2(self, target, pred, mask=None, **kwargs):
        loss = (pred - target) ** 2  # torch.Size([512, 64, 192])
        # loss per patch [N, L]
        loss = loss.sum(dim=-1)  # torch.Size([512, 64])

        # mask: [N, L], 0 is visible, 1 is reconstructed,
        loss = (loss * mask).sum() / mask.sum() if mask is not None else loss.mean()
        return loss

    def forward_loss_mae(self, target, pred, mask=None, **kwargs):
        loss = torch.abs(pred - target)  # torch.Size([512, 64, 192])
        # loss per patch [N, L]
        loss = loss.mean(dim=-1)  # torch.Size([512, 64])

        # mask: [N, L], 0 is visible, 1 is reconstructed,
        loss = (loss * mask).sum() / mask.sum() if mask is not None else loss.mean()
        return loss

    def forward_loss_l1(self, target, pred, mask=None, **kwargs):
        loss = torch.abs(pred - target)  # torch.Size([512, 64, 192])
        # loss per patch [N, L]
        loss = loss.sum(dim=-1)  # torch.Size([512, 64])

        # mask: [N, L], 0 is visible, 1 is reconstructed,
        loss = (loss * mask).sum() / mask.sum() if mask is not None else loss.mean()
        return loss

    def forward_loss_bce(self, target, pred, mask=None, **kwargs):
        # From Docs:
        # input: Tensor of arbitrary shape as unnormalized scores (often referred to as logits).
        # target: Tensor of the same shape as input with values between 0 and 1
        target = self.scale_01(target)

        loss = nn.functional.binary_cross_entropy_with_logits(
            pred, target, reduction="none"
        )
        # loss per patch [N, L]
        loss = loss.mean(dim=-1)

        # mask: [N, L], 0 is visible, 1 is reconstructed,
        loss = (loss * mask).sum() / mask.sum() if mask is not None else loss.mean()
        return loss

    def forward_loss_ssim(
        self, target, pred, mask=None, patch_embed_psize=None, input_channels=None
    ):
        """
        From Docs: https://github.com/VainF/pytorch-msssim
        "If you need to calculate MS-SSIM/SSIM on normalized images,
        please denormalize them to the range of [0, 1] or [0, 255] first.
        For ssim, it is recommended to set nonnegative_ssim=True to avoid negative results.
        However, this option is set to False by default to keep it consistent with tensorflow and skimage.
        For ms-ssim, there is no nonnegative_ssim option and the ssim reponses
        is forced to be non-negative to avoid NaN results."
        """
        # Pred, Target: [N, L, p*p*3]
        # Mask: [N, L]

        # SSIM and MS-SSIM functions require input to be in range [0, 1]
        target, pred = self.scale_01(target), self.scale_01(pred)

        # SSIM and MS-SSIM functions require input [N, C, H, W]
        target = self.unpatchify(target, patch_embed_psize, input_channels)
        pred = self.unpatchify(pred, patch_embed_psize, input_channels)

        # By default perform SSIM on reconstructed masked patches only (when used stand-alone)
        # Optionally, if mask is None, perform SSIM on all patches (reconstructed and visible)
        if mask is not None:
            if patch_embed_psize is None or input_channels is None:
                raise ValueError(
                    "patch_embed_psize and input_channels must be provided if mask is provided"
                )
            mask = mask.unsqueeze(-1).repeat(
                1, 1, patch_embed_psize**2 * 3
            )  # (N, H*W, p*p*3)
            mask = self.unpatchify(
                mask, p=patch_embed_psize, c=input_channels
            )  # 1 is removing, 0 is keeping

            target = target * mask
            pred = pred * mask

        return 1 - ssim(
            pred, target, data_range=1, size_average=True, nonnegative_ssim=True
        )

    def forward_loss_ms_ssim(
        self, target, pred, mask=None, patch_embed_psize=None, input_channels=None
    ):
        """
        From Docs: https://github.com/VainF/pytorch-msssim
        "If you need to calculate MS-SSIM/SSIM on normalized images,
        please denormalize them to the range of [0, 1] or [0, 255] first.
        For ssim, it is recommended to set nonnegative_ssim=True to avoid negative results.
        However, this option is set to False by default to keep it consistent with tensorflow and skimage.
        For ms-ssim, there is no nonnegative_ssim option and the ssim reponses
        is forced to be non-negative to avoid NaN results."
        """
        # Pred, Target: [N, L, p*p*3]
        # Mask: [N, L]

        # SSIM and MS-SSIM functions require input to be in range [0, 1]
        target, pred = self.scale_01(target), self.scale_01(pred)

        # SSIM and MS-SSIM functions require input [N, C, H, W]
        target = self.unpatchify(target, patch_embed_psize, input_channels)
        pred = self.unpatchify(pred, patch_embed_psize, input_channels)

        # By default perform SSIM on reconstructed masked patches only (when used stand-alone)
        # Optionally, if mask is None, perform SSIM on all patches (reconstructed and visible)
        if mask is not None:
            if patch_embed_psize is None or input_channels is None:
                raise ValueError(
                    "patch_embed_psize and input_channels must be provided if mask is provided"
                )
            mask = mask.unsqueeze(-1).repeat(
                1, 1, patch_embed_psize**2 * 3
            )  # (N, H*W, p*p*3)
            mask = self.unpatchify(
                mask, p=patch_embed_psize, c=input_channels
            )  # 1 is removing, 0 is keeping

            target = target * mask
            pred = pred * mask

        return 1 - ms_ssim(pred, target, data_range=1, size_average=True)

    def forward_loss_mse_ssim(self, target, pred, mask=None, weight=0.1, **kwargs):
        # combines mse and ssim loss
        # Loss on only reconstructed masked patches
        loss1 = self.forward_loss_mse(target, pred, mask=mask, **kwargs)
        # SSIM on whole reconstruction (not just reconstructed masked patches)
        # - to enforce global structural consistency
        loss2 = self.forward_loss_ssim(target, pred, mask=mask, **kwargs)
        # sum of the two losses
        return loss1 + weight * loss2

    def forward_loss_mse_ms_ssim(self, target, pred, mask=None, weight=0.1, **kwargs):
        # combines mse and ms-ssim loss
        # Loss on only reconstructed masked patches
        loss1 = self.forward_loss_mse(target, pred, mask=mask, **kwargs)
        # SSIM on whole reconstruction (not just reconstructed masked patches)
        # - to enforce global structural consistency
        loss2 = self.forward_loss_ms_ssim(target, pred, mask=mask, **kwargs)
        # sum of the two losses
        return loss1 + weight * loss2

    def forward_loss(
        self,
        target,
        pred,
        mask=None,
        patch_embed_psize=None,
        input_channels=None
    ):
        # patch_embed_psize and input_channels are required if passing in a full images as target
        #       Imgs: [N, C, H, W]   <= patchified by process_target below
        # Pred: [N, L, p*p*3]
        # Mask: [N, L]
        if patch_embed_psize is not None and input_channels is not None:
            target = self.process_target(target, patch_embed_psize, input_channels)

        return self.__forward_loss(
            target,
            pred,
            mask=mask,
            patch_embed_psize=patch_embed_psize,
            input_channels=input_channels
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}


class MAE_ViT_Baseline(MAE_ViT_Shared):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        input_size=112,
        input_channels=10,
        patch_size=8,  # Must be multiple of input_size
        mask_ratio=0.75,
        dim_model=256,
        # Encoder parameters
        encoder_num_layers=6,
        encoder_num_heads=4,  # Must be multiple of dim_model
        # Decoder parameters
        decoder_embed_dim=256,
        decoder_num_layers=4,
        decoder_num_heads=4,  # Must be multiple of decoder_embed_dim
        # Residual parameters
        residual_norm_style="post",
        residual_dropout=0.0,
        # Feedforward parameters
        ffn_name="MLP",  # Note: If use_xformers=False, only MLP is supported
        ffn_activation="gelu",  # Note: if use_xformers=False, only gelu is supported
        ffn_ratio=4,
        ffn_dropout=0.0,
        # Attention parameters
        attn_name="scaled_dot_product",
        attn_dropout=0.0,
        # Other parameters
        norm_layer=partial(
            nn.LayerNorm, eps=1e-6
        ),  # Note: Only used if use_xformers=False
        use_xformers=False,
        device=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.input_channels = input_channels
        self.patch_size = int(patch_size)
        self.dim_model = dim_model
        self.decoder_embed_dim = decoder_embed_dim
        self.mask_ratio = mask_ratio
        self.use_xformers = use_xformers
        self.device = device

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        assert input_size % self.patch_size == 0

        if not use_xformers:
            assert (
                attn_name == "scaled_dot_product"
            ), f"Attention {attn_name} not supported with use_xformers=False, as Timm's implementation uses scaled_dot_product"
            assert (
                ffn_name == "MLP"
            ), f"Feedforward {ffn_name} not supported with use_xformers=False, as Timm's implementation uses MLP"
            assert (
                ffn_activation == "gelu"
            ), f"Feedforward activation {ffn_activation} not supported with use_xformers=False, as Timm's implementation uses gelu"

        self.patch_embed = PatchEmbed(
            input_size, self.patch_size, input_channels, dim_model
        )
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_model))
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, dim_model), requires_grad=False
        )  # fixed sin-cos embedding

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(dim_model, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        print("Using Timm")
        encoder_blocks = [
            Block(
                dim=dim_model,
                num_heads=encoder_num_heads,
                mlp_ratio=ffn_ratio,
                qkv_bias=True,
                proj_drop=ffn_dropout,
                attn_drop=attn_dropout,
                norm_layer=norm_layer,
                drop_path=residual_dropout,
            )
            for _ in range(encoder_num_layers)
        ]
        self.encoder = nn.ModuleList(encoder_blocks)

        decoder_blocks = [
            Block(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=ffn_ratio,
                qkv_bias=True,
                proj_drop=ffn_dropout,
                attn_drop=attn_dropout,
                norm_layer=norm_layer,
                drop_path=residual_dropout,
            )
            for _ in range(decoder_num_layers)
        ]
        self.decoder = nn.ModuleList(decoder_blocks)

        # decoder to patch
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, self.patch_size**2 * input_channels, bias=True
        )
        # --------------------------------------------------------------------------
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.encoder_norm = norm_layer(dim_model)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        encoder_pos_embed = get_2d_sincos_pos_embed(
            self.encoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.encoder_pos_embed.data.copy_(
            torch.from_numpy(encoder_pos_embed).float().unsqueeze(0)
        )

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, mask_ratio):
        # TODO: Test out adding random noise to the input
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.encoder_pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.encoder_pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks
        if self.use_xformers:
            x = self.encoder(x)
        else:
            for blk in self.encoder:
                x = blk(x)
        # LayerNorm
        self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x_embed = x + self.decoder_pos_embed

        # apply Transformer blocks
        if self.use_xformers:
            x_embed = self.decoder(x_embed)
        else:
            for blk in self.decoder:
                x_embed = blk(x_embed)
        # LayerNorm
        x_embed = self.decoder_norm(x_embed)

        # predictor projection & remove cls token
        x_pred = self.decoder_pred(x_embed)[:, 1:, :]

        return x_pred, x_embed

    def forward(self, imgs, mask_ratio=0.75,
                mask_seed=None, return_embeds=False):
        if mask_seed is not None:
            torch.manual_seed(mask_seed)

        encoder_embed, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        decoder_pred, decoder_embed = self.forward_decoder(
            encoder_embed, ids_restore
        )  # [N, L, p*p*3]

        loss = self.forward_loss(
            imgs,
            decoder_pred,
            mask,
            self.patch_embed.patch_size[0],
            self.input_channels
        )

        if not return_embeds:
            return loss, decoder_pred, mask
        else:
            return loss, decoder_pred, mask, encoder_embed, decoder_embed


class MAE_ViT_MsLd(MAE_ViT_Baseline):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        # Range of random crop for Multi-scale training
        ms_range=(0.25, 0.75),
        # Reduction for decoder losses between original and cropped images
        ms_decoder_loss_reduction: str = "sum",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ms_decoder_loss_reduction = ms_decoder_loss_reduction.lower()

        self.allowed_reductions = ["mean", "sum"]
        assert (
            self.ms_decoder_loss_reduction in self.allowed_reductions
        ), f"ms_decoder_loss_reduction must be one of: {self.allowed_reductions}"

        print(f"ms_decoder_loss_reduction: {self.ms_decoder_loss_reduction}")

        self.crop = nn.Sequential(
            T.RandomResizedCrop(
                size=(self.input_size, self.input_size),
                scale=ms_range,
                antialias=True,
            )
        )

    def forward(
        self,
        imgs,
        mask_ratio=0.75,
        mask_seed: int = None,
        return_embeds=False,
        consistent_mask=False
    ):
        if mask_seed is not None:
            torch.manual_seed(mask_seed)
        elif consistent_mask:
            # Make sure the mask is consistent for all images in the batch
            mask_seed = torch.randint(0, 2**32 - 1, (1,)).item()

        # Random crop image
        imgs_crop = self.crop(imgs)

        # Forward Original image
        loss_orig, pred_orig, mask_orig, enc_emb_orig, dec_emb_orig = super().forward(
            imgs, mask_ratio=mask_ratio, mask_seed=mask_seed, return_embeds=True
        )
        # Forward Cropped image
        loss_crop, pred_crop, mask_crop, enc_emb_crop, dec_emb_crop = super().forward(
            imgs_crop, mask_ratio=mask_ratio, mask_seed=mask_seed, return_embeds=True
        )

        # Reconstruction loss combining original and cropped image
        loss_d = loss_orig + loss_crop
        if self.ms_decoder_loss_reduction == "mean":
            loss_d /= 2

        if not return_embeds:
            return loss_d, pred_orig, mask_orig

        return (
            loss_d,
            pred_orig,
            mask_orig,
            (enc_emb_orig, enc_emb_crop),
            (dec_emb_orig, dec_emb_crop),
        )


class MAE_ViT_MsLd_PAIRED(MAE_ViT_Baseline):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        # Range of random crop for Multi-scale training
        ms_range=(0.2, 0.8),
        # Reduction for decoder losses between original and cropped images
        ms_decoder_loss_reduction: str = "sum",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ms_decoder_loss_reduction = ms_decoder_loss_reduction.lower()

        self.allowed_reductions = ["mean", "sum"]
        assert (
            self.ms_decoder_loss_reduction in self.allowed_reductions
        ), f"ms_decoder_loss_reduction must be one of: {self.allowed_reductions}"

        print(f"ms_decoder_loss_reduction: {self.ms_decoder_loss_reduction}")

        # self.crop = nn.Sequential(
        #     T.RandomResizedCrop(
        #         size=(self.input_size, self.input_size),
        #         scale=ms_range,
        #         antialias=True,
        #     )
        # )

    def forward(
        self,
        imgs1,
        imgs2,
        mask_ratio=0.75,
        mask_seed: int = None,
        return_embeds=False,
        consistent_mask=False
    ):
        if mask_seed is not None:
            torch.manual_seed(mask_seed)
        elif consistent_mask:
            # Make sure the mask is consistent for all images in the batch
            mask_seed = torch.randint(0, 2**32 - 1, (1,)).item()

        # Forward Original image
        loss_orig, pred_orig, mask_orig, enc_emb_orig, dec_emb_orig = super().forward(
            imgs1, mask_ratio=mask_ratio, mask_seed=mask_seed, return_embeds=True
        )
        # Forward Cropped image
        loss_crop, pred_crop, mask_crop, enc_emb_crop, dec_emb_crop = super().forward(
            imgs2, mask_ratio=mask_ratio, mask_seed=mask_seed, return_embeds=True
        )

        # Reconstruction loss combining original and cropped image
        loss_d = loss_orig + loss_crop
        if self.ms_decoder_loss_reduction == "mean":
            loss_d /= 2

        if not return_embeds:
            return loss_d, pred_orig, mask_orig

        return (
            loss_d,
            pred_orig,
            mask_orig,
            (enc_emb_orig, enc_emb_crop),
            (dec_emb_orig, dec_emb_crop),
        )


class MAE_ViT_MsLdCeCd(MAE_ViT_MsLd):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        loss_cd=None,
        predictor_hidden_size=2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # If None, use the same loss as for the reconstruction (decoder projection head)
        self.loss_cd = loss_cd.lower() if loss_cd is not None else self.loss
        # get the loss function from class based on string
        self.__forward_loss_cd = getattr(self, f"forward_loss_{self.loss_cd}")
        print(f"__forward_loss_cd: {self.loss_cd} -> {self.__forward_loss_cd.__name__}")

        self.predictor = MLP(
            self.decoder_embed_dim, self.num_patches, predictor_hidden_size
        )

    def forward(
        self,
        imgs,
        mask_ratio=0.75,
        contr_bs=None,
        mask_seed: int = None,
        return_embeds=False,
        consistent_mask=False,
        **kwargs,
    ):
        (
            loss_d,
            pred_orig,
            mask_orig,
            (enc_emb_orig, enc_emb_crop),
            (dec_emb_orig, dec_emb_crop),
        ) = super().forward(
            imgs,
            mask_ratio=mask_ratio,
            mask_seed=mask_seed,
            return_embeds=True,
            consistent_mask=consistent_mask,
        )

        if contr_bs:
            bs = contr_bs
        else:
            bs = imgs.shape[0]

        # Cross decoder loss between original and crop
        cross_pred = self.predictor(dec_emb_crop[:, 1:, :])
        cross_target = dec_emb_orig[:, 1:, :]
        loss_cd = self.__forward_loss_cd(cross_target, cross_pred)

        # Contrastive encoder loss between original and crop
        contrast_criterian = NTXentLoss(bs, 0.5, cos_sim=True, device=pred_orig.device)

        f1 = torch.flatten(enc_emb_orig[:, 1:, :].mean(dim=1), 1)
        f2 = torch.flatten(enc_emb_crop[:, 1:, :].mean(dim=1), 1)
        # print('f1 shape:',f1.shape)
        # print('f2 shape:',f2.shape)

        loss_ce = contrast_criterian(f1, f2)

        # Reconstruction loss + cross decoder loss
        loss_d_cd_ce = loss_d + loss_cd + loss_ce
        # loss_d_cd_ce = loss_d + loss_cd

        if not return_embeds:
            return loss_d_cd_ce, pred_orig, mask_orig

        return (
            loss_d_cd_ce,
            pred_orig,
            mask_orig,
            (enc_emb_orig, enc_emb_crop),
            (dec_emb_orig, dec_emb_crop),
        )


class MAE_ViT_MsLdCeCd_PAIRED(MAE_ViT_MsLd_PAIRED):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        device="cuda:0",
        loss_cd=None,
        # bacth_size =128,
        predictor_hidden_size=2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # If None, use the same loss as for the reconstruction (decoder projection head)
        self.loss_cd = loss_cd.lower() if loss_cd is not None else self.loss
        # get the loss function from class based on string
        self.__forward_loss_cd = getattr(self, f"forward_loss_{self.loss_cd}")
        print(f"__forward_loss_cd: {self.loss_cd} -> {self.__forward_loss_cd.__name__}")

        # self.batch_size = bacth_size
        self.device = device

        self.predictor = MLP(
            self.decoder_embed_dim, self.num_patches, predictor_hidden_size
        )

        # self.contrast_criterian = NTXentLoss(self.batch_size, self.device, 0.5, cos_sim=True)

    def forward(
        self,
        imgs1,
        imgs2,
        mask_ratio=0.75,
        contr_bs=None,
        mask_seed: int = None,
        return_embeds=False,
        consistent_mask=False,
        **kwargs,
    ):
        (
            loss_d,
            pred_orig,
            mask_orig,
            (enc_emb_orig, enc_emb_crop),
            (dec_emb_orig, dec_emb_crop),
        ) = super().forward(
            imgs1,
            imgs2,
            mask_ratio=mask_ratio,
            mask_seed=mask_seed,
            return_embeds=True,
            consistent_mask=consistent_mask,
        )

        if contr_bs:
            bs = contr_bs
        else:
            bs = imgs1.shape[0]

        # Cross decoder loss between original and crop
        cross_pred = self.predictor(dec_emb_crop[:, 1:, :])
        cross_target = dec_emb_orig[:, 1:, :]
        loss_cd = self.__forward_loss_cd(cross_target, cross_pred)

        # Contrastive encoder loss between original and crop

        contrast_criterian = NTXentLoss(self.device, bs, 0.5, cos_sim=True)

        f1 = torch.flatten(enc_emb_orig[:, 1:, :].mean(dim=1), 1)
        f2 = torch.flatten(enc_emb_crop[:, 1:, :].mean(dim=1), 1)
        # print('f1 shape:',f1.shape)
        # print('f2 shape:',f2.shape)

        loss_ce = contrast_criterian(f1, f2)

        # Reconstruction loss + cross decoder loss
        loss_d_cd_ce = loss_d + loss_cd + loss_ce
        # loss_d_cd_ce = loss_d + loss_cd

        if not return_embeds:
            return loss_d_cd_ce, pred_orig, mask_orig

        return (
            loss_d_cd_ce,
            pred_orig,
            mask_orig,
            (enc_emb_orig, enc_emb_crop),
            (dec_emb_orig, dec_emb_crop),
        )
