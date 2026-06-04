# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

# import imp
# from functools import partial
from typing import Optional
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from timm.models.vision_transformer import Block, PatchEmbed
from timm.models.vision_transformer import DropPath, Mlp

from utils import (
    get_2d_sincos_pos_embed,
    get_2d_sincos_pos_embed_with_resolution,
)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn += mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GPTBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class FPNHead(nn.Module):
    def __init__(self, embed_dim, share_weights=False) -> None:
        super().__init__()
        self.share_weights = share_weights
        if self.share_weights:
            self.fpn1 = nn.Sequential(
                Norm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            self.do_fpn1 = lambda x: self.fpn1(self.fpn2(x))
        else:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                Norm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
        )

        # self.fpn3 = nn.Identity()

        # self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        InputL B X C X H X W
        """
        features = []
        if self.share_weights:
            ops = [
                self.do_fpn1,
                self.fpn2,
                # self.fpn3, self.fpn4
            ]
        else:
            ops = [
                self.fpn1,
                self.fpn2,
                # self.fpn3, self.fpn4
            ]
        for i in range(len(ops)):
            features.append(ops[i](x))

        return tuple(features)


class HFFB(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                hidden_dim, hidden_dim // 2, 3, padding=1, groups=hidden_dim // 2
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 1, padding=0),
        )
        self.residual = nn.Conv2d(hidden_dim, hidden_dim, 1)

    def forward(self, x):
        return self.convs(x) + self.residual(x)


class FCNHead(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers, out_channels) -> None:
        super().__init__()
        self.proj = nn.Conv2d(embed_dim, hidden_dim, 1)
        convs = []
        for _ in range(num_layers):
            convs.append(HFFB(hidden_dim))
        self.conv_blocks = nn.Sequential(*convs)
        self.pred = nn.Sequential(
            Norm2d(hidden_dim),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=4),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim // 2, hidden_dim // 4, 3, padding=1, groups=hidden_dim // 4
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 1, padding=0),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim // 2, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, xp):
        """
        InputL List[B X C X H X W], FPN features
        """
        out = []
        for x in xp:
            x = self.proj(x)
            out.append(self.pred(self.conv_blocks(x)))

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediate=False,
        return_layers=0,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.return_layers = return_layers  # 0 to return all layers

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            stacked_output = torch.stack(intermediate)
            if self.return_layers > 0:
                stacked_output = stacked_output[-self.return_layers :]
            return stacked_output

        return output.unsqueeze(0)


class MAEDecoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=4,
        num_decoder_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        return_layers=0,
    ) -> None:
        super().__init__()
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            return_layers=return_layers,
        )

    def forward(self, x, tgt):
        """
        x: N X L X d_emb
        tgt: N X T  X d_emb
        out: T X N X d_emb or N_layers X T X N X d_emb
        """
        n, k, d_emb = x.shape
        x = x.permute(1, 0, 2)  # L X N X d_emb
        tgt = tgt.permute(1, 0, 2)
        x = self.decoder(tgt, x)  # N_Layer X T X N X d_emb
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class PatchEmbedUnSafe(PatchEmbed):
    """Image to Patch Embedding"""

    def forward(self, x):
        B, C, H, W = x.shape
        # Dropped size check in timm
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MaskedAutoencoderViTScaleMAE(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=112,
        patch_size=8,
        in_chans=10,
        embed_dim=256,
        depth=6,
        num_heads=4,
        decoder_embed_dim=256,
        decoder_depth=4,
        decoder_num_heads=4,
        decoder_aux_loss_layers=1,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        use_mask_token=False,
        project_pos_emb=False,
        loss_masking=True,
        self_attention=False,
        absolute_scale=False,
        target_size=[240],
        fixed_output_size=0,
        fcn_dim=256,
        fcn_layers=3,
        independent_fcn_head=True,
        use_l1_loss=False,
        l1_loss_weight=1.0,
        band_config=[7, 56],
        progressive=False,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        assert len(band_config) == 2
        self.use_l1_loss = use_l1_loss
        self.l1_loss_weight = l1_loss_weight
        self.band_config = band_config
        self.patch_size = patch_size
        assert fixed_output_size % patch_size == 0
        self.fixed_output_size = fixed_output_size // patch_size
        self.multiscale = len(target_size) > 1
        self.patch_embed = PatchEmbedUnSafe(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.self_attention = self_attention
        self.absolute_scale = absolute_scale
        self.target_size = target_size
        self.independent_fcn_head = independent_fcn_head
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        if use_mask_token or 1:  # alwyas true
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token_decoder = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.use_mask_token = use_mask_token
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        self.project_pos_emb = project_pos_emb
        if project_pos_emb:
            self.pos_emb_projection = nn.Linear(
                decoder_embed_dim, decoder_embed_dim, bias=True
            )
        self.fpn = FPNHead(decoder_embed_dim, share_weights=progressive)
        if independent_fcn_head:
            self.fcn_high = FCNHead(decoder_embed_dim, fcn_dim, fcn_layers, in_chans)
            self.fcn_low = FCNHead(decoder_embed_dim, fcn_dim, fcn_layers, in_chans)
        else:
            self.fcn = FCNHead(decoder_embed_dim, fcn_dim, fcn_layers, in_chans)
        # Depending on the mode of decoding we are using, the decoder architecture is different
        if self.multiscale:
            self.decoder_blocks = nn.ModuleList(
                [
                    GPTBlock(
                        decoder_embed_dim,
                        decoder_num_heads,
                        mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer,
                    )
                    for _ in range(decoder_depth)
                ]
            )
        else:
            self.decoder_blocks = nn.ModuleList(
                [
                    Block(
                        decoder_embed_dim,
                        decoder_num_heads,
                        mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer,
                    )
                    for _ in range(decoder_depth)
                ]
            )

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.norm_pix_loss = norm_pix_loss
        self.loss_masking = loss_masking

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        if self.use_mask_token:
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

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 * C)
        """
        p = self.patch_embed.patch_size[0]
        C = imgs.shape[1]  # Use the actual channel count
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], C, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * C))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        C = x.shape[-1] // (p**2)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, C))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], C, h * p, h * p))
        return imgs

    def upsample_decoder(self, x, target_dim):
        """
        x: (N, L, num_patches**2, decoder_embed_dim)
        padded: (N, decoder_num_patches**2, decoder_embed_dim)
        """
        p = target_dim
        x = x.unsqueeze(dim=1)
        n, _, l_low, _ = x.shape
        l_low_dim = int(l_low**0.5)
        x = torch.nn.functional.interpolate(
            input=x.reshape(n, 1, l_low_dim, l_low_dim, self.decoder_embed_dim),
            size=(p, p, self.decoder_embed_dim),
            mode="nearest",
        ).view(n, 1, p**2, self.decoder_embed_dim)
        padded = x.squeeze(dim=1)
        return padded

    def find_closest_multiple(self, target_resolution):
        n = target_resolution + self.patch_embed.patch_size[0] / 2
        n = n - (n % self.patch_embed.patch_size[0])
        return int(n)

    def plot_decoder_vector(self, x):
        B, total_patches, _ = x.shape
        num_patches_per_axis = int(total_patches**0.5)
        patch_size = self.patch_embed.patch_size[0]
        embed_dim = self.decoder_embed_dim

        output_raster = torch.zeros(
            B, num_patches_per_axis * embed_dim, num_patches_per_axis
        )

        data = x.reshape(
            B, num_patches_per_axis, num_patches_per_axis, embed_dim
        )  # 4, 7, 7, 512

        data = data.permute(0, 3, 1, 2)  # 4, 512, 7, 7

        for img in range(B):
            for i in range(embed_dim):
                output_raster[
                    img, i * num_patches_per_axis : (i + 1) * num_patches_per_axis, :
                ] = data[img, i, :, :]

        return output_raster

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

    def forward_encoder(self, x, mask_ratio=0.0, input_res=None):
        # embed patches
        _, _, h, w = x.shape
        x = self.patch_embed(x)
        input_res = input_res.cpu()

        num_patches = int(
            (h * w) / (self.patch_embed.patch_size[0] * self.patch_embed.patch_size[1])
        )
        pos_embed = get_2d_sincos_pos_embed_with_resolution(
            x.shape[-1],
            int(num_patches**0.5),
            input_res,
            cls_token=True,
            device=x.device,
        )

        # get_2d_sincos_pos_embed(
        # pos_emb = torch.from_numpy(pos_emb).float().to(x.device) # n X L X d_emb
        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Added back to the mask token in decoder for decoding modes != "demasking"
        pos_embed_encoder = get_2d_sincos_pos_embed_with_resolution(
            self.decoder_embed_dim,
            int(num_patches**0.5),
            input_res,
            cls_token=True,
            device=x.device,
        )

        return x, mask, ids_restore, pos_embed_encoder

    def forward_decoder(
        self,
        x,
        ids_restore=[],
        target_res=[],
        target_dim=None,
        pos_embed_encoder=None,
        mask=None,
    ):
        # embed tokens
        x = self.decoder_embed(x)  # N X L X d_emb_decoder
        n, l, d = pos_embed_encoder.shape
        l_dim = int((l - 1) ** 0.5)

        pos_embed = get_2d_sincos_pos_embed_with_resolution(
            x.shape[-1], target_dim, target_res, cls_token=True, device=x.device
        )

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # # add pos embed again

        x = x + pos_embed_encoder

        if not self.use_mask_token:  # drop mask token from encoder
            num_masked = (mask == 0).sum(-1).min().item()
            mask_idx = torch.argsort(mask, dim=-1, descending=True)[:, :num_masked]
            mask_idx_n = torch.arange(n).reshape(-1, 1).repeat(1, num_masked)
            x = x[mask_idx_n, mask_idx]

        # pos_emb = torch.from_numpy(pos_emb).float().to(x.device) # n X ( L_t + 1) X d_emb
        pos_embed_raw = pos_embed
        if self.project_pos_emb:
            pos_emb = self.pos_emb_projection(pos_emb)
        # TODO: Consider adding a projection layer?
        ids = None
        if self.use_mask_token:
            x = x[:, 1:, :]

        n, p_2, d = x.shape
        p = int(p_2**0.5)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # standard decoder
        x = x.view(n, p, p, d).permute(0, 3, 1, 2).contiguous()  # B X C X H X W
        x = self.fpn(x)  # C2,C3,C4,C5
        if self.independent_fcn_head:
            x = [self.fcn_high([x[0]])[0], self.fcn_low([x[1]])[0]]
        else:
            x = self.fcn(x)

        # # apply Transformer blocks
        # for blk in self.decoder_blocks:
        #     x = blk(x)

        return x, pos_embed_raw, ids

    def split_pred(self, target_dim, pred, mean, var):
        pred_high, pred_low = pred
        pred_all = (
            nn.functional.interpolate(
                nn.functional.interpolate(
                    pred_low, (self.band_config[0], self.band_config[0]), mode="area"
                ),
                pred_high.shape[-2:],
                mode="bilinear",
            ) + pred_high
        )
        out = []
        for x in [pred_high, pred_low, pred_all]:
            out.append(self.patchify(x))
        return out

    @classmethod
    def random_crop(cls, seq, target_size, cls_token=False):
        # seq:
        if cls_token:
            seq, cls_tk = seq[:, 1:], seq[:, :1]
        n, l, _ = seq.shape
        dim = int(l**0.5)
        assert dim**2 == l
        if dim <= target_size:
            mask = None
        else:
            x0 = torch.randint(0, dim - target_size, (n,))  # n
            x1 = x0 + target_size
            y0 = torch.randint(0, dim - target_size, (n,))  # n
            y1 = y0 + target_size
            base = torch.zeros(n, dim, dim, 2)
            arr = torch.arange(dim)  # dim
            base[..., 1] += arr.view(1, dim, 1)  # y = h
            base[..., 0] += arr.view(1, 1, dim)  # x = w
            # base now is a grid
            xx = base[..., 0]
            yy = base[..., 1]
            mask = ((xx >= x0.view(n, 1, 1)) & (xx < x1.view(n, 1, 1))) & (
                (yy >= y0.view(n, 1, 1)) & (yy < y1.view(n, 1, 1))
            )  # n x dim x dim
            mask = mask.view(n, dim**2).long()  # N X L
            mask = torch.argsort(mask, dim=-1, descending=True)  # N X L
            mask = mask[:, : target_size**2]  # N X L_tgt
            mask, _ = torch.sort(mask, dim=-1)
            seq = cls.subsample(seq, mask)
        if cls_token:
            seq = torch.cat([cls_tk, seq], dim=1)
        return seq, mask

    @staticmethod
    def subsample(seq, mask):
        if mask is None:
            return seq
        n, l = seq.shape[:2]
        _, l_mask = mask.shape
        x_arr = torch.arange(n).view(n, 1).repeat(1, l_mask)
        seq = seq[x_arr, mask]
        return seq

    def set_fix_decoding_size(self, fixed_output_size):
        if type(fixed_output_size) == list:
            fixed_output_size = fixed_output_size[0]
        assert fixed_output_size % self.patch_size == 0
        self.fixed_output_size = fixed_output_size // self.patch_size

    def build_input_sequence(self, x, base_res, base_dim, pos_emb_base):
        p = self.patch_embed.patch_size[0]
        _, l_x, _ = x.shape
        _, length_pos_embed, _ = pos_emb_base.shape
        mask_tokens = self.mask_token_decoder.repeat(x.shape[0], length_pos_embed, 1)
        mask_tokens[:, :1] = x[:, :1]  # copy class token
        mask_tokens += pos_emb_base
        if self.fixed_output_size > 0:
            mask_tokens, mask = self.random_crop(
                mask_tokens, self.fixed_output_size, cls_token=True
            )
            _, length_pos_embed, _ = mask_tokens.shape
        else:
            mask = None
        new_x = [x, mask_tokens]  # first decoding has cls token

        # At the start, our array is [x, pos_embed]
        # We want this to be [x, pos_embed1, pos_embed2, ...]
        # We also want to return the sizes of each positional embedding (length_pos_embeds)
        atten_mask = [
            torch.ones(
                (l_x + length_pos_embed, l_x + length_pos_embed), device=x.device
            )
        ]
        length_pos_embeds = [length_pos_embed]
        ids = [mask]
        target_sizes = [x for x in self.target_size if x != max(self.target_size)]
        for d in target_sizes:
            d = d // p
            pos_emb = get_2d_sincos_pos_embed_with_resolution(
                x.shape[-1], d, base_res * d / base_dim, cls_token=True, device=x.device
            )
            _, length_pos_embed, _ = pos_emb.shape
            mask_tokens = self.mask_token_decoder.repeat(
                x.shape[0], length_pos_embed, 1
            )
            mask_tokens += pos_emb
            mask_tokens = mask_tokens[:, 1:]
            length_pos_embed = length_pos_embed - 1

            if self.fixed_output_size > 0:
                mask_tokens, mask = self.random_crop(
                    mask_tokens, self.fixed_output_size
                )
                _, length_pos_embed, _ = mask_tokens.shape
            else:
                mask = None
            new_x.append(mask_tokens)
            length_pos_embeds.append(length_pos_embed)
            ids.append(mask)
            atten_mask.append(
                torch.ones((length_pos_embed, length_pos_embed), device=x.device)
            )

        x = torch.cat(new_x, dim=1)
        atten_mask = torch.block_diag(*atten_mask)  # L X L
        atten_mask[:l_x] = 1
        atten_mask[:, :l_x] = 1
        atten_mask = 1 - atten_mask  # 0 no mask, 1 mask
        atten_mask[atten_mask == 1] = float("-inf")
        return x, length_pos_embeds, atten_mask, ids

    def forward_loss(self, imgs, pred, mask, target_dim, ids):
        """
        imgs: [N, 3, H, W]
        pred: [N_decoder_layers, L,N , p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        p = self.patch_embed.patch_size[0]
        dim1, dim2 = self.band_config  # 14,224
        pred_high, pred_low = pred
        n, _, _, _ = imgs.shape
        if dim2 != 224:
            target_low = nn.functional.interpolate(
                imgs, pred_low.shape[-2:], mode="area"
            )
        else:
            target_low = nn.functional.interpolate(
                nn.functional.interpolate(imgs, (dim2, dim2), mode="area"),
                pred_low.shape[-2:],
                mode="area",
            )
        target_high = imgs - nn.functional.interpolate(
            nn.functional.interpolate(imgs, (dim1, dim1), mode="area"),
            pred_high.shape[-2:],
            mode="bilinear",
        )
        n, l_low = mask.shape
        l_low_dim = int(l_low**0.5)
        mask = mask.reshape(n, 1, l_low_dim, l_low_dim)
        mask_low = torch.nn.functional.interpolate(mask, pred_low.shape[-2:])
        mask_high = torch.nn.functional.interpolate(mask, pred_high.shape[-2:])
        loss_l2 = mask_low * (target_low - pred_low) ** 2
        loss_l2 = loss_l2.sum() / (mask_low.sum() + 1e-9)

        if self.use_l1_loss:
            loss_l1 = (
                mask_high * torch.abs(target_high - pred_high) * self.l1_loss_weight
            )
        else:
            loss_l1 = mask_high * (target_high - pred_high) ** 2
        loss_l1 = loss_l1.sum() / (mask_high.sum() + 1e-9)
        return loss_l1 + loss_l2, 0, 1

    def set_target_size(self, target_size):
        self.target_size = target_size
        self.multiscale = len(target_size) > 1

    def forward(
        self,
        imgs,
        targets=None,
        mask_ratio=0.75,
        knn_feats=False,
        input_res=None,
        target_res=None,
        source_size=None,
    ):
        # images, targets: B X C X H0 X W0;  B X C X H1 X W1
        # input_res, target_res: []
        if source_size is not None:
            input_size = imgs.shape[2]
            assert (
                source_size % self.patch_size == 0
            ), "Source size must be a valid multiple of patch size"
            assert (
                source_size <= input_size
            ), "source size but be no greater than image size"
            if source_size < input_size:
                imgs = nn.functional.interpolate(
                    imgs, (source_size, source_size), mode="area"
                )  # downsample
                input_res = input_res * (input_size / source_size)
                target_size = targets.shape[2]
                target_size_new = int(target_size * (source_size / input_size))
                targets = nn.functional.interpolate(
                    imgs, (target_size_new, target_size_new), mode="area"
                )  # downsample
                target_res = target_res * (target_size / target_size_new)
        if self.absolute_scale:
            input_res = torch.ones_like(input_res).to(input_res.device)
        if knn_feats:
            latent, mask, ids_restore, _ = self.forward_encoder(imgs, 0.0, input_res)
            latent = latent[:, 0, :]  # take cls token
            # latent = latent[:,1:].mean(1)
            return latent
        if self.absolute_scale and target_res is not None:
            target_res = torch.ones_like(target_res).to(target_res.device)
        latent, mask, ids_restore, pos_embed_encoder = self.forward_encoder(
            imgs, mask_ratio, input_res
        )

        p = self.patch_embed.patch_size[0]
        target_dim = targets.shape[2] // p
        pred, pos_embed_decoder, ids = self.forward_decoder(
            latent,
            ids_restore=ids_restore,
            target_res=target_res,
            target_dim=target_dim,
            pos_embed_encoder=pos_embed_encoder,
            mask=mask,
        )  # [N_layers_decoder, L,N, p*p*3]
        loss, mean, var = self.forward_loss(targets, pred, mask, target_dim, ids)
        pred = self.split_pred(target_dim, pred, mean, var)
        return (loss, pred, mask, mean, var, pos_embed_encoder, pos_embed_decoder, imgs)
