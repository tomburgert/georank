import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.models.vision_transformer import PatchEmbed


class FlexiblePatchEmbed(PatchEmbed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ResNet(nn.Module):
    def __init__(self, resnet, weights, fc_size, num_cls, channels=10, pretrained=False):
        super().__init__()
        weights = weights.DEFAULT if pretrained else None
        self.fc_size = fc_size
        self.resnet = resnet(weights=weights)
        self.resnet.conv1 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(fc_size, num_cls)

    def forward(self, x):
        return self.resnet.forward(x)

    def register_hooks(self, get_activation):
        """Put hook avg_pool layer for feature extraction."""
        return self.resnet.avgpool.register_forward_hook(get_activation('avg_pool'))


class ViT(nn.Module):
    def __init__(self, vit, fc_size, num_cls, channels=10):
        super().__init__()
        self.fc_size = fc_size
        self.vit = vit
        self.vit.conv_proj = nn.Conv2d(channels, 1024, kernel_size=(16, 16), stride=(16, 16))
        self.vit.heads.head = nn.Linear(fc_size, num_cls)

    def forward(self, x):
        return self.vit.forward(x)


class DINOViT(nn.Module):
    def __init__(self, model_name, pretrained, img_size, in_chans, num_classes, patch_size=16):
        super().__init__()
        # Load timm ViT
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            in_chans=in_chans,
            num_classes=num_classes
        )
        self.vit.patch_embed = FlexiblePatchEmbed(
            img_size=self.vit.patch_embed.img_size, 
            patch_size=self.vit.patch_embed.patch_size,
            in_chans=self.vit.patch_embed.proj.in_channels,
            embed_dim=self.vit.patch_embed.proj.out_channels
        )

        # Store original image and patch size for interpolation logic
        self.orig_img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = self.vit.embed_dim
        self.patch_embed = self.vit.patch_embed
        self.cls_token = self.vit.cls_token
        self.pos_embed = self.vit.pos_embed
        self.pos_drop = self.vit.pos_drop
        self.blocks = self.vit.blocks
        self.norm = self.vit.norm

    def interpolate_positional_encoding(self, pos_embed, new_img_size):
        num_extra_tokens = 1  # CLS token
        cls_token = pos_embed[:, :num_extra_tokens]
        patch_pos_embed = pos_embed[:, num_extra_tokens:]

        orig_grid_size = self.orig_img_size // self.patch_size
        new_grid_size = new_img_size // self.patch_size

        patch_pos_embed = patch_pos_embed.reshape(1, orig_grid_size, orig_grid_size, -1).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed, size=(new_grid_size, new_grid_size), mode='bilinear', align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, new_grid_size * new_grid_size, -1)

        new_pos_embed = torch.cat((cls_token, patch_pos_embed), dim=1)
        return new_pos_embed

    def forward(self, x):
        B, C, H, W = x.shape
        # Interpolate positional embeddings if needed
        if H != self.orig_img_size or W != self.orig_img_size:
            pos_embed = self.interpolate_positional_encoding(self.vit.pos_embed, H)
        else:
            pos_embed = self.vit.pos_embed

        # Patch embedding
        x = self.vit.patch_embed(x)
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + pos_embed
        x = self.vit.pos_drop(x)

        # Transformer blocks
        x = self.vit.blocks(x)
        x = self.vit.norm(x)

        # Return CLS token output (standard for DINO)
        return x[:, 0]
