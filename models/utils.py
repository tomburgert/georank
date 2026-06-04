import timm
from torchvision import models

from models.classification import ResNet, ViT, DINOViT
from models.segmentation import ResNetSemSeg
from models.satmae_backbone import MaskedAutoencoderGroupChannelViT
from models.scalemae_backbone import MaskedAutoencoderViTScaleMAE
from models.crossscalemae_backbone import MAE_ViT_Baseline, MAE_ViT_MsLdCeCd
from models.geoclip import GeoCLIP
from models.croma_backbone import CROMA

from lightly.models.modules import MaskedVisionTransformerTIMM


def get_network(name, channels, pretrained, vit_image_size, vit_patch_size, num_cls):
    if name == 'resnet18':
        resnet = models.resnet18
        weights = models.ResNet18_Weights
        fc_size = 512
        return ResNet(resnet, weights, fc_size, num_cls=num_cls, channels=channels, pretrained=pretrained)
    if name == 'resnet18_semseg':
        resnet = models.resnet18
        weights = models.ResNet18_Weights
        fc_size = 512
        return ResNetSemSeg(resnet, weights, fc_size, num_cls=num_cls, channels=channels, pretrained=pretrained)
    elif name == 'resnet34':
        resnet = models.resnet34
        weights = models.ResNet34_Weights
        fc_size = 512
        return ResNet(resnet, weights, fc_size, num_cls=num_cls, channels=channels, pretrained=pretrained)
    elif name == 'resnet50':
        resnet = models.resnet50
        weights = models.ResNet18_Weights
        fc_size = 2048
        return ResNet(resnet, weights, fc_size, num_cls=num_cls, channels=channels, pretrained=pretrained)
    elif name == 'resnet101':
        resnet = models.resnet101
        weights = models.ResNet101_Weights
        fc_size = 2048
        return ResNet(resnet, weights, fc_size, num_cls=num_cls, channels=channels, pretrained=pretrained)
    elif name == 'dino_vit_base_world_encoding':
        vit = DINOViT('vit_base_patch16_224', pretrained=pretrained, img_size=vit_image_size, in_chans=channels, num_classes=0)
        return vit
    elif name == 'vit_base_world_encoding':
        vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained, img_size=vit_image_size, in_chans=channels, num_classes=0)
        return vit
    elif name == 'mae_lightly':
        model_name = "vit_base_patch16_224"
        vit = timm.create_model(model_name, pretrained=pretrained, img_size=vit_image_size, in_chans=channels, num_classes=0)
        backbone = MaskedVisionTransformerTIMM(vit=vit)
        return backbone
    elif name == 'mae':
        return MaskedAutoencoderGroupChannelViT(in_chans=channels)
    elif name == 'mae_scale':
        return MaskedAutoencoderViTScaleMAE()
    elif name == 'mae_crossscale':
        # return MAE_ViT_Baseline()
        return MAE_ViT_MsLdCeCd()
    elif name == 'geoclip_backbone':
        # return MAE_ViT_Baseline()
        return GeoCLIP()
    elif name == 'croma':
        return CROMA()
    if name == 'vitbX':
        fc_size = 1024
        weights = None  # no pretrained ViT available for image/patch size
        vit = models.vision_transformer._vision_transformer(
            patch_size=vit_image_size,
            image_size=vit_patch_size,
            num_layers=24,
            num_heads=16,
            hidden_dim=fc_size,
            mlp_dim=4096,
            weights=weights,
            progress=True,
        )
        return ViT(vit, fc_size, num_cls=num_cls, channels=channels)
