import torch

import torch.nn as nn

from typing import Optional
from torch import Tensor

import numpy as np
import torch.nn.functional as F

import timm

from transformers import CLIPModel, AutoProcessor

# Constants
A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
SF = 66.50336


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub.*')


def sample_b(sigma: float, size: tuple) -> Tensor:
    r"""Matrix of size :attr:`size` sampled from from :math:`\mathcal{N}(0, \sigma^2)`

    Args:
        sigma (float): standard deviation
        size (tuple): size of the matrix sampled

    See :class:`~rff.layers.GaussianEncoding` for more details
    """
    return torch.randn(size) * sigma


@torch.jit.script
def gaussian_encoding(
    v: torch.Tensor,
    b: torch.Tensor
) -> torch.Tensor:
    b = b.to(dtype=v.dtype, device=v.device)  # Ensure compatibility
    vp = 2 * torch.pi * v @ b.T
    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


@torch.jit.script
def basic_encoding(
        v: Tensor) -> Tensor:
    r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{v}} , \sin{2 \pi \mathbf{v}})`

    Args:
        v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`

    Returns:
        Tensor: mapped tensor of shape :math:`(N, *, 2 \cdot \text{input_size})`

    See :class:`~rff.layers.BasicEncoding` for more details.
    """
    vp = 2 * np.pi * v
    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


@torch.jit.script
def positional_encoding(
        v: Tensor,
        sigma: float,
        m: int) -> Tensor:
    r"""Computes :math:`\gamma(\mathbf{v}) = (\dots, \cos{2 \pi \sigma^{(j/m)} \mathbf{v}} , \sin{2 \pi \sigma^{(j/m)} \mathbf{v}}, \dots)`
        where :math:`j \in \{0, \dots, m-1\}`

    Args:
        v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`
        sigma (float): constant chosen based upon the domain of :attr:`v`
        m (int): [description]

    Returns:
        Tensor: mapped tensor of shape :math:`(N, *, 2 \cdot m \cdot \text{input_size})`

    See :class:`~rff.layers.PositionalEncoding` for more details.
    """
    j = torch.arange(m, device=v.device)
    coeffs = 2 * np.pi * sigma ** (j / m)
    vp = coeffs * torch.unsqueeze(v, -1)
    vp_cat = torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
    return vp_cat.flatten(-2, -1)


class GaussianEncoding(nn.Module):
    """Layer for mapping coordinates using random Fourier features"""

    def __init__(self, sigma: Optional[float] = None,
                 input_size: Optional[float] = None,
                 encoded_size: Optional[float] = None,
                 b: Optional[Tensor] = None):
        r"""
        Args:
            sigma (Optional[float]): standard deviation
            input_size (Optional[float]): the number of input dimensions
            encoded_size (Optional[float]): the number of dimensions the `b` matrix maps to
            b (Optional[Tensor], optional): Optionally specify a :attr:`b` matrix already sampled
        Raises:
            ValueError:
                If :attr:`b` is provided and one of :attr:`sigma`, :attr:`input_size`,
                or :attr:`encoded_size` is provided. If :attr:`b` is not provided and one of
                :attr:`sigma`, :attr:`input_size`, or :attr:`encoded_size` is not provided.
        """
        super().__init__()
        if b is None:
            if sigma is None or input_size is None or encoded_size is None:
                raise ValueError(
                    'Arguments "sigma," "input_size," and "encoded_size" are required.')

            b = sample_b(sigma, (encoded_size, input_size))
        elif sigma is not None or input_size is not None or encoded_size is not None:
            raise ValueError('Only specify the "b" argument when using it.')
        self.b = nn.parameter.Parameter(b, requires_grad=False)

    def forward(self, v: Tensor) -> Tensor:
        r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}} , \sin{2 \pi \mathbf{B} \mathbf{v}})`

        Args:
            v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`

        Returns:
            Tensor: Tensor mapping using random fourier features of shape :math:`(N, *, 2 \cdot \text{encoded_size})`
        """
        return gaussian_encoding(v, self.b)


class BasicEncoding(nn.Module):
    """Layer for mapping coordinates using the basic encoding"""

    def forward(self, v: Tensor) -> Tensor:
        r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{v}} , \sin{2 \pi \mathbf{v}})`

        Args:
            v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`

        Returns:
            Tensor: mapped tensor of shape :math:`(N, *, 2 \cdot \text{input_size})`
        """
        return basic_encoding(v)


class PositionalEncoding(nn.Module):
    """Layer for mapping coordinates using the positional encoding"""

    def __init__(self, sigma: float, m: int):
        r"""
        Args:
            sigma (float): frequency constant
            m (int): number of frequencies to map to
        """
        super().__init__()
        self.sigma = sigma
        self.m = m

    def forward(self, v: Tensor) -> Tensor:
        r"""Computes :math:`\gamma(\mathbf{v}) = (\dots, \cos{2 \pi \sigma^{(j/m)} \mathbf{v}} , \sin{2 \pi \sigma^{(j/m)} \mathbf{v}}, \dots)`

        Args:
            v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`

        Returns:
            Tensor: mapped tensor of shape :math:`(N, *, 2 \cdot m \cdot \text{input_size})`
        """
        return positional_encoding(v, self.sigma, self.m)


class ImageEncoderViT(nn.Module):
    def __init__(self, in_chans=10, patch_size=16, out_dim=512):
        super().__init__()

        # Load a standard ViT from timm (e.g., ViT-B/16)
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            in_chans=10,
            img_size=112,
            num_classes=0  # disables classification head
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.vit.embed_dim, self.vit.embed_dim),
            nn.ReLU(),
            nn.Linear(self.vit.embed_dim, out_dim)
        )

    def forward(self, x):
        # Input shape: [B, C, H, W] — where C = in_chans
        feats = self.vit.forward_features(x)  # typically returns [B, embed_dim]
        return self.mlp(feats)


class ImageEncoderCLIP(nn.Module):
    def __init__(self):
        super(ImageEncoderCLIP, self).__init__()
        self.CLIP = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        # Original 3-channel patch embedding
        old_conv = self.CLIP.vision_model.embeddings.patch_embedding  # Conv2d(3, 1024, 14, 14)

        # Create new Conv2d with 10 input channels
        new_conv = nn.Conv2d(
            in_channels=10,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        # Initialize: Map pretrained [R,G,B] to your band order [B,G,R,...]
        # Your order: [B02, B03, B04, ..., B12] = [B, G, R, ...] = [0, 1, 2, ..., 9]
        # So:
        # - pretrained R → index 2
        # - pretrained G → index 1
        # - pretrained B → index 0

        with torch.no_grad():
            new_conv.weight[:, 2] = old_conv.weight[:, 0]  # R → B04
            new_conv.weight[:, 1] = old_conv.weight[:, 1]  # G → B03
            new_conv.weight[:, 0] = old_conv.weight[:, 2]  # B → B02

            # Remaining 7 channels → zero
            if new_conv.in_channels > 3:
                new_conv.weight[:, 3:] = 0.0

        # Replace patch embedding in CLIP
        self.CLIP.vision_model.embeddings.patch_embedding = new_conv

        self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = nn.Sequential(nn.Linear(768, 768),
                                 nn.ReLU(),
                                 nn.Linear(768, 512))

        # Freeze CLIP
        for param in self.CLIP.parameters():
            param.requires_grad = False

    def preprocess_image(self, image):
        x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return x

    def forward(self, x):
        x = self.CLIP.get_image_features(pixel_values=x)
        x = self.mlp(x)
        return x


def equal_earth_projection(L):
    latitude = L[:, 0]
    longitude = L[:, 1]
    latitude_rad = torch.deg2rad(latitude)
    longitude_rad = torch.deg2rad(longitude)
    sin_theta = (torch.sqrt(torch.tensor(3.0)) / 2) * torch.sin(latitude_rad)
    theta = torch.asin(sin_theta)
    denominator = 3 * (9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1)
    x = (2 * torch.sqrt(torch.tensor(3.0)) * longitude_rad * torch.cos(theta)) / denominator
    y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta
    return (torch.stack((x, y), dim=1) * SF) / 180


class LocationEncoderCapsule(nn.Module):
    def __init__(self, sigma):
        super(LocationEncoderCapsule, self).__init__()
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
        self.km = sigma
        self.capsule = nn.Sequential(rff_encoding,
                                     nn.Linear(512, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU())
        self.head = nn.Sequential(nn.Linear(1024, 512))

    def forward(self, x):
        x = self.capsule(x)
        x = self.head(x)
        return x


class LocationEncoder(nn.Module):
    def __init__(self, sigma=[2**0, 2**4, 2**8], from_pretrained=True):
        super(LocationEncoder, self).__init__()
        self.sigma = sigma
        self.n = len(self.sigma)

        for i, s in enumerate(self.sigma):
            self.add_module('LocEnc' + str(i), LocationEncoderCapsule(sigma=s))

    def forward(self, location):
        location = equal_earth_projection(location)
        location_features = torch.zeros(location.shape[0], 512).to(location.device)

        for i in range(self.n):
            location_features += self._modules['LocEnc' + str(i)](location)
        
        return location_features


class GeoCLIP(nn.Module):
    def __init__(self, vision_encoder='clip', queue_size=4096):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if vision_encoder == 'clip':
            self.image_encoder = ImageEncoderCLIP()
        elif vision_encoder == 'vit':
            self.image_encoder = ImageEncoderViT()
        self.location_encoder = LocationEncoder()

        self._initialize_gps_queue(queue_size)

        self.device = "cpu"

    def to(self, device):
        self.device = device
        self.image_encoder.to(device)
        self.location_encoder.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
        return super().to(device)

    def _initialize_gps_queue(self, queue_size):
        self.queue_size = queue_size
        self.register_buffer("gps_queue", torch.randn(2, self.queue_size, dtype=torch.float32))
        self.gps_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.float32))

    @torch.no_grad()
    def dequeue_and_enqueue(self, gps):
        """ Update GPS queue

        Args:
            gps (torch.Tensor): GPS tensor of shape (batch_size, 2)
        """
        gps_batch_size = gps.shape[0]
        gps_ptr = int(self.gps_queue_ptr)
        
        assert self.queue_size % gps_batch_size == 0, f"Queue size {self.queue_size} should be divisible by batch size {gps_batch_size}"

        # Replace the GPS from ptr to ptr+gps_batch_size (dequeue and enqueue)
        self.gps_queue[:, gps_ptr:gps_ptr + gps_batch_size] = gps.t()
        gps_ptr = (gps_ptr + gps_batch_size) % self.queue_size  # move pointer
        self.gps_queue_ptr[0] = gps_ptr

    def get_gps_queue(self):
        return self.gps_queue.t()
                                             
    def forward(self, image, location):
        """ GeoCLIP's forward pass

        Args:
            image (torch.Tensor): Image tensor of shape (n, 3, 224, 224)
            location (torch.Tensor): GPS location tensor of shape (m, 2)

        Returns:
            logits_per_image (torch.Tensor): Logits per image of shape (n, m)
        """

        # Compute Features
        image_features = self.image_encoder(image)
        location_features = self.location_encoder(location)
        logit_scale = self.logit_scale.exp()
        
        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        location_features = F.normalize(location_features, dim=1)
        
        # Cosine similarity (Image Features & Location Features)
        logits_per_image = logit_scale * (image_features @ location_features.t())

        return logits_per_image
