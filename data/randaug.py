import torch
import transform as tf
from torch import Tensor
from torchvision.transforms import functional as F, InterpolationMode
from typing import List, Tuple, Optional, Dict


class RandAugment(torch.nn.Module):
    """Difference is that custom RandAug has no fixed magnitude, but up to max magnitude."""
    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        op_names: List[str] = ['all'],
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.op_names = op_names
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        augment_space = {
            # op_name: (magnitudes, signed)
            "AutoContrast": torch.tensor(0.0),
            "Brightness": torch.linspace(0.0, 0.9, num_bins),
            "Contrast": torch.linspace(0.0, 0.9, num_bins),
            "Equalize": torch.tensor(0.0),
            "Identity": torch.tensor(0.0),
            "Posterize": (8 - torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(),
            "RandomRotate": torch.linspace(0.0, 45.0, num_bins).int(),
            "Sharpness": torch.linspace(0.0, 0.9, num_bins),
            "ShearX": torch.linspace(0.0, 45.0, num_bins).int(),
            "ShearY": torch.linspace(0.0, 45.0, num_bins).int(),
            "Solarize": torch.linspace(255.0, 0.0, num_bins).int(),
            "TranslateX": torch.linspace(0.0, 0.4, num_bins),
            "TranslateY": torch.linspace(0.0, 0.4, num_bins),
        }
        if self.op_names == ['all']:
            pass
        else:
            augment_space = {key: augment_space[key] for key in self.op_names}
        return augment_space

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        for _ in range(self.num_ops):
            # img has shape (H, W, C)
            op_meta = self._augmentation_space(self.num_magnitude_bins)
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)
        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s


def _apply_op(
    img: Tensor, op_name: str, magnitude: float,
    interpolation: InterpolationMode, fill: Optional[List[float]]
):
    if op_name == "AutoContrast":
        Transform = tf.AutoContrast(p=1.0)
    elif op_name == "Brightness":
        Transform = tf.Brightness(brightness_limit=(-magnitude, magnitude), p=1.0)
    elif op_name == "Contrast":
        Transform = tf.Contrast(contrast_limit=(-magnitude, magnitude), p=1.0)
    elif op_name == "Equalize":
        Transform = tf.Equalize(p=1.0)
    elif op_name == "Posterize":
        Transform = tf.Posterize(num_bits=magnitude, p=1.0) 
    elif op_name == "RandomRotate":
        Transform = tf.RandomRotate(agnle=(-magnitude, magnitude), p=1.0)
    elif op_name == "Sharpness":
        Transform = tf.Sharpness(alpha=magnitude, p=1.0)
    elif op_name == "ShearX":
        Transform = tf.ShearX(shear_x=(-magnitude, magnitude), p=1.0)
    elif op_name == "ShearY":
        Transform = tf.ShearY(shear_y=(-magnitude, magnitude), p=1.0)
    elif op_name == "Solarize":
        Transform = tf.Solarize(threshold=magnitude, p=1.0) 
    elif op_name == "TranslateX":
        Transform = tf.TranslateX(pct_x=(-magnitude, magnitude), p=1.0)
    elif op_name == "TranslateY":
        Transform = tf.TranslateY(pct_y=(-magnitude, magnitude), p=1.0)
    elif op_name == "Identity":
        Transform = tf.Identity()
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    img = Transform(img)
    return img
