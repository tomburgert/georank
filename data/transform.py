import numpy as np

import albumentations as A
import torch
import torch.nn.functional as F

from torchvision import transforms

from data.utils import pixel_shift, custom_autocontrast, to_grayscale

from typing import Tuple, Union, List, Optional


class AutoContrast(object):
    """Constant Transformation."""
    def __init__(self, p: float = 0.5):
        self.AutoContrast = A.Lambda(image=custom_autocontrast, p=p)

    def __call__(self, image: np.array) -> np.array:
        return self.AutoContrast(image=image)['image']


class Brightness(object):
    """Range Transformation."""
    def __init__(self, brightness_limit : Union[float, Tuple[float, float]] = (-0.2, 0.2), p: float = 0.5):
        self.Brightness = A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=0,
            p=p
        )

    def __call__(self, image: np.array) -> np.array:
        return self.Brightness(image=image)['image']


class Contrast(object):
    """Range Transformation."""
    def __init__(self, contrast_limit : Union[float, Tuple[float, float]] = (-0.2, 0.2), p: float = 0.5):
        self.Contrast = A.RandomBrightnessContrast(
            brightness_limit=0,
            contrast_limit=contrast_limit,
            p=p
        )

    def __call__(self, image: np.array) -> np.array:
        return self.Contrast(image=image)['image']


class CutOut(object):
    """Range Transformation."""
    def __init__(self, max_edge: float = 0.5, min_edge: float = 0.2, p: float = 0.5):
        self.CutOut = A.CoarseDropout(
            max_holes=1,
            max_height=max_edge,
            max_width=max_edge,
            min_height=min_edge,
            min_width=min_edge,
            p=p)

    def __call__(self, image: np.array) -> np.array:
        return self.CutOut(image=image)['image']


class Equalize(object):
    """Constant Transformation."""
    def __init__(self, p: float = 0.5):
        self.Equalize = A.Equalize(p=p)

    def __call__(self, image: np.array) -> np.array:
        return self.Equalize(image=image)['image']


class Flip(object):
    """Constant Transformation. PyTorch Wrapper for Albumentation Flip."""
    def __init__(self, p: float = 0.5):
        self.Flip = A.Flip(p)

    def __call__(self, image: np.array) -> np.array:
        return self.Flip(image=image)['image']


# class GaussianBlur(object):
#     """Range Transformation."""
#     def __init__(self, sigma: Tuple[float, float] = (0.1, 2.0), p: float = 0.5):
#         self.GaussianBlur = A.GaussianBlur(blur_limit=(0, 0), sigma_limit=sigma, p=p)

#     def __call__(self, image: np.array) -> np.array:
#         return self.GaussianBlur(image=image)['image']

class GaussianBlur(object):
    """Range Transformation."""
    def __init__(self, sigma: Tuple[float, float] = (0.1, 2.0), p: float = 0.5):
        self.GaussianBlur = A.Lambda(image=self.gaussianblur_gpu, p=p)
        self.GaussianBlurPyTorch = transforms.GaussianBlur(kernel_size=(3, 3), sigma=sigma)

    def gaussianblur_gpu(self, image: np.array, **kwargs) -> np.array:
        dtype = image.dtype
        channel_pos = 0 if len(image.shape) == 3 else 1
        image = np.moveaxis(image, -1, channel_pos).astype(np.float32)
        image_gpu_tensor = torch.from_numpy(image).to(torch.device('cuda:0'))
        image_cpu_tensor = self.GaussianBlurPyTorch(image_gpu_tensor).cpu()
        image_np = image_cpu_tensor.numpy().astype(dtype)
        image_np = np.moveaxis(image_np, channel_pos, -1)
        return image_np

    def __call__(self, image: np.array) -> np.array:
        return self.GaussianBlur(image=image)['image']


class GaussianNoise(object):
    """Range Transformation."""
    def __init__(self, var_max: int = 100, per_channel: bool = True, p: float = 0.5):
        self.GaussianNoise = A.GaussNoise(var_limit=(0, var_max), mean=0, p=p)

    def __call__(self, image: np.array) -> np.array:
        return self.GaussianNoise(image=image)['image']


class GridShuffle(object):
    def __init__(self, grid_size: int = 3, p: float = 0.5):
        self.GridShuffle = A.RandomGridShuffle(grid=(grid_size, grid_size), p=p)

    def __call__(self, image: np.array) -> np.array:
        return self.GridShuffle(image=image)['image']


class GrayScale(object):
    def __init__(self, p: float = 0.5):
        self.GrayScale = A.Lambda(image=to_grayscale, p=p)

    def __call__(self, image: np.array) -> np.array:
        return self.GrayScale(image=image)['image']


# class GridShuffle(object):
#     """Constant/Range Transformation. Dynamic GridShuffle Method."""
#     def __init__(self, grid_size: int = 3, cell_flip_rot: bool = False, p: float = 0.5):
#         self.grid_size = grid_size
#         self.cell_flip_rot = cell_flip_rot
#         self.p = p

#     def __call__(self, image: np.array) -> np.array:
#         return shuffle_image(image, self.grid_size, self.cell_flip_rot, self.p)


class HorizontalFlip(object):
    """Constant Transformation. PyTorch Wrapper for Albumentation HorizontalFlip."""
    def __init__(self, p: float = 0.5):
        self.HorizontalFlip = A.HorizontalFlip(p)

    def __call__(self, image: np.array) -> np.array:
        return self.HorizontalFlip(image=image)['image']


class Identity(object):
    """Constant Transformation."""
    def __call__(self, image: np.array) -> np.array:
        return image


class MultiCutOut(object):
    """Range Transformation."""
    def __init__(
        self, max_holes: int = 8, min_holes: int = 3, max_edge: float = 0.2, min_edge: float = 0.1, p: float = 0.5
    ):
        self.MultiCutOut = A.CoarseDropout(
            max_holes=max_holes,
            min_holes=min_holes,
            max_height=max_edge,
            max_width=max_edge,
            min_height=min_edge,
            min_width=min_edge,
            p=p)

    def __call__(self, image: np.array) -> np.array:
        return self.MultiCutOut(image=image)['image']


class PixelDropout(object):
    """Custom Range Transformation."""
    def __init__(self, dropout_prob: float = 0.01, p: float = 0.5):
        dropout_prob_ = np.random.choice(np.arange(0.01, dropout_prob + 0.01, 0.01)).item()
        self.PixelDropout = A.PixelDropout(dropout_prob=dropout_prob_, p=p)

    def __call__(self, image: np.array) -> np.array:
        return self.PixelDropout(image=image)['image']


class PixelShift(object):
    """Range Transformation."""
    def __init__(self, shift: Union[Tuple[int, int], Tuple[float, float]] = (-10, 10), p: float = 0.5):
        self.shift = shift
        self.p = p

    def __call__(self, image: np.array) -> np.array:
        return pixel_shift(image, self.shift, self.p)


class Posterize(object):
    """Custom Range Transformation."""
    def __init__(self, num_bits: int = 4, p: float = 0.5):
        # not ideal: 8 should be changed to 7, only to meet PPDA experiment requirements.
        num_bits_ = np.random.choice(np.arange(8, num_bits - 1, -1)).item()
        self.Posterize = A.Posterize(num_bits=num_bits_, p=p)

    def __call__(self, image: np.array) -> np.array:
        return self.Posterize(image=image)['image']


class RandomResizedCrop(object):
    """Range Transformation."""
    def __init__(
        self,
        resize_size: Union[int, Tuple[int, int]] = (120, 120),
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio : Tuple[float, float] = (0.75, 1.3333333333333333),
        p: float = 0.5
    ):
        h, w = (resize_size, resize_size) if type(resize_size) != tuple else resize_size
        self.RandomResizedCrop = A.RandomResizedCrop(h, w, scale=scale, ratio=ratio, p=p)

    def __call__(self, image: np.array) -> np.array:
        return self.RandomResizedCrop(image=image)['image']


class CustomRandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.2, 1.0), ratio=(1.0, 1.0), interpolation=transforms.InterpolationMode.BILINEAR):
        super().__init__(size, scale=scale, ratio=ratio, interpolation=interpolation)
        self.last_crop_size = None  # Store the last crop size

    def forward(self, img):
        """
        Apply the transformation and store the crop dimensions.
        """
        np_type = img.dtype
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img_width, img_height = transforms.functional.get_image_size(img)  # Get original size

        # Use RandomResizedCrop's internal logic to compute crop params
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        # Compute the scale factor based on the crop width
        self.last_crop_size = w

        # Apply crop
        img = transforms.functional.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        img = img.permute(1, 2, 0).cpu().numpy().astype(np_type)
        return img


class RandomResizedCropReplay(object):
    """Replay-enabled RandomResizedCrop."""
    def __init__(
        self,
        resize_size: Union[int, Tuple[int, int]] = (120, 120),
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio : Tuple[float, float] = (0.75, 1.3333333333333333),
        p: float = 0.5
    ):
        h, w = (resize_size, resize_size) if not isinstance(resize_size, tuple) else resize_size
        self.transform = A.ReplayCompose([
            A.RandomResizedCrop(height=h, width=w, scale=scale, ratio=ratio, p=p)
        ])
        self.last_replay = None  # To store replay information

    def __call__(self, image: np.array) -> np.array:
        result = self.transform(image=image)
        # Store replay info for later access
        self.last_replay = result.get("replay", None)
        # Return only the transformed image for compatibility with Compose
        return result["image"]


class RandomRotate(object):
    """Range Transformation."""
    def __init__(self, angle: Union[int, Tuple[int, int]] = (-30, 30), p: float = 0.5):
        self.RandomRotate = A.Affine(rotate=angle, p=p)

    def __call__(self, image: np.array) -> np.array:
        return self.RandomRotate(image=image)['image']


class RandomRotate90(object):
    """Range Transformation. PyTorch Wrapper for Albumentation Flip."""
    def __init__(self, p: float = 0.5):
        self.RandomRotate90 = A.RandomRotate90(p)

    def __call__(self, image: np.array) -> np.array:
        return self.RandomRotate90(image=image)['image']


class Resize(object):
    """Range Transformation. PyTorch Wrapper for Albumentation Flip."""
    def __init__(self, size: Tuple[int, int]):
        self.Resize = A.Resize(size[0], size[1])

    def __call__(self, image: np.array) -> np.array:
        return self.Resize(image=image)['image']


class Sharpness(object):
    """Range Transformation."""
    def __init__(self, alpha: float = 0.2, p: float = 0.5):
        if np.sign(alpha) != np.sign(-1):
            self.Sharpness = A.Sharpen(alpha=(0, alpha), lightness=0.75, p=1.0)
        else:
            self.Sharpness = A.UnsharpMask(alpha=(0, -alpha), p=p)

    def __call__(self, image: np.array) -> np.array:
        return self.Sharpness(image=image)['image']


class ShearX(object):
    """Range Transformation."""
    def __init__(self, shear_x: Union[int, Tuple[int, int]] = (-20, 20), p: float = 0.5):
        self.ShearX = A.Affine(shear={'x': shear_x}, p=p)

    def __call__(self, image: np.array) -> np.array:
        return self.ShearX(image=image)['image']


class ShearY(object):
    """Range Transformation."""
    def __init__(self, shear_y: Union[int, Tuple[int, int]] = (-20, 20), p: float = 0.5):
        self.ShearY = A.Affine(shear={'y': shear_y}, p=p)

    def __call__(self, image: np.array) -> np.array:
        return self.ShearY(image=image)['image']


class Shear(object):
    """Range Transformation."""
    def __init__(self, shear: Union[int, Tuple[int, int]] = (-20, 20), p: float = 0.5):
        self.ShearX = A.Affine(shear={'x': shear}, p=p)
        self.ShearY = A.Affine(shear={'y': shear}, p=p)

    def __call__(self, image: np.array) -> np.array:
        if np.random.choice([0, 1]):
            return self.ShearX(image=image)['image']
        else:
            return self.ShearY(image=image)['image']


class Solarize(object):
    """Custom Range Transformation."""
    def __init__(self, threshold: int = 128, p: float = 0.5):
        threshold_ = np.random.choice(np.arange(254, threshold - 1, -1)).item()
        self.Solarize = A.Solarize(threshold=threshold_, p=p)

    def __call__(self, image: np.array) -> np.array:
        return self.Solarize(image=image)['image']


class TranslateX(object):
    """Range Transformation."""
    def __init__(self, pct_x: Union[float, Tuple[float, float]] = (-0.2, 0.2), p: float = 0.5):
        self.TranslateX = A.Affine(translate_percent={'x': pct_x}, p=p)

    def __call__(self, image: np.array) -> np.array:
        return self.TranslateX(image=image)['image']


class TranslateY(object):
    """Range Transformation."""
    def __init__(self, pct_y: Union[float, Tuple[float, float]] = (-0.2, 0.2), p: float = 0.5):
        self.TranslateY = A.Affine(translate_percent={'y': pct_y}, p=p)

    def __call__(self, image: np.array) -> np.array:
        return self.TranslateY(image=image)['image']


class Translate(object):
    """Range Transformation."""
    def __init__(self, pct: Union[float, Tuple[float, float]] = (-0.2, 0.2), p: float = 0.5):
        self.TranslateX = A.Affine(translate_percent={'x': pct}, p=p)
        self.TranslateY = A.Affine(translate_percent={'y': pct}, p=p)

    def __call__(self, image: np.array) -> np.array:
        if np.random.choice([0, 1]):
            return self.TranslateX(image=image)['image']
        else:
            return self.TranslateY(image=image)['image']


class VerticalFlip(object):
    """Constant Transformation. PyTorch Wrapper for Albumentation VerticalFlip."""
    def __init__(self, p: float = 0.5):
        self.VerticalFlip = A.VerticalFlip(p)

    def __call__(self, image: np.array) -> np.array:
        return self.VerticalFlip(image=image)['image']


class ToFloatS2(object):
    """Convert Satelite Band Values to float [0.0,1.0] range."""
    def __init__(self, normalize_factor: Union[List[int], int] = 8192):
        self.normalize_factor = normalize_factor

    def __call__(self, image: np.array) -> np.array:
        image_norm = image / self.normalize_factor
        image_clip = np.clip(image_norm, 0, 1)
        return image_clip


class ToInt8S2(object):
    """Convert Satelite Band Values to float [0.0,1.0] range."""
    def __call__(self, image: np.array) -> np.array:
        image = image * 255
        image = image.astype(np.uint8)
        return image


class RandAugment(torch.nn.Module):
    """Difference is that custom RandAug has no fixed magnitude, but up to max magnitude."""
    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        op_names: List[str] = [
            "AutoContrast", "Brightness", "Contrast", "Equalize", "Posterize", "RandomRotate",
            "Sharpness", "ShearX", "ShearX", "ShearY", "Solarize", "TranslateX", "TranslateY", "Identity"
        ],
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.op_names = op_names

    def forward(self, img: np.array) -> np.array:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        for _ in range(self.num_ops):
            # img has shape (H, W, C)
            op_index = np.random.randint(0, len(self.op_names))
            op_name = self.op_names[op_index]
            magnitude = get_magnitude(op_name, self.magnitude, self.num_magnitude_bins)
            img = _apply_op(img, op_name, magnitude)
        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f")"
        )
        return s


def _apply_op(img: np.array, op_name: str, magnitude: float):
    if op_name == "AutoContrast":
        Transform = AutoContrast(p=1.0)
    elif op_name == "Brightness":
        Transform = Brightness(brightness_limit=magnitude, p=1.0)
    elif op_name == "Contrast":
        Transform = Contrast(contrast_limit=magnitude, p=1.0)
    elif op_name == "CutOut":
        Transform = CutOut(max_edge=magnitude, min_edge=0.2, p=1.0)
    elif op_name == "Equalize":
        Transform = Equalize(p=1.0)
    elif op_name == "Flip":
        Transform = Flip(p=1.0)
    elif op_name == "GridShuffle":
        Transform = GridShuffle(grid_size=3, cell_flip_rot=False, p=1.0)
    elif op_name == "MultiCutOut":
        Transform = MultiCutOut(max_holes=magnitude[0], min_holes=3, max_edge=magnitude[1], min_edge=0.1, p=1.0)
    elif op_name == "PixelDropout":
        Transform = PixelDropout(dropout_prob=magnitude, p=1.0)
    elif op_name == "PixelShift":
        Transform = PixelShift(shift=magnitude, p=1.0)
    elif op_name == "Posterize":
        Transform = Posterize(num_bits=magnitude, p=1.0)
    elif op_name == "RandomResizedCrop":
        Transform = RandomResizedCrop(resize_size=(120, 120), scale=magnitude, ratio=(0.75, 1.34), p=1.0)
    elif op_name == "RandomRotate":
        Transform = RandomRotate(angle=magnitude, p=1.0)
    elif op_name == "RandomRotate90":
        Transform = RandomRotate90(p=1.0)
    elif op_name == "Sharpness":
        Transform = Sharpness(alpha=magnitude, p=1.0)
    elif op_name == "ShearX":
        Transform = ShearX(shear_x=magnitude, p=1.0)
    elif op_name == "ShearY":
        Transform = ShearY(shear_y=magnitude, p=1.0)
    elif op_name == "Solarize":
        Transform = Solarize(threshold=magnitude, p=1.0) 
    elif op_name == "TranslateX":
        Transform = TranslateX(pct_x=magnitude, p=1.0)
    elif op_name == "TranslateY":
        Transform = TranslateY(pct_y=magnitude, p=1.0)
    elif op_name == "Identity":
        Transform = Identity()
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    img = Transform(img)
    return img


def get_magnitude(
    op_name: str, magnitude: int, num_bins: int = 31
) -> Union[Union[Union[float, int], Tuple[float, float]], Tuple[int, int]]:
    """To-Do: Magnitudes Option for Geometric Transformations."""
    augment_space = {
        "AutoContrast": np.linspace(0.0, 0.0, num_bins),
        "Brightness": np.linspace(0.0, 0.6, num_bins),
        "Contrast": np.linspace(0.0, 0.6, num_bins),
        "CutOut": np.linspace(0.25, 0.7, num_bins),
        "GaussianBlur": np.linspace(0.0, 3.0, num_bins),
        "GaussianNoise": np.linspace(0.0, 500.0, num_bins).astype(int),
        "MultiCutOut": list(zip(np.linspace(3, 9, 31).astype(int), np.linspace(0.2, 0.5, 31))),
        "Equalize": np.linspace(0.0, 0.0, num_bins),
        "Flip": np.linspace(0.0, 0.0, num_bins),
        "GridShuffle": np.linspace(0.0, 0.0, num_bins),
        "Identity": np.linspace(0.0, 0.0, num_bins),
        "PixelDropout": np.linspace(0.0, 0.3, num_bins),
        "PixelShift": np.linspace(0.0, 128.0, num_bins).astype(int),
        "Posterize": 8 - (np.arange(num_bins) / ((num_bins - 1) / 4)).round().astype(int),
        "RandomResizedCrop": np.linspace(0.9, 0.08, num_bins),
        "RandomRotate": np.linspace(0.0, 60.0, num_bins).astype(int),
        "RandomRotate90": np.linspace(0.0, 0.0, num_bins),
        "ShearX": np.linspace(0.0, 60.0, num_bins).astype(int),
        "ShearY": np.linspace(0.0, 60.0, num_bins).astype(int),
        "Shear": np.linspace(0.0, 60.0, num_bins).astype(int),
        "Sharpness": np.linspace(0.0, 0.9, num_bins),
        "Solarize": np.linspace(255.0, 0.0, num_bins).astype(int),
        "TranslateX": np.linspace(0.0, 0.4, num_bins),
        "TranslateY": np.linspace(0.0, 0.4, num_bins),
        "Translate": np.linspace(0.0, 0.4, num_bins)
    }
    base_magnitude_ = augment_space[op_name][magnitude]
    if op_name == "MultiCutOut":
        base_magnitude = (base_magnitude_[0].item(), base_magnitude_[1].item())
    else:
        base_magnitude = base_magnitude_.item()

    if op_name in [
       "Brightness", "Contrast", "PixelShift", "RandomRotate", "ShearX", "ShearY",
       "Shear", "TranslateX", "TranslateY", "Translate"
       ]:
        magnitude = (-base_magnitude, base_magnitude)
    elif op_name == "GaussianBlur":
        magnitude = (0.0, base_magnitude)
    elif op_name == "RandomResizedCrop":
        magnitude = (base_magnitude, 1.0)
    else:
        magnitude = base_magnitude
    return magnitude


def get_augmentation_transforms(
    augmentations: str,
    p: float,
    p_list: Optional[List[int]],
    magnitude: Optional[int],
    brightness_limit : Optional[float],
    contrast_limit : Optional[float],
    max_edge: Optional[float],
    min_edge: Optional[float],
    sigma: Optional[List[float]],
    var_max: Optional[int],
    per_channel: Optional[bool],
    grid_size: Optional[int],
    max_holes: Optional[int],
    min_holes: Optional[int],
    dropout_prob: Optional[float],
    shift: Optional[int],
    num_bits: Optional[int],
    randaug_op_names: Optional[List[str]],
    randaug_magnitude: Optional[int],
    resize_size: Optional[List[int]],
    scale: Optional[List[float]],
    cond_rrc_min_scale: Optional[float],
    ratio : Optional[List[float]],
    angle: Optional[int],
    alpha: Optional[float],
    shear_x: Optional[int],
    shear_y: Optional[int],
    shear: Optional[int],
    threshold: Optional[int],
    pct_x: Optional[float],
    pct_y: Optional[float],
    pct: Optional[float],
) -> List[object]:
    """To-Do: Magnitudes Option for Geometric Transformations."""

    transform_names = list(augmentations.split('_'))
    compose = []

    if p_list is not None:
        assert len(transform_names) == len(p_list), "if p_list is provided, has to have same lentgh as num. augmentations"
        augment_ps = p_list
    else:
        augment_ps = [p] * len(transform_names)

    for transform_name, p in zip(transform_names, augment_ps):
        if 'autocontrast' == transform_name:
            compose += [AutoContrast(p=p)]
        if 'brightness' == transform_name:
            # ONLY HARDCODED WITH BRIGHTNESS
            mag = get_magnitude('Brightness', magnitude) if magnitude is not None else (-brightness_limit, brightness_limit)
            compose += [Brightness(brightness_limit=mag, p=p)]
        if 'contrast' == transform_name:
            mag = get_magnitude('Contrast', magnitude) if magnitude is not None else (-contrast_limit, contrast_limit)
            compose += [Contrast(contrast_limit=mag, p=p)]
        if 'cutout' == transform_name:
            mag = get_magnitude('CutOut', magnitude) if magnitude is not None else max_edge
            compose += [CutOut(max_edge=mag, min_edge=min_edge, p=p)]
        if 'equalize' == transform_name:
            compose += [Equalize(p=p)]
        if 'flip' == transform_name:
            compose += [Flip(p=p)]
        if 'gaussianblur' == transform_name:
            mag = get_magnitude('GaussianBlur', magnitude) if magnitude is not None else tuple(sigma)
            compose += [GaussianBlur(sigma=mag, p=p)]
        if 'gaussiannoise' == transform_name:
            mag = get_magnitude('GaussianNoise', magnitude) if magnitude is not None else var_max
            compose += [GaussianNoise(var_max=mag, per_channel=per_channel, p=p)]
        if 'grayscale' == transform_name:
            compose += [GrayScale(p=p)]
        if 'gridshuffle' == transform_name:
            compose += [GridShuffle(grid_size=grid_size, p=p)]
        if 'horizontalflip' == transform_name:
            compose += [HorizontalFlip(p=p)]
        if 'multicutout' == transform_name:
            mag = get_magnitude('MultiCutOut', magnitude) if magnitude is not None else (max_holes, max_edge)
            compose += [MultiCutOut(
                max_holes=mag[0], min_holes=min_holes, max_edge=mag[1], min_edge=min_edge, p=p
            )]
        if 'pixeldropout' == transform_name:
            mag = get_magnitude('PixelDropout', magnitude) if magnitude is not None else dropout_prob
            compose += [PixelDropout(dropout_prob=mag, p=p)]
        if 'pixelshift' == transform_name:
            mag = get_magnitude('PixelShift', magnitude) if magnitude is not None else (-shift, shift)
            compose += [PixelShift(shift=mag, p=p)]
        if 'posterize' == transform_name:
            mag = get_magnitude('Posterize', magnitude) if magnitude is not None else num_bits
            compose += [Posterize(num_bits=mag, p=p)]
        # if 'randaug' == transform_name:
        #    compose += [RandAugment(magnitude=randaug_magnitude, op_names=radnaug_op_names, p=p)]
        if 'randomresizedcrop' == transform_name:
            mag = get_magnitude('RandomResizedCrop', magnitude) if magnitude is not None else tuple(scale)
            compose += [RandomResizedCrop(
                scale=mag, resize_size=tuple(resize_size), p=p
            )]
        if 'randomrotate' == transform_name:
            mag = get_magnitude('RandomRotate', magnitude) if magnitude is not None else (-angle, angle)
            compose += [RandomRotate(angle=mag, p=p)]
        if 'randomrotate90' == transform_name:
            compose += [RandomRotate90(p=p)]
        if 'resize' == transform_name:
            compose += [Resize(tuple(resize_size))]
        if 'sharpness' == transform_name:
            mag = get_magnitude('Sharpness', magnitude) if magnitude is not None else alpha
            compose += [Sharpness(alpha=mag, p=p)]
        if 'shearx' == transform_name:
            mag = get_magnitude('Shear', magnitude) if magnitude is not None else (-shear_x, shear_x)
            compose += [ShearX(shear_x=mag, p=p)]
        if 'sheary' == transform_name:
            mag = get_magnitude('Shear', magnitude) if magnitude is not None else (-shear_y, shear_y)
            compose += [ShearY(shear_y=mag, p=p)]
        if 'shear' == transform_name:
            mag = get_magnitude('Shear', magnitude) if magnitude is not None else (-shear, shear)
            compose += [Shear(shear=mag, p=p)]
        if 'solarize' == transform_name:
            mag = get_magnitude('Solarize', magnitude) if magnitude is not None else threshold
            compose += [Solarize(threshold=mag, p=p)]
        if 'translatex' == transform_name:
            mag = get_magnitude('Translate', magnitude) if magnitude is not None else (-pct_x, pct_x)
            compose += [TranslateX(pct_x=mag, p=p)]
        if 'translatey' == transform_name:
            mag = get_magnitude('Translate', magnitude) if magnitude is not None else (-pct_y, pct_y)
            compose += [TranslateY(pct_y=mag, p=p)]
        if 'translate' == transform_name:
            mag = get_magnitude('Translate', magnitude) if magnitude is not None else (-pct, pct)
            compose += [Translate(pct=mag, p=p)]
        if 'verticalflip' == transform_name:
            compose += [VerticalFlip(p=p)]

    return compose


class BaseTransform(object):

    def __init__(self, cfg):
        self.cfg = cfg

    def add_data_transforms(
        self,
        means: List[float],
        stds: List[float],
        percentiles: List[float] = None,
        sentinel2: bool = False
    ) -> transforms.Compose:

        if sentinel2:
            self.pre_transforms = [ToFloatS2(percentiles), ToInt8S2()]

        self.post_transforms = [transforms.ToTensor(), transforms.Normalize(means, stds)]


class SingleTransform(BaseTransform):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.pre_transforms = []
        self.augmentation_transforms = get_augmentation_transforms(**self.cfg)
        self.post_transforms = []

    def setup_compose(self):
        transform_list = self.pre_transforms + self.augmentation_transforms + self.post_transforms
        self.transform_compose = transforms.Compose(transform_list)
        self.transform_compose_is_built = True

    def __call__(self, image: np.array) -> List[torch.Tensor]:
        assert self.transform_compose_is_built, 'Please call self.setup_compose() first.'
        return self.transform_compose(image)


class MoCoTransform(BaseTransform):

    def __init__(self, cfg, views: int):
        super().__init__(cfg)
        self.views = views
        self.pre_transforms = []
        self.augmentation_transforms = get_augmentation_transforms(**self.cfg)
        self.post_transforms = []

    def setup_compose(self):
        transform_list = self.pre_transforms + self.augmentation_transforms + self.post_transforms
        self.transform_compose = transforms.Compose(transform_list)
        self.transform_compose_is_built = True

    def __call__(self, image: Union[np.array, List[np.array]]) -> List[torch.Tensor]:
        assert self.transform_compose_is_built, 'Please call self.setup_compose() first.'
        if type(image) == list:
            return [self.transform_compose(img) for img in image]
        else:
            return [self.transform_compose(image) for i in range(self.views)]


class SatMAETransform(BaseTransform):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.pre_transforms = []
        self.augmentation_transforms = get_augmentation_transforms(**self.cfg)
        self.post_transforms = []

    def setup_compose(self):
        transform_list = self.pre_transforms + self.augmentation_transforms + self.post_transforms
        self.transform_compose = transforms.Compose(transform_list)
        self.transform_compose_is_built = True

    def __call__(self, image: Union[np.array, List[np.array]]) -> List[torch.Tensor]:
        assert self.transform_compose_is_built, 'Please call self.setup_compose() first.'
        img_dn_1x = self.transform_compose(image)
        img_dn_2x = F.interpolate(img_dn_1x.unsqueeze(0), scale_factor=2.0, mode='bilinear').squeeze(0)
        img_dn_4x = F.interpolate(img_dn_2x.unsqueeze(0), scale_factor=2.0, mode='bilinear').squeeze(0)
        return [img_dn_1x, img_dn_2x, img_dn_4x]


class ScaleMAETransform(BaseTransform):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.pre_transforms = []
        self.augmentation_transforms = get_augmentation_transforms(**self.cfg)
        self.post_transforms = []

        self.source_size = 112
        self.output_size = 224
        self.base_resolution = 10

        self.rrc_replay = RandomResizedCropReplay(scale=(0.2, 1.0), ratio=(1.0, 1.0), resize_size=(self.source_size, self.source_size), p=1.0)
        # self.rrc_replay = CustomRandomResizedCrop(size=(self.source_size, self.source_size))

        self.rrc_flip = transforms.Compose([
            self.rrc_replay,
            HorizontalFlip(p=0.5)
        ])
        self.resize = Resize((self.output_size, self.output_size))

    def setup_compose(self):
        transform_list = self.pre_transforms + self.augmentation_transforms + self.post_transforms
        self.transform_compose = transforms.Compose(transform_list)
        self.transform_compose_is_built = True

    def __call__(self, image: Union[np.array, List[np.array]]) -> List[torch.Tensor]:
        assert self.transform_compose_is_built, 'Please call self.setup_compose() first.'
        source_image = self.rrc_flip(image)
        replay_params = self.rrc_replay.last_replay
        # rcc_size = self.rrc_replay.last_crop_size
        resized_image = self.resize(source_image)  # size: output_size x output_size
        
        # Compute the resolution of the resized image.
        # (The resized image is a scaled version of the crop.)
        source_resolution = self.base_resolution * replay_params['transforms'][0]['params']['crop_width'] / self.source_size
        # source_resolution = self.base_resolution * rcc_size / self.source_size
        resized_resolution = source_resolution * (self.output_size / self.source_size)

        source_image = self.transform_compose(source_image)
        resized_image = self.transform_compose(resized_image)
        source_resolution = torch.tensor(source_resolution, dtype=torch.float32)
        resized_resolution = torch.tensor(resized_resolution, dtype=torch.float32)
        
        return source_image, source_resolution, resized_image, resized_resolution


class MoCoRRC2Transform(BaseTransform):

    def __init__(self, cfg, views: int):
        super().__init__(cfg)
        self.views = views
        self.pre_transforms = []
        self.augmentation_transforms = get_augmentation_transforms(**self.cfg)
        self.post_transforms = []
        self.conditional_rrc = RandomResizedCrop(
            scale=[cfg.cond_rrc_min_scale, 1.0], resize_size=tuple(cfg.resize_size), p=1.0
        )

    def setup_compose(self):
        transform_list = self.pre_transforms + self.augmentation_transforms + self.post_transforms
        self.transform_compose = transforms.Compose(transform_list)
        self.transform_compose_is_built = True

    def __call__(self, image: Union[np.array, List[np.array]]) -> List[torch.Tensor]:
        assert self.transform_compose_is_built, 'Please call self.setup_compose() first.'
        if type(image) == list:
            return [self.transform_compose(img) for img in image]
        else:
            image = self.conditional_rrc(image)
            return [self.transform_compose(image) for i in range(self.views)]


class DINOTransform(BaseTransform):

    def __init__(
        self,
        cfg,
        n_local_views: int,
        global_resize_size: Optional[List[int]],
        global_scale: Optional[List[float]],
        local_resize_size: Optional[List[int]],
        local_scale: Optional[List[float]],
    ):
        super().__init__(cfg)
        # when default DINO, hard-code blur probs per transform
        # gaussian_blur_probs = [1.0, 0.1, 0.5]
        self.n_local_views = n_local_views
        self.cfg.update(dict(resize_size=global_resize_size, scale=global_scale))
        self.global_augmentation_transforms_0 = get_augmentation_transforms(**self.cfg)
        self.global_augmentation_transforms_1 = get_augmentation_transforms(**self.cfg)

        self.cfg.update(dict(resize_size=local_resize_size, scale=local_scale))
        self.local_augmentation_transforms = get_augmentation_transforms(**self.cfg)
        self.pre_transforms = []
        self.post_transforms = []

    def setup_compose(self):
        global_transform_list_0 = self.pre_transforms + self.global_augmentation_transforms_0 + self.post_transforms
        global_transform_list_1 = self.pre_transforms + self.global_augmentation_transforms_1 + self.post_transforms
        local_transform_list = self.pre_transforms + self.local_augmentation_transforms + self.post_transforms
        self.global_transform_compose_0 = transforms.Compose(global_transform_list_0)
        self.global_transform_compose_1 = transforms.Compose(global_transform_list_1)
        self.local_transform_compose = transforms.Compose(local_transform_list)

        local_transforms = [self.local_transform_compose] * self.n_local_views
        self.transform_compose = [self.global_transform_compose_0, self.global_transform_compose_1]
        self.transform_compose.extend(local_transforms)
        self.transform_compose_is_built = True

    def __call__(self, image: Union[np.array, List[np.array]]) -> List[torch.Tensor]:
        assert self.transform_compose_is_built, 'Please call self.setup_compose() first.'
        if type(image) == list:
            return [transform(img) for img, transform in zip(image, self.transform_compose)]
        else:
            return [transform(image) for transform in self.transform_compose]


def get_self_transforms(ssl_cfg, aug_cfg):
    if ssl_cfg.algorithm == 'MoCoV2_RRC2':
        return MoCoRRC2Transform(aug_cfg, ssl_cfg.views)
    if ssl_cfg.algorithm == 'MoCoV2' or ssl_cfg.algorithm == 'VICReg' or ssl_cfg.algorithm == 'MAE' or ssl_cfg.algorithm == 'VICRegGeo' or ssl_cfg.algorithm == 'BYOL' or ssl_cfg.algorithm == 'SimSiam' or ssl_cfg.algorithm == 'SimCLR':
        # MAE augmentations: RRC [0.2,1] & Flip
        return MoCoTransform(aug_cfg, ssl_cfg.views)
    if ssl_cfg.algorithm == 'MoCoV3World':
        return MoCoTransform(aug_cfg, ssl_cfg.views)
    if ssl_cfg.algorithm == 'DINO' or ssl_cfg.algorithm == 'DINOWorld':
        return DINOTransform(
            aug_cfg,
            n_local_views=ssl_cfg.dino_n_local_views,
            global_resize_size=ssl_cfg.dino_global_resize_size,
            global_scale=ssl_cfg.dino_global_scale,
            local_resize_size=ssl_cfg.dino_local_resize_size,
            local_scale=ssl_cfg.dino_local_scale
        )
    if ssl_cfg.algorithm == 'Tile2Vec':
        return SingleTransform(aug_cfg)
    if ssl_cfg.algorithm == 'SatMAE':
        return SatMAETransform(aug_cfg)
    if ssl_cfg.algorithm == 'ScaleMAE':
        return ScaleMAETransform(aug_cfg)
    if ssl_cfg.algorithm == 'CrossScaleMAE':
        return SingleTransform(aug_cfg)
    if ssl_cfg.algorithm == 'GeoCLIP':
        return SingleTransform(aug_cfg)
    if ssl_cfg.algorithm == 'MAEWorld':
        return SingleTransform(aug_cfg)
    if ssl_cfg.algorithm == 'CROMA':
        return SingleTransform(aug_cfg)
