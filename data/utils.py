from typing import Union

import numpy as np
import cv2

from albumentations.augmentations import geometric


def add_mixed_noise(y_true, add_rate, sub_rate):

    y_noisy = y_true.copy()
    num_samples, num_classes = y_noisy.shape

    for cl in range(num_classes):
        cl_selection_pos = np.where(y_noisy[:, cl] == 1)[0]
        cl_selection_neg = np.where(y_noisy[:, cl] == 0)[0]

        max_addition = int(num_samples * 0.95) - len(cl_selection_pos)
        num_addition = min(int(len(cl_selection_pos) * add_rate), max_addition)
        num_deletion = int(len(cl_selection_pos) * sub_rate)

        cl_selection_pos_flip = np.random.choice(cl_selection_neg, num_addition, replace=False)
        cl_selection_neg_flip = np.random.choice(cl_selection_pos, num_deletion, replace=False)
        y_noisy[cl_selection_pos_flip, cl] = 1
        y_noisy[cl_selection_neg_flip, cl] = 0

    return y_noisy


def custom_autocontrast(img: np.array, **kwargs) -> np.array:
    h = cv2.calcHist([img], [0], None, [256], (0, 256)).ravel()

    for lo in range(256):
        if h[lo]:
            break
    for hi in range(255, -1, -1):
        if h[hi]:
            break

    if hi > lo:
        lut = np.zeros(256, dtype=np.uint8)
        scale_coef = 255.0 / (hi - lo)
        offset = -lo * scale_coef
        for ix in range(256):
            lut[ix] = int(np.clip(ix * scale_coef + offset, 0, 255))

        img = cv2.LUT(img, lut)

    return img


def to_grayscale(img: np.array, **kwars) -> np.array:
    img = img.astype(np.int32)
    num_channels = img.shape[2]
    mean_img = img.mean(2)
    out = np.repeat(mean_img[:, :, np.newaxis], num_channels, axis=2)
    out = out.astype(np.uint8)
    return out


def pixel_shift(img: np.array, shift: Union[int, float], p: float) -> np.array:
    """To-Do: Also in different direction (sign)."""
    assert img.dtype == np.uint8 or img.dtype == np.float32, "img has to be either of type np.uint8 or np.float32."
    # if img.dtype == np.uint8:
    #     assert type(shift[0]) == int, "if img is of type np.uint8, shift magnitude has to be of type int."
    # elif img.dtype == np.float32:
    #     assert type(shift[0]) == float, "if img is of type np.float32, shift magnitude has to be of type float."

    if img.dtype == np.uint8:
        img = img.astype(np.int32)

    if shift[0] >= shift[1]:
        shift_ = np.random.randint(shift[1] - 1, shift[1])
    else:
        shift_ = np.random.randint(shift[0], shift[1])

    if np.random.choice([0, 1], p=[1 - p, p]):
        img = img + shift_

    if img.dtype == np.int32:
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)

    return img


def shuffle_or_flip_rot(
    image: np.array,
    grid_size: int = 4,
    cell_flip_rot: bool = False,
    p: float = 0.5,
) -> np.array:
    should_apply = np.random.choice([0, 1], p=[1 - p, p])
    should_shuffle = np.random.choice([0, 1], p=[0.5, 0.5])
    output = image

    if should_apply:

        if should_shuffle:
            output = shuffle_image(image, grid_size, 1, cell_flip_rot)
        else:
            output = geometric.transforms.Flip(1)(image=image)['image']
            output = geometric.rotate.RandomRotate90(1)(image=output)['image']

    return output


def shuffle_image(image: np.array, grid_size: int = 4, cell_flip_rot: bool = False, p: float = 0.5) -> np.array:
    should_apply = np.random.choice([0, 1], p=[1 - p, p])
    output = image
    
    if should_apply:
        if grid_size == -1:
            grid_size = np.random.choice([2, 3, 4])

        new_order = np.random.permutation(np.arange(grid_size * grid_size))
        new_order = new_order.reshape(grid_size, grid_size)

        splits_row = np.split(image, grid_size, axis=0)
        splits_2d = np.concatenate(list(map(lambda x: np.split(x, grid_size, axis=1), splits_row)))
        
        if cell_flip_rot:
            splits_2d = list(map(lambda x: geometric.transforms.Flip()(image=x)['image'], splits_2d))
            splits_2d = list(map(lambda x: geometric.rotate.RandomRotate90()(image=x)['image'], splits_2d))
            splits_2d = np.array(splits_2d)
        
        output = np.concatenate(list(map(lambda x: np.concatenate(splits_2d[x]), new_order)), axis=1)        

    return output
