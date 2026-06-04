import rasterio
import os
from pathlib import Path
from typing import List

from tqdm import tqdm
import lmdb
import numpy as np

import pandas as pd


def read_tif(file_path, read_channels=None):
    """
    Path to `tif` file.

    If read_channel is `None` all channels/bands are read from the file.
    Otherwise the value of read_channel is used.
    """
    # https://gitlab.tubit.tu-berlin.de/rsim/bigearthnet-models-tf/blob/master/BigEarthNet.py
    # Needs to convert bands to
    with rasterio.open(file_path) as tif_image:
        num_channels = tif_image.count
        channels = range(1, num_channels + 1)
        read_channels = channels if read_channels is None else read_channels
        data = tif_image.read(read_channels)
    return data


def _write_lmdb(
    lmdb_path: Path = Path("lmdb.db"),
    lmdb_max_size: int = 2**40
) -> None:
    """
    The function writes an LMDB archive.
    A list of patch paths is processed individually.
    The output archive is written to `lmdb_path`.

    `lmdb_max_size` is the _theoretical_ upper size for the LMDB archive.
    Usually, there should be no need to change this default, as 1TiB (the default)
    is big enough for most encodings and shouldn't even bother if not as much disk-space
    is available. Though some users have reported that they had to change this value
    to get it working which I couldn't reproduce.
    """
    env = lmdb.open(str(lmdb_path), map_size=lmdb_max_size * 2, readonly=False)
    
    with env.begin(write=True) as txn:
        for split in ['train', 'val', 'test_gt']:
            split_path = os.path.join('/workspace/datasets/fMoW/fmow-sentinel', split)
            class_names = sorted(os.listdir(split_path))
            for class_name in class_names:
                print(class_name)
                class_path = os.path.join(split_path, class_name)
                class_locations = sorted(os.listdir(class_path))
                for class_location in class_locations:
                    location_path = os.path.join(class_path, class_location)
                    timestamps = sorted(os.listdir(location_path))
                    for timestamp in timestamps:
                        image_path = os.path.join(location_path, timestamp)
                        image_name = '{}_{}'.format(split, timestamp)
                        tifs = read_tif(image_path)
                        tifs = np.moveaxis(tifs, 0, -1)
                        txn.put(image_name.encode(), tifs.dumps())

    env.close()


_write_lmdb(Path('/workspace/datasets/fMoW/lmdb.db'))
