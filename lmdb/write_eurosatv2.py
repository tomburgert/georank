import os
from pathlib import Path
from typing import List
from tqdm import tqdm

import lmdb
import numpy as np

import sys
sys.path.append('/workspace/modules/mlc_noise/data')
from s2_interface import AnyS2_Patch


def _write_lmdb_big(
    patch_paths: List[Path],
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
    all_band_names = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

    if len(patch_paths) == 0:
        raise ValueError(
            "No patches were provided! Maybe provided wrong Sentinel directory?"
        )
    env = lmdb.open(str(lmdb_path), map_size=lmdb_max_size, readonly=False)
    
    with env.begin(write=True) as txn:
        for patch_path in tqdm(patch_paths):
            patch_name = patch_path.stem
            array = np.load(patch_path)

            sample = {}
            for i, band_name in enumerate(["B02", "B03", "B04", "B08"]):
                sample[band_name] = array['gsd_10'][:, :, i]
            for i, band_name in enumerate(["B05", "B06", "B07", "B8A", "B11", "B12"]):
                sample[band_name] = array['gsd_20'][:, :, i]
            for i, band_name in enumerate(["B01", "B09"]):
                sample[band_name] = array['gsd_60'][:, :, i]

            eurosat_patch = AnyS2_Patch(
                *[sample[band_name] for band_name in all_band_names],
                band10_shape=(192, 192),
                band20_shape=(96, 96),
                band60_shape=(32, 32)
            )

            txn.put(patch_name.encode(), eurosat_patch.dumps())

    env.close()


def _write_lmdb_small(
    patch_paths: List[Path],
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
    all_band_names = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

    if len(patch_paths) == 0:
        raise ValueError(
            "No patches were provided! Maybe provided wrong Sentinel directory?"
        )
    env = lmdb.open(str(lmdb_path), map_size=lmdb_max_size, readonly=False)
    
    with env.begin(write=True) as txn:
        for patch_path in tqdm(patch_paths):
            patch_name = patch_path.stem
            array = np.load(patch_path)

            sample = {}
            for i, band_name in enumerate(["B02", "B03", "B04", "B08"]):
                sample[band_name] = array['gsd_10'][64:128, 64:128, i]
            for i, band_name in enumerate(["B05", "B06", "B07", "B8A", "B11", "B12"]):
                sample[band_name] = array['gsd_20'][32:64:, 32:64, i]
            for i, band_name in enumerate(["B01", "B09"]):
                sample[band_name] = array['gsd_60'][10:21, 10:21, i]

            eurosat_patch = AnyS2_Patch(
                *[sample[band_name] for band_name in all_band_names],
                band10_shape=(64, 64),
                band20_shape=(32, 32),
                band60_shape=(11, 11)
            )

            txn.put(patch_name.encode(), eurosat_patch.dumps())

    env.close()


base_path = '/workspace/datasets/EuroSAT_V2/smartdata/data/sentinel/EuroSAT/resamples'
paths = []

for cl_name in os.listdir(base_path):
    paths += list(Path(os.path.join(base_path, cl_name)).glob('*.npz'))

_write_lmdb_big(paths, Path('/workspace/datasets/EuroSAT_V2/lmdb_196x196.db'))

_write_lmdb_small(paths, Path('/workspace/datasets/EuroSAT_V2/lmdb_64x64.db'))
