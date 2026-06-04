import os
import numpy as np
import lmdb
import rasterio

import sys
sys.path.append('/home/tomburgert/research_code/modules/custom_modules/mlc_noise/data')
from s2_interface import SeCo_Patch, AnyS2_Patch
from bigearthnet_patch_interface.s2_interface import BigEarthNet_S2_Patch


SECO_95TH_PERCENTILES = {
    "B1.tif": 5000,
    "B2.tif": 2470.588134765625,
    "B3.tif": 3411.764892578125,
    "B4.tif": 4392.15673828125,
    "B8.tif": 4801.470703125,
    "B5.tif": 4897.05859375,
    "B6.tif": 5044.11767578125,
    "B7.tif": 5176.470703125,
    "B8A.tif": 5127.451171875,
    "B9.tif": 5000,
    "B11.tif": 5884.80419921875,
    "B12.tif": 5257.35302734375
}


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


base_path = '/mnt/storagecube/data/shared/datasets/SeCo/seasonal_contrast_1m'

bands = [
    'B1.tif',
    'B2.tif',
    'B3.tif',
    'B4.tif',
    'B5.tif',
    'B6.tif',
    'B7.tif',
    'B8.tif',
    'B8A.tif',
    'B9.tif',
    'B11.tif',
    'B12.tif'
]

# write lmdb, load patches from all hdf5 files
lmdb_path = '/mnt/storagecube/data/shared/datasets/SeCo/lmdb_uint16_60x60.db'
# lmdb_path = '/mnt/storagecube/data/shared/datasets/SeCo/lmdb_uint16_120x120.db'
# lmdb_path = '/mnt/storagecube/data/shared/datasets/SeCo/lmdb_uint16_264x264.db'
env = lmdb.open(str(lmdb_path), map_size=(2**40), readonly=False)
patch_names_train = []
patch_names_test = []

with env.begin(write=True) as txn:
    for i, dir in enumerate(sorted(os.listdir(base_path))):
        if i % 10000 == 0:
            print(i)
        dir_path = os.path.join(base_path, dir)
        for patch_name in sorted(os.listdir(dir_path)):
            patch_path = os.path.join(dir_path, patch_name)
            unique_patch_name = '{}_{}'.format(dir, patch_name)
            tif_bands = []
            for band in bands:
                # percentile = SECO_95TH_PERCENTILES[band]
                uint16_values = (read_tif(os.path.join(patch_path, band)).astype(np.float32)[0] / 255 * 10000).astype(np.uint16)

                # uint8_values = (np.clip(uint16_values / percentile, 0, 1) * 255).astype(np.uint8)

                # if uint8_values.shape[0] == 264:
                #     uint8_values = uint8_values[72:192, 72:192]
                # elif uint8_values.shape[0] == 132:
                #     uint8_values = uint8_values[36:96, 36:96]
                # elif uint8_values.shape[0] == 44:
                #     uint8_values = uint8_values[12:32, 12:32]

                # if uint16_values.shape[0] == 264:
                #     uint16_values = uint16_values[72:192, 72:192]
                # elif uint16_values.shape[0] == 132:
                #     uint16_values = uint16_values[36:96, 36:96]
                # elif uint16_values.shape[0] == 44:
                #     uint16_values = uint16_values[12:32, 12:32]

                if uint16_values.shape[0] == 264:
                    uint16_values = uint16_values[102:162, 102:162]
                elif uint16_values.shape[0] == 132:
                    uint16_values = uint16_values[51:81, 51:81]
                elif uint16_values.shape[0] == 44:
                    uint16_values = uint16_values[17:27, 17:27]

                tif_bands.append(uint16_values)

            # seco_patch = BigEarthNet_S2_Patch(*tif_bands)
            # seco_patch = SeCo_Patch(*tif_bands)
            seco_patch = AnyS2_Patch(*tif_bands, band10_shape=(60, 60), band20_shape=(30, 30), band60_shape=(10, 10))
            txn.put(unique_patch_name.encode(), seco_patch.dumps())

env.close()
