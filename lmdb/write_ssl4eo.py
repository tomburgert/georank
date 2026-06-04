import os
import pandas as pd
import lmdb
import rasterio

from skimage.transform import resize

import sys
sys.path.append('/home/tomburgert/research_code/modules/custom_modules/mlc_noise/data')
from s2_interface import SeCo_Patch
from bigearthnet_patch_interface.s2_interface import BigEarthNet_S2_Patch


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


base_path = '/mnt/storagecube/data/shared/datasets/SSL4EO-S12/s2a'

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
lmdb_path_full = '/mnt/storagecube/data/shared/datasets/SSL4EO-S12/s2a_resized_lmdb.db'
lmdb_path_120x120 = '/data/datasets/SSL4EO-S12/s2a_resized_lmdb_120x120.db'
env_full = lmdb.open(str(lmdb_path_full), map_size=(2**40) * 2, readonly=False)
env_120x120 = lmdb.open(str(lmdb_path_120x120), map_size=(2**40) * 2, readonly=False)

patch_names_train = []
groups = []


with env_full.begin(write=True) as txn_full:
    with env_120x120.begin(write=True) as txn_120x120:

        for i, directory in enumerate(sorted(os.listdir(base_path))):
            if i % 10000 == 0:
                print(i)
            dir_path = os.path.join(base_path, directory)
            for patch_name in sorted(os.listdir(dir_path)):
                patch_path = os.path.join(dir_path, patch_name)
                unique_patch_name = '{}_{}'.format(directory, patch_name)

                tif_bands = []
                for band in bands:
                    band_tif = read_tif(os.path.join(patch_path, band))[0]
                    if band in ['B2.tif', 'B3.tif', 'B4.tif', 'B8.tif']:
                        if band_tif.shape != (264, 264):
                            band_tif = resize(band_tif / 30000, (264, 264), mode='reflect') * 30000
                    if band in ['B5.tif', 'B6.tif', 'B7.tif', 'B8A.tif', 'B11.tif', 'B12.tif']:
                        if band_tif.shape != (132, 132):
                            band_tif = resize(band_tif / 30000, (132, 132), mode='reflect') * 30000
                    if band in ['B1.tif', 'B9.tif']:
                        if band_tif.shape != (44, 44):
                            band_tif = resize(band_tif / 30000, (44, 44), mode='reflect') * 30000

                    tif_bands.append(band_tif)
                
                ssl4eo_patch = SeCo_Patch(*tif_bands)

                txn_full.put(unique_patch_name.encode(), ssl4eo_patch.dumps())
                groups.append(directory)
                patch_names_train.append(unique_patch_name)

                bands_center_crop = [
                    ssl4eo_patch.band01.data[12:32, 12:32],
                    ssl4eo_patch.band02.data[72:192, 72:192],
                    ssl4eo_patch.band03.data[72:192, 72:192],
                    ssl4eo_patch.band04.data[72:192, 72:192],
                    ssl4eo_patch.band05.data[36:96, 36:96],
                    ssl4eo_patch.band06.data[36:96, 36:96],
                    ssl4eo_patch.band07.data[36:96, 36:96],
                    ssl4eo_patch.band08.data[72:192, 72:192],
                    ssl4eo_patch.band8A.data[36:96, 36:96],
                    ssl4eo_patch.band09.data[12:32, 12:32],
                    ssl4eo_patch.band11.data[36:96, 36:96],
                    ssl4eo_patch.band12.data[36:96, 36:96]
                ]
                    
                ssl4eo_patch_120x120 = BigEarthNet_S2_Patch(*bands_center_crop)
                txn_120x120.put(unique_patch_name.encode(), ssl4eo_patch_120x120.dumps())


env_full.close()
env_120x120.close()

df = pd.DataFrame({'group': groups, 'name': patch_names_train})
df.to_parquet('/mnt/storagecube/data/shared/datasets/SSL4EO-S12/temporal_groups.parquet')
