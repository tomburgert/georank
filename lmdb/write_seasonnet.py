import os
import lmdb

import rasterio

import sys
sys.path.append('/home/tomburgert/research_code/modules/custom_modules/mlc_noise/data')
from s2_interface import AnyS2_Patch


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


base_dir = '/data/tomburgert/data/datasets/SeasonNet'
labels_list = []
patch_names_list = []

lmdb_path = '/data/tomburgert/data/datasets/SeasonNet/lmdb_full_year.db'
env = lmdb.open(str(lmdb_path), map_size=(2**40) * 2, readonly=False)


lmdb_path_map = '/data/tomburgert/data/datasets/SeasonNet/ref_map_lmdb_full_year.db'
env_map = lmdb.open(str(lmdb_path_map), map_size=(2**40), readonly=False)


with env.begin(write=True) as txn:
    with env_map.begin(write=True) as txn_map:

        for dir_name in ['summer', 'spring', 'fall', 'winter']:

            season_path = os.path.join(base_dir, dir_name)
            grid_path = os.path.join(season_path, 'grid1')

            patch_names = sorted(os.listdir(grid_path))

            for i, patch_name in enumerate(patch_names):

                if i % 10000 == 0:
                    print(i)

                patch_path = os.path.join(grid_path, patch_name)

                # read RGB
                bands_rgb = read_tif(os.path.join(patch_path, patch_name + '_10m_RGB.tif'))

                # read IR
                bands_ir  = read_tif(os.path.join(patch_path, patch_name + '_10m_IR.tif'))

                # read bands 20m
                bands_20m = read_tif(os.path.join(patch_path, patch_name + '_20m.tif'))

                # read bands 60m
                bands_60m = read_tif(os.path.join(patch_path, patch_name + '_60m.tif'))

                # read the map
                label_map = read_tif(os.path.join(patch_path, patch_name + '_labels.tif'))

                # match the band oder: 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'
                patch_bands = [bands_60m[0], bands_rgb[2], bands_rgb[1], bands_rgb[0], bands_20m[0], bands_20m[1],
                               bands_20m[2], bands_ir[0], bands_20m[3], bands_60m[1], bands_20m[4], bands_20m[5]]

                s4a_patch = AnyS2_Patch(*patch_bands, band10_shape=(120, 120), band20_shape=(60, 60), band60_shape=(20, 20))
                txn.put(patch_name.encode(), s4a_patch.dumps())

                txn_map.put(patch_name.encode(), label_map.dumps())

            
env_map.close()
env.close()
