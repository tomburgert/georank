import os
import rasterio
import pandas as pd
import numpy as np


def extract_tif(path):
    with rasterio.open(path) as src:
        location = src.bounds
    return np.mean([location.bottom, location.top]), np.mean([location.left, location.right])


base_path = '/mnt/storagecube/data/shared/datasets/SSL4EO-S12/s2a'
ssl4eo_clouds = pd.read_parquet('/mnt/storagecube/data/shared/datasets/SSL4EO-S12/dummy_clouds_labels.parquet')

longs, lats = [], []

for i, name in enumerate(ssl4eo_clouds.name.to_list()):
    if i % 10000 == 0:
        print(i)
    tmp_path1 = os.path.join(base_path, name[:7])
    tmp_path2 = os.path.join(tmp_path1, name[8:])
    tif_path = os.path.join(tmp_path2, 'B1.tif')
    longitude, latitude = extract_tif(tif_path)
    longs.append(longitude)
    lats.append(latitude)

df_centers = pd.DataFrame(data=dict(patch_id=ssl4eo_clouds.name, longitude=longs, latitude=lats))
df_centers.to_parquet('/mnt/storagecube/data/shared/datasets/SSL4EO-S12/s2a/ssl4eo_gps_centers.parquet')
