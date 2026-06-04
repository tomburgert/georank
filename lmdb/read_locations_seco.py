import os
import rasterio
import pandas as pd
import numpy as np


def extract_tif(path):
    with rasterio.open(path) as src:
        location = src.bounds
    return np.mean([location.bottom, location.top]), np.mean([location.left, location.right])


base_path = '/mnt/storagecube/data/shared/datasets/SeCo/seasonal_contrast_1m'
seco_train = pd.read_csv('/data/datasets/SeCo/splits/train_multi_loc.csv', header=None)

longs, lats = [], []

for i, name in enumerate(seco_train[0].to_list()):
    if i % 10000 == 0:
        print(i)
    tmp_path1 = os.path.join(base_path, name[:6])
    tmp_path2 = os.path.join(tmp_path1, name[7:])
    tif_path = os.path.join(tmp_path2, 'B1.tif')
    longitude, latitude = extract_tif(tif_path)
    longs.append(longitude)
    lats.append(latitude)

df_centers = pd.DataFrame(data=dict(patch_id=seco_train[0], longitude=longs, latitude=lats))
df_centers.to_parquet('/mnt/storagecube/data/shared/datasets/SeCo/seco_gps_centers.parquet')
