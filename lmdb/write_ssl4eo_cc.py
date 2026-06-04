import os
import lmdb
import numpy as np
import pandas as pd

import sys
sys.path.append('/home/tomburgert/research_code/modules/custom_modules/mlc_noise/data')
from s2_interface import SeCo_Patch, AnyS2_Patch
from bigearthnet_patch_interface.s2_interface import BigEarthNet_S2_Patch


# write lmdb, load patches from all hdf5 files
old_lmdb_path = '/mnt/storagecube/data/shared/datasets/SSL4EO-S12/s2a_resized_lmdb.db'
# new_lmdb_path = '/data/datasets/SSL4EO-S12/s2a_resized_lmdb_120x120.db'
new_lmdb_path = '/data/datasets/SSL4EO-S12/s2a_resized_lmdb_60x60.db'

base_path = '/mnt/storagecube/data/shared/datasets/SSL4EO-S12/s2a'
csv_path = '/mnt/storagecube/data/shared/datasets/SSL4EO-S12/splits/train_multi_loc.csv'
patch_names = pd.read_csv(csv_path, header=None).to_numpy()[:, 0]

env_old = lmdb.open(str(old_lmdb_path), readonly=True, lock=False, meminit=False, readahead=True)
env_new = lmdb.open(str(new_lmdb_path), map_size=(2**40), readonly=False)


with env_new.begin(write=True) as txn_new:
    for patch_name in patch_names:
        with env_old.begin(write=False) as txn_old:
            byteflow = txn_old.get(patch_name.encode('utf-8'))

        s2_patch = SeCo_Patch.loads(byteflow)

        bands_center_crop = [
            s2_patch.band01.data[17:27, 17:27],
            s2_patch.band02.data[102:162, 102:162],
            s2_patch.band03.data[102:162, 102:162],
            s2_patch.band04.data[102:162, 102:162],
            s2_patch.band05.data[51:81, 51:81],
            s2_patch.band06.data[51:81, 51:81],
            s2_patch.band07.data[51:81, 51:81],
            s2_patch.band08.data[102:162, 102:162],
            s2_patch.band8A.data[51:81, 51:81],
            s2_patch.band09.data[17:27, 17:27],
            s2_patch.band11.data[51:81, 51:81],
            s2_patch.band12.data[51:81, 51:81]
        ]

        ssl4eo_patch = AnyS2_Patch(*bands_center_crop, band10_shape=(60, 60), band20_shape=(30, 30), band60_shape=(10, 10))

        # bands_center_crop = [
        #     s2_patch.band01.data[12:32, 12:32],
        #     s2_patch.band02.data[72:192, 72:192],
        #     s2_patch.band03.data[72:192, 72:192],
        #     s2_patch.band04.data[72:192, 72:192],
        #     s2_patch.band05.data[36:96, 36:96],
        #     s2_patch.band06.data[36:96, 36:96],
        #     s2_patch.band07.data[36:96, 36:96],
        #     s2_patch.band08.data[72:192, 72:192],
        #     s2_patch.band8A.data[36:96, 36:96],
        #     s2_patch.band09.data[12:32, 12:32],
        #     s2_patch.band11.data[36:96, 36:96],
        #     s2_patch.band12.data[36:96, 36:96]
        # ]
            
        # ssl4eo_patch = BigEarthNet_S2_Patch(*bands_center_crop)
        txn_new.put(patch_name.encode(), ssl4eo_patch.dumps())

env_new.close()
env_old.close()
