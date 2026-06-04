import pickle

from tqdm import tqdm
import lmdb
import numpy as np
import pandas as pd

csv_path1 = '/workspace/temp/deepglobe_version1_train.csv'
csv_path2 = '/workspace/temp/deepglobe_version1_val.csv'
csv_path3 = '/workspace/temp/deepglobe_version1_test.csv'

lmdb_path_write = '/workspace/temp/deepglobe_patches_uint8.lmdb'
lmdb_path_read = '/workspace/datasets/DeepGlobe/ML_DeepGlobe/patches.lmdb'

patch_names1 = list(pd.read_csv(csv_path1, header=None).to_numpy()[:, 0])
patch_names2 = list(pd.read_csv(csv_path2, header=None).to_numpy()[:, 0])
patch_names3 = list(pd.read_csv(csv_path3, header=None).to_numpy()[:, 0])
patch_names = patch_names1 + patch_names2 + patch_names3

env_write = lmdb.open(lmdb_path_write, map_size=2**40, readonly=False)
env_read = lmdb.open(lmdb_path_read, readonly=True, lock=False, meminit=False, readahead=True)

for patch_name in tqdm(patch_names):
    
    with env_read.begin(write=False) as txn:
        byteflow = txn.get(patch_name.encode('utf-8'))
        patch = pickle.loads(byteflow)
        patch = patch.astype(np.uint8)

    with env_write.begin(write=True) as txn:
        txn.put(patch_name.encode(), patch.dumps())

env_write.close()
