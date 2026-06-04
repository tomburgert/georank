import h5py
import lmdb
import numpy as np

# TODO: change culture_10 with str € {random,block} and lmdb file name
split = 'block'

# write lmdb, load patches from all hdf5 files
lmdb_path = '/mnt/storagecube/data/shared/datasets/So2Sat/lmdb_{}_uint16.db'.format(split)
env = lmdb.open(str(lmdb_path), map_size=2**40, readonly=False)
patch_names_train = []
patch_names_test = []


with env.begin(write=True) as txn:
    with h5py.File('/mnt/storagecube/data/shared/datasets/So2Sat/m1613658/{}/training.h5'.format(split), 'r') as f:
        for idx, patch in enumerate(f['sen2']):
            if idx % 1000 == 0:
                print(idx)
            patch_name = "train_patch_{}".format(idx)
            patch_names_train.append(patch_name)
            patch = (patch * 10000).astype(int)
            patch = patch.astype(np.uint16)
            txn.put(patch_name.encode(), patch.dumps())
    with h5py.File('/mnt/storagecube/data/shared/datasets/So2Sat/m1613658/{}/testing.h5'.format(split), 'r') as f:
        for idx, patch in enumerate(f['sen2']):
            if idx % 1000 == 0:
                print(idx)
            patch_name = "test_patch_{}".format(idx)
            patch_names_test.append(patch_name)
            patch = (patch * 10000).astype(int)
            patch = patch.astype(np.uint16)
            txn.put(patch_name.encode(), patch.dumps())

env.close()
