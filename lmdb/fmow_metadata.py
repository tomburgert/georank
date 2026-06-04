import numpy as np
import pandas as pd
import lmdb
import pickle


names1 = pd.read_csv('/workspace/datasets/fMoW/splits/train_full.csv', header=None).to_numpy()[:, 0]
names2 = pd.read_csv('/workspace/datasets/fMoW/splits/val_full.csv', header=None).to_numpy()[:, 0]
names3 = pd.read_csv('/workspace/datasets/fMoW/splits/test_full.csv', header=None).to_numpy()[:, 0]

patch_names = np.concatenate([names1, names2, names3])


env = lmdb.open(
    str('/workspace/datasets/fMoW/lmdb.db'),
    readonly=True,
    lock=False,
    meminit=False,
    readahead=True,
)

enough_dimension = []
dimension1 = []
dimension2 = []

with env.begin(write=False) as txn:
    c = 0
    res = []
    for patch_name in patch_names:
        c += 1
        if c % 1000 == 0:
            print(c)
        byteflow = txn.get(patch_name.encode('utf-8'))
        patch = pickle.loads(byteflow)

        enough_dimension.append(patch.shape[2] == 13)
        dimension1.append(patch.shape[0])
        dimension2.append(patch.shape[1])

df = pd.DataFrame(dict(name=patch_names, enough_dimension=enough_dimension, dim1=dimension1, dim2=dimension2))
df.to_parquet('/workspace/datasets/fMoW/meta_selection.parquet')
