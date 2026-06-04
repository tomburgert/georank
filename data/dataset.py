from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import pickle

import os
import lmdb
import numpy as np
import pandas as pd

from skimage.transform import resize
from bigearthnet_patch_interface.s2_interface import BigEarthNet_S2_Patch

from torch.utils.data import Dataset

from safetensors.numpy import load as safetensor_load

import torch
import torch.nn.functional as F

from constants import (
    BEN19_NAME2IDX,
    DEEPGLOBE_NAME2IDX,
    TREESATAI_NAME2IDX,
    EUROSAT_NAME2IDX,
    SO2SAT_NAME2IDX,
    FMOW_NAME2IDX,
    S4A_COMMON_NAME2IDX,
    S4A_SEMSEG_LOW_NAME2IDX,
    S4A_HIGH2LOW_STR,
    S4A_LOW_NAME2IDX,
    BEN19CLOUDS_NAME2IDX,
    S4A_NAME2ENCODING
)
from s2_interface import SeCo_Patch, S4A_Patch, AnyS2_Patch


class BaseDataset(Dataset):
    def __init__(self, lmdb_path, csv_path, labels_path, eval_resize=None, temporal_views_path=None, transform=None, split='train'):
        """
        Parameter
        ---------
        lmdb_path      : path to the LMDB file for efficiently loading the patches.
        csv_path       : path to a csv file containing the patch names that will make up this split
        transform_mode:  specifies the image transform mode which determines the augmentations
                         to be applied to the image
        """
        self.env = None
        self.split = split
        self.lmdb_path = lmdb_path
        self.patch_names = self.read_csv(csv_path)
        self.eval_resize = eval_resize
        self.transform = transform
        self.temporal_views = self.read_temporal_views(temporal_views_path)

    def read_csv(self, csv_data):
        return pd.read_csv(csv_data, header=None).to_numpy()[:, 0]

    def read_labels(self, meta_data_path, patch_names):
        df = pd.read_parquet(meta_data_path)
        df_subset = df.set_index('name').loc[self.patch_names].reset_index(inplace=False)
        string_labels = df_subset.labels.tolist()
        multihot_labels = np.array(list(map(self.convert_to_multihot, string_labels)))
        return multihot_labels

    def read_temporal_views(self, temporal_views_path):
        temporal_views = None
        if temporal_views_path is not None:
            temporal_views = pd.read_parquet(temporal_views_path)
        return temporal_views
    
    def convert_to_multihot(self, labels):
        raise NotImplementedError

    def __getitem__(self, idx):
        """Get item at position idx of Dataset."""
        if self.env is None:
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
            )

        patch_name = self.patch_names[idx]

        if patch_name == 'noise':
            patch = self.sample_noise().astype(np.float32)
            patch = np.moveaxis(patch, 0, -1)
            label = self.sample_label()
        else:
            with self.env.begin(write=False) as txn:
                byteflow = txn.get(patch_name.encode('utf-8'))

            patch = pickle.loads(byteflow)
            label = self.labels[idx]

        patch = self.transform(patch) if self.transform is not None else patch
        return patch, label, idx

    def __len__(self):
        """Get length of Dataset."""
        return len(self.patch_names)


class Ben19V2Dataset(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, eval_resize=None, temporal_views_path=None, transform=None, split='train', 
                 active_classes=None, s1_mm=False, s1_only=False, discard_empty_labels=True):
        super().__init__(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)
        """
        Parameter
        ---------
        lmdb_path      : path to the LMDB file for efficiently loading the patches.
        csv_path       : path to a csv file containing the patch names that will make up this split
        transform_mode:  specifies the image transform mode which determines the augmentations
                         to be applied to the image
        """
        # self.active_classes = active_classes
        # self.labels = self.select_active_classes(self.labels)
        self.s1_mm = s1_mm
        self.s1_only = s1_only

        self.band_ordering = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
        if self.s1_mm:
            self.band_ordering = self.band_ordering + ['VH', 'VV']
        if self.s1_only:
            print('TRACE1')
            self.band_ordering = ['VH', 'VV']

        self.return_patchname = False
        self.reader = BENv2LDMBReader(
            image_lmdb_file=lmdb_path,
            metadata_file=labels_path,
            bands=self.band_ordering,
            process_bands_fn=None,
            process_labels_fn=self.convert_to_multihot,
        )

    def convert_to_multihot(self, labels):
        multihot = np.zeros(19)
        indices = [BEN19_NAME2IDX[label] for label in labels]
        multihot[indices] = 1
        return multihot

    def __getitem__(self, idx):
        patch_name = self.patch_names[idx]
        patch, label = self.reader[patch_name]
        patch = np.moveaxis(patch, 0, -1)  # s.t. ToTensor can reverse to beginning

        if self.s1_mm:
            patch[:, :, -2] = patch[:, :, -2] + 30  # hack to put VH into [0,1] via clipping
            patch[:, :, -1] = patch[:, :, -1] + 25  # hack to put VV into [0,1] via clipping

        if self.eval_resize is not None:
            shape = tuple(self.eval_resize) + (10, )
            patch = F.interpolate(torch.Tensor(patch.astype(np.float32)).unsqueeze(0).unsqueeze(0), shape, mode='nearest').squeeze().numpy()
        patch = self.transform(patch) if self.transform is not None else patch

        return patch, label, idx


class Ben19V2DatasetIB(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, eval_resize=None, temporal_views_path=None, transform=None,
                 split='train', active_classes=None, s1_mm=False, discard_empty_labels=True,
                 noise_image_pct=0.05, p_base=0.05, classes_for_p=[], p_classwise=[]
                 ):
        super().__init__(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)
        """
        Parameter
        ---------
        lmdb_path      : path to the LMDB file for efficiently loading the patches.
        csv_path       : path to a csv file containing the patch names that will make up this split
        transform_mode:  specifies the image transform mode which determines the augmentations
                         to be applied to the image
        """
        # self.active_classes = active_classes
        # self.labels = self.select_active_classes(self.labels)
        self.band_ordering = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
        self.return_patchname = False
        self.reader = BENv2LDMBReader(
            image_lmdb_file=lmdb_path,
            metadata_file=labels_path,
            bands=self.band_ordering,
            process_bands_fn=None,
            process_labels_fn=self.convert_to_multihot,
        )

        if self.split == 'train':
            self.noise_image_pct = noise_image_pct
            noise_names = np.array(['noise' for i in range(int(len(self.patch_names) * self.noise_image_pct))])
            self.patch_names = np.concatenate([self.patch_names, noise_names])
            self.weights = np.load('/mnt/storagecube/tomburgert/bigearthnet_stats/benv2_kdes/random_choice_norm.npy')

            self.p_base = p_base
            self.p_list = np.array([p_base for i in range(19)])
            self.classes_for_p = classes_for_p
            self.p_classwise = p_classwise
            self.p_list[self.classes_for_p] = self.p_classwise

    def convert_to_multihot(self, labels):
        multihot = np.zeros(19)
        indices = [BEN19_NAME2IDX[label] for label in labels]
        multihot[indices] = 1
        return multihot

    # def read_pickled_kde(path):
    #     with open(path, 'rb') as f:
    #         return cloudpickle.load(f)

    def sample_noise(self):
        return np.array([np.random.choice(np.arange(15000), size=(120, 120), p=self.weights[i]) for i in range(10)])

    def sample_label(self):
        label = np.zeros(19)
        while label.sum() == 0:
            label = np.array([np.random.choice([0, 1], p=[1 - p, p]) for p in self.p_list])
        return label.astype(np.float64)

    def __getitem__(self, idx):
        patch_name = self.patch_names[idx]
        if patch_name == 'noise':
            patch = self.sample_noise().astype(np.float32)
            patch = np.moveaxis(patch, 0, -1)  # s.t. ToTensor can reverse to beginning
            label = self.sample_label()
        else:
            patch, label = self.reader[patch_name]
            patch = np.moveaxis(patch, 0, -1)  # s.t. ToTensor can reverse to beginning
            if self.eval_resize is not None:
                shape = tuple(self.eval_resize) + (10, )
                patch = F.interpolate(torch.Tensor(patch.astype(np.float32)).unsqueeze(0).unsqueeze(0), shape, mode='nearest').squeeze().numpy()
        
        patch = self.transform(patch) if self.transform is not None else patch
        return patch, label, idx


class Ben19Dataset(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, eval_resize=None, temporal_views_path=None, transform=None, split='train', 
                 active_classes=None, s1_mm=False, discard_empty_labels=True):
        super().__init__(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)
        """
        Parameter
        ---------
        lmdb_path      : path to the LMDB file for efficiently loading the patches.
        csv_path       : path to a csv file containing the patch names that will make up this split
        transform_mode:  specifies the image transform mode which determines the augmentations
                         to be applied to the image
        """
        self.active_classes = active_classes
        self.labels = self.read_labels(labels_path, self.patch_names)
        self.labels = self.select_active_classes(self.labels)
        self.s1_mm = s1_mm
        self.band_ordering = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']

        if discard_empty_labels:
            self.discard_empty_labels()

    def read_labels(self, meta_data_path, patch_names):
        df = pd.read_parquet(meta_data_path)
        df_subset = df.set_index('name').loc[self.patch_names].reset_index(inplace=False)
        string_labels = df_subset.new_labels.tolist()
        multihot_labels = np.array(list(map(self.convert_to_multihot, string_labels)))
        return multihot_labels
    
    def convert_to_multihot(self, labels):
        multihot = np.zeros(19)
        indices = [BEN19_NAME2IDX[label] for label in labels]
        multihot[indices] = 1
        return multihot

    def interpolate_bands(self, bands, img10_shape=[120, 120]):
        """Interpolate bands. See: https://github.com/lanha/DSen2/blob/master/utils/patches.py."""
        bands_interp = np.zeros([bands.shape[0]] + img10_shape).astype(np.float32)
        for i in range(bands.shape[0]):
            bands_interp[i] = resize(bands[i] / 30000, img10_shape, mode='reflect') * 30000
        return bands_interp

    def select_active_classes(self, multihot):
        if self.active_classes is not None:
            multihot = multihot[:, self.active_classes]
        return multihot

    def discard_empty_labels(self):
        empty_idx = np.argwhere(self.labels.sum(axis=1) == 0).flatten()
        print("Removing {} patches without any active label.".format(len(empty_idx)))
        self.patch_names = np.delete(self.patch_names, empty_idx)
        self.labels = np.delete(self.labels, empty_idx, axis=0)

    def __getitem__(self, idx):
        """Get item at position idx of Dataset."""
        if self.env is None:
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
            )

        patch_name = self.patch_names[idx]
            
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode('utf-8'))
        
        s2_patch = BigEarthNet_S2_Patch.loads(byteflow)
        label = self.labels[idx]

        bands10 = s2_patch.get_stacked_10m_bands()
        bands10 = bands10.astype(np.float32)
        bands20 = s2_patch.get_stacked_20m_bands()
        bands20 = self.interpolate_bands(bands20)
        bands20 = bands20.astype(np.float32)

        # put channel to last axis s.t. toTensor can flip them to first axis again
        patch = np.moveaxis(np.concatenate([bands10, bands20]), 0, -1)
        if self.eval_resize is not None:
            shape = tuple(self.eval_resize) + (10, )
            patch = F.interpolate(torch.Tensor(patch.astype(np.float32)).unsqueeze(0).unsqueeze(0), shape, mode='nearest').squeeze().numpy()

        patch = self.transform(patch) if self.transform is not None else patch

        # if self.s1_mm:
        #     patch = patch[[2, 1, 0], ...]

        return patch, label, idx


class SeCoDataset(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, eval_resize=None, temporal_views_path=None, transform=None, split='train'):
        super().__init__(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)
        """
        Parameter
        ---------
        lmdb_path      : for SeCo LMBD PATH = seasonal dependency
        csv_path       : path to a csv file containing the patch names that will make up this split
        transform_mode:  specifies the image transform mode which determines the augmentations
                         to be applied to the image
        """
        self.labels = self.read_labels(labels_path, self.patch_names)
        self.band_ordering = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
        if self.temporal_views is not None:
            self.temporal_groupby = self.temporal_views.groupby('group')

    def read_labels(self, meta_data_path, patch_names):
        # not needed, unsupervised dataset
        return np.array(0)

    def interpolate_bands(self, bands, img10_shape=[264, 264]):
        """Interpolate bands. See: https://github.com/lanha/DSen2/blob/master/utils/patches.py."""
        bands_interp = np.zeros([bands.shape[0]] + img10_shape).astype(np.float32)
        for i in range(bands.shape[0]):
            bands_interp[i] = resize(bands[i] / 30000, img10_shape, mode='reflect') * 30000
        return bands_interp

    def __getitem__(self, idx):
        """Get item at position idx of Dataset."""
        if self.env is None:
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
                max_spare_txns=8,
            )

        patch_name = self.patch_names[idx]
            
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode('utf-8'))

        s2_patch = SeCo_Patch.loads(byteflow)
        label = 0

        bands10 = s2_patch.get_stacked_10m_bands()
        bands10 = bands10.astype(np.float32)
        bands10_shape = list(bands10.shape[-2:])
        bands20 = s2_patch.get_stacked_20m_bands()
        bands20 = self.interpolate_bands(bands20, img10_shape=bands10_shape)
        bands20 = bands20.astype(np.float32)

        # put channel to last axis s.t. toTensor can flip them to first axis again
        patch = np.moveaxis(np.concatenate([bands10, bands20]), 0, -1)
        patch = patch.astype(np.uint8)  # only for test reasons here, normalize the lmdb issue!

        if self.temporal_views is not None:
            group = patch_name.split('_')[0]
            view_candidates = self.temporal_groupby.get_group(group).name.to_numpy()
            view_candidates = np.delete(view_candidates, np.argwhere(view_candidates == patch_name))
            patch_name2 = np.random.choice(view_candidates)

            with self.env.begin(write=False) as txn:
                byteflow = txn.get(patch_name2.encode('utf-8'))

            s2_patch = SeCo_Patch.loads(byteflow)
            label = 0

            bands10 = s2_patch.get_stacked_10m_bands()
            bands10 = bands10.astype(np.float32)
            bands20 = s2_patch.get_stacked_20m_bands()
            bands20 = self.interpolate_bands(bands20, img10_shape=bands10_shape)
            bands20 = bands20.astype(np.float32)

            # put channel to last axis s.t. toTensor can flip them to first axis again
            patch2 = np.moveaxis(np.concatenate([bands10, bands20]), 0, -1)
            patch = [patch, patch2]
            patch = patch.astype(np.uint8)  # only for test reasons here, normalize the lmdb issue!

        patch = self.transform(patch)

        return patch, label, idx


class SeCoDatasetTemporal(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, eval_resize=None, temporal_views_path=None, transform=None, split='train'):
        super().__init__(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)
        """
        Parameter
        ---------
        lmdb_path      : for SeCo LMBD PATH = seasonal dependency
        csv_path       : path to a csv file containing the patch names that will make up this split
        transform_mode:  specifies the image transform mode which determines the augmentations
                         to be applied to the image
        """
        self.labels = self.read_labels(labels_path, self.patch_names)
        self.band_ordering = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
        if self.temporal_views is not None:
            self.temporal_groupby = self.temporal_views.groupby('group')
            self.groups = np.unique(list(map(lambda x: x.split('_')[0], self.patch_names)))

    def read_labels(self, meta_data_path, patch_names):
        # not needed, unsupervised dataset
        return np.array(0)

    def interpolate_bands(self, bands, img10_shape=[264, 264]):
        """Interpolate bands. See: https://github.com/lanha/DSen2/blob/master/utils/patches.py."""
        bands_interp = np.zeros([bands.shape[0]] + img10_shape).astype(np.float32)
        for i in range(bands.shape[0]):
            bands_interp[i] = resize(bands[i] / 30000, img10_shape, mode='reflect') * 30000
        return bands_interp

    def load_patch(self, patch_name):
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode('utf-8'))

        s2_patch = SeCo_Patch.loads(byteflow)
        label = 0

        bands10 = s2_patch.get_stacked_10m_bands()
        bands10 = bands10.astype(np.float32)
        bands10_shape = list(bands10.shape[-2:])
        bands20 = s2_patch.get_stacked_20m_bands()
        bands20 = self.interpolate_bands(bands20, img10_shape=bands10_shape)
        bands20 = bands20.astype(np.float32)

        # put channel to last axis s.t. toTensor can flip them to first axis again
        patch = np.moveaxis(np.concatenate([bands10, bands20]), 0, -1)
        patch = patch.astype(np.uint8)  # only for test reasons here, normalize the lmdb issue!
        return patch, label

    def __getitem__(self, idx):
        """Get item at position idx of Dataset."""
        if self.env is None:
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
                max_spare_txns=8,
            )

        if self.temporal_views is None:
            patch_name = self.patch_names[idx]
            patch, label = self.load_patch(patch_name)

        if self.temporal_views is not None:
            group = self.groups[idx]
            view_candidates = self.temporal_groupby.get_group(group).name.to_numpy()
            patch_names = np.random.choice(view_candidates, 2, replace=False)
            patch1, label1 = self.load_patch(patch_names[0])
            patch2, label2 = self.load_patch(patch_names[1])
            patch = [patch1, patch2]
            label = label1

        patch = self.transform(patch)

        return patch, label, idx

    def __len__(self):
        N = len(self.patch_names) if self.temporal_views is None else len(self.groups)
        return N


class SeCo264x264Dataset(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, eval_resize=None, temporal_views_path=None, transform=None, split='train'):
        super().__init__(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)
        """
        Parameter
        ---------
        lmdb_path      : for SeCo LMBD PATH = seasonal dependency
        csv_path       : path to a csv file containing the patch names that will make up this split
        transform_mode:  specifies the image transform mode which determines the augmentations
                         to be applied to the image
        """
        self.labels = self.read_labels(labels_path, self.patch_names)
        self.band_ordering = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
        if self.temporal_views is not None:
            self.temporal_groupby = self.temporal_views.groupby('group')

    def read_labels(self, meta_data_path, patch_names):
        # not needed, unsupervised dataset
        return np.array(0)

    def interpolate_bands(self, bands, img10_shape=[264, 264]):
        """Interpolate bands. See: https://github.com/lanha/DSen2/blob/master/utils/patches.py."""
        bands_interp = np.zeros([bands.shape[0]] + img10_shape).astype(np.float32)
        for i in range(bands.shape[0]):
            bands_interp[i] = resize(bands[i] / 30000, img10_shape, mode='reflect') * 30000
        return bands_interp

    def __getitem__(self, idx):
        """Get item at position idx of Dataset."""
        if self.env is None:
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
                max_spare_txns=8,
            )

        patch_name = self.patch_names[idx]
            
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode('utf-8'))

        s2_patch = SeCo_Patch.loads(byteflow)
        label = 0

        bands10 = s2_patch.get_stacked_10m_bands()
        bands10 = bands10.astype(np.float32)
        bands10_shape = list(bands10.shape[-2:])
        bands20 = s2_patch.get_stacked_20m_bands()
        bands20 = self.interpolate_bands(bands20, img10_shape=bands10_shape)
        bands20 = bands20.astype(np.float32)

        # put channel to last axis s.t. toTensor can flip them to first axis again
        patch = np.moveaxis(np.concatenate([bands10, bands20]), 0, -1)

        if self.temporal_views is not None:
            group = patch_name.split('_')[0]
            view_candidates = self.temporal_groupby.get_group(group).name.to_numpy()
            view_candidates = np.delete(view_candidates, np.argwhere(view_candidates == patch_name))
            patch_name2 = np.random.choice(view_candidates)

            with self.env.begin(write=False) as txn:
                byteflow = txn.get(patch_name2.encode('utf-8'))

            s2_patch = SeCo_Patch.loads(byteflow)
            label = 0

            bands10 = s2_patch.get_stacked_10m_bands()
            bands10 = bands10.astype(np.float32)
            bands20 = s2_patch.get_stacked_20m_bands()
            bands20 = self.interpolate_bands(bands20, img10_shape=bands10_shape)
            bands20 = bands20.astype(np.float32)

            # put channel to last axis s.t. toTensor can flip them to first axis again
            patch2 = np.moveaxis(np.concatenate([bands10, bands20]), 0, -1)
            patch = [patch, patch2]

        patch = self.transform(patch)

        return patch, label, idx


class SSL4EODataset(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, eval_resize=None, temporal_views_path=None, transform=None, split='train'):
        super().__init__(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)
        """
        Parameter
        ---------
        lmdb_path      : for SeCo LMBD PATH = seasonal dependency
        csv_path       : path to a csv file containing the patch names that will make up this split
        transform_mode:  specifies the image transform mode which determines the augmentations
                         to be applied to the image
        """
        self.labels = self.read_labels(labels_path, self.patch_names)
        self.band_ordering = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
        if self.temporal_views is not None:
            self.temporal_groupby = self.temporal_views.groupby('group')
            self.groups = np.unique(list(map(lambda x: x.split('_')[0], self.patch_names)))

    def read_labels(self, meta_data_path, patch_names):
        # not needed, unsupervised dataset
        return np.array(0)

    def interpolate_bands(self, bands, img10_shape=[264, 264]):
        """Interpolate bands. See: https://github.com/lanha/DSen2/blob/master/utils/patches.py."""
        bands_interp = np.zeros([bands.shape[0]] + img10_shape).astype(np.float32)
        for i in range(bands.shape[0]):
            bands_interp[i] = resize(bands[i] / 30000, img10_shape, mode='reflect') * 30000
        return bands_interp

    def load_patch(self, patch_name):
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode('utf-8'))

        if self.lmdb_path[-5] == '64.db':
            s2_patch = SeCo_Patch.loads(byteflow)
        elif self.lmdb_path[-5] == '60.db':
            s2_patch = AnyS2_Patch.loads(byteflow)
        else:
            s2_patch = BigEarthNet_S2_Patch.loads(byteflow)
        label = 0

        bands10 = s2_patch.get_stacked_10m_bands()
        bands10 = bands10.astype(np.float32)
        bands10_shape = list(bands10.shape[-2:])
        bands20 = s2_patch.get_stacked_20m_bands()
        bands20 = self.interpolate_bands(bands20, img10_shape=bands10_shape)
        bands20 = bands20.astype(np.float32)

        # put channel to last axis s.t. toTensor can flip them to first axis again
        patch = np.moveaxis(np.concatenate([bands10, bands20]), 0, -1)
        # patch = patch.astype(np.uint8)  # only for test reasons here, normalize the lmdb issue!
        return patch, label

    def __getitem__(self, idx):
        """Get item at position idx of Dataset."""
        if self.env is None:
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
                max_spare_txns=8,
            )

        if self.temporal_views is None:
            patch_name = self.patch_names[idx]
            patch, label = self.load_patch(patch_name)

        if self.temporal_views is not None:
            group = self.groups[idx]
            view_candidates = self.temporal_groupby.get_group(group).name.to_numpy()
            replace = False if len(view_candidates) > 1 else True
            patch_names = np.random.choice(view_candidates, 2, replace=replace)
            patch1, label1 = self.load_patch(patch_names[0])
            patch2, label2 = self.load_patch(patch_names[1])
            patch = [patch1, patch2]
            label = label1

        patch = self.transform(patch)

        return patch, label, idx

    def __len__(self):
        N = len(self.patch_names) if self.temporal_views is None else len(self.groups)
        return N


class DeepGlobeDataset(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, eval_resize=None, temporal_views_path=None, transform=None, split='train', 
                 use_balancing=False, noise_image_pct=0.05, p_base=0.05, classes_for_p=[], p_classwise=[]):
        super().__init__(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)
        self.labels = self.read_labels(labels_path, self.patch_names)

        if use_balancing and self.split == 'train':
            self.use_balancing = use_balancing
            self.noise_image_pct = noise_image_pct
            noise_names = np.array(['noise' for i in range(int(len(self.patch_names) * noise_image_pct))])
            self.patch_names = np.concatenate([self.patch_names, noise_names])
            self.weights = np.load('/mnt/storagecube/data/shared/datasets/deepGlobe/ML_DeepGlobe/kdes/random_choice_norm.npy')

            self.p_base = p_base
            self.p_list = np.array([p_base for i in range(6)])
            self.classes_for_p = classes_for_p
            self.p_classwise = p_classwise
            self.p_list[self.classes_for_p] = self.p_classwise

    def sample_noise(self):
        return np.array([np.random.choice(np.arange(256), size=(120, 120), p=self.weights[i]) for i in range(3)])

    def sample_label(self):
        label = np.zeros(6)
        while label.sum() == 0:
            label = np.array([np.random.choice([0, 1], p=[1 - p, p]) for p in self.p_list])
        return label.astype(np.float64)
        
    def convert_to_multihot(self, labels):
        multihot = np.zeros(6)
        indices = [DEEPGLOBE_NAME2IDX[label] for label in labels]
        multihot[indices] = 1
        return multihot
    
    
class TreeSatAIDataset(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, eval_resize=None, temporal_views_path=None, transform=None, split='train',
                 use_balancing=False, noise_image_pct=0.05, p_base=0.05, classes_for_p=[], p_classwise=[]):
        super().__init__(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)
        self.labels = self.read_labels(labels_path, self.patch_names)

        if use_balancing and self.split == 'train':
            self.use_balancing = use_balancing
            self.noise_image_pct = noise_image_pct
            noise_names = np.array(['noise' for i in range(int(len(self.patch_names) * noise_image_pct))])
            self.patch_names = np.concatenate([self.patch_names, noise_names])
            self.weights = np.load('/mnt/storagecube/data/rsim_member/datasets_private/TreeSatAI/kdes/random_choice_norm.npy')

            self.p_base = p_base
            self.p_list = np.array([p_base for i in range(15)])
            self.classes_for_p = classes_for_p
            self.p_classwise = p_classwise
            self.p_list[self.classes_for_p] = self.p_classwise

    def sample_noise(self):
        return np.array([np.random.choice(np.arange(256), size=(304, 304), p=self.weights[i]) for i in range(4)])

    def sample_label(self):
        label = np.zeros(15)
        while label.sum() == 0:
            label = np.array([np.random.choice([0, 1], p=[1 - p, p]) for p in self.p_list])
        return label.astype(np.float64)
    
    def convert_to_multihot(self, labels):
        multihot = np.zeros(15)
        indices = [TREESATAI_NAME2IDX[label] for label in labels]
        multihot[indices] = 1
        return multihot


class EuroSATDataset(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, eval_resize=None, temporal_views_path=None, transform=None, split='train'):
        super().__init__(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)
        self.band_ordering = [
            'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B010', 'B11', 'B12'
        ]
        self.ben19_ordering = [1, 2, 3, 7, 4, 5, 6, 8, 11, 12]
        self.labels = self.read_labels(labels_path, self.patch_names)
    
    def convert_to_multihot(self, labels):
        return EUROSAT_NAME2IDX[labels[0]]  # hack for single-label classification

    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
            )

        patch_name = self.patch_names[idx]
            
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode('utf-8'))

        patch = pickle.loads(byteflow)

        label = self.labels[idx]
        patch = patch[:, :, self.ben19_ordering]
        if self.eval_resize is not None:
            shape = tuple(self.eval_resize) + (10, )
            patch = F.interpolate(torch.Tensor(patch.astype(np.float32)).unsqueeze(0).unsqueeze(0), shape, mode='nearest').squeeze().numpy()
        patch = self.transform(patch) if self.transform is not None else patch
        return patch, label, idx


class EuroSATV2Dataset(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, eval_resize=None, temporal_views_path=None, transform=None, split='train'):
        super().__init__(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)
        self.labels = self.read_labels(labels_path, self.patch_names)
    
    def convert_to_multihot(self, labels):
        return EUROSAT_NAME2IDX[labels[0]]  # hack for single-label classification

    def interpolate_bands(self, bands, img10_shape=[120, 120]):
        """Interpolate bands. See: https://github.com/lanha/DSen2/blob/master/utils/patches.py."""
        bands_interp = np.zeros([bands.shape[0]] + img10_shape).astype(np.float32)
        for i in range(bands.shape[0]):
            bands_interp[i] = resize(bands[i] / 30000, img10_shape, mode='reflect') * 30000
        return bands_interp

    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
            )

        patch_name = self.patch_names[idx]
            
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode('utf-8'))

        s2_patch = AnyS2_Patch.loads(byteflow)
        label = self.labels[idx]

        bands10 = s2_patch.get_stacked_10m_bands()
        bands10 = bands10.astype(np.float32)
        bands20 = s2_patch.get_stacked_20m_bands()
        bands20 = self.interpolate_bands(bands20, img10_shape=list(bands10.shape[1:]))
        bands20 = bands20.astype(np.float32)

        # put channel to last axis s.t. toTensor can flip them to first axis again
        patch = np.moveaxis(np.concatenate([bands10, bands20]), 0, -1)
        if self.eval_resize is not None:
            shape = tuple(self.eval_resize) + (10, )
            patch = F.interpolate(torch.Tensor(patch.astype(np.float32)).unsqueeze(0).unsqueeze(0), shape, mode='nearest').squeeze().numpy()
        patch = self.transform(patch) if self.transform is not None else patch
        return patch, label, idx


class So2SatDataset(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, eval_resize=None, temporal_views_path=None, transform=None, split='train'):
        super().__init__(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)
        self.band_ordering = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
        self.ben19_ordering = [0, 1, 2, 6, 3, 4, 5, 7, 8, 9]
        self.labels = self.read_labels(labels_path, self.patch_names)
    
    def convert_to_multihot(self, labels):
        return SO2SAT_NAME2IDX[labels[0]]  # hack for single-label classification

    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
            )

        patch_name = self.patch_names[idx]
            
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode('utf-8'))

        patch = pickle.loads(byteflow)

        label = self.labels[idx]
        patch = patch[:, :, self.ben19_ordering]
        if self.eval_resize is not None:
            shape = tuple(self.eval_resize) + (10, )
            patch = F.interpolate(torch.Tensor(patch.astype(np.float32)).unsqueeze(0).unsqueeze(0), shape, mode='nearest').squeeze().numpy()
        patch = self.transform(patch) if self.transform is not None else patch

        return patch, label, idx


class FMoWDataset(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, eval_resize=None, temporal_views_path=None, transform=None, split='train'):
        super().__init__(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)
        self.band_ordering = [
            'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B010', 'B11', 'B12'
        ]
        self.ben19_ordering = [1, 2, 3, 7, 4, 5, 6, 8, 11, 12]
        self.labels = self.read_labels(labels_path, self.patch_names)
    
    def convert_to_multihot(self, labels):
        return FMOW_NAME2IDX[labels[0]]  # hack for single-label classification

    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
            )

        patch_name = self.patch_names[idx]
            
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode('utf-8'))

        patch = pickle.loads(byteflow)

        label = self.labels[idx]
        patch = patch[:, :, self.ben19_ordering]
        patch = resize(patch / 30000, (64, 64), mode='reflect') * 30000

        patch = self.transform(patch) if self.transform is not None else patch

        return patch, label, idx


class S4ADataset(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, eval_resize=None, temporal_views_path=None, transform=None, split='train'):
        super().__init__(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)
        """
        Parameter
        ---------
        lmdb_path      : path to the LMDB file for efficiently loading the patches.
        csv_path       : path to a csv file containing the patch names that will make up this split
        transform_mode:  specifies the image transform mode which determines the augmentations
                         to be applied to the image
        """
        self.band_ordering = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
        self.labels = self.read_labels(labels_path, self.patch_names)
    
    def convert_to_multihot(self, labels):
        multihot = np.zeros(9)
        indices = [S4A_LOW_NAME2IDX[S4A_HIGH2LOW_STR[label]] for label in labels]
        multihot[indices] = 1
        return multihot

    def interpolate_bands(self, bands, img10_shape=[122, 122]):
        """Interpolate bands. See: https://github.com/lanha/DSen2/blob/master/utils/patches.py."""
        bands_interp = np.zeros([bands.shape[0]] + img10_shape).astype(np.float32)
        for i in range(bands.shape[0]):
            bands_interp[i] = resize(bands[i] / 30000, img10_shape, mode='reflect') * 30000
        return bands_interp

    def __getitem__(self, idx):
        """Get item at position idx of Dataset."""
        if self.env is None:
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
            )

        patch_name = self.patch_names[idx]
            
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode('utf-8'))

        s2_patch = S4A_Patch.loads(byteflow)
        label = self.labels[idx]

        bands10 = s2_patch.get_stacked_10m_bands()
        bands10 = bands10.astype(np.float32)
        bands20 = s2_patch.get_stacked_20m_bands()
        bands20 = self.interpolate_bands(bands20)
        bands20 = bands20.astype(np.float32)

        # put channel to last axis s.t. toTensor can flip them to first axis again
        patch = np.moveaxis(np.concatenate([bands10, bands20]), 0, -1)
        if self.eval_resize is not None:
            shape = tuple(self.eval_resize) + (10, )
            patch = F.interpolate(torch.Tensor(patch.astype(np.float32)).unsqueeze(0).unsqueeze(0), shape, mode='nearest').squeeze().numpy()

        patch = self.transform(patch)
        return patch, label, idx


class S4ASegmentationDataset(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, eval_resize=None, temporal_views_path=None, transform=None, split='train'):
        super().__init__(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)
        """
        Parameter
        ---------
        lmdb_path      : path to the LMDB file for efficiently loading the patches.
        csv_path       : path to a csv file containing the patch names that will make up this split
        transform_mode:  specifies the image transform mode which determines the augmentations
                         to be applied to the image
        """
        self.band_ordering = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
        self.lmdb_path_maps = '/data/tomburgert/data/additional_data/Sen4AgriNet/ref_map_lmdb.db'
        self.env_maps = None
        self.id2name = {v: k for k, v in S4A_NAME2ENCODING.items()}

    def convert_labels_to_reference_map(self, labels):
        return np.vectorize(lambda label: S4A_SEMSEG_LOW_NAME2IDX[S4A_HIGH2LOW_STR[self.id2name[label]]])(labels)

    def interpolate_bands(self, bands, img10_shape=[122, 122]):
        """Interpolate bands. See: https://github.com/lanha/DSen2/blob/master/utils/patches.py."""
        bands_interp = np.zeros([bands.shape[0]] + img10_shape).astype(np.float32)
        for i in range(bands.shape[0]):
            bands_interp[i] = resize(bands[i] / 30000, img10_shape, mode='reflect') * 30000
        return bands_interp

    def __getitem__(self, idx):
        """Get item at position idx of Dataset."""
        if self.env is None:
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
            )

        if self.env_maps is None:
            self.env_maps = lmdb.open(
                str(self.lmdb_path_maps),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
            )

        patch_name = self.patch_names[idx]
        label_name = '_'.join(patch_name.split('_')[:5] + patch_name.split('_')[-1:])
            
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode('utf-8'))

        s2_patch = S4A_Patch.loads(byteflow)

        with self.env_maps.begin(write=False) as txn:
            byteflow_label = txn.get(label_name.encode('utf-8'))

        label_map = pickle.loads(byteflow_label)
        label_map = self.convert_labels_to_reference_map(label_map)

        bands10 = s2_patch.get_stacked_10m_bands()
        bands10 = bands10.astype(np.float32)
        bands20 = s2_patch.get_stacked_20m_bands()
        bands20 = self.interpolate_bands(bands20)
        bands20 = bands20.astype(np.float32)

        # put channel to last axis s.t. toTensor can flip them to first axis again
        patch = np.moveaxis(np.concatenate([bands10, bands20]), 0, -1)
        if self.eval_resize is not None:
            shape = tuple(self.eval_resize) + (10, )
            patch = F.interpolate(torch.Tensor(patch.astype(np.float32)).unsqueeze(0).unsqueeze(0), shape, mode='nearest').squeeze().numpy()

        patch = self.transform(patch)
        return patch, label_map, idx


class SeasonNetDataset(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, eval_resize=None, temporal_views_path=None, transform=None, split='train'):
        super().__init__(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)
        """
        Parameter
        ---------
        lmdb_path      : path to the LMDB file for efficiently loading the patches.
        csv_path       : path to a csv file containing the patch names that will make up this split
        transform_mode:  specifies the image transform mode which determines the augmentations
                         to be applied to the image
        """
        self.band_ordering = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
        self.lmdb_path_maps = '/data/tomburgert/data/datasets/SeasonNet/ref_map_lmdb.db'
        self.env_maps = None

    def interpolate_bands(self, bands, img10_shape=[120, 120]):
        """Interpolate bands. See: https://github.com/lanha/DSen2/blob/master/utils/patches.py."""
        bands_interp = np.zeros([bands.shape[0]] + img10_shape).astype(np.float32)
        for i in range(bands.shape[0]):
            bands_interp[i] = resize(bands[i] / 30000, img10_shape, mode='reflect') * 30000
        return bands_interp

    def __getitem__(self, idx):
        """Get item at position idx of Dataset."""
        if self.env is None:
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
            )

        if self.env_maps is None:
            self.env_maps = lmdb.open(
                str(self.lmdb_path_maps),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
            )

        patch_name = self.patch_names[idx]
            
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode('utf-8'))

        s2_patch = S4A_Patch.loads(byteflow)

        with self.env_maps.begin(write=False) as txn:
            byteflow_label = txn.get(patch_name.encode('utf-8'))

        label_map = pickle.loads(byteflow_label)[0]
        label_map = label_map.astype(np.int64)
        # ensure that first label is label 0
        label_map = label_map - 1 
        # label_map = self.convert_labels_to_reference_map(label_map)

        bands10 = s2_patch.get_stacked_10m_bands()
        bands10 = bands10.astype(np.float32)
        bands20 = s2_patch.get_stacked_20m_bands()
        bands20 = self.interpolate_bands(bands20)
        bands20 = bands20.astype(np.float32)

        # put channel to last axis s.t. toTensor can flip them to first axis again
        patch = np.moveaxis(np.concatenate([bands10, bands20]), 0, -1)
        if self.eval_resize is not None:
            shape = tuple(self.eval_resize) + (10, )
            patch = F.interpolate(torch.Tensor(patch.astype(np.float32)).unsqueeze(0).unsqueeze(0), shape, mode='nearest').squeeze().numpy()

        patch = self.transform(patch)
        return patch, label_map, idx


class Ben19CloudsDataset(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, eval_resize=None, temporal_views_path=None, transform=None, split='train', active_classes=None,
                 s1_mm=False):
        super().__init__(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)
        """
        Parameter
        ---------
        lmdb_path      : path to the LMDB file for efficiently loading the patches.
        csv_path       : path to a csv file containing the patch names that will make up this split
        transform_mode:  specifies the image transform mode which determines the augmentations
                         to be applied to the image
        """
        self.active_classes = active_classes
        self.labels = self.read_labels(labels_path, self.patch_names)
        self.s1_mm = s1_mm
        self.band_ordering = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]
    
    def convert_to_multihot(self, labels):
        return [float(BEN19CLOUDS_NAME2IDX[labels[0]])]

    def interpolate_bands(self, bands, img10_shape=[120, 120]):
        """Interpolate bands. See: https://github.com/lanha/DSen2/blob/master/utils/patches.py."""
        bands_interp = np.zeros([bands.shape[0]] + img10_shape).astype(np.float32)
        for i in range(bands.shape[0]):
            bands_interp[i] = resize(bands[i] / 30000, img10_shape, mode='reflect') * 30000
        return bands_interp

    def __getitem__(self, idx):
        """Get item at position idx of Dataset."""
        if self.env is None:
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
            )

        patch_name = self.patch_names[idx]
            
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode('utf-8'))

        s2_patch = BigEarthNet_S2_Patch.loads(byteflow)
        label = self.labels[idx]

        bands10 = s2_patch.get_stacked_10m_bands()
        bands10 = bands10.astype(np.float32)
        bands20 = s2_patch.get_stacked_20m_bands()
        bands20 = self.interpolate_bands(bands20)
        bands20 = bands20.astype(np.float32)

        # put channel to last axis s.t. toTensor can flip them to first axis again
        patch = np.moveaxis(np.concatenate([bands10, bands20]), 0, -1)
        patch = self.transform(patch)
        # if self.s1_mm:
        #     patch = patch[[2, 1, 0], ...]

        return patch, label, idx


def stack_and_interpolate(
    bands: Dict[str, np.ndarray],
    order: Optional[Iterable[str]] = None,
    img_size: int = 120,
    upsample_mode: str = "nearest",
) -> np.array:
    """
    Supports 2D input (as values in the dict) with "nearest", "bilinear" and "bicubic" interpolation
    """

    def _interpolate(img_data):
        if not img_data.shape[-2:] == (img_size, img_size):
            return F.interpolate(
                torch.Tensor(np.float32(img_data)).unsqueeze(0).unsqueeze(0),
                (img_size, img_size),
                mode=upsample_mode,
                align_corners=True if upsample_mode in ["bilinear", "bicubic"] else None,
            ).squeeze().numpy()
        else:
            return np.float32(img_data)

    # if order is None, order is alphabetical
    if order is None:
        order = sorted(bands.keys())
    return np.stack([_interpolate(bands[x]) for x in order])


_s1_bandnames = ["VH", "VV"]
_s2_bandnames = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"]
_all_bandnames = _s2_bandnames + _s1_bandnames

STANDARD_BANDS = {
    "S1": _s1_bandnames,
    "S2": _s2_bandnames,
    "ALL": _all_bandnames,
    "RGB": ["B04", "B03", "B02"],
    "10m": ["B02", "B03", "B04", "B08"],
    "20m": ["B05", "B06", "B07", "B11", "B12", "B8A"],
    "60m": ["B01", "B09"],
    2: _s1_bandnames,
    10: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A"],
    12: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A", "VH", "VV"],
    3: ["B04", "B03", "B02"],
    4: ["B04", "B03", "B02", "B08"],
}


def resolve_band_combi(bands: Union[Iterable, str, int]) -> list:
    """
    Resolves a predefined combination of bands or a list of bands into a list of
    individual bands and checks if all bands contained are actual S1/S2 band names.

    :param bands: a combination of bands as defined in BAND_COMBINATION_PREDEFINTIONS
        or a list of bands, or a single band, e.g. "B02", ["B02", "B03"], 2, "S1", "S2"
    :return: a list of bands contained in the predefinition or the single band as list
    """
    if isinstance(bands, str):
        if bands in _s1_bandnames or bands in _s2_bandnames:
            bands = [bands]
        else:
            assert bands in STANDARD_BANDS.keys(), (
                "Band combination unknown, please use a list of strings or one of " f"{STANDARD_BANDS.keys()}"
            )
            bands = STANDARD_BANDS[bands]
    elif isinstance(bands, int):
        assert bands in STANDARD_BANDS.keys(), (
            "Band combination unknown, please use a list of strings or one of " f"{STANDARD_BANDS.keys()}"
        )
        bands = STANDARD_BANDS[bands]
    elif isinstance(bands, Iterable):
        for band in bands:
            assert band in _all_bandnames, f"Band '{band}' unknown"
    else:
        raise ValueError(f"Unknown type of bands: {type(bands)}")
    assert isinstance(bands, list), "Bands should be a list"
    return bands


class BENv2LDMBReader:
    def __init__(
        self,
        image_lmdb_file: Union[str, Path],
        metadata_file: Union[str, Path],
        metadata_snow_cloud_file: Optional[Union[str, Path]] = None,
        bands: Optional[Union[Iterable, str, int]] = None,
        process_bands_fn: Optional[Callable[[Dict[str, np.ndarray], List[str]], Any]] = None,
        process_labels_fn: Optional[Callable[[List[str]], Any]] = None,
    ):
        self.image_lmdb_file = image_lmdb_file
        self.env = None

        self.bands = bands if bands is not None else _all_bandnames
        self.bands = resolve_band_combi(self.bands)
        self.uses_s1 = any([x in _s1_bandnames for x in self.bands])
        self.uses_s2 = any([x in _s2_bandnames for x in self.bands])

        self.metadata = pd.read_parquet(metadata_file)
        if metadata_snow_cloud_file is not None:
            metadata_snow_cloud = pd.read_parquet(metadata_snow_cloud_file)
            self.metadata = pd.concat([self.metadata, metadata_snow_cloud])

        # self.lbls = {row["patch_id"]: row["labels"] for idx, row in self.metadata.iterrows()}
        self.lbls = {p: l for p, l in zip(self.metadata["patch_id"], self.metadata["labels"])}
        self.lbl_key_set = set(self.lbls.keys())
        # self.mapping = {row["patch_id"]: row["s1_name"] for idx, row in self.metadata.iterrows()}
        self.mapping = {p: s for p, s in zip(self.metadata["patch_id"], self.metadata["s1_name"])}

        # set mean and std based on bands selected
        self.mean = None
        self.std = None

        self.process_bands_fn = stack_and_interpolate
        self.process_labels_fn = process_labels_fn if process_labels_fn is not None else lambda x: x

        self._keys: Optional[set] = None
        self._S2_keys: Optional[set] = None
        self._S1_keys: Optional[set] = None

    def open_env(self):
        if self.env is None:
            self.env = lmdb.open(
                str(self.image_lmdb_file),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
                map_size=8 * 1024**3,  # 8GB blocked for caching
                max_spare_txns=16,  # expected number of concurrent transactions (e.g. threads/workers)
            )

    def keys(self, update: bool = False):
        self.open_env()
        if self._keys is None or update:
            assert self.env is not None, "Environment not opened yet"
            with self.env.begin() as txn:
                self._keys = set(txn.cursor().iternext(values=False))
            self._keys = {x.decode() for x in self._keys}
        return self._keys

    def S2_keys(self, update: bool = False):
        if self._S2_keys is None or update:
            self._S2_keys = {key for key in self.keys(update) if key.startswith("S2")}
        return self._S2_keys

    def S1_keys(self, update: bool = False):
        if self._S1_keys is None or update:
            self._S1_keys = {key for key in self.keys(update) if key.startswith("S1")}
        return self._S1_keys

    def __getitem__(self, key: str):
        # the key is the name of the S2v2 patch

        # open lmdb file if not opened yet
        self.open_env()
        img_data_dict: dict = {}
        if self.uses_s2:
            assert self.env is not None, "Environment not opened yet"
            # read image data for S2v2
            with self.env.begin(write=False, buffers=True) as txn:
                byte_data = txn.get(key.encode())
            img_data_dict.update(safetensor_load(bytes(byte_data)))

        if self.uses_s1:
            # read image data for S1
            assert self.mapping is not None, "S1 bands are used, but no mapping is provided"
            s1_key = self.mapping[key]
            assert self.env is not None, "Environment not opened yet"
            with self.env.begin(write=False, buffers=True) as txn:
                byte_data = txn.get(s1_key.encode())
            img_data_dict.update(safetensor_load(bytes(byte_data)))

        assert isinstance(self.bands, list), "Bands should be a list"
        img_data_dict = {k: v for k, v in img_data_dict.items() if k in self.bands}

        img_data = self.process_bands_fn(img_data_dict, self.bands)
        labels = self.lbls[key] if key in self.lbl_key_set else []
        labels = self.process_labels_fn(labels)

        return img_data, labels
