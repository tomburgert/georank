import os
import copy
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.dataset import (
    Ben19Dataset,
    TreeSatAIDataset,
    DeepGlobeDataset,
    EuroSATDataset,
    EuroSATV2Dataset,
    So2SatDataset,
    FMoWDataset,
    SeCoDataset,
    SeCoDatasetTemporal,
    SeCo264x264Dataset,
    SSL4EODataset,
    S4ADataset,
    S4ASegmentationDataset,
    SeasonNetDataset,
    Ben19CloudsDataset,
    Ben19V2Dataset,
    Ben19V2DatasetIB
)
# from data.utils import add_mixed_noise
from data.constants import ACTIVE_CLASSES
from data.constants import (
    BAND_99TH_PERCENTILES,
    EUROSAT_99TH_PERCENTILES,
    EUROSATV2_99TH_PERCENTILES,
    SO2SAT_99TH_PERCENTILES,
    FMOW_99TH_PERCENTILES,
    S4A_99TH_PERCENTILES,
    SECO_95TH_PERCENTILES,
    SSL4EO_95TH_PERCENTILES
)
from data.constants import (
    BAND_NORM_STATS,
    EUROSAT_NORM_STATS,
    EUROSATV2_NORM_STATS,
    SO2SAT_NORM_STATS,
    FMOW_NORM_STATS,
    S4A_NORM_STATS,
    SECO_95TH_NORM_STATS,
    SSL4EO_95TH_NORM_STATS,
    CLOUD_NORM_STATS,
)


class DataModule(pl.LightningDataModule):

    def __init__(self, cfg, transform_tr, transform_te, eval_resize=None):
        super().__init__()
        self.cfg = cfg
        self.transform_tr = transform_tr
        self.transform_te = transform_te
        self.eval_resize = eval_resize

    def get_dataset(self, lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split):
        raise NotImplementedError

    def setup(self):

        if self.cfg.dataset.eval_on_test:
            self.cfg.dataset.val_csv = self.cfg.dataset.test_csv
            self.transform_val = self.transform_te

        self.trainset_tr = self.get_dataset(
            lmdb_path=self.cfg.dataset.lmdb_path,
            csv_path=self.cfg.dataset.train_csv,
            labels_path=self.cfg.dataset.labels_path,
            eval_resize=self.eval_resize,
            temporal_views_path=self.cfg.dataset.temporal_views_path,
            transform=self.transform_tr,
            split='train',
        )
        self.trainset_te = self.get_dataset(
            lmdb_path=self.cfg.dataset.lmdb_path,
            csv_path=self.cfg.dataset.train_csv,
            labels_path=self.cfg.dataset.labels_path,
            eval_resize=self.eval_resize,
            temporal_views_path=self.cfg.dataset.temporal_views_path,
            transform=self.transform_te,
            split='train',
        )
        self.valset = self.get_dataset(
            lmdb_path=self.cfg.dataset.lmdb_path,
            csv_path=self.cfg.dataset.val_csv,
            labels_path=self.cfg.dataset.labels_path,
            eval_resize=self.eval_resize,
            temporal_views_path=self.cfg.dataset.temporal_views_path,
            transform=self.transform_te,
            split='validation',
        )
        self.testset = self.get_dataset(
            lmdb_path=self.cfg.dataset.lmdb_path,
            csv_path=self.cfg.dataset.test_csv,
            labels_path=self.cfg.dataset.labels_path,
            eval_resize=self.eval_resize,
            temporal_views_path=self.cfg.dataset.temporal_views_path,
            transform=self.transform_te,
            split='test',
        )

        # self.trainset_tr.labels = add_mixed_noise(
        #     self.trainset_tr.labels,
        #     self.cfg.dataset.addn,
        #     self.cfg.dataset.subn
        # )

        # self.trainset_te.labels = self.trainset_tr.labels.copy()

    def get_loader(self, dataset, drop_last):
        shuffle = True if dataset == self.trainset_tr else False
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            shuffle=shuffle,
            pin_memory=self.cfg.dataset.pin_memory,
            drop_last=drop_last
        )
        return dataloader

    def train_dataloader(self, drop_last=False):
        return self.get_loader(self.trainset_tr, drop_last)

    def train_te_dataloader(self, drop_last=False):
        return self.get_loader(self.trainset_te, drop_last)

    def val_dataloader(self, drop_last=False):
        return self.get_loader(self.valset, drop_last)

    def test_dataloader(self, drop_last=False):
        return self.get_loader(self.testset, drop_last)


class BigEarthNetV2DataModule(DataModule):

    def __init__(self, cfg, transform_tr, transform_te, eval_resize=None):
        super().__init__(cfg, transform_tr, transform_te, eval_resize)
        self.cfg = cfg
        self.init_transforms()
        self.num_cls = 19
        # self.init_active_classes()

    def get_dataset(self, lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split):
        return Ben19V2Dataset(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split, s1_mm=self.cfg.dataset.s1_mm, s1_only=self.cfg.dataset.s1_only)

    def init_transforms(self):
        keys = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
        if self.cfg.dataset.s1_mm:
            keys = keys + ['VH', 'VV']
        if self.cfg.dataset.s1_only:
            keys = ['VH', 'VV']

        self.train_country = os.path.basename(self.cfg.dataset.train_csv).split('_')[0].capitalize()
        self.test_country = list(map(lambda x: os.path.basename(x).split('_')[0].capitalize(), self.cfg.dataset.test_csv))
        if self.cfg.dataset.global_pctl:
            print('Global Pre-Normalization.')
            tr_percentiles = 10000
        elif self.cfg.dataset.all_percentiles and not self.cfg.dataset.global_pctl:
            print('All BEN Pre-Normalization.')
            all_percentiles = BAND_99TH_PERCENTILES['All']
            tr_percentiles = [all_percentiles[k] for k in keys]
        else:
            print('Country Pre-Normalization.')
            all_percentiles = BAND_99TH_PERCENTILES['All']
            tr_percentiles = [all_percentiles[k] for k in keys]

            # tr_percentiles = list(BAND_99TH_PERCENTILES['All'].values())  # cheap hack (solve)
            # tr_percentiles = list(BAND_99TH_PERCENTILES[self.train_country].values())
        # te_percentiles = list(map(lambda x: list(BAND_99TH_PERCENTILES[x].values()), self.train_country))

        # currently train and test normalized by train norms (!)
        channel_global = 'Global' if self.cfg.dataset.global_pctl else 'Channel'
        if self.cfg.dataset.all_percentiles:
            print('All Percentiles', channel_global)
            all_means = BAND_NORM_STATS[channel_global]['All']['mean']
            all_stds = BAND_NORM_STATS[channel_global]['All']['std']
            means = [all_means[k] for k in keys]
            stds  = [all_stds[k] for k in keys]
        else:
            print('Country Percentiles', channel_global)
            all_means = BAND_NORM_STATS[channel_global]['All']['mean']
            all_stds = BAND_NORM_STATS[channel_global]['All']['std']
            means = [all_means[k] for k in keys]
            stds  = [all_stds[k] for k in keys]

            # means = list(BAND_NORM_STATS[channel_global]['All']['mean'].values())  # cheap hack (solve)
            # stds  = list(BAND_NORM_STATS[channel_global]['All']['std'].values())  # cheap hack (solve)
            # means = list(BAND_NORM_STATS[channel_global][self.train_country]['mean'].values())
            # stds  = list(BAND_NORM_STATS[channel_global][self.train_country]['std'].values())

        # add data transforms to transform_tr
        self.transform_tr.add_data_transforms(means, stds, tr_percentiles, sentinel2=True)
        self.transform_tr.setup_compose()

        # self.transform_val.add_data_transforms(means, stds, tr_percentiles, sentinel2=True)
        # self.transform_val.setup_compose()

        self.transform_te.add_data_transforms(means, stds, tr_percentiles, sentinel2=True)
        self.transform_te.setup_compose()

        # transform_va_list = []
        # for te_percentile in range(len(self.test_country)):
        #     transform = copy.deepcopy(self.transform_te)
        #     transform.add_data_transforms(means, stds, tr_percentiles, sentinel2=True)
        #     transform.setup_compose()
        #     transform_va_list.append(transform)
        # self.transform_val = transform_va_list

        # # add data transforms to (all) transform_te's
        # transform_te_list = []
        # for te_percentile in range(len(self.test_country)):
        #     transform = copy.deepcopy(self.transform_te)
        #     transform.add_data_transforms(means, stds, tr_percentiles, sentinel2=True)
        #     transform.setup_compose()
        #     transform_te_list.append(transform)
        # self.transform_te = transform_te_list


class BigEarthNetV2IBDataModule(BigEarthNetV2DataModule):
    def __init__(self, cfg, transform_tr, transform_te, eval_resize=None):
        super().__init__(cfg, transform_tr, transform_te, eval_resize)

    def get_dataset(self, lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split):
        return Ben19V2DatasetIB(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split,
                                noise_image_pct=self.cfg.imbalances.noise_image_pct,
                                p_base=self.cfg.imbalances.p_base, 
                                classes_for_p=self.cfg.imbalances.classes_for_p,
                                p_classwise=self.cfg.imbalances.p_classwise)


class BigEarthNetDataModule(DataModule):

    def __init__(self, cfg, transform_tr, transform_te, eval_resize=None):
        super().__init__(cfg, transform_tr, transform_te, eval_resize)
        self.cfg = cfg
        self.init_transforms()
        self.init_active_classes()

    def get_dataset(self, lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split):
        return Ben19Dataset(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split, self.active_classes)

    def init_transforms(self):
        self.train_country = os.path.basename(self.cfg.dataset.train_csv).split('_')[0].capitalize()
        self.test_country = list(map(lambda x: os.path.basename(x).split('_')[0].capitalize(), self.cfg.dataset.test_csv))
        if self.cfg.dataset.global_pctl:
            print('Global Pre-Normalization.')
            tr_percentiles = 10000
        elif self.cfg.dataset.all_percentiles and not self.cfg.dataset.global_pctl:
            print('All BEN Pre-Normalization.')
            tr_percentiles = list(BAND_99TH_PERCENTILES['All'].values())
        else:
            print('Country Pre-Normalization.')
            tr_percentiles = list(BAND_99TH_PERCENTILES[self.train_country].values())
        # te_percentiles = list(map(lambda x: list(BAND_99TH_PERCENTILES[x].values()), self.train_country))

        # currently train and test normalized by train norms (!)
        channel_global = 'Global' if self.cfg.dataset.global_pctl else 'Channel'
        if self.cfg.dataset.all_percentiles:
            print('All Percentiles', channel_global)
            means = list(BAND_NORM_STATS[channel_global]['All']['mean'].values())
            stds  = list(BAND_NORM_STATS[channel_global]['All']['std'].values())
        else:
            print('Country Percentiles', channel_global)
            means = list(BAND_NORM_STATS[channel_global][self.train_country]['mean'].values())
            stds  = list(BAND_NORM_STATS[channel_global][self.train_country]['std'].values())

        # add data transforms to transform_tr
        self.transform_tr.add_data_transforms(means, stds, tr_percentiles, sentinel2=True)
        self.transform_tr.setup_compose()

        transform_va_list = []
        for te_percentile in range(len(self.test_country)):
            transform = copy.deepcopy(self.transform_te)
            transform.add_data_transforms(means, stds, tr_percentiles, sentinel2=True)
            transform.setup_compose()
            transform_va_list.append(transform)
        self.transform_val = transform_va_list

        # add data transforms to (all) transform_te's
        transform_te_list = []
        for te_percentile in range(len(self.test_country)):
            transform = copy.deepcopy(self.transform_te)
            transform.add_data_transforms(means, stds, tr_percentiles, sentinel2=True)
            transform.setup_compose()
            transform_te_list.append(transform)
        self.transform_te = transform_te_list

    def init_active_classes(self):
        active_classes_tr = set(ACTIVE_CLASSES[self.train_country])
        active_classes_te = list(map(lambda x: set(ACTIVE_CLASSES[x]), self.test_country))
        active_classes_te = set.intersection(*active_classes_te)
        self.active_classes = list(active_classes_tr.intersection(active_classes_te))

        if self.cfg.dataset.intersection_8country:
            self.active_classes = ACTIVE_CLASSES['Intersection_8Country']
        self.num_cls = len(self.active_classes)


class BigEarthNetCloudsDataModule(DataModule):

    def __init__(self, cfg, transform_tr, transform_te, eval_resize=None):
        super().__init__(cfg, transform_tr, transform_te, eval_resize)
        self.cfg = cfg
        self.num_cls = 1
        self.init_transforms()

    def get_dataset(self, lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split):
        return Ben19CloudsDataset(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)

    def init_transforms(self):
        tr_percentiles = list(BAND_99TH_PERCENTILES['All'].values())
        te_percentiles = [list(BAND_99TH_PERCENTILES['All'].values())]
        # currently train and test normalized by train norms (!)
        means = list(CLOUD_NORM_STATS['All']['mean'].values())
        stds  = list(CLOUD_NORM_STATS['All']['std'].values())

        # add data transforms to transform_tr
        self.transform_tr.add_data_transforms(means, stds, tr_percentiles, sentinel2=False)
        self.transform_tr.setup_compose()

        # add data transforms to (all) transform_te's
        transform_te_list = []
        for te_percentile in te_percentiles:
            transform = copy.deepcopy(self.transform_te)
            transform.add_data_transforms(means, stds, te_percentile, sentinel2=False)
            transform.setup_compose()
            transform_te_list.append(transform)
        self.transform_te = transform_te_list


class SeCoDataModule(DataModule):

    def __init__(self, cfg, transform_tr, transform_te, eval_resize=None):
        super().__init__(cfg, transform_tr, transform_te, eval_resize)
        self.cfg = cfg
        self.num_cls = 0
        self.init_transforms()

    def get_dataset(self, lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split):
        return SeCoDataset(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)

    def init_transforms(self):
        percentiles = list(SECO_95TH_PERCENTILES.values())
        # currently train and test normalized by train norms (!)
        means = list(SECO_95TH_NORM_STATS['mean'].values())
        stds  = list(SECO_95TH_NORM_STATS['std'].values())

        # add data transforms to transform_tr
        self.transform_tr.add_data_transforms(means, stds, percentiles, sentinel2=False)  # hard-coded no sentinel2 pre-norm
        self.transform_tr.setup_compose()

        # no validation loader, cheap hack: copy transform_tr
        self.transform_te = copy.deepcopy(self.transform_tr)


class SeCo264x264DataModule(DataModule):

    def __init__(self, cfg, transform_tr, transform_te, eval_resize=None):
        super().__init__(cfg, transform_tr, transform_te, eval_resize)
        self.cfg = cfg
        self.num_cls = 0
        self.init_transforms()

    def get_dataset(self, lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split):
        return SeCo264x264Dataset(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)

    def init_transforms(self):
        percentiles = list(SECO_95TH_PERCENTILES.values())
        # currently train and test normalized by train norms (!)
        means = list(SECO_95TH_NORM_STATS['mean'].values())
        stds  = list(SECO_95TH_NORM_STATS['std'].values())

        # add data transforms to transform_tr
        self.transform_tr.add_data_transforms(means, stds, percentiles, sentinel2=True)  # hard-coded no sentinel2 pre-norm
        self.transform_tr.setup_compose()

        # no validation loader, cheap hack: copy transform_tr
        self.transform_te = copy.deepcopy(self.transform_tr)


class SeCoDataModuleTemporal(DataModule):

    def __init__(self, cfg, transform_tr, transform_te, eval_resize=None):
        super().__init__(cfg, transform_tr, transform_te, eval_resize)
        self.cfg = cfg
        self.num_cls = 0
        self.init_transforms()

    def get_dataset(self, lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split):
        return SeCoDatasetTemporal(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)

    def init_transforms(self):
        percentiles = list(SECO_95TH_PERCENTILES.values())
        # currently train and test normalized by train norms (!)
        means = list(SECO_95TH_NORM_STATS['mean'].values())
        stds  = list(SECO_95TH_NORM_STATS['std'].values())

        # add data transforms to transform_tr
        self.transform_tr.add_data_transforms(means, stds, percentiles, sentinel2=False)  # hard-coded no sentinel2 pre-norm
        self.transform_tr.setup_compose()

        # no validation loader, cheap hack: copy transform_tr
        self.transform_te = copy.deepcopy(self.transform_tr)


class SSL4EODataModule(DataModule):

    def __init__(self, cfg, transform_tr, transform_te, eval_resize=None):
        super().__init__(cfg, transform_tr, transform_te, eval_resize)
        self.cfg = cfg
        self.num_cls = 0
        self.init_transforms()

    def get_dataset(self, lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split):
        return SSL4EODataset(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)

    def init_transforms(self):
        percentiles = list(SSL4EO_95TH_PERCENTILES.values())
        # currently train and test normalized by train norms (!)
        means = list(SSL4EO_95TH_NORM_STATS['mean'].values())
        stds  = list(SSL4EO_95TH_NORM_STATS['std'].values())

        # add data transforms to transform_tr
        self.transform_tr.add_data_transforms(means, stds, percentiles, sentinel2=True)  # hard-coded no sentinel2 pre-norm
        self.transform_tr.setup_compose()

        # no validation loader, cheap hack: copy transform_tr
        self.transform_te = copy.deepcopy(self.transform_tr)


class DeepGlobeDataModule(DataModule):

    def __init__(self, cfg, transform_tr, transform_te, eval_resize=None):
        super().__init__(cfg, transform_tr, transform_te, eval_resize)
        self.cfg = cfg
        self.num_cls = 6
        self.init_transforms()

    def get_dataset(self, lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split):
        return DeepGlobeDataset(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split,
                                self.cfg.imbalances.use_balancing, self.cfg.imbalances.noise_image_pct,
                                self.cfg.imbalances.p_base, self.cfg.imbalances.classes_for_p,
                                self.cfg.imbalances.p_classwise)

    def init_transforms(self):
        """Currently hard-coded always only one test-dataloader."""
        means = [0.4095, 0.3808, 0.2836]
        stds  = [0.1509, 0.1187, 0.1081]

        # add data transforms to transform_tr
        self.transform_tr.add_data_transforms(means, stds)
        self.transform_tr.setup_compose()

        # add data transforms to (all) transform_te's
        self.transform_te.add_data_transforms(means, stds)
        self.transform_te.setup_compose()


class TreeSatAIDataModule(DataModule):

    def __init__(self, cfg, transform_tr, transform_te, eval_resize=None):
        super().__init__(cfg, transform_tr, transform_te, eval_resize)
        self.cfg = cfg
        self.num_cls = 15
        self.init_transforms()

    def get_dataset(self, lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split):
        return TreeSatAIDataset(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split,
                                self.cfg.imbalances.use_balancing, self.cfg.imbalances.noise_image_pct,
                                self.cfg.imbalances.p_base, self.cfg.imbalances.classes_for_p,
                                self.cfg.imbalances.p_classwise)

    def init_transforms(self):
        """Currently hard-coded always only one test-dataloader."""
        means = [0.5929, 0.3647, 0.3333, 0.3172]
        stds  = [0.1441, 0.1085, 0.0882, 0.1049]
        
        # add data transforms to transform_tr
        self.transform_tr.add_data_transforms(means, stds)
        self.transform_tr.setup_compose()

        # add data transforms to (all) transform_te's
        self.transform_te.add_data_transforms(means, stds)
        self.transform_te.setup_compose()


class EuroSATDataModule(DataModule):

    def __init__(self, cfg, transform_tr, transform_te, eval_resize=None):
        super().__init__(cfg, transform_tr, transform_te, eval_resize)
        self.cfg = cfg
        self.num_cls = 10
        self.init_transforms()

    def get_dataset(self, lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split):
        return EuroSATDataset(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)

    def init_transforms(self):
        keys = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']

        if self.cfg.dataset.pretrain_norm:
            all_percentiles = BAND_99TH_PERCENTILES['All']
            percentiles = [all_percentiles[k] for k in keys]

            all_means = BAND_NORM_STATS['Channel']['All']['mean']
            all_stds = BAND_NORM_STATS['Channel']['All']['std']
            means = [all_means[k] for k in keys]
            stds  = [all_stds[k] for k in keys]

            # percentiles = list(BAND_99TH_PERCENTILES['All'].values())
            # # currently train and test normalized by train norms (!)
            # means = list(BAND_NORM_STATS['Channel']['All']['mean'].values())
            # stds  = list(BAND_NORM_STATS['Channel']['All']['std'].values())
        else:
            percentiles = list(EUROSAT_99TH_PERCENTILES.values())
            # currently train and test normalized by train norms (!)
            means = list(EUROSAT_NORM_STATS['mean'].values())
            stds  = list(EUROSAT_NORM_STATS['std'].values())

        # add data transforms to transform_tr
        self.transform_tr.add_data_transforms(means, stds, percentiles, sentinel2=True)
        self.transform_tr.setup_compose()

        # add data transforms to (all) transform_te's
        self.transform_te.add_data_transforms(means, stds, percentiles, sentinel2=True)
        self.transform_te.setup_compose()

        self.transform_val = self.transform_te


class EuroSATV2DataModule(DataModule):

    def __init__(self, cfg, transform_tr, transform_te, eval_resize=None):
        super().__init__(cfg, transform_tr, transform_te, eval_resize)
        self.cfg = cfg
        self.num_cls = 10
        self.init_transforms()

    def get_dataset(self, lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split):
        return EuroSATV2Dataset(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)

    def init_transforms(self):
        keys = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']

        all_percentiles = BAND_99TH_PERCENTILES['All']
        percentiles = [all_percentiles[k] for k in keys]

        all_means = BAND_NORM_STATS['Channel']['All']['mean']
        all_stds = BAND_NORM_STATS['Channel']['All']['std']
        means = [all_means[k] for k in keys]
        stds  = [all_stds[k] for k in keys]

        # percentiles = list(BAND_99TH_PERCENTILES['All'].values())
        # # currently train and test normalized by train norms (!)
        # means = list(BAND_NORM_STATS['Channel']['All']['mean'].values())
        # stds  = list(BAND_NORM_STATS['Channel']['All']['std'].values())

        # if self.cfg.dataset.pretrain_norm:
        #     percentiles = list(BAND_99TH_PERCENTILES['All'].values())
        #     # currently train and test normalized by train norms (!)
        #     means = list(BAND_NORM_STATS['Channel']['All']['mean'].values())
        #     stds  = list(BAND_NORM_STATS['Channel']['All']['std'].values())
        # else:
        #     percentiles = list(EUROSATV2_99TH_PERCENTILES.values())
        #     # currently train and test normalized by train norms (!)
        #     means = list(EUROSATV2_NORM_STATS['mean'].values())
        #     stds  = list(EUROSATV2_NORM_STATS['std'].values())

        # add data transforms to transform_tr
        self.transform_tr.add_data_transforms(means, stds, percentiles, sentinel2=True)
        self.transform_tr.setup_compose()

        # add data transforms to (all) transform_te's
        self.transform_te.add_data_transforms(means, stds, percentiles, sentinel2=True)
        self.transform_te.setup_compose()

        self.transform_val = self.transform_te


class So2SatDataModule(DataModule):

    def __init__(self, cfg, transform_tr, transform_te, eval_resize=None):
        super().__init__(cfg, transform_tr, transform_te, eval_resize)
        self.cfg = cfg
        self.num_cls = 17
        self.init_transforms()

    def get_dataset(self, lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split):
        return So2SatDataset(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)

    def init_transforms(self):
        if self.cfg.dataset.pretrain_norm:
            percentiles = list(BAND_99TH_PERCENTILES['All'].values())
            # currently train and test normalized by train norms (!)
            means = list(BAND_NORM_STATS['Channel']['All']['mean'].values())
            stds  = list(BAND_NORM_STATS['Channel']['All']['std'].values())
        else:
            percentiles = list(SO2SAT_99TH_PERCENTILES.values())
            # currently train and test normalized by train norms (!)
            means = list(SO2SAT_NORM_STATS['mean'].values())
            stds  = list(SO2SAT_NORM_STATS['std'].values())

        # add data transforms to transform_tr
        self.transform_tr.add_data_transforms(means, stds, percentiles, sentinel2=True)
        self.transform_tr.setup_compose()

        # add data transforms to (all) transform_te's
        self.transform_te.add_data_transforms(means, stds, percentiles, sentinel2=True)
        self.transform_te.setup_compose()

        self.transform_val = self.transform_te


class FMoWDataModule(DataModule):

    def __init__(self, cfg, transform_tr, transform_te, eval_resize=None):
        super().__init__(cfg, transform_tr, transform_te, eval_resize)
        self.cfg = cfg
        self.num_cls = 62
        self.init_transforms()

    def get_dataset(self, lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split):
        return FMoWDataset(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)

    def init_transforms(self):
        percentiles = list(FMOW_99TH_PERCENTILES.values())
        # currently train and test normalized by train norms (!)
        means = list(FMOW_NORM_STATS['mean'].values())
        stds  = list(FMOW_NORM_STATS['std'].values())

        # add data transforms to transform_tr
        self.transform_tr.add_data_transforms(
            means, stds, percentiles, sentinel2=True, global_pctl=self.cfg.dataset.global_pctl
        )
        self.transform_tr.setup_compose()

        # add data transforms to (all) transform_te's
        self.transform_te.add_data_transforms(
            means, stds, percentiles, sentinel2=True, global_pctl=self.cfg.dataset.global_pctl
        )
        self.transform_te.setup_compose()
        self.transform_te = [self.transform_te]


class S4ADataModule(DataModule):

    def __init__(self, cfg, transform_tr, transform_te, eval_resize=None):
        super().__init__(cfg, transform_tr, transform_te, eval_resize)
        self.cfg = cfg
        self.num_cls = 9
        self.init_transforms()

    def get_dataset(self, lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split):
        # add here parameter for high-level or low-level classes
        return S4ADataset(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)

    def init_transforms(self):
        if self.cfg.dataset.pretrain_norm:
            percentiles = list(BAND_99TH_PERCENTILES['All'].values())
            # currently train and test normalized by train norms (!)
            means = list(BAND_NORM_STATS['Channel']['All']['mean'].values())
            stds  = list(BAND_NORM_STATS['Channel']['All']['std'].values())
        else:
            percentiles = list(S4A_99TH_PERCENTILES.values())
            # currently train and test normalized by train norms (!)
            means = list(S4A_NORM_STATS['mean'].values())
            stds  = list(S4A_NORM_STATS['std'].values())

        # add data transforms to transform_tr
        self.transform_tr.add_data_transforms(means, stds, percentiles, sentinel2=True)
        self.transform_tr.setup_compose()

        # add data transforms to (all) transform_te's
        self.transform_te.add_data_transforms(means, stds, percentiles, sentinel2=True)
        self.transform_te.setup_compose()

        self.transform_val = self.transform_te


class S4ASegmentationDataModule(DataModule):

    def __init__(self, cfg, transform_tr, transform_te, eval_resize=None):
        super().__init__(cfg, transform_tr, transform_te, eval_resize)
        self.cfg = cfg
        self.num_cls = 10
        self.init_transforms()

    def get_dataset(self, lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split):
        # add here parameter for high-level or low-level classes
        return S4ASegmentationDataset(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)

    def init_transforms(self):
        if self.cfg.dataset.pretrain_norm:
            percentiles = list(BAND_99TH_PERCENTILES['All'].values())
            # currently train and test normalized by train norms (!)
            means = list(BAND_NORM_STATS['Channel']['All']['mean'].values())
            stds  = list(BAND_NORM_STATS['Channel']['All']['std'].values())
        else:
            percentiles = list(S4A_99TH_PERCENTILES.values())
            # currently train and test normalized by train norms (!)
            means = list(S4A_NORM_STATS['mean'].values())
            stds  = list(S4A_NORM_STATS['std'].values())

        # add data transforms to transform_tr
        self.transform_tr.add_data_transforms(means, stds, percentiles, sentinel2=True)
        self.transform_tr.setup_compose()

        # add data transforms to (all) transform_te's
        self.transform_te.add_data_transforms(means, stds, percentiles, sentinel2=True)
        self.transform_te.setup_compose()

        self.transform_val = self.transform_te


class SeasonNetDataModule(DataModule):

    def __init__(self, cfg, transform_tr, transform_te, eval_resize=None):
        super().__init__(cfg, transform_tr, transform_te, eval_resize)
        self.cfg = cfg
        self.num_cls = 33
        self.init_transforms()

    def get_dataset(self, lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split):
        # add here parameter for high-level or low-level classes
        return SeasonNetDataset(lmdb_path, csv_path, labels_path, eval_resize, temporal_views_path, transform, split)

    def init_transforms(self):
        if self.cfg.dataset.pretrain_norm:
            percentiles = list(BAND_99TH_PERCENTILES['All'].values())
            # currently train and test normalized by train norms (!)
            means = list(BAND_NORM_STATS['Channel']['All']['mean'].values())
            stds  = list(BAND_NORM_STATS['Channel']['All']['std'].values())
        else:
            percentiles = list(S4A_99TH_PERCENTILES.values())
            # currently train and test normalized by train norms (!)
            means = list(S4A_NORM_STATS['mean'].values())
            stds  = list(S4A_NORM_STATS['std'].values())

        # add data transforms to transform_tr
        self.transform_tr.add_data_transforms(means, stds, percentiles, sentinel2=True)
        self.transform_tr.setup_compose()

        # add data transforms to (all) transform_te's
        self.transform_te.add_data_transforms(means, stds, percentiles, sentinel2=True)
        self.transform_te.setup_compose()

        self.transform_val = self.transform_te


def get_datamodule(dataset):
    if dataset == 'BigEarthNet':
        return BigEarthNetDataModule
    if dataset == 'BigEarthNetV2':
        return BigEarthNetV2DataModule
    if dataset == 'BigEarthNetV2IB':
        return BigEarthNetV2IBDataModule
    if dataset == 'BigEarthNetClouds':
        return BigEarthNetCloudsDataModule
    if dataset == 'DeepGlobe':
        return DeepGlobeDataModule
    if dataset == 'TreeSatAI':
        return TreeSatAIDataModule
    if dataset == 'EuroSAT':
        return EuroSATDataModule
    if dataset == 'EuroSATV2':
        return EuroSATV2DataModule
    if dataset == 'So2Sat_Random' or dataset == 'So2Sat_Block' or dataset == 'So2Sat_Culture10':
        return So2SatDataModule
    if dataset == 'fMoW':
        return FMoWDataModule
    if dataset == 'Sen4AgriNet_Corr' or dataset == 'Sen4AgriNet_Uncorr':
        return S4ADataModule
    if dataset == 'SeasonNet':
        return SeasonNetDataModule
    if dataset == 'Sen4AgriNet_Segmentation_Corr' or dataset == 'Sen4AgriNet_Segmentation_Uncorr':
        return S4ASegmentationDataModule
    if dataset == 'SeCo' or dataset == 'SeCoFull':
        return SeCoDataModule
    if dataset == 'SeCoTemporal':
        return SeCoDataModuleTemporal
    if dataset == 'SeCo264x264':
        return SeCo264x264DataModule
    if dataset == 'SSL4EO':
        return SSL4EODataModule
