import os
import numpy as np
import pandas as pd

import torch
import torch.optim as optim

from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AveragePrecision
from torchmetrics.segmentation import DiceScore, GeneralizedDiceScore, HausdorffDistance, MeanIoU

import pytorch_lightning as pl

from utils import LinearWarmupCosineAnnealingLR


def handle_determinism_for_metric(func):
    """
    Decorator to handle deterministic algorithm settings for segmentation tasks.
    If the task is segmentation, it temporarily deactivates deterministic algorithms.
    """
    def wrapper(self, batch, *args, **kwargs):
        if torch.are_deterministic_algorithms_enabled():
            torch.use_deterministic_algorithms(False)  # Deactivate deterministic algorithms for segmentation
            result = func(self, batch, *args, **kwargs)
            torch.use_deterministic_algorithms(True)   # Reactivate deterministic algorithms after the operation
        else:
            result = func(self, batch, *args, **kwargs)
        return result

    return wrapper


def handle_determinism_for_segmentation(func):
    """
    Decorator to handle deterministic algorithm settings for segmentation tasks.
    If the task is segmentation, it temporarily deactivates deterministic algorithms.
    """
    def wrapper(self, batch, *args, **kwargs):
        if self.cfg.dataset.task == 'segmentation' and torch.are_deterministic_algorithms_enabled():
            torch.use_deterministic_algorithms(False)  # Deactivate deterministic algorithms for segmentation
            result = func(self, batch, *args, **kwargs)
            torch.use_deterministic_algorithms(True)   # Reactivate deterministic algorithms after the operation
        else:
            result = func(self, batch, *args, **kwargs)
        return result

    return wrapper


class BaseModel(pl.LightningModule):
    def __init__(self, cfg, datamodule, network):
        super().__init__()
        self.cfg = cfg

        self.model = network
        self.save_hyperparameters('cfg')
        
        self.datamodule = datamodule
        self.criterion = self.init_criterion()

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        metrics = self.init_metrics(self.datamodule.num_cls)
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        self.should_extract_features = False
        self.inference = False

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        # self.log_noisy_labels()
        if self.cfg.tracking.apply_feature_extraction:
            self.make_features_save_dirs()

    @handle_determinism_for_segmentation
    def apply_criterion(self, logits, y):
        return self.criterion(logits, y)

    def put_on_cpu_if_segmentation(self, output):
        if self.cfg.dataset.task == 'segmentation':
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    if k == 'loss':
                        output[k] = v
                    else:
                        output[k] = v.detach().cpu()
        return output

    def training_step(self, batch, batch_idx):
        x, y, idx = batch
        logits    = self.forward(x)
        loss      = self.apply_criterion(logits, y)
        probs     = torch.sigmoid(logits)
        preds     = torch.argmax(logits, dim=1)
        output    = dict(y=y, idx=idx, loss=loss, probs=probs, preds=preds)
        output    = self.put_on_cpu_if_segmentation(output)
        self.training_step_outputs.append(output)
        return output

    def validation_step(self, batch, batch_idx):
        x, y, idx = batch
        logits    = self.forward(x)
        loss      = self.apply_criterion(logits, y)
        probs     = torch.sigmoid(logits)
        preds     = torch.argmax(logits, dim=1)
        output    = dict(y=y, idx=idx, loss=loss, probs=probs, preds=preds)
        output    = self.put_on_cpu_if_segmentation(output)
        self.validation_step_outputs.append(output)
        return output

    def test_step(self, batch, batch_idx):
        x, y, idx = batch
        logits    = self.forward(x)
        loss      = self.apply_criterion(logits, y)
        probs     = torch.sigmoid(logits)
        preds     = torch.argmax(logits, dim=1)
        output    = dict(y=y, idx=idx, loss=loss, probs=probs, preds=preds)
        output    = self.put_on_cpu_if_segmentation(output)
        self.test_step_outputs.append(output)
        return output

    def unpack_step_outputs(self, step_outs):
        y     = torch.cat(list(map(lambda x: x['y'], step_outs)), dim=0)
        idx   = torch.cat(list(map(lambda x: x['idx'], step_outs)), dim=0)
        probs = torch.cat(list(map(lambda x: x['probs'], step_outs)), dim=0)
        preds = torch.cat(list(map(lambda x: x['preds'], step_outs)), dim=0)
        loss  = torch.stack(list(map(lambda x: x['loss'], step_outs)))
        return y, idx, probs, preds, loss

    def on_train_epoch_start(self):
        if self.cfg.tracking.apply_feature_extraction and self.current_epoch in self.cfg.tracking.feature_extraction_epochs:
            self.should_extract_features = True
            if self.cfg.tracking.feature_extraction_with_transformation:
                self.enable_feature_extraction()       

    def on_train_epoch_end(self):
        y, idx, probs, preds, loss = self.unpack_step_outputs(self.training_step_outputs)
        self.training_step_outputs.clear()

        output = self.apply_metric(self.train_metrics, probs, preds, y.long())
        self.log_metrics(loss, output, 'train')

        if self.cfg.tracking.should_track_train_probs:
            self.track_probs(idx, probs, y.long(), 'train')

        if self.should_extract_features:

            if not self.cfg.tracking.feature_extraction_with_transformation:
                self.enable_feature_extraction()
                dataloader = self.train_te_dataloader()
                _, _, _ = self.infer(dataloader)

            self.disable_feature_extraction()
            self.should_extract_features = False
            idx = idx.cpu().numpy()
            self.save_features(self.feature_matrix[idx], train=True)

    def on_validation_epoch_end(self):
        y, idx, probs, preds, loss = self.unpack_step_outputs(self.validation_step_outputs)
        self.validation_step_outputs.clear()

        output = self.apply_metric(self.val_metrics, probs, preds, y.long())
        self.log_metrics(loss, output, 'val')

        if self.cfg.tracking.should_track_val_probs and not self.trainer.sanity_checking:
            self.track_probs(idx, probs, y.long(), 'val')

    def on_test_epoch_end(self):
        y, idx, probs, preds, loss = self.unpack_step_outputs(self.test_step_outputs)
        self.test_step_outputs.clear()

        output = self.apply_metric(self.test_metrics, probs, preds, y.long())
        self.log_metrics(loss, output, 'test')

        self.track_probs(idx, probs, y.long(), 'test')

    def infer(self, dataloader):
        """Used to infer train probabilities or embeddings without augmentation."""
        self.on_validation_model_eval()  # calls `model.eval()`
        torch.set_grad_enabled(False)
        self.inference = True

        idx, probs, targets = [], [], []
        for batch_idx, batch in enumerate(dataloader):
            batch = self.on_before_batch_transfer(batch, 0)
            batch = self.transfer_batch_to_device(batch, self.device, 0)
            out = self.training_step(batch, batch_idx)
            idx.append(out['idx'])
            probs.append(out['probs'])
            targets.append(out['y'])

        self.on_validation_model_train()  # calls `model.train()`
        torch.set_grad_enabled(True)
        self.inference = False

        idx = torch.cat(idx, dim=0)
        probs = torch.cat(probs, dim=0)
        targets = torch.cat(targets, dim=0)
        return idx, probs, targets

    def configure_optimizers(self):
        # optimizer = optim.SGD(
        #     self.model.parameters(),
        #     lr=0.014,
        #     momentum=0.9,
        #     weight_decay=0.00005,
        #     nesterov=True
        # )
        # lr_scheduler = {'scheduler': optim.lr_scheduler.MultiStepLR(
        #     optimizer,
        #     milestones=self.cfg.optim.milestones,
        #     gamma=self.cfg.optim.gamma,        
        # ), 'interval': 'epoch'
        # }
        # return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.optim.min_lr,
            weight_decay=self.cfg.optim.weight_decay
        )

        max_intervals = int(self.trainer.max_epochs * len(self.datamodule.trainset_tr) / self.cfg.dataset.batch_size)
        warmup = 10000 if max_intervals > 10000 else 100 if max_intervals > 100 else 0

        lr_scheduler = {'scheduler': LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup,
            max_epochs=max_intervals,
            warmup_start_lr=self.cfg.optim.min_lr / 10,
            eta_min=self.cfg.optim.min_lr / 10
        ), 'name': 'learning_rate', 'interval': "step", 'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def init_criterion(self):
        if self.cfg.dataset.task == 'single_label' or self.cfg.dataset.task == 'segmentation':
            return torch.nn.CrossEntropyLoss()
        elif self.cfg.dataset.task == 'multi_label':
            return torch.nn.BCEWithLogitsLoss(pos_weight=None)
        elif self.cfg.dataset.task == 'binary':
            y = self.datamodule.trainset_tr.labels
            pos_weight = torch.from_numpy(np.sum(y == 0, axis=0) / np.sum(y == 1, axis=0)) if self.cfg.params.pos_weight else None
            return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def init_metrics(self, num_cls):
        if self.cfg.dataset.task == 'binary':
            metrics = MetricCollection({
                'AP'         : AveragePrecision(task='binary'),
                'acc'        : Accuracy(task='binary'),
                'f1'         : F1Score(task='binary'),
                'prec'       : Precision(task='binary'),
                'rec'        : Recall(task='binary'),
            })
        if self.cfg.dataset.task == 'single_label':
            metrics = MetricCollection({
                'accmac'        : Accuracy(num_classes=num_cls, task='multiclass', average='macro'),
                'accmic'        : Accuracy(num_classes=num_cls, task='multiclass', average='micro'),
                'f1mac'         : F1Score(num_classes=num_cls, task='multiclass', average='macro'),
                'precmac'       : Precision(num_classes=num_cls, task='multiclass', average='macro'),
                'recmic'        : Recall(num_classes=num_cls, task='multiclass', average='macro'),
                'accclasses'    : Accuracy(num_classes=num_cls, task='multiclass', average=None),
                'f1classes'     : F1Score(num_classes=num_cls, task='multiclass', average=None),
                'precclasses'   : Precision(num_classes=num_cls, task='multiclass', average=None),
                'recclasses'    : Recall(num_classes=num_cls, task='multiclass', average=None),
            })
        elif self.cfg.dataset.task == 'multi_label':
            metrics = MetricCollection({
                'APmic'     : AveragePrecision(num_labels=num_cls, task='multilabel', average='micro'),
                'APmac'     : AveragePrecision(num_labels=num_cls, task='multilabel', average='macro'),
                'APclasses' : AveragePrecision(num_labels=num_cls, task='multilabel', average=None),
                'f1mic'     : F1Score(num_labels=num_cls, task='multilabel', average='micro', threshold=0.5),
                'f1mac'     : F1Score(num_labels=num_cls, task='multilabel', average='macro', threshold=0.5),
                'f1classes' : F1Score(num_labels=num_cls, task='multilabel', average=None, threshold=0.5)
            })
        elif self.cfg.dataset.task == 'segmentation':
            metrics = MetricCollection({
                'DICEmic'   : DiceScore(num_classes=num_cls, average='micro', include_background=False),
                'DICEmac'   : DiceScore(num_classes=num_cls, average='macro', include_background=False),
                'GenDICE'   : GeneralizedDiceScore(num_classes=num_cls, include_background=False),
                # 'HausDist' : HausdorffDistance(num_classes=num_cls, include_background=False, input_format='index'),
                'IoU'       : MeanIoU(num_classes=num_cls, include_background=False),
                'IoU_wB'    : MeanIoU(num_classes=num_cls, include_background=True),
                'PAmic'     : Accuracy(num_classes=num_cls, task='multiclass', average='micro', ignore_index=0),
                'PAmac'     : Accuracy(num_classes=num_cls, task='multiclass', average='macro', ignore_index=0),
                'PAmic_wB'  : Accuracy(num_classes=num_cls, task='multiclass', average='micro'),  # , ignore_index=0),
                'PAmac_wB'  : Accuracy(num_classes=num_cls, task='multiclass', average='macro')  # , ignore_index=0),
            })
        return metrics

    @handle_determinism_for_metric
    def apply_metric(self, metrics, probs, preds, y):
        if self.cfg.dataset.task == 'single_label':
            out = metrics(preds, y)
        elif self.cfg.dataset.task == 'multi_label':
            out = metrics(probs, y)
        elif self.cfg.dataset.task == 'segmentation':
            out = self.chunk_wise_metric_calculation(preds, y, metrics)
        return out

    def chunk_wise_metric_calculation(self, preds, y, metrics, chunk_size=1000):
        # Reset the metrics before accumulating new results
        if hasattr(metrics, 'reset'):
            metrics.reset()

        # Process data in chunks
        for i in range(0, len(preds), chunk_size):
            probs_chunk = preds[i:i + chunk_size].to(self.device) 
            y_chunk = y[i:i + chunk_size].to(self.device) .long()
            metrics.update(probs_chunk, y_chunk)

        # Compute the final metric result after all updates
        return metrics.compute()

    #################
    # LOGGING MODULE
    #################

    def track_probs(self, idx, probs, targets, log_str='train'):
        if log_str == 'train' and not self.cfg.tracking.train_track_with_transformation:
            idx, probs, _ = self.infer(self.train_te_dataloader())
        self.log_inferences(idx, probs, log_str)

    def log_inferences(self, idx, probs, split):
        idx = list(idx.cpu().numpy())
        probs = list(map(list, probs.detach().cpu().numpy().astype(float)))
        df = pd.DataFrame(index=idx, data={str(self.current_epoch): probs})
        path = os.path.join(self.logger.log_dir, 'tracking_{}.parquet'.format(split))
        if os.path.exists(path):
            df_old = pd.read_parquet(path)
            df = pd.concat([df_old, df], axis=1)
        df.to_parquet(path)

    def log_list(self, prefix, metric_list):
        for cl_idx, cl_value in zip(np.arange(self.datamodule.num_cls), metric_list):
            self.log('{}{}'.format(prefix, cl_idx), cl_value, on_epoch=True, on_step=False)

    def log_parquet(self, index, data, name):
        df = pd.DataFrame(index=index, data=data)
        if df.columns.dtype == int:
            df.columns = df.columns.astype(str) 
        path = os.path.join(self.logger.log_dir, name)
        df.to_parquet(path)

    def log_metrics(self, loss, output, log_str='train'):
        if self.cfg.dataset.task == 'binary' or self.cfg.dataset.task == 'segmentation':
            self.log('{}_loss'.format(log_str), loss.mean(), prog_bar=True)
            self.log_dict(output)

        elif self.cfg.dataset.task == 'single_label':
            acc_classes = output.pop('{}_accclasses'.format(log_str))
            f1_classes = output.pop('{}_f1classes'.format(log_str))
            rec_classes = output.pop('{}_recclasses'.format(log_str))
            prec_classes = output.pop('{}_precclasses'.format(log_str))

            self.log('{}_loss'.format(log_str), loss.mean(), prog_bar=True)
            self.log_dict(output)
            self.log_list('{}_acc_cl'.format(log_str), acc_classes)
            self.log_list('{}_f1_cl'.format(log_str), f1_classes)
            self.log_list('{}_rec_cl'.format(log_str), rec_classes)
            self.log_list('{}_prec_cl'.format(log_str), prec_classes)

        elif self.cfg.dataset.task == 'multi_label':
            ap_classes = output.pop('{}_APclasses'.format(log_str))
            f1_classes = output.pop('{}_f1classes'.format(log_str))

            self.log('{}_loss'.format(log_str), loss.mean(), prog_bar=True)
            self.log_dict(output)
            self.log_list('{}_AP_cl'.format(log_str), ap_classes)
            self.log_list('{}_f1_cl'.format(log_str), f1_classes)

    def log_noisy_labels(self):
        noisy_label = self.datamodule.trainset_tr.labels
        patch_ids = self.datamodule.trainset_tr.patch_names
        idx = np.arange(len(noisy_label))
        data = {'patch_id': patch_ids, 'labels': list(map(list, noisy_label))}
        self.log_parquet(idx, data, 'noisy_labels.parquet')

    ####################
    # Feature Extraction
    ####################

    def init_feature_matrix(self):
        self.p = 0
        self.feature_matrix = torch.zeros(len(self.datamodule.trainset_tr), self.model.fc_size)

    def get_activation(self, name):
        """To-Do: Sort activations if shuffle is true for feature extraction data loader."""
        def hook(model, input, output):
            # only save features on training set
            if self.training or self.inference:
                batch_size = output.shape[0]
                features = torch.reshape(output.detach().cpu(), tuple(output.shape[0:2]))
                self.feature_matrix[self.p:self.p + batch_size, :] = features
                self.p += batch_size
        return hook

    def register_feature_extraction_hook(self):
        self.handle = self.model.register_hooks(self.get_activation)

    def remove_feature_extract_hook(self):
        self.handle.remove()

    def enable_feature_extraction(self):
        self.init_feature_matrix()
        self.register_feature_extraction_hook()

    def disable_feature_extraction(self):
        self.remove_feature_extract_hook()

    def make_features_save_dirs(self):
        features_save_dir = os.path.join(self.logger.log_dir, 'features')
        if not os.path.exists(features_save_dir):
            os.mkdir(features_save_dir)

        for epoch in self.cfg.tracking.feature_extraction_epochs:
            epoch_save_dir = os.path.join(features_save_dir, 'epoch_{}'.format(epoch))
            if not os.path.exists(epoch_save_dir):
                os.mkdir(epoch_save_dir)

    def save_features(self, X, train):
        log_str = 'train' if train else 'val'
        fname = 'features/epoch_{}/{}_features.npy'.format(self.current_epoch, log_str)
        fpath = os.path.join(self.logger.log_dir, fname)
        np.save(fpath, X)

    ####################
    # DATA RELATED HOOKS
    ####################

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def train_te_dataloader(self):
        return self.datamodule.train_te_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()
