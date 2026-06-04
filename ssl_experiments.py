import sys
import os
import copy
import yaml

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from config import SelfSupervisedCConfig


print(hydra)

cs = ConfigStore.instance()
cs.store(name="base_config", node=SelfSupervisedCConfig)


def update_dataset_parameter(cfg, dataset, loc='single_loc', dataset_size='100', image_size='normal_size', v='v1'):

    with open('conf/datasets.yaml', "r") as f:
        yaml_file = yaml.safe_load(f)

    cfg.params.dataset       = dataset
    cfg.dataset.lmdb_path    = yaml_file[dataset]['lmdb_path'][image_size]
    cfg.dataset.labels_path  = yaml_file[dataset]['labels_path']
    cfg.dataset.train_csv    = yaml_file[dataset]['train_csv'][loc][dataset_size][v]
    cfg.dataset.task         = yaml_file[dataset]['task']
    cfg.dataset.val_csv      = yaml_file[dataset]['val_csv']
    cfg.dataset.test_csv     = yaml_file[dataset]['test_csv']
    cfg.dataset.num_classes  = yaml_file[dataset]['num_classes']
    cfg.dataset.num_channels = yaml_file[dataset]['num_channels']

    return cfg


@hydra.main(version_base=None, config_path="conf", config_name="ssl_config")
def main(cfg):

    if cfg.params.slurm_bypass:
        print('Bypassing slurm, using GPU: {}'.format(cfg.params.cuda_no))
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.params.cuda_no
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    print(OmegaConf.to_yaml(cfg))

    import torch

    if 'gaussianblur' in cfg.dataaug.augmentations:
        print('Activate multiprocessing for GaussianBlur on GPU.')
        try:
            torch.multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass

    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    from lightly.utils.dist import print_rank_zero

    from models.utils import get_network
    from ssl_base import get_ssl_model
    from eval_base import KNNClassifier, LinearClassifier, FinetuneLinearClassifier, UPerSegmentationModel, KNN_Segmentation

    sys.path.append('data')
    from data.datamodule import get_datamodule  # noqa: E402
    from data.transform import SingleTransform, get_self_transforms

    seed_everything(cfg.params.seed, workers=True)

    test_dataaug_cfg = copy.deepcopy(cfg.dataaug)
    test_dataaug_cfg.update({'augmentations': 'notpretrained', 'p_list': [0]})
    transforms_tr = get_self_transforms(cfg.ssl, cfg.dataaug)
    transforms_te = SingleTransform(test_dataaug_cfg)

    # be careful: hardcoded for BENV2 different pre-training size!!
    cfg = update_dataset_parameter(
        cfg=cfg,
        dataset=cfg.params.dataset,
        loc=cfg.dataset.pretrain_loc,
        dataset_size=cfg.dataset.pretrain_dataset_size,
        image_size=cfg.dataset.pretrain_image_size,
        v=cfg.dataset.pretrain_split_version
    )

    DataModule = get_datamodule(cfg.params.dataset)
    dm_train = DataModule(cfg, transforms_tr, transforms_te)
    dm_train.setup()
    print('DataModule initialized.')

    network = get_network(**cfg.model, num_cls=dm_train.num_cls)
    model = get_ssl_model(cfg, dm_train, network)
    print('Model initialized.')

    if cfg.params.max_epochs <= 0:
        print_rank_zero("Epochs <= 0, skipping pretraining.")
        if cfg.logging.ckpt_path is not None:
            model.load_state_dict(torch.load(cfg.logging.ckpt_path)["state_dict"], strict=False)
    else:
        if cfg.logging.ckpt_path is not None:
            model = type(model).load_from_checkpoint(cfg.logging.ckpt_path, cfg=cfg, datamodule=dm_train, network=network)
            # model.load_state_dict(torch.load(cfg.logging.ckpt_path)["state_dict"], strict=False)

        if cfg.logging.disable_early_stopping:
            callbacks = []
        else:
            callbacks = [EarlyStopping(monitor="train_loss", patience=10, check_finite=True)]
        if cfg.logging.save_checkpoint:
            if cfg.logging.save_every_epoch:
                callbacks += [ModelCheckpoint(
                    monitor="train_loss",
                    filename='model-epoch{epoch}',
                    save_top_k=-1,
                    every_n_epochs=1
                )]
            else:
                callbacks += [ModelCheckpoint(monitor="train_loss", filename='best_model')]
                # callbacks += [ModelCheckpoint(monitor=None, every_n_epochs=10, filename='model-{epoch}', save_top_k=-1)]
            # callbacks += [ModelCheckpoint(monitor="first_epoch", mode='max')]  # hard-coded

        print('All callbacks calibrated.')
        trainer = Trainer(
            accelerator='gpu',
            callbacks=callbacks,
            devices=[0],
            enable_checkpointing=cfg.logging.save_checkpoint,
            max_epochs=cfg.params.max_epochs,
            logger=CSVLogger(save_dir=cfg.logging.exp_dir, name='pre-training'),
            num_sanity_val_steps=0,
            deterministic=True
        )
        pre_training_version = os.path.basename(os.path.normpath(trainer._loggers[0].log_dir))
        print('Waiting for model training to start....')
        trainer.fit(model)

    for eval_dataset in cfg.ssl.eval_datasets:

        cfg = update_dataset_parameter(
            cfg=cfg,
            dataset=eval_dataset,
            loc='single_loc',
            dataset_size='100',
            image_size='normal_size',
            v='v1'
        )

        if cfg.ssl.algorithm in ['SatMAE', 'ScaleMAE', 'CrossScaleMAE', 'GeoCLIP', 'CROMA']:
            test_dataaug_cfg = copy.deepcopy(cfg.dataaug)
            test_dataaug_cfg.update({'augmentations': 'resize', 'p_list': [1.0]})
            train_dataaug_cfg = copy.deepcopy(cfg.dataaug)
            train_dataaug_cfg.update({'augmentations': 'resize_flip', 'p_list': [1.0, 0.8]})
            if cfg.ssl.test_flip:
                train_dataaug_cfg.update({'augmentations': 'resize_flip', 'p_list': [1.0, 0.8]})
            else:
                train_dataaug_cfg.update({'augmentations': 'resize', 'p_list': [1.0]})
        else:
            test_dataaug_cfg = copy.deepcopy(cfg.dataaug)
            test_dataaug_cfg.update({'augmentations': 'notpretrained', 'p_list': [0]})
            train_dataaug_cfg = copy.deepcopy(cfg.dataaug)
            # train_dataaug_cfg.update({'augmentations': 'notpretrained', 'p_list': [0]})
            if cfg.ssl.test_flip:
                train_dataaug_cfg.update({'augmentations': 'flip', 'p_list': [0.8]})
            else:
                train_dataaug_cfg.update({'augmentations': 'none', 'p_list': [0.0]})

        transforms_tr = SingleTransform(train_dataaug_cfg)
        transforms_te = SingleTransform(test_dataaug_cfg)

        DataModuleEval = get_datamodule(eval_dataset)
        dm_eval = DataModuleEval(cfg, transforms_tr, transforms_te, cfg.ssl.eval_resize)
        dm_eval.setup()

        model = get_ssl_model(cfg, dm_eval, network)
        if cfg.logging.ckpt_path is None:
            model_path = "pre-training/{}/checkpoints/best_model.ckpt".format(pre_training_version)
            cfg.logging.ckpt_path = os.path.join(cfg.logging.exp_dir, model_path)
        model.load_state_dict(torch.load(cfg.logging.ckpt_path)["state_dict"], strict=False)

        if cfg.ssl.algorithm in ['SatMAE', 'ScaleMAE', 'CrossScaleMAE']:
            model_backbone = model.network
            model_feature_dim = model.network.cls_token.shape[-1]
        elif cfg.ssl.algorithm == 'GeoCLIP':
            model_backbone = model.network
            model_feature_dim = model.network.image_encoder.CLIP.vision_model.config.hidden_size
        if cfg.ssl.algorithm in ['CROMA']:
            model_backbone = model.network
            model_feature_dim = None
        else:
            model_backbone = model.backbone
            model_feature_dim = model.feature_dim

        if cfg.ssl.skip_knn_eval:
            print_rank_zero("Skipping KNN eval.")
        else:
            # KNNModel = get_knn_classifier(cfg.dataset.task)

            if cfg.dataset.task == 'multi_label' or cfg.dataset.task == 'single_label':

                classifier = KNNClassifier(
                    cfg=cfg,
                    datamodule=dm_eval,
                    network=model_backbone,
                    num_classes=dm_eval.num_cls,
                    knn_k=cfg.ssl.knn_k,
                    knn_t=cfg.ssl.knn_t,
                    feature_dtype=torch.float16,
                )

                # Run KNN evaluation.
                trainer = Trainer(
                    max_epochs=1,
                    accelerator='gpu',
                    devices=[0],
                    logger=CSVLogger(save_dir=os.path.join(cfg.logging.exp_dir, eval_dataset), name='knn_eval'),
                    enable_checkpointing=False,
                    num_sanity_val_steps=0,
                )
                trainer.fit(model=classifier)

            elif cfg.dataset.task == 'segmentation':

                if cfg.ssl.semseg_knn:

                    classifier = KNN_Segmentation(
                        cfg=cfg,
                        datamodule=dm_eval,
                        network=model_backbone,
                        num_classes=dm_eval.num_cls,
                        knn_k=cfg.ssl.knn_k,
                        knn_t=cfg.ssl.knn_t,
                        feature_dtype=torch.float16,
                    )

                    # Run KNN evaluation.
                    trainer = Trainer(
                        max_epochs=1,
                        accelerator='gpu',
                        # limit_train_batches=5,
                        # limit_val_batches=5,
                        devices=[0],
                        logger=CSVLogger(save_dir=os.path.join(cfg.logging.exp_dir, eval_dataset), name='segmentation_eval_knn'),
                        enable_checkpointing=False,
                        num_sanity_val_steps=0,
                    )
                    trainer.fit(model=classifier)

                else:
                
                    classifier = UPerSegmentationModel(
                        cfg=cfg,
                        datamodule=dm_eval,
                        network=model_backbone,
                        num_classes=dm_eval.num_cls,
                    )

                    trainer = Trainer(
                        max_epochs=20,
                        accelerator='gpu',
                        devices=[0],
                        logger=CSVLogger(save_dir=os.path.join(cfg.logging.exp_dir, eval_dataset), name='segmentation_eval_uper'),
                        enable_checkpointing=False,
                        num_sanity_val_steps=0,
                    )
                    trainer.fit(model=classifier)

        if cfg.ssl.skip_linear_eval:
            print_rank_zero("Skipping linear eval.")
        else:
            classifier = LinearClassifier(
                cfg=cfg,
                datamodule=dm_eval,
                network=model_backbone,
                batch_size_per_device=cfg.ssl.eval_batch_size,
                feature_dim=model_feature_dim,
                num_classes=dm_eval.num_cls,
                freeze_model=True,
            )
            trainer = Trainer(
                # max_epochs=30,
                max_epochs=30,
                accelerator='gpu',
                devices=[0],
                logger=CSVLogger(save_dir=os.path.join(cfg.logging.exp_dir, eval_dataset), name='linear_eval'),
                enable_checkpointing=False,
                num_sanity_val_steps=0,
            )
            trainer.fit(model=classifier)

        if cfg.ssl.skip_finetune_eval:
            print_rank_zero("Skipping fine-tune eval.")
        else:
            classifier = FinetuneLinearClassifier(
                cfg=cfg,
                datamodule=dm_eval,
                network=model_backbone,
                batch_size_per_device=cfg.ssl.eval_batch_size,
                feature_dim=model_feature_dim,
                num_classes=dm_eval.num_cls,
                freeze_model=False,
            )
            trainer = Trainer(
                max_epochs=30,
                accelerator='gpu',
                devices=[0],
                logger=CSVLogger(save_dir=os.path.join(cfg.logging.exp_dir, eval_dataset), name='finetune_eval'),
                enable_checkpointing=False,
                num_sanity_val_steps=0,
            )
            trainer.fit(model=classifier)


if __name__ == "__main__":
    main()
