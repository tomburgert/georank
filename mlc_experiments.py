import sys
import os
import copy
import yaml

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from config import NoiseMLCConfig


print(hydra)

cs = ConfigStore.instance()
cs.store(name="base_config", node=NoiseMLCConfig)


def update_dataset_parameter(cfg, dataset, loc='single_loc', dataset_size='100', image_size='normal_size', v='v1'):

    with open('conf/datasets.yaml', "r") as f:
        yaml_file = yaml.safe_load(f)

    cfg.dataset.lmdb_path    = yaml_file[dataset]['lmdb_path'][image_size]
    cfg.dataset.labels_path  = yaml_file[dataset]['labels_path']
    cfg.dataset.train_csv    = yaml_file[dataset]['train_csv'][loc][dataset_size][v]
    cfg.dataset.task         = yaml_file[dataset]['task']
    cfg.dataset.val_csv      = yaml_file[dataset]['val_csv']
    cfg.dataset.test_csv     = yaml_file[dataset]['test_csv']
    cfg.dataset.num_classes  = yaml_file[dataset]['num_classes']
    cfg.dataset.num_channels = yaml_file[dataset]['num_channels']

    return cfg


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    if cfg.params.slurm_bypass:
        print('Bypassing slurm, using GPU: {}'.format(cfg.params.cuda_no))
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.params.cuda_no
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    print(OmegaConf.to_yaml(cfg))

    import torch

    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.callbacks import ModelCheckpoint

    from base import BaseModel
    from models.utils import get_network
    from eval_base import KNNClassifier

    sys.path.append('data')
    from data.datamodule import get_datamodule  # noqa: E402
    from data.transform import SingleTransform

    seed_everything(cfg.params.seed, workers=True)

    torch.set_float32_matmul_precision('high')

    cfg = update_dataset_parameter(
        cfg=cfg,
        dataset=cfg.params.dataset,
        loc=cfg.dataset.pretrain_loc,
        dataset_size=cfg.dataset.pretrain_dataset_size,
        image_size=cfg.dataset.pretrain_image_size,
        v=cfg.dataset.pretrain_split_version
    )

    test_dataaug_cfg = copy.deepcopy(cfg.dataaug)
    test_dataaug_cfg.update({'augmentations': 'notpretrained', 'p_list': [0]})
    transforms_tr = SingleTransform(cfg.dataaug)
    transforms_te = SingleTransform(test_dataaug_cfg)

    DataModule = get_datamodule(cfg.params.dataset)
    dm = DataModule(cfg, transforms_tr, transforms_te)
    
    dm.setup()

    network = get_network(**cfg.model, num_cls=dm.num_cls)

    model = BaseModel(cfg, dm, network)

    if not cfg.params.skip_training:
        callbacks = []
        if cfg.logging.save_checkpoint:
            if cfg.dataset.task == 'multi_label':
                metric_name = 'val_APmac'
            elif cfg.dataset.task == 'binary':
                metric_name = 'val_f1'
            elif cfg.dataset.task == 'segmentation':
                metric_name = 'val_PAmac'
            elif cfg.dataset.task == 'single_label':
                metric_name = 'val_accmac'
            callbacks += [ModelCheckpoint(monitor=metric_name, filename='best_model_val0', mode='max')]
            # callbacks += [ModelCheckpoint(monitor="val1_APmac", filename='best_model_val1', mode='max')]

        trainer = Trainer(
            accelerator='gpu',
            callbacks=callbacks,
            devices=[0],
            enable_checkpointing=cfg.logging.save_checkpoint,
            max_epochs=cfg.params.max_epochs,
            logger=CSVLogger(save_dir=cfg.logging.exp_dir),
            deterministic=True
        )

        trainer.fit(model)

        # for i in range(len(callbacks)):
        #     # load best model
        #     model.load_state_dict(torch.load(callbacks[i].best_model_path)["state_dict"])
        #     trainer.test(model)

    if cfg.params.test:
        trainer = Trainer(
            accelerator='gpu',
            callbacks=[],
            devices=[0],
            max_epochs=cfg.params.max_epochs,
            logger=CSVLogger(save_dir=cfg.logging.exp_dir),
            deterministic=True
        )

        model.load_state_dict(torch.load(cfg.logging.ckpt_path)["state_dict"])
        trainer.test(model)

    if cfg.params.knn_evaluation:
        train_dataaug_cfg = copy.deepcopy(cfg.dataaug)
        train_dataaug_cfg.update({'augmentations': 'flip', 'p_list': [0.8]})

        transforms_tr = SingleTransform(train_dataaug_cfg)
        dm = DataModule(cfg, transforms_tr, transforms_te)
        dm.setup()

        if cfg.params.skip_training:
            model.load_state_dict(torch.load(cfg.logging.ckpt_path)["state_dict"])

        classifier = KNNClassifier(
            cfg=cfg,
            datamodule=dm,
            network=model,
            num_classes=dm.num_cls,
            knn_k=cfg.params.knn_k,
            knn_t=cfg.params.knn_k,
            feature_dtype=torch.float16,
        )

        # Run KNN evaluation.
        trainer = Trainer(
            max_epochs=1,
            accelerator='gpu',
            devices=[0],
            logger=CSVLogger(save_dir=cfg.logging.exp_dir, name='knn_eval'),
            enable_checkpointing=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(model=classifier)


if __name__ == "__main__":
    main()
