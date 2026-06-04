from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
from dataclasses import field


@dataclass
class GeneralParameter:
    seed: int = 1
    dataset: str = ''
    populations: int = 5
    cuda_no: str = 2
    max_epochs: int = 20
    model: str = ''
    pos_weight: bool = False
    start: int = 0
    skip_training: bool = False
    knn_evaluation: bool = False
    knn_k: int = 10
    knn_t: float = 0.9
    log_on_epoch: bool = True
    test: bool = False
    percentage: Optional[int] = None
    slurm_bypass: bool = False


@dataclass
class DatasetParameter:
    lmdb_path: str = ''
    labels_path: str = ''
    train_csv: str = ''
    temporal_views_path: Optional[str] = None
    val_csv: str = ''
    test_csv: str = ''
    task: str = ''
    num_classes: Optional[int] = None
    num_channels: Optional[int] = None
    pretrain_loc: str = 'single_loc'
    pretrain_dataset_size: str = '100'
    pretrain_image_size: str = 'normal_size'
    pretrain_split_version: str = 'v1'
    intersection_8country: bool = True
    all_percentiles: bool = False
    pretrain_norm: bool = False
    global_pctl: bool = False
    num_workers: int = 4
    batch_size: int = 128
    pin_memory: bool = True
    eval_on_test: bool = False
    s1_mm: bool = False
    s1_only: bool = False


@dataclass
class SelfSupervisedParamter:
    algorithm: str = 'MoCoV2'
    eval_datasets: List[str] = field(default_factory=list)
    views: int = 2
    eval_batch_size: Optional[int] = 512
    skip_knn_eval: bool = True
    skip_linear_eval: bool = True
    skip_finetune_eval: bool = True
    semseg_knn: bool = False
    knn_k: int = 10
    knn_t: float = 0.9
    mae_knn_eval: str = 'use_cls_token'
    eval_resize: Optional[List[int]] = None
    gcl_test: bool = False
    gcl_geo_loss: str = 'geo_pred_head_string'
    gcl_geo_distance_measure: str = 'haversine'
    mae_use_lr_scheduling: bool = False
    gcl_alpha: float = 0.5
    gcl_geo_label_type: str = ''
    gcl_geo_k: int = 10
    gcl_ntxend_temperature: float = 0.1
    gcl_mse_spacing: float = 0.1
    gcl_mse_min_dist_with_weight: float = 0.5
    gcl_mse_max_dist_with_weight: float = 1.0
    gcl_mse_soft_margin: float = 0.01
    gcl_softrank_reg_strength: float = 0.001
    gcl_weights_style: int = 1
    gcl_min_dist_km: int = 0
    gcl_max_dist_km: int = 3000
    gcl_rank_soft_margin: int = 1
    gcl_rank_loss_type: str = 'mse'
    gcl_reach: int = 400
    gcl_loss_batch_size: int = 64
    gcl_lower_bound: Optional[float] = None
    gcl_upper_bound: Optional[float] = None
    gcl_pct_threshold: float = 0.0
    moco_lr: float = 0.4
    moco_memory_bank_size: int = 4092
    moco_temperature: float = 0.04
    vicreg_lr: float = 0.4
    vicreg_lambda: float = 25.0
    vicreg_mu: float = 25.0
    vicreg_nu: float = 1.0
    mae_mask_ratio: float = 0.75
    mae_lr: float = 0.00015
    dino_warmup_teacher_temp: float = 0.04
    dino_teacher_temp: float = 0.04  # < 0.7
    dino_warmup_teacher_temp_epochs: int = 5
    dino_student_temp: float = 0.1
    dino_center_momentum: float = 0.9
    dino_lr: float = 0.03
    dino_n_local_views: Optional[int] = None
    dino_global_resize_size: Optional[List[int]] = field(default_factory=list)
    dino_global_scale: Optional[List[float]] = field(default_factory=list)
    dino_local_resize_size: Optional[List[int]] = field(default_factory=list)
    dino_local_scale: Optional[List[float]] = field(default_factory=list)
    use_geography_loss: bool = False
    disable_geo_loss: bool = False
    disable_ssl_loss: bool = False
    test_flip: bool = True
    use_world_encoding: bool = False
    world_input_mode: str = 'spherical'  # or fourier, raw
    world_proj_type: str = 'frozen_linear'  # or linear, mlp
    injection_mode: str = 'add'  # or add, concat, cls_concat
    world_freeze: bool = True
    mae_decoder_dim: int = 512
    tile2vec_margin: float = 1.0
    tile2vec_pos_radius: float = 15.0
    tile2vec_neg_radius: float = 50.0
    tile2vec_lr: float = 0.001


@dataclass
class DataAugmentationParameter:
    augmentations: str = 'none'
    p: float = 0.5
    p_list: Optional[List[float]] = None
    magnitude: Optional[int] = None
    brightness_limit : Optional[float] = None
    contrast_limit : Optional[float] = None
    max_edge: Optional[float] = None
    min_edge: Optional[float] = None
    sigma: Optional[List[float]] = field(default_factory=list)
    var_max: Optional[int] = None
    per_channel: Optional[bool] = None
    grid_size: Optional[int] = 3
    max_holes: Optional[int] = None
    min_holes: Optional[int] = None
    dropout_prob: Optional[float] = None
    shift: Optional[int] = None
    num_bits: Optional[int] = None
    randaug_op_names: Optional[List[str]] = field(default_factory=list)
    randaug_magnitude: Optional[int] = None
    resize_size: Optional[List[int]] = field(default_factory=list)
    scale: Optional[List[float]] = field(default_factory=list)
    cond_rrc_min_scale: Optional[float] = 1.0
    ratio : Optional[List[float]] = field(default_factory=list)
    angle: Optional[int] = None
    alpha: Optional[float] = None
    shear_x: Optional[int] = None
    shear_y: Optional[int] = None
    shear: Optional[int] = None
    threshold: Optional[int] = None
    pct_x: Optional[float] = None
    pct_y: Optional[float] = None
    pct: Optional[float] = None


@dataclass
class Network:
    name: str = 'resnet18'
    channels: int = 10
    pretrained: bool = False
    vit_image_size: int = 224
    vit_patch_size: int = 16


@dataclass
class Optimizer:
    min_lr: float = 0.002
    momentum: float = 0.9
    weight_decay: float = 0.005
    gamma: float = 0.1
    milestones: List[int] = field(default_factory=list)


@dataclass
class Logging:
    exp_dir: str = ''
    ckpt_path: Optional[str] = None
    save_checkpoint : bool = False
    save_every_epoch : bool = False
    disable_early_stopping: bool = False


@dataclass
class Tracking:
    train_track_with_transformation: bool = False
    should_track_train_probs: bool = False
    should_track_val_probs: bool = False
    apply_feature_extraction: bool = False
    feature_extraction_epochs: List[int] = field(default_factory=list)
    feature_extraction_with_transformation: bool = True


@dataclass
class Imbalances:
    use_balancing: bool = False
    noise_image_pct: float = 0.05
    p_base: float = 0.05
    classes_for_p: List[int] = field(default_factory=list)
    p_classwise: List[float] = field(default_factory=list)


@dataclass
class NoiseMLCConfig:
    params: GeneralParameter = field(default_factory=GeneralParameter)
    dataset: DatasetParameter = field(default_factory=DatasetParameter)
    model: Network = field(default_factory=Network)
    optim: Optimizer = field(default_factory=Optimizer)
    logging: Logging = field(default_factory=Logging)
    tracking: Tracking = field(default_factory=Tracking)
    dataaug: DataAugmentationParameter = field(default_factory=DataAugmentationParameter)
    imbalances: Imbalances = field(default_factory=Imbalances)


@dataclass
class SelfSupervisedCConfig:
    params: GeneralParameter = field(default_factory=GeneralParameter)
    dataset: DatasetParameter = field(default_factory=DatasetParameter)
    ssl: SelfSupervisedParamter = field(default_factory=SelfSupervisedParamter)
    model: Network = field(default_factory=Network)
    optim: Optimizer = field(default_factory=Optimizer)
    logging: Logging = field(default_factory=Logging)
    tracking: Tracking = field(default_factory=Tracking)
    dataaug: DataAugmentationParameter = field(default_factory=DataAugmentationParameter)
    imbalances: Imbalances = field(default_factory=Imbalances)
