_base_ = [
    "../swin/swin-base-patch4-window12-in22k-384x384-pre_upernet_8xb2-160k_ade20k-512x512.py"
]

# model
checkpoint_file = "../../models/mmsegmentation/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_20210531_125459-429057bf.pth"
num_classes = 3
short_side = 512
crop_size = (short_side, short_side)

_base_.data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=_base_.data_preprocessor,
    backbone=dict(init_cfg=dict(checkpoint=checkpoint_file)),
    decode_head=dict(num_classes=num_classes),
    auxiliary_head=dict(num_classes=num_classes),
)

# dataset
dataset_type = "OilleakDataset"
data_root = "../../data/sly_oilleak_20251010_v7.4.3/mmseg"

_base_.train_pipeline[1] = dict(type="LoadAnnotations", reduce_zero_label=False)
_base_.train_pipeline[2] = dict(
    type="RandomResize", scale=(2048, short_side), ratio_range=(0.5, 2.0), keep_ratio=True
)
_base_.train_pipeline[3] = dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75)
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(type=dataset_type, data_root=data_root, pipeline=_base_.train_pipeline)
)

_base_.test_pipeline[1] = dict(type="Resize", scale=(2048, short_side), keep_ratio=True)
_base_.test_pipeline[2] = dict(type="LoadAnnotations", reduce_zero_label=False)
val_dataloader = dict(
    dataset=dict(type=dataset_type, data_root=data_root, pipeline=_base_.test_pipeline)
)

test_dataloader = val_dataloader

# schedule
warmup = 141
max_iters = 42300  # 120 * ceil(number_of_training_images / (2 * 8))
val_interval = 1410  # 10 * ceil(number_of_training_images / (2 * 8))

_base_.param_scheduler[0]["end"] = warmup
_base_.param_scheduler[1]["begin"] = warmup
_base_.param_scheduler[1]["end"] = max_iters

train_cfg = dict(max_iters=max_iters, val_interval=val_interval)
default_hooks = dict(checkpoint=dict(interval=val_interval))

# runtime
vis_backends = [dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]
_base_.visualizer["vis_backends"] = vis_backends
