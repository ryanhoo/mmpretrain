data_root = 'data/birds_1369_train'
data_root_val = 'data/birds_1369_val'
work_dir = './work_dirs/efficientnetv2_s_2nd'
image_size = 384
num_classes=1395
batch_size=32
max_epochs = 100

auto_scale_lr = dict(base_batch_size=batch_size)
data_preprocessor = dict(
    mean=[
        127.5,
        127.5,
        127.5,
    ],
    num_classes=num_classes,
    std=[
        127.5,
        127.5,
        127.5,
    ],
    to_rgb=True)
dataset_type = 'CustomDataset'
# default_hooks = dict(
#     # only keeps the latest 3 checkpoints
#     checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook', max_keep_ckpts=10),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
# load_from = 'work_dirs/efficientnetv2_s_2nd/latest.pth'
log_level = 'INFO'
model = dict(
    backbone=dict(arch='s', type='EfficientNetV2'),
    head=dict(
        in_channels=1280,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=num_classes,
        topk=(
            1,
            5,
        ),
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(lr=0.1, momentum=0.9, type='SGD', weight_decay=0.0001))
param_scheduler = dict(
    by_epoch=True, gamma=0.1, milestones=[
        30,
        60,
        90,
    ], type='MultiStepLR')
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=batch_size,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='data/birds_val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(crop_padding=0, crop_size=image_size, type='EfficientNetCenterCrop'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=5,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    topk=(
        1,
        5,
    ), type='Accuracy')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(crop_padding=0, crop_size=image_size, type='EfficientNetCenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)
train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        data_root=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(crop_padding=0, scale=300, type='EfficientNetRandomCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=5,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(crop_padding=0, scale=300, type='EfficientNetRandomCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        data_root=data_root_val,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(crop_padding=0, crop_size=image_size, type='EfficientNetCenterCrop'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=5,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    topk=(
        1,
        5,
    ), type='Accuracy')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
