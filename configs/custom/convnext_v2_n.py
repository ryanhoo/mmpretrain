_base_ = [
    '../_base_/models/convnext_v2/nano.py',
    # '../_base_/datasets/imagenet_bs64_swin_384.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

data_root_train = 'data/birds_1451/train'
data_root_val = 'data/birds_1451/val'
work_dir = './work_dirs/convnext_v2_n'
image_size = 384
num_classes=1451
num_workers=32
batch_size=32
batch_size_val=32
max_epochs = 200

resume = True

# dataset settings
dataset_type = 'CustomDataset'
model = dict(
    backbone=dict(
        arch='nano',
        drop_path_rate=0.1,
        layer_scale_init_value=0.0,
        type='ConvNeXt',
        use_grn=True),
    head=dict(
        in_channels=640,
        init_cfg=None,
        loss=dict(label_smooth_val=0.2, type='LabelSmoothLoss'),
        num_classes=num_classes,
        type='LinearClsHead'),
    init_cfg=dict(
        bias=0.0, layer=[
            'Conv2d',
            'Linear',
        ], std=0.02, type='TruncNormal'),
    type='ImageClassifier')
data_preprocessor = dict(
    num_classes=num_classes,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=image_size,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=image_size, backend='pillow', interpolation='bicubic'),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root_train,
        # split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=batch_size_val,
    num_workers=num_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root_val,
        # split='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# schedule setting
optim_wrapper = dict(
    optimizer=dict(lr=8e-4, weight_decay=0.3),
    clip_grad=None,
)

# learning policy
param_scheduler = [dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True)]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]
