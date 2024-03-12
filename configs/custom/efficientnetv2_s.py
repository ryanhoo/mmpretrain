_base_ = [
    '../_base_/models/efficientnet_v2/efficientnetv2_s.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]

# dataset settings
dataset_type = 'CustomDataset'
data_root = 'data/birds'
data_preprocessor = dict(
    num_classes=27,
    # RGB format normalization parameters
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetRandomCrop', scale=300, crop_padding=0),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetCenterCrop', crop_size=384, crop_padding=0),
    dict(type='PackInputs'),
]

# >>>>>>>>>>>>>>> 在此重载数据设置 >>>>>>>>>>>>>>>>>>>
# train_dataloader = dict(
#     batch_size=128,
#     num_workers=32,
#     dataset=dict(
#         type=dataset_type,
#         data_root='data/birds',
#         ann_file='',       # 我们假定使用子文件夹格式，因此需要将标注文件置空
#         data_prefix='',    # 使用 `data_root` 路径下所有数据
#         with_label=True,
#     )
# )
# val_dataloader = dict(
#     batch_size=64,
#     num_workers=32,
#     dataset=dict(
#         type=dataset_type,
#         data_root='data/birds_val',
#         # split='val',
#         ann_file='',       # 我们假定使用子文件夹格式，因此需要将标注文件置空
#         data_prefix='',    # 使用 `data_root` 路径下所有数据
#         with_label=True,
#         pipeline=test_pipeline),
#     sampler=dict(type='DefaultSampler', shuffle=False),
# )
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
train_cfg = dict(max_epochs=100)
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))