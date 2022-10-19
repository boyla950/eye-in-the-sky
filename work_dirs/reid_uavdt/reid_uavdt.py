base_ = ['../_base_/default_runtime.py']
TRAIN_REID = True
model = dict(
    type='BaseReID',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling', kernel_size=(8, 4), stride=1),
    head=dict(
        type='LinearReIDHead',
        num_fcs=1,
        in_channels=2048,
        fc_channels=1024,
        out_channels=128,
        num_classes=1050,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss_pairwise=dict(type='TripletLoss', margin=0.3, loss_weight=1.0),
        norm_cfg=dict(type='BN1d'),
        act_cfg=dict(type='ReLU')))
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[5])
total_epochs = 6
dataset_type = 'ReIDDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(
        type='SeqResize',
        img_scale=(128, 256),
        share_params=False,
        keep_ratio=False,
        bbox_clip_border=False,
        override=False),
    dict(
        type='SeqRandomFlip',
        share_params=False,
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='SeqNormalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='VideoCollect', keys=['img', 'gt_label']),
    dict(type='ReIDFormatBundle')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(128, 256), keep_ratio=False),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'], meta_keys=[])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='ReIDDataset',
        triplet_sampler=dict(num_ids=8, ins_per_id=4),
        ann_file=
        './UAVDT_ReID/meta/train_80.txt',
        data_prefix='./UAVDT_ReID/imgs',
        pipeline=[
            dict(type='LoadMultiImagesFromFile', to_float32=True),
            dict(
                type='SeqResize',
                img_scale=(128, 256),
                share_params=False,
                keep_ratio=False,
                bbox_clip_border=False,
                override=False),
            dict(
                type='SeqRandomFlip',
                share_params=False,
                flip_ratio=0.5,
                direction='horizontal'),
            dict(
                type='SeqNormalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='VideoCollect', keys=['img', 'gt_label']),
            dict(type='ReIDFormatBundle')
        ]),
    val=dict(
        type='ReIDDataset',
        triplet_sampler=None,
        ann_file=
        './UAVDT_ReID/meta/val_20.txt',
        data_prefix='./UAVDT_ReID/imgs',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(128, 256), keep_ratio=False),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=[])
        ]),
    test=dict(
        type='ReIDDataset',
        triplet_sampler=None,
        ann_file=
        './UAVDT_ReID/meta/val_20.txt',
        data_prefix='./UAVDT_ReID/imgs',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(128, 256), keep_ratio=False),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=[])
        ]))
evaluation = dict(interval=10, metric=None)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth'
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/reid_uavdt'
gpu_ids = [0]
