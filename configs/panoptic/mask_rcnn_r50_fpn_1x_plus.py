# model settings
model = dict(
    type='PanopticDetector',
    pretrained='modelzoo://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='PanopticFPN',
        in_channels=[256, 512, 1024, 2048],
        in_embeds=16,
        out_channels=256,
        stuff_num_classes=54,
        num_proto_convs=3,
        num_segm_convs=3,
        start_level=1,
        add_extra_convs=True,
        num_outs=5,
        upsample_ratio=2),
    bbox_head=dict(
        type='YolactBboxHeadPlus',
        num_classes=81,
        inst_embeds=16,
        in_channels=256,
        num_cls_convs=3,
        num_reg_convs=3,
        num_emb_convs=3,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict(
    bbox_head=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    proposals=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        score_thr=0.01,
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=100),
    proto=dict(
        stride=4.,
        final_stride=1.),
    mask_head=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=False)))
test_cfg = dict(
    bbox_head=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=100),
    proto=dict(
        stride=4.,
        final_stride=1.,
        score_thr=0.5,
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=100,
        mask_thr_binary=0.5))
# dataset settings
dataset_type = 'PanopticCocoDataset'        # change .
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadPanopticAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    # add for semantic seg
    # use 1/4 scales to surprivised
    dict(type='SegResizeFlipPadRescale', scale_factor=1.),  # 1/4 scales
    dict(type='DefaultFormatBundle'),
    # add gt_semantic_seg collections.
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'],
                 meta_keys=['filename', 'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'img_norm_cfg', 'cat_info',
                            'id']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        category_file=data_root + 'annotations/panoptic_coco_category_infos.json',
        img_prefix=data_root + 'train2017/',
        seg_prefix=data_root + 'annotations/panoptic_train2017_stuff_semantic_segmentation_pngs',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        category_file=data_root + 'annotations/panoptic_coco_category_infos.json',
        img_prefix=data_root + 'val2017/',
        seg_prefix=data_root + 'annotations/panoptic_train2017_stuff_semantic_segmentation_pngs',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        category_file=data_root + 'annotations/panoptic_coco_category_infos.json',
        img_prefix=data_root + 'val2017/',
        seg_prefix=data_root + 'annotations/panoptic_train2017_stuff_semantic_segmentation_pngs',
        pipeline=test_pipeline))
# optimizer
# 0.02 for 8GPUs
# for panoptic and segmentation use 0.01
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
evaluation = dict(interval=1)
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/mask_rcnn_r50_fpn_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
