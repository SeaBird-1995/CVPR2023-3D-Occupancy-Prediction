'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-01-15 15:04:22
Email: haimingzhang@link.cuhk.edu.cn
Description: Using the occupancy GT as inputs, and predict the occupancy in the future
based on the ViDAR transformer.
'''

_base_ = [
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
voxel_size = [0.2, 0.2, 8]



img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 4 # each sequence contains `queue_length` frames.

base_channel = 64
expansion = 8

return_len_train = 11
return_len_ = 11

start_frame = 0
mid_frame = 5
end_frame = 11
num_future_pred = end_frame - mid_frame

model = dict(
    type='OccWorldViDAR',
    start_frame=start_frame,
    mid_frame=mid_frame,
    end_frame=end_frame,
    video_test_mode=True,
    
    occ_encoder=dict(
        type='OccEncoder',
        num_classes=18,
        expansion=expansion,
        encoder_cfg=dict(
            type='CustomResNet',
            numC_input=16*expansion,
            num_channels=[base_channel * 2, base_channel * 4, base_channel * 8]),
        encoder_neck=dict(
            type='FPN_LSS',
            in_channels=base_channel * 8 + base_channel * 2,
            out_channels=256),
    ),
    # the history encoder
    history_encoder=dict(
        type='OccHistoryEncoder',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        pc_range=point_cloud_range,
        transformer=dict(
            type='TransformerOccGT',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerOccEncoder',
                num_layers=3,
                pc_range=point_cloud_range,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialOccCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='CustomMSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        # model training and testing settings
        train_cfg=dict(pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range)))),
    ## The future decoder
    future_decoder=dict(
        type='BEVFutureDecoderHead',
        use_mask=False,
        num_classes=18,
        pc_range=point_cloud_range,
        num_future_pred=num_future_pred,
        bev_h=bev_h_,
        bev_w=bev_w_,

        loss_occ= dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        
        transformer=dict(
            type='FutureTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVPredictorEncoder',
                num_layers=6,
                pc_range=point_cloud_range,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVPredictorLayer',
                    attn_cfgs=[
                        dict(
                            type='DeformSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='TemporalCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='CustomMSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        occ_head=dict(
            type='OccHead',
            num_classes=18
        ),),
    )


dataset_type = 'NuScenesOccWorldDataset'
data_path = 'data/nuscenes/'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_path = data_path,
        return_len = return_len_train+1, 
        offset = 0,
        start_frame=start_frame,
        mid_frame=mid_frame,
        end_frame=end_frame,
        imageset = 'data/nuscenes_infos_train_temporal_v3_scene.pkl'),
    val=dict(type='nuScenesSceneDatasetLidarTraverse',
             data_path = data_path,
             return_len = return_len_+1, 
             offset = 0,
             imageset = 'data/nuscenes_infos_val_temporal_v3_scene.pkl',
             test_mode=True),
    test=dict(type='nuScenesSceneDatasetLidarTraverse',
              data_path = data_path,
              return_len = return_len_+1, 
              offset = 0,
              imageset = 'data/nuscenes_infos_val_temporal_v3_scene.pkl',
              test_mode=True),
    shuffler_sampler=dict(type='PytorchDistributedSampler'),
    nonshuffler_sampler=dict(type='PytorchDistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# custom_hooks = [
#     dict(
#         type='OccupancyEvalHook',
#     ),
# ]

checkpoint_config = dict(interval=2)
