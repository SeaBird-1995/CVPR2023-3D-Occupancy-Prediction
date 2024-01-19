'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-01-17 09:22:34
Email: haimingzhang@link.cuhk.edu.cn
Description: The occupancy decoder. Modified from BEVFormer occupancy baseline.
'''
import copy
import torch
import torch.nn as nn
from einops import rearrange

from mmdet.models import HEADS
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn import Conv2d, Conv3d, ConvModule
from mmcv.runner import force_fp32, auto_fp16


@HEADS.register_module()
class OccHead(BaseModule):
    """Modified from BEVFormer Occ.

    Args:
        BaseModule (_type_): _description_
    """
    def __init__(self,
                 norm_cfg_3d=dict(type='BN3d', ),
                 embed_dims=256,
                 out_dim=32,
                 pillar_h=16,
                 act_cfg=dict(type='ReLU',inplace=True),
                 num_classes=18,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        
        self.embed_dims = embed_dims
        self.out_dim = out_dim
        self.pillar_h = pillar_h

        use_bias_3d = norm_cfg_3d is None

        self.middle_dims = self.embed_dims // pillar_h
        self.decoder = nn.Sequential(
            ConvModule(
                self.middle_dims,
                self.out_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias_3d,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=norm_cfg_3d,
                act_cfg=act_cfg),
            # ConvModule(
            #     self.out_dim,
            #     self.out_dim,
            #     kernel_size=3,
            #     stride=1,
            #     padding=1,
            #     bias=use_bias_3d,
            #     conv_cfg=dict(type='Conv3d'),
            #     norm_cfg=norm_cfg_3d,
            #     act_cfg=act_cfg),
        )
        self.predicter = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim*2),
            nn.Softplus(),
            nn.Linear(self.out_dim*2, num_classes),
        )

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    @auto_fp16(apply_to=('x'))
    def forward(self, x):
        assert x.dim() == 4

        bs, c, bev_h, bev_w = x.shape

        bev_embed = x.view(bs, -1, self.pillar_h, bev_h, bev_w)  # (bs, C, Z, Y, X)
        outputs = self.decoder(bev_embed)
        outputs = outputs.permute(0, 4, 3, 2, 1) # to (bs, X, Y, Z, C)
        outputs = self.predicter(outputs)
        return outputs

