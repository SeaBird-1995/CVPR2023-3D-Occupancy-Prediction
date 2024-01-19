'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-01-19 15:00:53
Email: haimingzhang@link.cuhk.edu.cn
Description: The occupancy encoder.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from mmcv.runner import BaseModule, force_fp32
from mmdet3d.models.builder import BACKBONES
from mmdet3d.models.builder import NECKS


@BACKBONES.register_module()
class OccEncoder(BaseModule):

    def __init__(self,
                 encoder_cfg, 
                 encoder_neck=None,
                 num_classes=18,
                 expansion=8, 
                 init_cfg=None):
        super().__init__(init_cfg)

        self.expansion = expansion
        self.num_cls = num_classes

        self.encoder = BACKBONES.build(encoder_cfg)
        if encoder_neck is not None:
            self.encoder_neck = NECKS.build(encoder_neck)
        self.class_embeds = nn.Embedding(num_classes, expansion)

    @property
    def with_encoder_neck(self):
        return hasattr(self,
                       'encoder_neck') and self.encoder_neck is not None
    
    def forward_encoder(self, x):
        # x: bs, F, H, W, D
        bs, F, H, W, D = x.shape
        x = self.class_embeds(x) # bs, F, H, W, D, c
        x = rearrange(x, 'bs F x y z c -> (bs F) c z y x')
        x = rearrange(x, 'b c z y x -> b (c z) y x')
        
        z = self.encoder(x)
        return z
        
    def forward(self, x, **kwargs):
        z = self.forward_encoder(x)

        if self.with_encoder_neck:
            z = self.encoder_neck(z)
            if type(x) in [list, tuple]:
                x = x[0]
        return z