'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-01-10 22:48:25
Email: haimingzhang@link.cuhk.edu.cn
Description: Predict the future BEV features, based on historial BEV features and
future BEV queries.
'''

import copy
import torch
import torch.nn as nn
from einops import rearrange

from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.models import HEADS
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn.bricks.transformer import build_positional_encoding
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models.utils import build_transformer
from mmdet3d.models.builder import build_head
from mmdet.models.builder import build_loss


@HEADS.register_module()
class BEVFutureDecoderHead(BaseModule):
    """BEVFutureDecoder Head
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 transformer,
                 positional_encoding,
                 pc_range,
                 occ_head=None,
                 num_future_pred=6,
                 bev_h=30,
                 bev_w=30,
                 init_cfg=None,
                 use_mask=False,
                 num_classes=18,
                 loss_occ=None,
                 **kwargs):
        super(BEVFutureDecoderHead, self).__init__(init_cfg=init_cfg)

        self.transformer = build_transformer(transformer)
        self.positional_encoding = build_positional_encoding(positional_encoding)

        self.num_classes = num_classes
        self.embed_dims = self.transformer.embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.use_mask = use_mask
        
        self.num_future_pred = num_future_pred
        if occ_head is not None:
            self.occ_head = build_head(occ_head)
            self.loss_occ = build_loss(loss_occ)

        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()

    def predict_future_feats(self, curr_bev_feat, img_metas):
        bs, N, C = curr_bev_feat.shape
        dtype = curr_bev_feat.dtype
        
        # get the bev queries
        future_bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=future_bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        # Future decoder
        bev_embed = self.transformer(
            curr_bev_feat,
            future_bev_queries,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h,
                         self.real_w / self.bev_w),
            bev_pos=bev_pos,
            img_metas=img_metas)
        return bev_embed

    @auto_fp16(apply_to=('bev_feat'))
    def forward(self, bev_feat_dict, img_metas_list):
        """Forward function.
        Args:
            bev_feat (Tensor): current bev feature (B, num_query, C).
            img_metas_list: list of dict, list length is the batch size.
        Returns:
        """
        bev_feat = bev_feat_dict['bev_embed']

        future_bev_feats = []
        # 1) loop the number of future predictions
        for i in range(self.num_future_pred):
            img_metas = [each[i] for each in img_metas_list]
            # 2) predict the future bev features autoregressively
            future_bev_embed = self.predict_future_feats(bev_feat, img_metas)  # (bs, num_query, c)
            future_bev_feats.append(future_bev_embed)
            bev_feat = future_bev_embed

        # 3) decode the occupancy
        future_bev_feats = torch.stack(future_bev_feats, dim=1) # to (bs, num_future_pred, num_query, c)
        future_bev_feats = rearrange(future_bev_feats, 'b n (h w) c -> (b n) c h w', h=self.bev_h, w=self.bev_w)

        occ_logits = self.occ_head(future_bev_feats)
        occ_logits = rearrange(occ_logits, '(b n) x y z c -> b n x y z c', 
                               n=self.num_future_pred)
        outs = {
            'bev_embed': future_bev_feats,
            'occ_pred': occ_logits,
        }

        return outs

    def loss_single(self, preds, voxel_semantics, mask_camera=None):
        voxel_semantics=voxel_semantics.long()
        if self.use_mask:
            assert mask_camera is not None
            voxel_semantics=voxel_semantics.reshape(-1)
            preds=preds.reshape(-1,self.num_classes)
            mask_camera=mask_camera.reshape(-1)
            num_total_samples=mask_camera.sum()
            loss_occ=self.loss_occ(preds,voxel_semantics,mask_camera, avg_factor=num_total_samples)
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics,)
        return loss_occ

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             future_occ_gt,
             mask_camera,
             preds_dicts,
             img_metas=None):
        """"Loss function.
        Args:

            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        occ_pred = preds_dicts['occ_pred']
        b, num_pred, h, w, d, c = occ_pred.shape

        occ_pred = occ_pred.view(-1, h, w, d, c)
        voxel_semantics = future_occ_gt.view(-1, h, w, d)

        loss_dict = dict()
        assert voxel_semantics.min() >= 0 and voxel_semantics.max()<=17
        losses = self.loss_single(occ_pred, voxel_semantics, mask_camera=None)
        loss_dict['loss_occ'] = losses
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_occupancy(self, preds_dicts):
        """Get the occupancy prediction.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            list[dict]: Decoded occupancy prediction.
        """

        occ_pred = preds_dicts['occ_pred']  # (bs, num_pred, h, w, d, c)
        occ_score = occ_pred.softmax(-1)
        occ_score = occ_score.argmax(-1)

        return occ_score