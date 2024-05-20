'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-05-20 22:00:55
Email: haimingzhang@link.cuhk.edu.cn
Description: Support the OpenScene occupancy dataset.
'''

import copy
import os
from pickle import TRUE
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import mmcv
from os import path as osp, stat
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .nuscnes_eval import Axis, NuScenesEval_custom
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.parallel import DataContainer as DC
import random
from torch.nn import functional as F
from nuscenes.utils.geometry_utils import transform_matrix
from mmcv.runner import force_fp32
from collections import defaultdict
import math
from tqdm import tqdm
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from .openscene_occ_metrics import Metric_mIoU, Metric_FScore
from .nuscenes_occ import NuSceneOcc


@DATASETS.register_module()
class CustomNuDataset(NuSceneOcc):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.
        Args:
            ann_file (str): Path of the annotation file.
        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """

        data_infos = mmcv.load(ann_file)
        data_infos = data_infos[::self.load_interval]
        return data_infos
    
    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_bboxes_3d = info['anns']['gt_boxes'][mask]
            gt_names_3d = info['anns']['gt_names'][mask]
        else:
            gt_bboxes_3d = info['anns']['gt_boxes']
            gt_names_3d = info['anns']['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results
        
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            # pts_filename=info['lidar_path'],
            sweeps=osp.join(self.data_root, info['lidar_path']),
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['sample_prev'],
            next_idx=info['sample_next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )


        if 'occ_gt_final_path' in info:
            input_dict['occ_gt_final_path'] = info['occ_gt_final_path']
        if 'flow_gt_final_path' in info:
            input_dict['flow_gt_final_path'] = info['flow_gt_final_path']

        lidar2ego_rotation = info['lidar2ego_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        ego2lidar = transform_matrix(translation=lidar2ego_translation, rotation=Quaternion(lidar2ego_rotation),
                                     inverse=True)
        input_dict['ego2lidar'] = ego2lidar

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return input_dict
    
    def evaluate_miou(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        from .pipelines.loading import openscene_occ_to_voxel

        if show_dir is not None:
            if not os.path.exists(show_dir):
                os.mkdir(show_dir)
            print('\nSaving output and gt in {} for visualization.'.format(show_dir))
            begin=eval_kwargs.get('begin',None)
            end=eval_kwargs.get('end',None)
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=12,
            use_lidar_mask=False,
            use_image_mask=False)
        if self.eval_fscore:
            self.fscore_eval_metrics = Metric_FScore(
                leaf_size=10,
                threshold_acc=0.4,
                threshold_complete=0.4,
                voxel_size=[0.4, 0.4, 0.4],
                range=[-40, -40, -1, 40, 40, 5.4],
                void=[17, 255],
                use_lidar_mask=False,
                use_image_mask=True,
            )
        print('\nStarting Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]

            occ_gt = np.load(info['occ_gt_final_path'])
            gt_semantics = openscene_occ_to_voxel(occ_gt).numpy()

            if show_dir is not None:
                if begin is not None and end is not None:
                    if index>= begin and index<end:
                        sample_token = info['token']
                        save_path = os.path.join(show_dir,str(index).zfill(4))
                        np.savez_compressed(save_path, pred=occ_pred, gt=occ_gt, sample_token=sample_token)
                else:
                    sample_token=info['token']
                    save_path=os.path.join(show_dir,str(index).zfill(4))
                    np.savez_compressed(save_path,pred=occ_pred,gt=occ_gt,sample_token=sample_token)

            # occ_pred = occ_pred
            self.occ_eval_metrics.add_batch(occ_pred, gt_semantics, None, None)
            if self.eval_fscore:
                self.fscore_eval_metrics.add_batch(occ_pred, gt_semantics, None, None)

        self.occ_eval_metrics.count_miou()
        if self.eval_fscore:
            self.fscore_eval_metrics.count_fscore()