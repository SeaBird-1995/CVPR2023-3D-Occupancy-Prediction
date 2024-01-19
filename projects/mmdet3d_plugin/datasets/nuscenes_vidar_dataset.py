'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-01-18 12:06:37
Email: haimingzhang@link.cuhk.edu.cn
Description: The ViDAR nuscenes dataset, use the historial images to predict
the future BEV feature maps.
'''
import copy

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import mmcv
from os import path as osp
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .nuscnes_eval import NuScenesEval_custom
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.parallel import DataContainer as DC
import random
from .nuscenes_occ import NuSceneOcc


@DATASETS.register_module()
class NuScenesViDARDataset(NuSceneOcc):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, 
                 return_len,
                 offset,
                 times=1,
                 start_frame=0,
                 mid_frame=5,
                 end_frame=8,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.return_len = return_len
        self.offset = offset
        self.times = times

        self.scene_names = list(self.data_infos.keys())
        scene_lens = [len(self.data_infos[sn]) for sn in self.scene_names]
        self.scene_lens = [l - self.return_len - self.offset for l in scene_lens]
        if not self.test_mode:
            self.flag = np.zeros(len(self), dtype=np.uint8)
        
        self.origin_data_infos = copy.deepcopy(self.data_infos)
        self.start_frame = start_frame
        self.mid_frame = mid_frame
        self.end_frame = end_frame

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(28130, dtype=np.uint8)

    def __len__(self):
        'Denotes the total number of samples'
        return sum(self.scene_lens)

    def load_annotations(self, ann_file):
        """Load annotations from ann_file without sorting.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = data['infos']
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos
    
    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        ## TODO, do not use the loop
        for i, scene_len in enumerate(self.scene_lens):
            if index < scene_len:
                scene_name = self.scene_names[i]
                idx = index
                break
            else:
                index -= scene_len
        
        self.data_infos = self.origin_data_infos[scene_name]

        index_list = list(range(self.return_len + self.offset))
        index_list = [idx + i for i in index_list]
        queue = []
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            queue.append(example)

        return self.union2one(queue)

    def fetch_data(self, queue, key, start, end):
        res = [queue[idx][key].data for idx in range(start, end)]
        return res

    def union2one(self, queue):
        start_frame, mid_frame, end_frame = self.start_frame, self.mid_frame, self.end_frame

        imgs_list = self.fetch_data(queue, 'img', start_frame, mid_frame)
        # the future infos
        occ_list = [queue[idx]['voxel_semantics'] for idx in range(mid_frame, end_frame)]

        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        
        result = dict()
        result['img'] = DC(torch.stack(imgs_list[start_frame:mid_frame]), cpu_only=False, stack=True)

        history_metas = dict()
        for i in range(start_frame, mid_frame):
            history_metas[i] = metas_map[i]
        
        future_metas = dict()
        for i in range(mid_frame, end_frame):
            new_idx = i - mid_frame
            future_metas[new_idx] = metas_map[i]

        result['img_metas'] = DC(history_metas, cpu_only=True)
        result['future_img_metas'] = DC(future_metas, cpu_only=True)
        result['future_voxel_semantics'] = np.stack(occ_list)

        return result
    
    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_train_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

