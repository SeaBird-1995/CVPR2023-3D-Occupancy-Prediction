'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-01-15 11:58:57
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import copy
import os, numpy as np, pickle
from copy import deepcopy
from tqdm import tqdm
from mmdet.datasets import DATASETS
import torch
from torch.utils.data import Dataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes, Box3DMode
from mmcv.parallel import DataContainer as DC
from mmdet3d.datasets.pipelines import Compose
from mmdet.utils import get_root_logger
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .pipelines.transform_3d import CustomCollect3D
from .occworld_metrics import MeanIoU, multi_step_MeanIou


@DATASETS.register_module()
class NuScenesOccWorldDataset(Dataset):
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')
    def __init__(
            self, 
            data_path,
            return_len, 
            offset,
            imageset='train', 
            nusc=None,
            times=5,
            test_mode=False,
            input_dataset='gts',
            output_dataset='gts',
            start_frame=0,
            mid_frame=5,
            end_frame=11
        ):
        super().__init__()
        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        self.nusc_infos = data['infos']
        self.scene_names = list(self.nusc_infos.keys())
        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]
        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        self.nusc = nusc
        self.times = times
        self.test_mode = test_mode
        assert input_dataset in ['gts', 'tpv_dense', 'tpv_sparse']
        assert output_dataset == 'gts', f'only used for evaluation, output_dataset should be gts, but got {output_dataset}'
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset

        pipeline = [dict(type='CustomCollect3D', keys=[])]
        self.pipeline = Compose(pipeline)

        self.start_frame = start_frame
        self.mid_frame = mid_frame
        self.end_frame = end_frame
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)*self.times

    def __getitem__(self, index):
        index = index % len(self.nusc_infos)
        scene_name = self.scene_names[index]
        scene_len = self.scene_lens[index]
        idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.input_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        input_occs = np.stack(occs)
        
        ## target occupancy
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.output_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        output_occs = np.stack(occs)

        ## get meta infos
        metas_list = []
        for i in range(self.return_len + self.offset):
            input_dict = self.nusc_infos[scene_name][idx + i]
            ## get can bus
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

            example = self.pipeline(input_dict)
            metas_list.append(example)

        metas = self.union2one(metas_list)
        
        ## convert to tensor
        input, target = input_occs[:self.return_len], output_occs[self.offset:]
        input = torch.from_numpy(input).to(torch.int64)
        target = torch.from_numpy(target).to(torch.int64)

        start_frame, mid_frame, end_frame = self.start_frame, self.mid_frame, self.end_frame

        data_dict = dict(
            history_occ=input[start_frame:mid_frame],
            future_occ=input[mid_frame:end_frame]
        )
        data_dict.update(metas)
        return data_dict

    def union2one(self, queue):
        start_frame, mid_frame, end_frame = self.start_frame, self.mid_frame, self.end_frame

        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if prev_scene_token is None:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = "placeholder"
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
        history_metas = dict()
        for i in range(start_frame, mid_frame):
            history_metas[i] = metas_map[i]
        
        future_metas = dict()
        for i in range(mid_frame, end_frame):
            new_idx = i - mid_frame
            future_metas[new_idx] = metas_map[i]

        result['img_metas'] = DC(history_metas, cpu_only=True)
        result['future_img_metas'] = DC(future_metas, cpu_only=True)

        return result

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        print('\nStarting Evaluation...')
        from .occ_metrics import OccWorldEvaluator

        evaluator = OccWorldEvaluator(eval_length=eval_kwargs['eval_length'])
        
        for occ_pred, target_occs in tqdm(zip(occ_results[0], occ_results[1])):
            ## obtain the occ prediction
            # (bs, n, 200, 200, 16)
            ## compute the metrics
            evaluator.after_step(occ_pred, target_occs)

        evaluator.after_epoch()
    
    def get_meta_data(self, scene_name, idx):
        gt_modes = []
        xys = []
        for i in range(self.return_len + self.offset):
            xys.append(self.nusc_infos[scene_name][idx+i]['gt_ego_fut_trajs'][0]) #1*2
            gt_modes.append(self.nusc_infos[scene_name][idx+i]['pose_mode'])
        xys = np.asarray(xys)
        gt_modes = np.asarray(gt_modes)
        return {'rel_poses': xys, 'gt_mode': gt_modes}
    
    def get_image_info(self, scene_name, idx):
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        # import pdb; pdb.set_trace()
        input_dict = dict(
            sample_idx=info['token'],
            ego2global_translation = info['ego2global_translation'],
            ego2global_rotation = info['ego2global_rotation'],
        )
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        cam_positions = []
        focal_positions = []
        
        lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
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
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            # import pdb; pdb.set_trace()
            ego2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix)
            ego2cam_t = cam_info['sensor2ego_translation'] @ ego2cam_r.T
            ego2cam_rt = np.eye(4)
            ego2cam_rt[:3, :3] = ego2cam_r.T
            ego2cam_rt[3, :3] = -ego2cam_t
            
            
            cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            #cam_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            #focal_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])
        
        
        
        
        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                ego2lidar=ego2lidar,
                cam_positions=cam_positions,
                focal_positions=focal_positions,
                lidar2ego=lidar2ego,
            ))
        
        return input_dict


@DATASETS.register_module()
class nuScenesSceneDatasetLidarTraverse(NuScenesOccWorldDataset):
    def __init__(
        self,
        data_path,
        return_len,
        offset,
        imageset='train',
        nusc=None,
        times=1,
        test_mode=False,
        use_valid_flag=True,
        input_dataset='gts',
        output_dataset='gts',
    ):
        super().__init__(data_path, return_len, offset, 
                         imageset, nusc, times, 
                         test_mode, input_dataset, output_dataset)
        self.scene_lens = [l - self.return_len - self.offset for l in self.scene_lens]
        self.use_valid_flag = use_valid_flag
        self.CLASSES = [
            'noise', 'animal' ,'human.pedestrian.adult', 'human.pedestrian.child',
            'human.pedestrian.construction_worker',
            'human.pedestrian.personal_mobility',
            'human.pedestrian.police_officer',
            'human.pedestrian.stroller', 'human.pedestrian.wheelchair',
            'movable_object.barrier', 'movable_object.debris',
            'movable_object.pushable_pullable', 'movable_object.trafficcone',
            'static_object.bicycle_rack', 'vehicle.bicycle',
            'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car',
            'vehicle.construction', 'vehicle.emergency.ambulance',
            'vehicle.emergency.police', 'vehicle.motorcycle',
            'vehicle.trailer', 'vehicle.truck', 'flat.driveable_surface',
            'flat.other', 'flat.sidewalk', 'flat.terrain', 'flat.traffic_marking',
            'static.manmade', 'static.other', 'static.vegetation',
            'vehicle.ego'
        ]
        self.with_velocity = True
        self.with_attr = True
        self.box_mode_3d = Box3DMode.LIDAR
    
    def __len__(self):
        'Denotes the total number of samples'
        return sum(self.scene_lens)
    
    def __getitem__(self, index):
        for i, scene_len in enumerate(self.scene_lens):
            if index < scene_len:
                scene_name = self.scene_names[i]
                idx = index
                break
            else:
                index -= scene_len
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.input_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        input_occs = np.stack(occs)
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.output_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        output_occs = np.stack(occs)
        
        ## get meta infos
        metas_list = []
        for i in range(self.return_len + self.offset):
            input_dict = self.nusc_infos[scene_name][idx + i]
            ## get can bus
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

            example = self.pipeline(input_dict)
            metas_list.append(example)

        metas = self.union2one(metas_list)
        
        ## convert to tensor
        input, target = input_occs[:self.return_len], output_occs[self.offset:]
        input = torch.from_numpy(input).to(torch.int64)
        target = torch.from_numpy(target).to(torch.int64)

        data_dict = dict(
            input_occs=input,
            # target_occs=target,
            metas=metas
        )
        return data_dict
    
    def get_meta_info(self, scene_name, idx):
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
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        fut_valid_flag = info['valid_flag']
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        '''gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
                print(f'Warning: {cat} not in CLASSES')
        gt_labels_3d = np.array(gt_labels_3d)
        '''
        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        
        if self.with_attr:
            gt_fut_trajs = info['gt_agent_fut_trajs'][mask]
            gt_fut_masks = info['gt_agent_fut_masks'][mask]
            gt_fut_goal = info['gt_agent_fut_goal'][mask]
            gt_lcf_feat = info['gt_agent_lcf_feat'][mask]
            gt_fut_yaw = info['gt_agent_fut_yaw'][mask]
            attr_labels = np.concatenate(
                [gt_fut_trajs, gt_fut_masks, gt_fut_goal[..., None], gt_lcf_feat, gt_fut_yaw], axis=-1
            ).astype(np.float32)
        
        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            #gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            attr_labels=attr_labels,
            fut_valid_flag=fut_valid_flag,)
        
        return anns_results
        
        
        
    def get_image_info(self, scene_name, idx):
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        # import pdb; pdb.set_trace()
        input_dict = dict(
            sample_idx=info['token'],
            ego2global_translation = info['ego2global_translation'],
            ego2global_rotation = info['ego2global_rotation'],
        )
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        cam_positions = []
        focal_positions = []
        
        lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
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
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            # import pdb; pdb.set_trace()
            ego2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix)
            ego2cam_t = cam_info['sensor2ego_translation'] @ ego2cam_r.T
            ego2cam_rt = np.eye(4)
            ego2cam_rt[:3, :3] = ego2cam_r.T
            ego2cam_rt[3, :3] = -ego2cam_t
            
            
            cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            #cam_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            #focal_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])
        
        
        
        
        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                ego2lidar=ego2lidar,
                cam_positions=cam_positions,
                focal_positions=focal_positions,
                lidar2ego=lidar2ego,
            ))
        
        return input_dict
        