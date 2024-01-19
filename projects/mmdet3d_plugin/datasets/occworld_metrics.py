'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-01-17 15:03:15
Email: haimingzhang@link.cuhk.edu.cn
Description: Compute the occupancy metrics. Borrowed from OccWorld.
'''

import numpy as np
from copy import deepcopy
import torch
import torch.distributed as dist
from mmdet.utils import get_root_logger


class MeanIoU:

    def __init__(self,
                 class_indices,
                 ignore_label: int,
                 label_str,
                 name
                 # empty_class: int
        ):
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        self.ignore_label = ignore_label
        self.label_str = label_str
        self.name = name

    def reset(self) -> None:
        self.total_seen = torch.zeros(self.num_classes).cuda()
        self.total_correct = torch.zeros(self.num_classes).cuda()
        self.total_positive = torch.zeros(self.num_classes).cuda()

    def _after_step(self, outputs, targets):
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]

        for i, c in enumerate(self.class_indices):
            self.total_seen[i] += torch.sum(targets == c).item()
            self.total_correct[i] += torch.sum((targets == c)
                                               & (outputs == c)).item()
            self.total_positive[i] += torch.sum(outputs == c).item()

    def _after_epoch(self):
        dist.all_reduce(self.total_seen)
        dist.all_reduce(self.total_correct)
        dist.all_reduce(self.total_positive)

        ious = []

        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i]
                                                   + self.total_positive[i]
                                                   - self.total_correct[i])
                ious.append(cur_iou.item())

        miou = np.mean(ious)
        logger = get_root_logger('INFO')
        # logger = MMLogger.get_current_instance()
        logger.info(f'Validation per class iou {self.name}:')
        for iou, label_str in zip(ious, self.label_str):
            logger.info('%s : %.2f%%' % (label_str, iou * 100))
        
        return miou * 100


class multi_step_MeanIou:
    def __init__(self,
                 class_indices,
                 ignore_label: int,
                 label_str,
                 name,
                 times=1):
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        self.ignore_label = ignore_label
        self.label_str = label_str
        self.name = name
        self.times = times
        
        self.reset()
        
    def reset(self) -> None:
        self.total_seen = torch.zeros(self.times, self.num_classes).cuda()
        self.total_correct = torch.zeros(self.times, self.num_classes).cuda()
        self.total_positive = torch.zeros(self.times, self.num_classes).cuda()
    
    def _after_step(self, outputses, targetses):
        
        assert outputses.shape[1] == self.times, f'{outputses.shape[1]} != {self.times}'
        assert targetses.shape[1] == self.times, f'{targetses.shape[1]} != {self.times}'
        for t in range(self.times):
            outputs = outputses[:,t, ...][targetses[:,t, ...] != self.ignore_label].cuda()
            targets = targetses[:,t, ...][targetses[:,t, ...] != self.ignore_label].cuda()
            for j, c in enumerate(self.class_indices):
                self.total_seen[t, j] += torch.sum(targets == c).item()
                self.total_correct[t, j] += torch.sum((targets == c)
                                                      & (outputs == c)).item()
                self.total_positive[t, j] += torch.sum(outputs == c).item()
    
    def _after_epoch(self):
        dist.all_reduce(self.total_seen)
        dist.all_reduce(self.total_correct)
        dist.all_reduce(self.total_positive)
        mious = []
        for t in range(self.times):
            ious = []
            for i in range(self.num_classes):
                if self.total_seen[t, i] == 0:
                    ious.append(1)
                else:
                    cur_iou = self.total_correct[t, i] / (self.total_seen[t, i]
                                                          + self.total_positive[t, i]
                                                          - self.total_correct[t, i])
                    ious.append(cur_iou.item())
            miou = np.mean(ious)
            logger = get_root_logger('INFO')
            logger.info(f'per class iou {self.name} at time {t}:')
            for iou, label_str in zip(ious, self.label_str):
                logger.info('%s : %.2f%%' % (label_str, iou * 100))
            logger.info(f'mIoU {self.name} at time {t}: %.2f%%' % (miou * 100))
            mious.append(miou * 100)
        return mious, np.mean(mious)
    

class OccWorldEvaluator(object):
    def __init__(self, eval_length=6):
        unique_label = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        unique_label_str = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', \
                            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck', \
                            'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation']
        
        self.CalMeanIou_sem = multi_step_MeanIou(
            unique_label, 
            -100, 
            unique_label_str, 
            'sem', 
            times=eval_length)
        self.CalMeanIou_vox = multi_step_MeanIou(
            [1], 
            -100, 
            ['occupied'], 
            'vox', 
            times=eval_length)
    
    def after_step(self, pred_occs, target_occs):
        target_occs_iou = deepcopy(target_occs)
        target_occs_iou[target_occs_iou != 17] = 1
        target_occs_iou[target_occs_iou == 17] = 0
        
        if isinstance(pred_occs, dict):
            self.CalMeanIou_sem._after_step(pred_occs['sem_pred'], target_occs)
            self.CalMeanIou_vox._after_step(pred_occs['iou_pred'], target_occs_iou)
        else:
            pred_iou = deepcopy(pred_occs)
            pred_iou[pred_iou != 17] = 1
            pred_iou[pred_iou == 17] = 0
            self.CalMeanIou_sem._after_step(pred_occs, target_occs)
            self.CalMeanIou_vox._after_step(pred_iou, target_occs_iou)
    
    def after_epoch(self):
        logger = get_root_logger('INFO')

        val_miou, _ = self.CalMeanIou_sem._after_epoch()
        val_iou, _ = self.CalMeanIou_vox._after_epoch()

        logger.info(f'Current val iou is {val_iou}')
        logger.info(f'Current val miou is {val_miou}')
        # logger.info(f'avg val iou is {(val_iou[1]+val_iou[3]+val_iou[5])/3}')
        # logger.info(f'avg val miou is {(val_miou[1]+val_miou[3]+val_miou[5])/3}')
