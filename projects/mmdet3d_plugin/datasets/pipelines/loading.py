import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
import os
import torch


@PIPELINES.register_module()
class LoadOccGTFromFile(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    note that we read image in BGR style to align with opencv.imread
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
            self,
            data_root,
        ):
        self.data_root = data_root

    def __call__(self, results):
        # print(results.keys())
        if 'occ_gt_path' in results:
            occ_gt_path = results['occ_gt_path']
            occ_gt_path = os.path.join(self.data_root,occ_gt_path)

            occ_labels = np.load(occ_gt_path)
            semantics = occ_labels['semantics']
            mask_lidar = occ_labels['mask_lidar']
            mask_camera = occ_labels['mask_camera']
        else:
            semantics = np.zeros((200,200,16),dtype=np.uint8)
            mask_lidar = np.zeros((200,200,16),dtype=np.uint8)
            mask_camera = np.zeros((200, 200, 16), dtype=np.uint8)

        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera


        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)
    

def openscene_occ_to_voxel(occ_data,
                           point_cloud_range=[-50.0, -50.0, -4.0, 50.0, 50.0, 4.0], 
                           occupancy_size=[0.5, 0.5, 0.5],
                           occupancy_classes=11):
    if isinstance(occ_data, np.ndarray):
        occ_data = torch.from_numpy(occ_data)
    
    occ_data = occ_data.long()
    occ_xdim = int((point_cloud_range[3] - point_cloud_range[0]) / occupancy_size[0])
    occ_ydim = int((point_cloud_range[4] - point_cloud_range[1]) / occupancy_size[1])
    occ_zdim = int((point_cloud_range[5] - point_cloud_range[2]) / occupancy_size[2])

    voxel_num = occ_xdim * occ_ydim * occ_zdim

    gt_occupancy = (torch.ones(voxel_num, dtype=torch.long) * occupancy_classes).to(occ_data.device)
    gt_occupancy[occ_data[:, 0]] = occ_data[:, 1]

    gt_occupancy = gt_occupancy.reshape(occ_zdim, occ_ydim, occ_xdim)

    gt_occupancy = gt_occupancy.permute(2, 1, 0)
    return gt_occupancy


@PIPELINES.register_module()
class LoadOpenSceneOccupancy(object):
    """load occupancy GT data
       gt_type: index_class, store the occ index and occ class in one file with shape (n, 2)
    """
    def __call__(self, results):
        occ_gt_path = results['occ_gt_final_path']
        occ_gts = torch.from_numpy(np.load(occ_gt_path))  # (n, 2)

        results['voxel_semantics'] = openscene_occ_to_voxel(occ_gts).numpy()
        return results