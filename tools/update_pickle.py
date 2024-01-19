'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-01-16 11:41:31
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import os
import os.path as osp
import pickle


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    data_infos = data['infos']
    metadata = data['metadata']
    print(len(data_infos))
    return data_infos, metadata


def get_scene_sequence_data(data_infos):
    scene_name_list = []
    total_scene_seq = []
    curr_seq = []
    for idx, data in enumerate(data_infos):
        scene_token = data['scene_token']
        next_idx = min(idx + 1, len(data_infos) - 1)
        next_scene_token = data_infos[next_idx]['scene_token']

        curr_seq.append(data)

        if next_scene_token != scene_token:
            total_scene_seq.append(curr_seq)
            scene_name_list.append(scene_token)
            curr_seq = []

    total_scene_seq.append(curr_seq)
    scene_name_list.append(scene_token)
    return scene_name_list, total_scene_seq


def save_scene_pickle(pickle_file):
    from nuscenes import NuScenes

    data_infos, metadata = load_pickle(pickle_file)
    scene_token_list, total_scene_seq = get_scene_sequence_data(data_infos)
    print(len(scene_token_list), len(total_scene_seq))

    version = 'v1.0-trainval'
    data_root = 'data/occ3d-nus'
    nusc = NuScenes(version, data_root)

    data_dict = {}
    for idx, scene_token in enumerate(scene_token_list):
        scene_data = total_scene_seq[idx]
        scene_rec = nusc.get('scene', scene_token)
        scene_name = scene_rec['name']
        data_dict[scene_name] = scene_data
    
    results = dict()
    results['infos'] = data_dict
    results['metadata'] = metadata

    save_path = osp.join(data_root, 'occ_infos_temporal_train_scene.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    pickle_file1 = "data/occ3d-nus/occ_infos_temporal_train.pkl"
    # pickle_file1 = "data/occ3d-nus/occ_infos_temporal_val.pkl"
    save_scene_pickle(pickle_file1)

    exit(0)

    
    data_infos1 = load_pickle(pickle_file1)

    pickle_file2 = "data/nuscenes_infos_train_temporal_v3_scene.pkl"
    data_infos2 = load_pickle(pickle_file2)

    print(len(data_infos1))

    

    scene_data = data_infos2['scene-0001']
    for item in scene_data:
        sample_token = item['token']

        data1 = None
        for data in data_infos1:
            if data['token'] == sample_token:
                print(data['token'])
                data1 = data
                break
        
        data2 = item
        print(data1['timestamp'], data2['timestamp'])
