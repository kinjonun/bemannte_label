import pdb

import numpy as np
import mmcv
import random

ann_file = "/home/sun/MapTR/data/argoverse2/sensor/maptrv2_argo_geo_train.pkl"
load_interval = 1
# val: 23522;  test: 23542;  train: 108972.        geo_split:  val:23794  test: 23541, train: 108701
# data = mmcv.load(ann_file, file_format='pkl')
# samples = data['samples']
# data_infos = list(sorted(data['samples'], key=lambda e: e['timestamp']))

# print(len(data['samples']))
# ['e2g_translation', 'e2g_rotation', 'cams', 'lidar_path', 'timestamp', 'log_id', 'token', 'annotation', 'sample_idx']
# 'timestamp': '315966889759625000', 'log_id': '0526e68e-2ff1-3e53-b0f8-45df02e45a93', 'token': '0526e68e-2ff1-3e53-b0f8-45df02e45a93_315966889759625000',
# print(samples[0])

output_file = '/home/sun/bemapnet/assets/splits/argoverse2/av2_geo_val_list.txt'

# with open(output_file, 'w', encoding='utf-8') as f:
#     for i in range(len(samples)):
#         sample = samples[i]
#         timestamp = sample['timestamp']
#         f.write(timestamp + '\n')
nus_all_path = '/home/sun/Bev/BeMapNet/assets/splits/nuscenes/all.txt'
nus_val_path = '/home/sun/Bev/BeMapNet/assets/splits/nuscenes/val.txt'

old_train_scenes_path = '/home/sun/Bev/BeMapNet/assets/splits/argoverse2/scenes/old/train_scenes_list.txt'
new_train_scenes_path = '/home/sun/Bev/geographical-splits/near_extrapolation_splits/argoverse2/argo_geo_txts/train.txt'

old_test_scenes_path = '/home/sun/Bev/BeMapNet/assets/splits/argoverse2/scenes/old/test_scenes_list.txt'

geo_val_path = '/home/sun/Bev/BeMapNet/assets/splits/argoverse2/av2_geo_val_list.txt'
old_val_path = '/home/sun/Bev/BeMapNet/assets/splits/argoverse2/val_timestamp_list.txt'
def read_timestamps_from_file(file_path):
    """读取文件中的所有时间戳，并将其存储在一个集合中。"""
    with open(file_path, 'r') as file:
        timestamps = file.read().splitlines()
    return set(timestamps)

# origin_split = read_timestamps_from_file('/home/sun/bemapnet/assets/splits/argoverse2/av2_all_timestamp_list.txt')
old_train_scenes = read_timestamps_from_file(old_train_scenes_path)
new_train_scenes = read_timestamps_from_file(new_train_scenes_path)
old_test_scenes = read_timestamps_from_file(old_test_scenes_path)
old_val = read_timestamps_from_file(old_val_path)

overlap_val = read_timestamps_from_file('/home/sun/Bev/BeMapNet/assets/splits/argoverse2/scenes/overlap_val_und_val.txt')
# random_files = random.sample(overlap_val, 20)
# print(random_files)
# pdb.set_trace()
nus_all = read_timestamps_from_file(nus_all_path)
nus_val = read_timestamps_from_file(nus_val_path)

new_val_scenes = '/home/sun/Bev/BeMapNet/assets/splits/argoverse2/av2_geo_val_list.txt'
old_val_scenes = '/home/sun/Bev/BeMapNet/assets/splits/argoverse2/scenes/old/val_scenes_list.txt'
# aus_split =
# pdb.set_trace()
nus_geo_train = '/home/sun/Bev/geographical-splits/near_extrapolation_splits/nuscenes/samples/nusc_samples_geo_txts/train.txt'
nus_geo_val = '/home/sun/Bev/geographical-splits/near_extrapolation_splits/nuscenes/samples/nusc_samples_geo_txts/val.txt'
nus_geo_test = '/home/sun/Bev/geographical-splits/near_extrapolation_splits/nuscenes/samples/nusc_samples_geo_txts/test.txt'

j = 0
for token in open(new_val_scenes).readlines():
    # pdb.set_trace()
    timestamp = token.strip()
    if timestamp in old_val:
        j += 1
        print(timestamp)

print(j)


