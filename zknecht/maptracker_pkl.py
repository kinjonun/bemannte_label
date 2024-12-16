import numpy as np
import mmcv
import random
import pdb

anno_file = '/home/sun/Bev/maptracker/datasets/av2/av2_map_infos_train_newsplit.pkl'
data = mmcv.load(anno_file, file_format='pkl')       # dict_keys(['samples', 'id2map'])    train:117171
pdb.set_trace()
samples = data['samples']
# keys:['e2g_translation', 'e2g_rotation', 'cams', 'lidar_fpath', 'prev', 'token', 'log_id', 'scene_name', 'sample_idx']
# id2map: ['bb110668-5037-3c04-bd34-34cf1ace8d0f', '8beeb8db-28f9-396c-b752-17f906505948', ....      746
# data['id2map']['bb110668-5037-3c04-bd34-34cf1ace8d0f']: datasets/av2/train/bb110668-5037-3c04-bd34-34cf1ace8d0f/map/log_map_archive_bb110668-5037-3c04-bd34-34cf1ace8d0f____ATX_city_72134.json
# data_infos = list(sorted(data['samples'], key=lambda e: e['timestamp']))