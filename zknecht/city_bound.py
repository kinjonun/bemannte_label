import numpy as np
import os
import pdb
import mmcv

city_list = ['WDC', 'MIA', 'PAO', 'PIT', 'ATX', 'DTW']
nus_city_list = ['singapore-onenorth', 'boston-seaport', 'singapore-queenstown', 'singapore-hollandvillage']

av2_pkl = "/home/sun/Bev/maptracker/datasets/av2/av2_map_infos_train_geosplit.pkl"

# for city in city_list:
#     data = mmcv.load(av2_pkl, file_format="pkl")
#     samples = data['samples']
#     x, y = 2000, 2000
#     for i in range(len(samples)):
#         sample = samples[i]
#         map_location = sample['map_location']
#         if map_location != city:
#             continue
#         e2g_translation = sample['e2g_translation']
#         if e2g_translation[0] < x:
#             x = e2g_translation[0]
#         if e2g_translation[1] < y:
#             y = e2g_translation[1]
#     print(f"'{city}': np.array([{x}, {y}]) - bev_radius,")


Nus_pkl = "/home/sun/Bev/maptracker/datasets/nuscenes/nuscenes_map_infos_val.pkl"
nus_data = mmcv.load(Nus_pkl, file_format="pkl")
for city in nus_city_list:
    x, y = -20000, -20000
    i = 0
    for data in nus_data:
        location = data['location']
        if location != city:
            continue
        e2g_translation = data['e2g_translation']
        if e2g_translation[0] > x:
            x = e2g_translation[0]
        if e2g_translation[1] > y:
            y = e2g_translation[1]
    print(f"'{city}': np.array([{x}, {y}]) - bev_radius,")

# Min train
# 'singapore-onenorth': np.array([118.57026480010829, 420.875645627154]) - bev_radius,
# 'boston-seaport': np.array([298.1007339647131, 328.68013429321843]) - bev_radius,
# 'singapore-queenstown': np.array([347.6363536016149, 862.9616321764047]) - bev_radius,
# 'singapore-hollandvillage': np.array([442.6842717189943, 902.7211942929146]) - bev_radius,

# max
# 'singapore-onenorth': np.array([1231.6395131271931, 1776.7152476390472]) - bev_radius,
# 'boston-seaport': np.array([2526.7565904062476, 1895.6798285992836]) - bev_radius,
# 'singapore-queenstown': np.array([2685.4630285066164, 3297.9240638856163]) - bev_radius,
# 'singapore-hollandvillage': np.array([2489.1602184455824, 2838.527681290493]) - bev_radius,


# test  max
# 'singapore-onenorth': np.array([1226.8376863460462, 1701.5241956192117]) - bev_radius,
# 'boston-seaport': np.array([2526.6375700990047, 1896.1430101392557]) - bev_radius,
# 'singapore-queenstown': np.array([1995.4277378137795, 3413.556410335427]) - bev_radius,
# 'singapore-hollandvillage': np.array([2459.2487881480365, 2835.4977534353775]) - bev_radius,


# min
# 'singapore-onenorth': np.array([276.4417818762576, 387.1475249899082]) - bev_radius,
# 'boston-seaport': np.array([418.11065560176917, 783.5572018906381]) - bev_radius,
# 'singapore-queenstown': np.array([347.65529809041254, 987.9988815492643]) - bev_radius,
# 'singapore-hollandvillage': np.array([608.5075861224198, 2007.316488052058]) - bev_radius,