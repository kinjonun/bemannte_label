import pdb
import mmcv
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os.path as osp


val_map_file = "/hrmap/av2_geosplit_12_10_epoch_3.pkl"
img_path = '/home/sun/Bev/BeMapNet/hrmap'
av2_cities = ['WDC', 'MIA', 'PAO', 'PIT', 'ATX', 'DTW']
try:
    val_data = torch.load(val_map_file)
    # WDC = data['WDC']   # [11545, 10030, 3]
    # MIA = data['MIA']
    # PAO = data['PAO'].cpu().numpy()   # 069cc46d-38bb-309d-88cf-296a3d0c0820   5159 * 3409
    # PIT = data['PIT'].cpu().numpy()
    # ATX = data['ATX'].cpu().numpy()
    # DTW = data['DTW'].cpu().numpy()
    # for i in range(PAO.shape[0]):
    #     for j in range(PAO.shape[1]):
    #         point = PAO[i, j][0]
    #         if point > 100:
    #             print(i, j, point)
    # for city in av2_cities:
    for city in av2_cities:
        city_map = val_data[city].cpu().numpy()
        sidewalk_cmap = LinearSegmentedColormap.from_list('sidewalk', ['lightblue', 'blue'])
        boundary_cmap = LinearSegmentedColormap.from_list('boundary', ['lightgreen', 'green'])
        lane_cmap = LinearSegmentedColormap.from_list('lane', ['lightcoral', 'red'])

        sidewalk_mask = (city_map[:, :, 1] > 100)
        boundary_mask = (city_map[:, :, 2] > 100)
        lane_mask = (city_map[:, :, 0] > 100)

        image = np.ones((city_map.shape[0], city_map.shape[1], 3))
        image[sidewalk_mask] = sidewalk_cmap((city_map[:, :, 1][sidewalk_mask] - 100) / 30)[:, :3]
        image[boundary_mask] = boundary_cmap((city_map[:, :, 2][boundary_mask] - 100) / 30)[:, :3]
        image[lane_mask] = lane_cmap((city_map[:, :, 0][lane_mask] - 100) / 30)[:, :3]

        image = np.flip(np.transpose(image, (1, 0, 2)), axis=0)
        plt.figure(figsize=(200, 200 * city_map.shape[1]/city_map.shape[0]))
        plt.gca().set_aspect('equal')
        plt.imshow(image)

        map_path = osp.join(img_path, f'{city}_map.png')
        plt.savefig(map_path, bbox_inches='tight', format='png', dpi=120)
        print(f'Saved {city}_map.png')
        plt.close()
    # PIT = data['PIT']   # 0526e68e-2ff1-3e53-b0f8-45df02e45a93   3404 * 2626
    # ATX = data['ATX']
    # DTW = data['DTW']
    # print("Data loaded successfully:", data)
except Exception as e:
    print(f"Error loading data: {e}")
