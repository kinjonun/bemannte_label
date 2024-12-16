import pdb
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import os
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely.geometry import LineString
import cv2


def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords



folder_path = "/home/sun/Bev/BeMapNet/data/nuscenes/customer/bemapnet_test/0f71bfd0614643e48b74a1f7cd52129e.npz"
# folder_path_1 = '/home/sun/Bev/BeMapNet/data/nuscenes/customer/bemapnet_lidar/1b9a789e08bb4b7b89eacb2176c70840.npz'
dataset_path = '/media/sun/z/nuscenes/nuscenes'

data = np.load(os.path.join(folder_path), allow_pickle=True)
# data_1 = np.load(os.path.join(folder_path_1), allow_pickle=True)
# print(data['lidar2img_rts'])
# lidar2img_rts = data['lidar2img_rts']
ego_vectors = data['ego_vectors']
lidar2img_FRONT = data['lidar2img_rts'][1]
img_FRONT_path = data['image_paths'][1]
img_path = os.path.join(dataset_path, img_FRONT_path)
image = cv2.imread(img_path)
# print(img_FRONT_path)
line = ego_vectors[11]
# {'pts': array([[ 3.24756585, 15.        ],
#        [ 3.25267024, 14.00001303],
#        [ 3.25777463, 13.00002605],
#        [ 3.26287903, 12.00003908],
#        [ 3.26798342, 11.00005211],
#        [ 3.27308781, 10.00006514],
#        [ 3.2781922 ,  9.00007816]]), 'pts_num': 7, 'type': 0}

line_pts = line['pts']
# print(ego_vectors)
# pdb.set_trace()
line_ = LineString(line['pts'])
distances = np.linspace(0, line_.length, 200)
coords = np.array([list(line_.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
pts_num = coords.shape[0]
zeros = np.zeros((pts_num, 1))
zeros[:] = -1.4                         # 地面相对 lidar 的高度
ones = np.ones((pts_num, 1))

lidar_coords = np.concatenate([coords, zeros, ones], axis=1).transpose(1, 0)         # (4, 200)
pix_coords = perspective(lidar_coords, lidar2img_FRONT)

pix_coords = pix_coords.astype(np.int32)         # 将像素坐标转换为整数坐标（如果有必要）

cv2.polylines(image, [pix_coords], isClosed=False, color=255, thickness=8)      # 绘制多边形

# cv2.fillPoly(mask, [pix_coords], color=1)    # 如果想要填充多边形，可以使用 fillPoly

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()