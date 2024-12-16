import pdb
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import os
import random
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely.geometry import LineString
import cv2

color = {0: 'orange', 1: 'blue', 2: 'red'}

# '02a00399-3857-444e-8db3-a8f58489c394_315966070559696000.npz'
# '02a00399-3857-444e-8db3-a8f58489c394_315966086160209000.npz'

def av2_npz():
    folder_path = '/home/sun/Bev/BeMapNet/data/argoverse2/customer'
    file_names = os.listdir(folder_path)
    # random_file = random.choice(file_names)
    random_file = '315968330760061000.npz'  # 11332: 315966329559911000  无boundary   366：315970547759868000 无ped_crossing
    # random_file = '/home/sun/Bev/maptracker/zknecht/customer/val/315968330760061000.npz'
    # print("av2_npz:", random_file)   # 0a8a4cfa-4902-3a76-8301-08698d6290a2_315968251759912000

    file_path = os.path.join(folder_path, random_file)  # bb9be2e6-8f0e-3bb3-8bb9-5d9aa9df384d_315967053759951000
    file_path = '/home/sun/Bev/BeMapNet/data/argoverse2/geosplits_interval_1/02a00399-3857-444e-8db3-a8f58489c394_315966070559696000.npz'
    pred_npz_path = '/home/sun/Bev/BeMapNet/outputs/bemapnet_av2_res50/0912/evaluation/results/0aa4e8f5-2f9a-39a1-8f80-c2fdde4405a2_315971843759817000.npz'
    data = np.load(file_path, allow_pickle=True)
    # ['input_dict', 'instance_mask', 'instance_mask8', 'semantic_mask', 'ctr_points', 'ego_points', 'map_vectors']
    data_dict = {key: data[key].tolist() for key in data.files}

    input_dict = data_dict["input_dict"]
    ego2global_translation = input_dict["ego2global_translation"]
    ego2global_rotation = input_dict["ego2global_rotation"]
    seq_info = input_dict["seq_info"]
    # print(input_dict['map_geoms']['ped_crossing'])
    pdb.set_trace()  # 'a47ba6a9-ffa1-3979-bb40-512339284b8b'
    timestamp = "av2 "  # + input_dict["timestamp"]
    # intrinsic = np.stack([np.eye(3) for _ in range(len(input_dict["camera_intrinsics"]))], axis=0)
    # camera_intrinsics = np.array(input_dict['camera_intrinsics'])
    # intrinsic[:, :, :] = camera_intrinsics[:, :3, :3]

    instance_mask8 = data_dict["instance_mask8"]
    # print("instance_mask8.shape: ", len(instance_mask8))
    ctr_points = data_dict["ctr_points"]
    ego_points = data_dict["ego_points"]
    # print("ctr_points:", ctr_points)
    return ctr_points, ego_points, instance_mask8, timestamp


def nuscenes_npz():
    folder_path_1 = '/home/sun/Bev/BeMapNet/data/nuscenes/customer/bemapnet_location'
    folder_path = '/home/sun/Bev/BeMapNet/data/nuscenes/customer/bemapnet'
    file_names = os.listdir(folder_path)
    # random_file = random.choice(file_names)
    # print("nuscenes_npz:", random_file)
    # random_file = "8fa07b104b904d22970d2f8946a0b771.npz"
    random_file = "80d56e801c7e465995bdb116b3e678aa.npz"
    # random_file = 'fd8420396768425eabec9bdddf7e64b6.npz'

    file_path = os.path.join(folder_path, random_file)
    data = np.load(file_path, allow_pickle=True)
    data = {key: data[key].tolist() for key in data.files}

    file_path_lidar = os.path.join(folder_path_1, random_file)
    data_lidar = np.load(file_path_lidar, allow_pickle=True)
    data_lidar = {key: data_lidar[key].tolist() for key in data_lidar.files}
    # ['image_paths', 'trans', 'rots', 'intrins', 'semantic_mask', 'instance_mask', 'instance_mask8', 'ego_vectors',
    # 'map_vectors', 'ctr_points', 'cam_ego_pose_trans', 'cam_ego_pose_rots', 'lidar_filename', 'lidar_tran',
    # 'lidar_rot', 'lidar_ego_pose_tran', 'lidar_ego_pose_rot']
    # pdb.set_trace()
    image_paths = data['image_paths']
    # print(image_paths)
    trans = data['trans']
    rots = np.zeros([6, 3, 3], )
    # for i in range(len(data['rots'])):
    #     rots[i] = Quaternion(data['rots'][i]).rotation_matrix  # ["camera2ego"]
    # rots_Quat = data['rots']
    # intrins = data['intrins']
    #
    # lidar_rot = Quaternion(data['lidar_rot']).rotation_matrix  # 四元数   lidar2ego
    # lidar_rot_Quat = data['lidar_rot']
    # lidar_tran = data['lidar_tran']

    # cam_ego_pose_trans = data['cam_ego_pose_trans']  # ["came_go2global"]
    # cam_ego_pose_rots = np.zeros([6, 3, 3])
    # for i in range(len(data['cam_ego_pose_rots'])):
    #     cam_ego_pose_rots[i] = Quaternion(data['cam_ego_pose_rots'][i]).rotation_matrix
    # cam_ego_pose_rots_Quat = data['cam_ego_pose_rots']

    # lidar_ego_pose_tran = data['lidar_ego_pose_tran']
    # lidar_ego_pose_rot = Quaternion(data['lidar_ego_pose_rot']).rotation_matrix  # ego2global
    # lidar_ego_pose_rot_Quat = data['lidar_ego_pose_rot']
    # # pdb.set_trace()
    # ego2global = np.eye(4)
    # ego2global[:3, :3] = lidar_ego_pose_rot
    # ego2global[:3, 3] = lidar_ego_pose_tran
    # lidar2ego = np.eye(4)
    # lidar2ego[:3, :3] = lidar_rot
    # lidar2ego[:3, 3] = lidar_tran
    # lidar2global = ego2global @ lidar2ego

    semantic_mask = data['semantic_mask']  # (3, 400, 200)
    instance_mask = data['instance_mask']  # (3, 400, 200)
    instance_mask8 = data['instance_mask8']
    instance_mask8_1 = data_lidar['instance_mask8']

    # label = instance_mask8_1 == instance_mask8
    # if False in label:
    #     print(1)
    # pdb.set_trace()
    ego_vectors = data['ego_vectors']
    ctr_points = data['ctr_points']
    ctr_points_lidar = data_lidar['ctr_points']
    ego_vectors_lidar = data_lidar['ego_vectors']
    map_vectors = data['map_vectors']
    # pdb.set_trace()
    flip_ctr_points = []
    for i in range(len(data['ctr_points'])):
        point = np.zeros_like(data['ctr_points'][i]['pts'])
        for j in range(point.shape[0]):
            point[j, 0] = data['ctr_points'][i]['pts'][j, 0]
            point[j, 1] = 60 - data['ctr_points'][i]['pts'][j, 1]
        flip_ctr_points.append({'pts': point, 'pts_num': len(point), 'type': data['ctr_points'][i]['type']})

    flip_ins_mask = []
    for i in range(len(data['instance_mask8'])):
        mask = np.flip(data['instance_mask8'][i], axis=0)
        flip_ins_mask.append(mask)

    # print('ego_vectors', flip_ins_mask[2])
    # print('ctr_points', data_1['image_paths'])
    return ctr_points, ego_vectors, instance_mask8, instance_mask8_1, random_file


def plot_ctr_points(ctr_points, ego_vectors, timestamp):
    plt.figure(figsize=(4, 8))
    # plt.title(timestamp)
    car_img = Image.open('/home/sun/Bev/BeMapNet/assets/figures/lidar_car.png')
    plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
    for item in ctr_points:
        pts = item['pts']
        y = [pt[0] - 15 for pt in pts]
        x = [-pt[1] + 30 for pt in pts]
        plt.scatter(y, x, c=color[item['type']])

    # for item in ego_vectors:
    #     pts = item['pts']
    #     for i in range(len(pts) - 1):
    #         plt.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]], c=color[item['type']])


def plot_instance_mask(instance_mask8, instance_mask8_1, timestamp):
    plt.figure(figsize=(8, 12))
    # plt.title(timestamp)
    plt.subplot(2, 3, 1)
    plt.imshow(instance_mask8[0], cmap='gray')
    plt.subplot(2, 3, 2)
    # print("instance_mask8:", len(instance_mask8[0]))
    # print("instance_mask8[1]", instance_mask8[1])
    plt.imshow(instance_mask8[1], cmap='gray')
    plt.subplot(2, 3, 3)
    plt.imshow(instance_mask8[2], cmap='gray')
    plt.subplot(2, 3, 4)
    plt.imshow(instance_mask8_1[0], cmap='gray')
    plt.subplot(2, 3, 5)
    plt.imshow(instance_mask8_1[1], cmap='gray')
    plt.subplot(2, 3, 6)
    plt.imshow(instance_mask8_1[2], cmap='gray')


# av2_ctr_points, av2_ego_points, av2_instance_mask8, av2_timestamp = av2_npz()
# cross = av2_instance_mask8[2]
# for i in range(len(cross)):
#     for j in range(len(cross[i])):
#         aa = np.int16(cross[i][j])
#         if aa > 0:
#             print(aa)
#         pdb.set_trace()
# print('av2_ctr_points', av2_ctr_points)
# plot_ctr_points(av2_ctr_points, av2_ego_points, av2_timestamp)
# plot_instance_mask(av2_instance_mask8, av2_instance_mask8, av2_timestamp)
# plt.show()
#
# nus_ctr_points, nus_ego_points, nus_instance_mask8, instance_mask8_1, nus_timestamp = nuscenes_npz()
# # print(nus_timestamp)
# plot_ctr_points(nus_ctr_points, nus_ego_points, nus_timestamp)
# plot_instance_mask(nus_instance_mask8, instance_mask8_1, nus_timestamp)
# plt.show()

ctr_points, ego_points, instance_mask8, timestamp = av2_npz()
