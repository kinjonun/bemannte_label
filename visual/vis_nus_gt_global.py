import pdb
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import cv2
from PIL import Image, ImageDraw
from pyquaternion import Quaternion
from shapely.geometry import LineString, Point
from vis_av2_merged import get_prev2curr_vectors, plot_line
import torch
from tqdm import tqdm


def get_matrix(data_dict, inverse=False):

    e2g_trans = torch.tensor(data_dict['lidar_ego_pose_tran'], dtype=torch.float64)
    e2g_rot = torch.tensor(Quaternion(data_dict['lidar_ego_pose_rot']).rotation_matrix, dtype=torch.float64)

    matrix = torch.eye(4, dtype=torch.float64)
    if inverse:  # g2e_matrix
        matrix[:3, :3] = e2g_rot.T
        matrix[:3, 3] = -(e2g_rot.T @ e2g_trans)
    else:
        matrix[:3, :3] = e2g_rot
        matrix[:3, 3] = e2g_trans
    return matrix


def vis_nus_gt_local(samples, save_path, anno_path, scene):
    color = {0: 'g', 1: 'r', 2: 'b', 3: 'g', 4: "c", }
    save_path = osp.join(save_path, scene)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    for j, sample in enumerate(samples):
        gt_data = np.load(osp.join(anno_path, f"{sample}.npz"), allow_pickle=True)
        data_dict = {key: gt_data[key].tolist() for key in gt_data.files}
        ego_points = data_dict["ego_vectors"]
        pdb.set_trace()
        plt.figure(figsize=(100, 35))
        plt.xlim(-31, 31)
        plt.ylim(-16, 16)
        for item in ego_points:
            pts_type = item['type']
            pts = item['pts'][:, :2][:, ::-1]
            for i in range(len(pts) - 1):
                plt.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]], c=color[pts_type + 1])

        map_path = osp.join(save_path, f'local_{j}.png')
        plt.gca().set_aspect('equal')
        # plt.axis('off')
        plt.savefig(map_path, bbox_inches='tight', format='png', dpi=40)
        plt.close()
        # print(f'Saved {map_path}')


def main():
    anno_path = '/home/sun/Bev/BeMapNet/data/nuscenes/customer/bemapnet_location'
    car_img = Image.open('/home/sun/Bev/maptracker/resources/car-orange.png')
    vis_path = '/home/sun/Bev/BeMapNet/visual_global/original/nus'
    save_path = '/home/sun/Bev/BeMapNet/paper_imgs/nus_local'
    color = {0: 'g', 1: 'r', 2: 'b', 3: 'g', 4: "c", }

    with open('/home/sun/Bev/BeMapNet/data/nuscenes/nus_scene2sample.pkl', 'rb') as f:
        nus_scene2sample_dict = pickle.load(f)

    scene_list = list(nus_scene2sample_dict.keys())
    samples = nus_scene2sample_dict[scene_list[0]]
    # pdb.set_trace()
    # print(samples)
    # pdb.set_trace()
    # sample_data = np.load(osp.join(anno_path, f"{samples[0]}.npz"), allow_pickle=True)
    # sample_data_dict = {key: sample_data[key].tolist() for key in sample_data.files}
    # # dict_keys(['image_paths', 'trans', 'rots', 'intrins', 'semantic_mask', 'instance_mask', 'instance_mask8',
    # # 'ego_vectors', 'map_vectors', 'ctr_points', 'cam_ego_pose_trans', 'cam_ego_pose_rots', 'lidar_filename',
    # # 'lidar_tran', 'lidar_rot', 'lidar_ego_pose_tran', 'lidar_ego_pose_rot', 'lidar2global_r', 'lidar2global_t',
    # # 'lidar2cam_rts', 'lidar2img_rts', 'map_location'])
    # matrix = get_matrix(sample_data_dict)
    #
    # print(matrix)
    curr_data = np.load(osp.join(anno_path, f"{samples[-1]}.npz"), allow_pickle=True)
    curr_data_dict = {key: curr_data[key].tolist() for key in curr_data.files}
    # pdb.set_trace()
    curr_ego_points = curr_data_dict["ego_vectors"]
    curr_g2e_matrix = get_matrix(curr_data_dict, inverse=True)
    for scene in tqdm(scene_list[2:]):
        samples = nus_scene2sample_dict[scene]
        vis_nus_gt_local(samples, save_path, anno_path, scene)
    # vis_nus_gt_local(samples, save_path, anno_path, scene)
    # pdb.set_trace()
    rand_points, prev2curr_gt, prev2curr_gt_types = [], [], []
    for i, sample in enumerate(samples[-2::-1]):
        prev_data = np.load(osp.join(anno_path, f"{sample}.npz"), allow_pickle=True)
        prev_data_dict = {key: prev_data[key].tolist() for key in prev_data.files}
        prev_ego_points = prev_data_dict["ego_vectors"]

        ego_pts, ego_types = [], []
        for item in prev_ego_points:
            pts_type = item['type']
            pts = item['pts'][:, :2][:, ::-1]
            if len(pts) < 2:
                continue
            polyline = LineString(pts)
            distances = np.linspace(0, polyline.length, 50)
            sampled_points = np.array([list(polyline.interpolate(distance).coords) for distance in distances]).reshape(
                -1, 2)
            ego_pts.append(sampled_points)
            ego_types.append(pts_type)

        prev_e2g_matrix = get_matrix(prev_data_dict, False)
        prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix
        prev2curr_gt_vectors = get_prev2curr_vectors(ego_pts, prev2curr_matrix)
        # pdb.set_trace()
        prev2curr_gt.append(prev2curr_gt_vectors)
        prev2curr_gt_types.append(ego_types)

        for j, points in enumerate(prev2curr_gt_vectors):
            rand_points.append(np.array(points[0]))
            rand_points.append(np.array(points[-1]))

    plt.figure(figsize=(200, 70))
    for m in range(len(prev2curr_gt_types)):
        prev2curr_gt_vector = prev2curr_gt[m]
        prev2curr_gt_type = prev2curr_gt_types[m]
        for n in range(prev2curr_gt_vector.shape[0]):
            pts = prev2curr_gt_vector[n]
            pts_type = prev2curr_gt_type[n]
            for i in range(len(pts) - 1):
                plt.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]], c=color[pts_type + 1])

    map_path = osp.join(save_path, f'{scene}_gt_unmerged.png')
    plt.gca().set_aspect('equal')
    # plt.axis('off')
    plt.savefig(map_path, bbox_inches='tight', format='png', dpi=40)
    plt.close()
    print(f'Saved {map_path}')

    # for points in curr_ego_points:
    #     rand_points.append(np.array(points[0]))
    #     rand_points.append(np.array(points[-1]))
    ss = np.stack(rand_points)
    s1 = np.array([np.floor(min(ss[:, 0])), np.ceil(max(ss[:, 0]))], dtype=int)
    s2 = np.array([np.floor(min(ss[:, 1])), np.ceil(max(ss[:, 1]))], dtype=int)
    s3 = np.array([(s1[1]-s1[0]), (s2[1]-s2[0])], dtype=float)
    s4 = np.array([s1[0], s2[0]], dtype=float)
    print("s3, s4: ", s3, s4)
    # plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
    # plt.text(-15, 31, 'GT', color='red', fontsize=12)
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    main()
