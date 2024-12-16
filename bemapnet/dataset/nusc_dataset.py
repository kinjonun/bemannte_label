import os
import pdb
import torch
import numpy as np
from PIL import Image
from copy import deepcopy
from skimage import io as skimage_io
from torch.utils.data import Dataset
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from nuscenes import NuScenes
import matplotlib.pyplot as plt
import cv2
from shapely.geometry import LineString, box


class NuScenesMapDataset(Dataset):
    def __init__(self, img_key_list, map_conf, ida_conf, bezier_conf, transforms, data_split="training"):
        super().__init__()
        self.img_key_list = img_key_list
        self.map_conf = map_conf
        self.ida_conf = ida_conf
        self.bez_conf = bezier_conf
        self.ego_size = map_conf["ego_size"]
        self.mask_key = map_conf["mask_key"]
        self.nusc_root = map_conf["nusc_root"]
        self.anno_root = map_conf["anno_root"]
        self.split_dir = map_conf["split_dir"]
        self.num_degree = bezier_conf["num_degree"]
        self.max_pieces = bezier_conf["max_pieces"]
        self.max_instances = bezier_conf["max_instances"]
        self.split_mode = 'train' if data_split == "training" else 'val'
        split_path = os.path.join(self.split_dir, f'{self.split_mode}.txt')
        self.tokens = [token.strip() for token in open(split_path).readlines()]
        self.transforms = transforms

    def __getitem__(self, idx: int):
        token = self.tokens[idx]
        sample = np.load(os.path.join(self.anno_root, f'{token}.npz'), allow_pickle=True)
        resize_dims, crop, flip, rotate = self.sample_ida_augmentation()
        images, ida_mats = [], []
        for im_view in self.img_key_list:
            for im_path in sample['image_paths']:
                if im_path.startswith(f'samples/{im_view}/'):
                    im_path = os.path.join(self.nusc_root, im_path)
                    img = skimage_io.imread(im_path)
                    img, ida_mat = self.img_transform(img, resize_dims, crop, flip, rotate)
                    images.append(img)
                    ida_mats.append(ida_mat)
        # pdb.set_trace()

        extrinsic = np.stack([np.eye(4) for _ in range(sample["trans"].shape[0])], axis=0)
        rots = []
        if sample['rots'].shape[1] == 4:
            for rot in sample['rots']:
                extrin = Quaternion(rot).rotation_matrix
                rots.append(extrin)
            extrinsic[:, :3, :3] = np.array(rots)
        else:
            extrinsic[:, :3, :3] = sample["rots"]
        extrinsic[:, :3, 3] = sample["trans"]
        intrinsic = sample['intrins']

        map_location = str(sample['map_location'])
        lidar2global = np.eye(4)
        lidar2global[:3, :3] = sample['lidar2global_r']
        lidar2global[:3, 3] = sample['lidar2global_t']

        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(sample['lidar_ego_pose_rot']).rotation_matrix
        ego2global[:3, 3] = sample['lidar_ego_pose_tran']

        # image_paths = torch.tensor(sample['image_paths'].tolist())
        ctr_points = np.zeros((self.max_instances, max(self.max_pieces) * max(self.num_degree) + 1, 2), dtype=np.float)
        ins_labels = np.zeros((self.max_instances, 3), dtype=np.int16) - 1

        for ins_id, ctr_info in enumerate(sample['ctr_points']):
            cls_id = int(ctr_info['type'])
            ctr_pts_raw = np.array(ctr_info['pts'])
            max_points = self.max_pieces[cls_id] * self.num_degree[cls_id] + 1
            num_points = max_points if max_points <= ctr_pts_raw.shape[0] else ctr_pts_raw.shape[0]
            assert num_points >= self.num_degree[cls_id] + 1
            ctr_points[ins_id][:num_points] = np.array(ctr_pts_raw[:num_points])
            ins_labels[ins_id] = [cls_id, (num_points - 1) // self.num_degree[cls_id] - 1, num_points]

        masks = sample[self.mask_key]

        if flip:
            new_order = [2, 1, 0, 5, 4, 3]
            img_key_list = [self.img_key_list[i] for i in new_order]
            images = [images[i] for i in new_order]
            ida_mats = [ida_mats[i] for i in new_order]
            extrinsic = [extrinsic[i] for i in new_order]
            intrinsic = [intrinsic[i] for i in new_order]
            masks = [np.flip(mask, axis=1) for mask in masks]
            ctr_points = self.point_flip(ctr_points, ins_labels, self.ego_size)
        img_metas = {'map_location': map_location, 'lidar2global': lidar2global}
        item = dict(
            images=images, targets=dict(masks=masks, points=ctr_points, labels=ins_labels),
            extrinsic=np.stack(extrinsic), intrinsic=np.stack(intrinsic), ida_mats=np.stack(ida_mats),
            extra_infos=dict(token=token, img_key_list=self.img_key_list, map_size=self.ego_size, do_flip=flip,
                             img_metas=img_metas),
        )
        if self.transforms is not None:
            item = self.transforms(item)
        return item

    def __len__(self):
        return len(self.tokens)

    def sample_ida_augmentation(self):
        """Generate ida augmentation values based on ida_config."""
        resize_dims = w, h = self.ida_conf["resize_dims"]
        crop = (0, 0, w, h)
        if self.ida_conf["up_crop_ratio"] > 0:
            crop = (0, int(self.ida_conf["up_crop_ratio"] * h), w, h)
        flip, color, rotate_ida = False, False, 0
        if self.split_mode == "train":
            if self.ida_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            if self.ida_conf["rot_lim"]:
                assert isinstance(self.ida_conf["rot_lim"], (tuple, list))
                rotate_ida = np.random.uniform(*self.ida_conf["rot_lim"])
        return resize_dims, crop, flip, rotate_ida

    def img_transform(self, img, resize_dims, crop, flip, rotate):
        img = Image.fromarray(img)
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        W, H = img.size
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        scales = torch.tensor([resize_dims[0] / W, resize_dims[1] / H])
        ida_rot *= torch.Tensor(scales)
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = ida_rot.new_zeros(3, 3)
        ida_mat[2, 2] = 1
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return np.asarray(img), ida_mat

    @staticmethod
    def point_flip(points, labels, map_shape):

        def _flip(pts):
            pts[:, 0] = map_shape[1] - pts[:, 0]
            return pts.copy()

        points_ret = deepcopy(points)
        for ins_id in range(points.shape[0]):
            end = labels[ins_id, 2]
            points_ret[ins_id][:end] = _flip(points[ins_id][:end])

        return points_ret

    @staticmethod
    def get_rot(h):
        return torch.Tensor([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])


def depth_transform(cam_depth, img_shape, resize_dims, crop, flip, rotate):   # (900, 1600)
    """Transform depth based on ida augmentation configuration.

    Args:
        cam_depth (np array): Nx3, 3: x,y,d.
        resize (float): Resize factor.
        resize_dims (list): Final dimension.
        crop (list): x1, y1, x2, y2
        flip (bool): Whether to flip.
        rotate (float): Rotation value.

    Returns:
        np array: [h/down_ratio, w/down_ratio, d]
    """
    # pdb.set_trace()
    resize_dims = np.array(resize_dims)[::-1]
    H, W = resize_dims                                 # (512, 896)
    resize_h = H /img_shape[0]
    resize_w = W /img_shape[1]
    cam_depth[:, 0] = cam_depth[:, 0] * resize_w       # (3379, 3)
    cam_depth[:, 1] = cam_depth[:, 1] * resize_h       # (3379, 3)
    cam_depth[:, 0] -= crop[0]
    cam_depth[:, 1] -= crop[1]                         # (3379, 3)
    if flip:
        cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

    cam_depth[:, 0] -= W / 2.0                         # 移动原点到中心便于做旋转
    cam_depth[:, 1] -= H / 2.0
    h = rotate / 180 * np.pi
    rot_matrix = [[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)],]
    cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T
    cam_depth[:, 0] += W / 2.0
    cam_depth[:, 1] += H / 2.0

    depth_coords = cam_depth[:, :2].astype(np.int16)

    depth_map = np.zeros((crop[3]-crop[1], crop[2]-crop[0]))    # [384, 896]
    valid_mask = ((depth_coords[:, 1] < resize_dims[0]) & (depth_coords[:, 0] < resize_dims[1])
                  & (depth_coords[:, 1] >= 0) & (depth_coords[:, 0] >= 0))
    depth_map[depth_coords[valid_mask, 1], depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

    return torch.Tensor(depth_map)


def map_pointcloud_to_image(lidar_points, img, lidar_calibrated_sensor, lidar_ego_pose, cam_calibrated_sensor,
                            cam_ego_pose,  min_dist: float = 0.0,):

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    lidar_points = LidarPointCloud(lidar_points.T)
    lidar_points.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_calibrated_sensor['translation']))

    # Second step: transform from ego to the global frame.
    lidar_points.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_ego_pose['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    lidar_points.translate(-np.array(cam_ego_pose['translation']))
    lidar_points.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    lidar_points.translate(-np.array(cam_calibrated_sensor['translation']))
    lidar_points.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud. Grab the depths (camera frame z axis points away from the camera).
    depths = lidar_points.points[2, :]
    coloring = depths
    # pdb.set_trace()
    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(lidar_points.points[:3, :], np.array(cam_calibrated_sensor['camera_intrinsic']), normalize=True)  # (3, 34752)

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make sure points are at least 1m in front of the camera to
    # avoid seeing the lidar points on the camera casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    # mask = np.logical_and(mask, points[0, :] < img.size[0] - 1)      # 1600
    mask = np.logical_and(mask, points[0, :] < img[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    # mask = np.logical_and(mask, points[1, :] < img.size[1] - 1)      # 900
    mask = np.logical_and(mask, points[1, :] < img[0] - 1)

    points = points[:, mask]         # (3, 3379)
    coloring = coloring[mask]        # (3379,)
    return points, coloring


def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords


def line_ego_to_pvmask(line_ego, mask, lidar2feat, color=1, thickness=1, z=-1.6):
    # pdb.set_trace()
    distances = np.linspace(0, line_ego.length, 200)
    coords = np.array([list(line_ego.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
    pts_num = coords.shape[0]                # 200
    zeros = np.zeros((pts_num, 1))
    zeros[:] = z
    ones = np.ones((pts_num, 1))
    lidar_coords = np.concatenate([coords, zeros, ones], axis=1).transpose(1, 0)     # (4, 200)
    pix_coords = perspective(lidar_coords, lidar2feat)
    cv2.polylines(mask, np.int32([pix_coords]), False, color=color, thickness=thickness)


class NuScenesMapDatasetDepth(Dataset):
    def __init__(self, img_key_list, map_conf, ida_conf, bezier_conf, aux_seg_cfg, transforms, data_split="training"):
        super().__init__()
        self.img_key_list = img_key_list
        self.map_conf = map_conf
        self.ida_conf = ida_conf
        self.bez_conf = bezier_conf
        self.ego_size = map_conf["ego_size"]
        self.mask_key = map_conf["mask_key"]
        self.nusc_root = map_conf["nusc_root"]
        self.anno_root = map_conf["anno_root"]
        self.split_dir = map_conf["split_dir"]
        self.num_degree = bezier_conf["num_degree"]
        self.max_pieces = bezier_conf["max_pieces"]
        self.max_instances = bezier_conf["max_instances"]
        self.split_mode = 'train' if data_split == "training" else 'val'
        split_path = os.path.join(self.split_dir, f'{self.split_mode}.txt')
        # split_path = os.path.join('/home/sun/Bev/BeMapNet/assets/splits/nuscenes/ein.txt')
        self.tokens = [token.strip() for token in open(split_path).readlines()]
        self.transforms = transforms
        self.return_depth = True
        self.aux_seg = aux_seg_cfg
        self.feat_down_sample = aux_seg_cfg["feat_down_sample"]

    def __getitem__(self, idx: int):
        token = self.tokens[idx]
        sample = np.load(os.path.join(self.anno_root, f'{token}.npz'), allow_pickle=True)
        resize_dims, crop, flip, rotate = self.sample_ida_augmentation()   # (896, 512), (0, 128, 896, 512),
        # pdb.set_trace()

        map_location = str(sample['map_location'])
        images, ida_mats, lidar_depth = [], [], []
        lidar_filename = np.array(sample['lidar_filename']).tolist()
        lidar_calibrated_sensor = dict(
            rotation=sample['lidar_rot'],                 # 四元数    lidar2ego
            translation=sample['lidar_tran'])
        lidar_ego_pose = dict(
            rotation=sample['lidar_ego_pose_rot'],        # 四元数    ego2global
            translation=sample['lidar_ego_pose_tran'])

        intrinsic = sample['intrins']
        rots = []
        for rot in sample['rots']:
            extrin = Quaternion(rot).rotation_matrix
            rots.append(extrin)
        extrinsic = np.stack([np.eye(4) for _ in range(sample["trans"].shape[0])], axis=0)
        extrinsic[:, :3, :3] = np.array(rots)
        extrinsic[:, :3, 3] = sample["trans"]

        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(sample['lidar_ego_pose_rot']).rotation_matrix
        ego2global[:3, 3] = sample['lidar_ego_pose_tran']

        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = np.array(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
        lidar2ego[:3, 3] = lidar_calibrated_sensor['translation']
        # lidar2global_r = ego2global[:3, :3] @ lidar2ego[:3, :3]
        # lidar2global_t = ego2global[:3, 3] + ego2global[:3, :3] @ lidar2ego[:3, 3]

        lidar2global = np.eye(4)
        lidar2global[:3, :3] = sample['lidar2global_r']
        lidar2global[:3, 3] = sample['lidar2global_t']

        lidar2cam_rts = sample['lidar2cam_rts']
        lidar2img_rts = sample['lidar2img_rts']
        ego_vectors = sample['ego_vectors']

        lidar_points = np.fromfile(os.path.join(self.nusc_root, lidar_filename), dtype=np.float32,
                                   count=-1).reshape(-1, 5)[..., :4]
        # pdb.set_trace()
        cam_ego2global = np.stack([np.eye(4) for _ in range(sample["trans"].shape[0])], axis=0)
        # lidar2cam_rts = np.stack([np.eye(4) for _ in range(sample["trans"].shape[0])], axis=0)
        # lidar2img_rts = np.stack([np.eye(4) for _ in range(sample["trans"].shape[0])], axis=0)

        # pdb.set_trace()
        for i in range(len(sample['image_paths'])):
            im_path = sample['image_paths'][i]
            im_path = os.path.join(self.nusc_root, im_path)
            img = skimage_io.imread(im_path)
            cam_calibrated_sensor = dict(
                rotation=sample['rots'][i],                    # 四元数    ["camera2ego"]
                translation=sample['trans'][i],
                camera_intrinsic=sample['intrins'][i])         # [3, 3]
            cam_ego_pose = dict(
                rotation=sample['cam_ego_pose_rots'][i],
                translation=sample['cam_ego_pose_trans'][i])
            cam_ego2global_r = Quaternion(sample['cam_ego_pose_rots'][i]).rotation_matrix
            cam_ego2global_t = sample['cam_ego_pose_trans'][i]
            cam_ego2global[i, :3, :3] = cam_ego2global_r               # ["cam_ego2global"]
            cam_ego2global[i, :3, 3] = cam_ego2global_t
            # # pdb.set_trace()
            # cam2global_r = cam_ego2global_r @ Quaternion(sample['rots'][i]).rotation_matrix
            # cam2global_t = cam_ego2global_t + cam_ego2global_r @ sample['trans'][i]
            #
            # lidar2cam_r = np.linalg.inv(cam2global_r) @ lidar2global_r
            # lidar2cam_t = cam2global_r.T @ (lidar2global_t - cam2global_t)
            # lidar2cam_rts[i, :3, :3] = lidar2cam_r
            # lidar2cam_rts[i, :3, 3] = lidar2cam_t
            #
            # viewpad = np.eye(4)
            # viewpad[:intrinsic[i].shape[0], :intrinsic[i].shape[1]] = intrinsic[i]
            # lidar2img_rt = (viewpad @ lidar2cam_rts[i])
            # lidar2img_rts[i] = lidar2img_rt

            # pdb.set_trace()
            if self.return_depth:
                pts_img, depth = map_pointcloud_to_image(lidar_points.copy(), img.shape[:2], lidar_calibrated_sensor.copy(),
                                                     lidar_ego_pose.copy(), cam_calibrated_sensor, cam_ego_pose)
                point_depth = np.concatenate([pts_img[:2, :].T, depth[:, None]], axis=1).astype(np.float32)
                point_depth_augmented = depth_transform(point_depth, img.shape[:2], resize_dims, crop, flip, rotate)
                lidar_depth.append(point_depth_augmented)

            img, ida_mat = self.img_transform(img, resize_dims, crop, flip, rotate)
            images.append(img)
            ida_mats.append(ida_mat)

        # for i in range(len(lidar_depth)):
        #     plt.imshow(lidar_depth[i], cmap='hot')
        #     plt.colorbar()  # 添加颜色条
        #     plt.show()  # 显示图片
        # pdb.set_trace()

        num_cam = len(sample['image_paths'])
        img_size = images[0].shape[:2]
        # pdb.set_trace()
        if self.aux_seg['pv_seg']:
            vector_num_list = {cls_idx: [] for cls_idx in range(3)}       # map-type -> list
            for vector in ego_vectors:
                if vector['pts_num'] >= 2:
                    vector_num_list[vector['type']].append(LineString(vector['pts'][:vector['pts_num']]))

            gt_pv_semantic_mask = np.zeros((num_cam, 1, img_size[0] // self.feat_down_sample,
                                            img_size[1] // self.feat_down_sample), dtype=np.uint8)
            lidar2img = lidar2img_rts
            scale_factor = np.eye(4)
            scale_factor[0, 0] *= 1 / self.feat_down_sample
            scale_factor[1, 1] *= 1 / self.feat_down_sample
            lidar2feat = [scale_factor @ l2i for l2i in lidar2img]

            for instance_type, instance in vector_num_list.items():
                if instance_type != -1:
                    for line_ego in instance:
                        # pdb.set_trace()
                        if line_ego.geom_type == 'LineString':
                            for cam_index in range(num_cam):
                                line_ego_to_pvmask(line_ego, gt_pv_semantic_mask[cam_index][0], lidar2feat[cam_index],
                                                            color=1, thickness=self.aux_seg['pv_thickness'])
                        else:
                            print(line_ego.geom_type)

        # pdb.set_trace()
        # image_paths = torch.tensor(sample['image_paths'].tolist())
        ctr_points = np.zeros((self.max_instances, max(self.max_pieces) * max(self.num_degree) + 1, 2), dtype=np.float)
        ins_labels = np.zeros((self.max_instances, 3), dtype=np.int16) - 1

        for ins_id, ctr_info in enumerate(sample['ctr_points']):
            cls_id = int(ctr_info['type'])
            ctr_pts_raw = np.array(ctr_info['pts'])
            max_points = self.max_pieces[cls_id] * self.num_degree[cls_id] + 1
            num_points = max_points if max_points <= ctr_pts_raw.shape[0] else ctr_pts_raw.shape[0]
            assert num_points >= self.num_degree[cls_id] + 1
            ctr_points[ins_id][:num_points] = np.array(ctr_pts_raw[:num_points])
            ins_labels[ins_id] = [cls_id, (num_points - 1) // self.num_degree[cls_id] - 1, num_points]

        masks = sample[self.mask_key]

        if flip:
            new_order = [2, 1, 0, 5, 4, 3]
            img_key_list = [self.img_key_list[i] for i in new_order]
            images = [images[i] for i in new_order]
            ida_mats = [ida_mats[i] for i in new_order]
            extrinsic = [extrinsic[i] for i in new_order]
            intrinsic = [intrinsic[i] for i in new_order]
            masks = [np.flip(mask, axis=1) for mask in masks]
            ctr_points = self.point_flip(ctr_points, ins_labels, self.ego_size)

        item = dict(
            images=images, targets=dict(masks=masks, points=ctr_points, labels=ins_labels),
            extrinsic=np.stack(extrinsic), intrinsic=np.stack(intrinsic), ida_mats=np.stack(ida_mats),
            extra_infos=dict(token=token, img_key_list=self.img_key_list, map_size=self.ego_size, do_flip=flip),
            cam_ego_pose=cam_ego2global, lidar2ego=lidar2ego, lidar2cam=lidar2cam_rts, lidar2img=lidar2img_rts,
            map_location=map_location, lidar2global=lidar2global,
        )
        if self.transforms is not None:
            item = self.transforms(item)

        if self.return_depth:
            item['lidar_depth'] = np.stack(lidar_depth)    # [6, 384, 896]
        # pdb.set_trace()
        if self.aux_seg['pv_seg']:
            item['targets']['gt_pv_semantic_mask'] = gt_pv_semantic_mask
        return item

    def __len__(self):
        return len(self.tokens)

    def sample_ida_augmentation(self):
        """Generate ida augmentation values based on ida_config."""
        resize_dims = w, h = self.ida_conf["resize_dims"]
        crop = (0, 0, w, h)
        if self.ida_conf["up_crop_ratio"] > 0:
            crop = (0, int(self.ida_conf["up_crop_ratio"] * h), w, h)
        flip, color, rotate_ida = False, False, 0
        if self.split_mode == "train":
            if self.ida_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            if self.ida_conf["rot_lim"]:
                assert isinstance(self.ida_conf["rot_lim"], (tuple, list))
                rotate_ida = np.random.uniform(*self.ida_conf["rot_lim"])
        return resize_dims, crop, flip, rotate_ida

    def img_transform(self, img, resize_dims, crop, flip, rotate):      # (896, 512), (0, 128, 896, 512)
        img = Image.fromarray(img)
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        w, h = img.size
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        scales = torch.tensor([resize_dims[0] / w, resize_dims[1] / h])
        ida_rot *= torch.Tensor(scales)
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = ida_rot.new_zeros(3, 3)
        ida_mat[2, 2] = 1
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return np.asarray(img), ida_mat

    @staticmethod
    def point_flip(points, labels, map_shape):

        def _flip(pts):
            pts[:, 0] = map_shape[1] - pts[:, 0]
            return pts.copy()

        points_ret = deepcopy(points)
        for ins_id in range(points.shape[0]):
            end = labels[ins_id, 2]
            points_ret[ins_id][:end] = _flip(points[ins_id][:end])

        return points_ret

    @staticmethod
    def get_rot(h):
        return torch.Tensor([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])
