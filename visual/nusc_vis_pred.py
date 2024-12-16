import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
import os.path as osp
from nuscenes import NuScenes
import cv2
import random
from tqdm import tqdm
import glob

def save_pred_visual(file, folder_path, sample_path):
    color = {0: 'r', 1: 'orange', 2: 'b', 3: 'g', 4: "c", 5: "m", 6: "k", 7: "y", 8: "deeppink"}
    npz_path = os.path.join(folder_path, 'results', file)
    _, _, epoch_num = npz_path.split('/')[-3].rpartition('_')
    # pdb.set_trace()
    data = np.load(npz_path, allow_pickle=True)
    dt_res = data['dt_res'].tolist()
    res = dict(dt_res)
    points = res["map"]
    label = res["pred_label"]

    plt.figure(figsize=(3, 6))
    plt.ylim(-30, 30)
    plt.xlim(-15, 15)

    skla = 15 / 100
    for i in range(1, len(res["map"])):
        ins = res["map"][i]
        for j in range(len(ins) - 1):
            x = ins[j][0]
            y = 400 - ins[j][1]

            plt.plot([(ins[j][0] - 100) * skla, (ins[j + 1][0] - 100) * skla],
                     [(200 - ins[j][1]) * skla, (200 - ins[j + 1][1]) * skla], c=color[label[i]])

    car_img = Image.open('/home/sun/Bev/BeMapNet/assets/figures/lidar_car.png')
    plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

    plt.text(-15, 31, 'PRED_epoch_'+epoch_num, color='red', fontsize=12)
    map_path = osp.join(sample_path, 'PRED_epoch_'+epoch_num+'.png')
    plt.savefig(map_path, bbox_inches='tight', format='png', dpi=1200)
    plt.close()

def save_GT_visual(file, anno_path, sample_path):
    gt_path = os.path.join(anno_path, file)
    data = np.load(gt_path, allow_pickle=True)

    plt.figure(figsize=(3, 6))
    plt.ylim(-30, 30)
    plt.xlim(-15, 15)
    # plt.axis('off')

    # for item in data['ctr_points']:
    #     pts = item['pts']
    #     y = [pt[0] - 15 for pt in pts]
    #     x = [-pt[1] + 30 for pt in pts]
    #     plt.scatter(y, x, c=color[item['type']+1])

    color = {0: 'r', 1: 'orange', 2: 'b', 3: 'g', 4: "c", 5: "m", 6: "k", 7: "y", 8: "deeppink"}
    for item in data['ego_vectors']:
        pts = item['pts']
        for i in range(len(pts) - 1):
            plt.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]], c=color[item['type']+1])

    car_img = Image.open('/home/sun/Bev/BeMapNet/assets/figures/lidar_car.png')
    plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
    plt.text(-15, 31, 'GT', color='red', fontsize=12)
    plt.tight_layout()

    map_path = osp.join(sample_path, 'GT.png')
    plt.savefig(map_path, bbox_inches='tight', format='png', dpi=1200)

def save_surroud(token, dataroot, sample_path, nusc):
    img_key_list = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
    sample = nusc.get('sample', token)
    row_1_list = []
    row_2_list = []
    image_name_list = []
    for cam in img_key_list[:3]:
        img = nusc.get('sample_data', sample['data'][cam])
        filename = img['filename']
        pdb.set_trace()
        # print(filename)
        img_path = os.path.join(dataroot, filename)
        img = cv2.imread(img_path)
        row_1_list.append(img)
        image_name_list.append(filename)

    for cam in img_key_list[3:]:
        img = nusc.get('sample_data', sample['data'][cam])
        filename = img['filename']
        img_path = os.path.join(dataroot, filename)
        img = cv2.imread(img_path)
        row_2_list.append(img)
        image_name_list.append(filename)

    row_1_img = cv2.hconcat(row_1_list)       # 水平拼接成一张图像
    row_2_img = cv2.hconcat(row_2_list)
    cams_img = cv2.vconcat([row_1_img, row_2_img])
    cams_img_path = osp.join(sample_path, 'surroud_view.jpg')
    cv2.imwrite(cams_img_path, cams_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

    img_name_path = osp.join(sample_path, 'images_name_list.txt')
    with open(img_name_path, "w") as f:
        for item in image_name_list:
            f.write("%s\n" % item)

def concat(sample_path):
    gt_path = osp.join(sample_path, 'GT.png')
    surroud_path = osp.join(sample_path, 'surroud_view.jpg')

    pred_paths = sorted(glob.glob(osp.join(sample_path, 'PRED*')))
    pred_images = []
    for pred_path in pred_paths:
        img = cv2.imread(pred_path)
        pred_images.append(img)

    gt = cv2.imread(gt_path)
    pred = cv2.imread(pred_path)
    surroud = cv2.imread(surroud_path)

    surroud_h, surroud_w, _ = surroud.shape
    pred_h, pred_w, _ = pred.shape
    resize_ratio = surroud_h / pred_h

    resized_w = pred_w * resize_ratio
    resized_pred = [cv2.resize(pred, (int(resized_w), int(surroud_h))) for pred in pred_images]
    resized_gt_map_img = cv2.resize(gt, (int(resized_w), int(surroud_h)))

    img = cv2.hconcat([surroud, resized_gt_map_img] + resized_pred)

    cams_img_path = osp.join(sample_path, 'Sample_vis.jpg')
    cv2.imwrite(cams_img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])


def main():
    anno_path = "/home/sun/Bev/BeMapNet/data/nuscenes/customer/bemapnet_lidar_coord"
    eval_path = "/home/sun/Bev/BeMapNet/outputs/bemapnet_nuscenes_res50_lss/2024-07-16"
    dataroot = "/media/sun/z/nuscenes/nuscenes"

    # file_list = os.listdir("/home/sun/Bev/BeMapNet/outputs/bemapnet_nuscenes_res50/2024-05-03T10:33:35/evaluation/results")
    # random_files = random.sample(file_list, 20)
    # print(random_files)
    # pdb.set_trace()

    random_files = ['d056b9bdd56f44669540c6c323042d30.npz', '02fa71fc16d449c382c24b1b7deef3ba.npz', 'e47f3fd631fb41d69932f13be883640c.npz',
                    '1b120b73e0374cddbf17ac9e6656dcb6.npz', 'f69afa3eed11447eb559f49c325f02d7.npz', 'c51b9abd585d421b85dbf4a95eec4936.npz',
                    '3097b27c66c448d58aee1c54958466ad.npz', 'de0625ead6144de9bea3608ea9cc02d3.npz', '5cba7499a7004bea8e3318d5dc97e132.npz',
                    'd528a32b7367451e84f119068d4568a0.npz', 'a11122da55ec4b27b580c23e8473e874.npz', 'c0ffb6f6b7e841f3bad545afdaf298b8.npz',
                    '9f5cb220fe7e4702a9b8033347a03755.npz', '378d5e91029d490684ccc9f9208b60e0.npz', 'f443bc4e97744a29888ec3b18f042d7b.npz',
                    'abe33290dd64401786b25024293ae4dc.npz', '4afb1a21cfd44947b98f9d2f778657d6.npz', 'dd29902d904d4bd1a68e5e6926cd6dfc.npz',
                    '4f38413567d14272b4fc9c89cd8405f3.npz', '4ffbb8951b9841429bfd3aee39c6a2be.npz']
    # random_files = ['0a0d6b8c2e884134a3b48df43d54c36a.npz']
    vis_path = os.path.join(eval_path, 'visual')
    if not os.path.exists(vis_path):
        os.makedirs(vis_path, exist_ok=True)

    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
    num_visual = 0

    for file in tqdm(random_files, desc="Processing files"):
        # print(file)
        token = os.path.splitext(file)[0]                 # a4354e58aaaa454493ec48f11176530c
        sample_path = os.path.join(vis_path, token)
        if not os.path.exists(sample_path):
            os.makedirs(sample_path, exist_ok=True)

        all_eval = glob.glob(osp.join(eval_path, 'evaluation*'))
        # pdb.set_trace()
        for path in all_eval:
            save_pred_visual(file, path, sample_path)

        save_GT_visual(file, anno_path, sample_path)
        save_surroud(token, dataroot, sample_path, nusc)
        concat(sample_path)


        # num_visual = num_visual + 1
        # if num_visual >= 2:
        #     break


if __name__ == '__main__':
    main()