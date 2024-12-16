import pdb
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import os.path as osp
import cv2
from pathlib import Path
import random
from tqdm import tqdm

caption_by_cam = {
    'ring_front_center': 'CAM_FRONT_CENTER',
    'ring_front_right': 'CAM_FRONT_RIGHT',
    'ring_front_left': 'CAM_FRONT_LEFT',
    'ring_rear_right': 'CAM_REAR_RIGHT',
    'ring_rear_left': 'CAM_REAT_LEFT',
    'ring_side_right': 'CAM_SIDE_RIGHT',
    'ring_side_left': 'CAM_SIDE_LEFT',
}


def save_gt_visual(data_dict, car_img, sample_path, vis_maptracker_gt=False):
    color = {0: 'g', 1: 'orange', 2: 'b', 3: 'r', 4: "c", "ped_crossing": "b", "divider": "r", "boundary": "g"}

    ego_points = data_dict["ego_points"]
    # ctr_points = data_dict["ctr_points"]
    local_index = data_dict["input_dict"]["seq_info"][1]

    # for item in ctr_points:
    #     pts = item['pts']
    #     y = [pt[0] - 15 for pt in pts]
    #     x = [-pt[1] + 30 for pt in pts]
    #     plt.scatter(y, x, c=color[item['type'] + 1])
    if vis_maptracker_gt:
        map_geoms = data_dict["input_dict"]["map_geoms"]
        plt.figure(figsize=(30, 15))
        plt.xlim(-31, 31)
        plt.ylim(-16, 16)
        for key, value in map_geoms.items():
            for pts in value:
                for i in range(pts.shape[0] - 1):
                    plt.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]], c=color[key], linewidth=4)
                    if key == "ped_crossing":
                        plt.scatter(pts[i][0], pts[i][1], c=color[key])
        plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
        plt.tight_layout()
        map_path = osp.join(sample_path, 'maptracker_gt.png')
        plt.savefig(map_path, bbox_inches='tight', format='png', dpi=100)
        plt.close()

    plt.figure(figsize=(15, 30))
    plt.ylim(-31, 31)
    plt.xlim(-16, 16)

    for item in ego_points:
        pts = item['pts']
        x = np.array([pt[0] for pt in pts])
        y = np.array([-pt[1] for pt in pts])
        plt.plot(y, x, c=color[item['type'] + 1], linewidth=4)

    plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
    plt.text(-15, 31, 'GT', color='red', fontsize=12)
    plt.tight_layout()

    map_path = osp.join(sample_path, f'GT.png')
    plt.savefig(map_path, bbox_inches='tight', format='png', dpi=100)
    plt.close()


def save_pred_visual(pred_path, sample_path, data_dict):
    _, _, epoch_num = pred_path.split('/')[-3].rpartition('_')

    data = np.load(pred_path, allow_pickle=True)
    dt_res = data['dt_res'].tolist()
    res = dict(dt_res)
    map_points = res["map"]
    label = res["pred_label"]
    ego_points = data_dict["ego_points"]   # gt

    plt.figure(figsize=(15, 30))
    plt.ylim(-31, 31)
    plt.xlim(-16, 16)
    color = {0: 'g', 1: 'orange', 2: 'b', 3: 'r', 4: "c", }

    for item in ego_points:    # plot gt as background
        pts = item['pts']
        x = np.array([pt[0] for pt in pts])
        y = np.array([-pt[1] for pt in pts])
        plt.plot(y, x, c='#aaaaaa', linewidth=3)

    scale = 15 / 100
    for i in range(1, len(map_points)):  # 第0个是None
        ins = np.int16(map_points[i])
        x = [(pt[0]-100) * scale for pt in ins]
        y = [(200-pt[1]) * scale for pt in ins]
        plt.plot(x, y, c=color[np.int16(label[i])], linewidth=4)
        # for j in range(len(ins) - 1):
        #     plt.plot([(ins[j][0] - 100) * scale, (ins[j + 1][0] - 100) * scale],
        #              [(200 - ins[j][1]) * scale, (200 - ins[j + 1][1]) * scale], c=color[np.int16(label[i])], linewidth=4)

    car_img = Image.open('/home/sun/Bev/BeMapNet/assets/figures/lidar_car.png')
    plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

    plt.text(-15, 31, 'PRED_epoch_' + epoch_num, color='red', fontsize=40)
    map_path = osp.join(sample_path, 'PRED_epoch_' + epoch_num + '.png')
    plt.savefig(map_path, bbox_inches='tight', format='png', dpi=100)
    plt.close()


def save_surround(cams_dict, sample_dir, timestamp):
    rendered_cams_dict = {}
    for key, cam_dict in cams_dict.items():
        cam_img = cv2.imread(cam_dict)
        # render_anno_on_pv(cam_img, pred_anno, cam_dict['lidar2img'])
        if 'front' not in key:
            #         cam_img = cam_img[:,::-1,:]
            cam_img = cv2.flip(cam_img, 1)
        lw = 8
        tf = max(lw - 1, 1)
        w, h = cv2.getTextSize(caption_by_cam[key], 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        p1 = (0, 0)
        p2 = (w, h + 3)
        color = (0, 0, 0)
        txt_color = (255, 255, 255)
        cv2.rectangle(cam_img, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(cam_img,
                    caption_by_cam[key], (p1[0], p1[1] + h + 2),
                    0,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)
        rendered_cams_dict[key] = cam_img

    new_image_height = 2048
    new_image_width = 1550 + 2048 * 2
    color = (255, 255, 255)
    first_row_canvas = np.full((new_image_height, new_image_width, 3), color, dtype=np.uint8)
    first_row_canvas[(2048 - 1550):, :2048, :] = rendered_cams_dict['ring_front_left']
    first_row_canvas[:, 2048:(2048 + 1550), :] = rendered_cams_dict['ring_front_center']
    first_row_canvas[(2048 - 1550):, 3598:, :] = rendered_cams_dict['ring_front_right']

    new_image_height = 1550
    new_image_width = 2048 * 4
    color = (255, 255, 255)
    second_row_canvas = np.full((new_image_height, new_image_width, 3), color, dtype=np.uint8)
    second_row_canvas[:, :2048, :] = rendered_cams_dict['ring_side_left']
    second_row_canvas[:, 2048:4096, :] = rendered_cams_dict['ring_rear_left']
    second_row_canvas[:, 4096:6144, :] = rendered_cams_dict['ring_rear_right']
    second_row_canvas[:, 6144:, :] = rendered_cams_dict['ring_side_right']

    resized_first_row_canvas = cv2.resize(first_row_canvas, (8192, 2972))
    full_canvas = np.full((2972 + 1550, 8192, 3), color, dtype=np.uint8)
    full_canvas[:2972, :, :] = resized_first_row_canvas
    full_canvas[2972:, :, :] = second_row_canvas
    cams_img_path = osp.join(sample_dir, 'surround_view.jpg')
    cv2.imwrite(cams_img_path, full_canvas, [cv2.IMWRITE_JPEG_QUALITY, 50])


def concat(sample_path):
    gt_path = glob.glob(osp.join(sample_path, 'GT*'))
    # pdb.set_trace()
    surround_path = osp.join(sample_path, 'surround_view.jpg')
    pred_paths = glob.glob(osp.join(sample_path, 'PRED*'))

    pred_images = []
    for pred_path in pred_paths:
        img = cv2.imread(pred_path)
        pred_images.append(img)

    gt = cv2.imread(gt_path[0])
    # pred = cv2.imread(pred_path)
    surround = cv2.imread(surround_path)

    surround_h, surround_w, _ = surround.shape
    pred_h, pred_w, _ = gt.shape
    resize_ratio = surround_h / pred_h

    resized_w = pred_w * resize_ratio
    resized_pred = [cv2.resize(pred, (int(resized_w), int(surround_h))) for pred in pred_images]
    resized_gt_map_img = cv2.resize(gt, (int(resized_w), int(surround_h)))

    img = cv2.hconcat([surround, resized_gt_map_img] + resized_pred)

    cams_img_path = osp.join(sample_path, 'Sample_vis.jpg')
    cv2.imwrite(cams_img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 80])


def main():
    project_path = "/home/sun/Bev/BeMapNet"
    anno_path = "/home/sun/Bev/maptracker/datasets/av2/80_X_50/val"
    anno_path = "/home/sun/Bev/BeMapNet/data/argoverse2/geo_interval_1"
    output_path = "/home/sun/Bev/BeMapNet/outputs/bemapnet_av2_res50_geosplit_interval_1/1004_perfect"
    car_img = Image.open('/home/sun/Bev/BeMapNet/assets/figures/lidar_car.png')

    vis_path = os.path.join(output_path, 'visual')
    # vis_path = osp.join('/home/sun/Bev/BeMapNet/paper_imgs')
    if not os.path.exists(vis_path):
        os.makedirs(vis_path, exist_ok=True)

    num_visual = 10
    index = 0

    # file_list = os.listdir("/home/sun/Bev/BeMapNet/outputs/bemapnet_av2_res50_geo_splits_interval_4/0902/evaluation_epoch_5/results")
    # random_files = random.sample(file_list, 20)
    # print(random_files)
    # pdb.set_trace()
    random_files_geo = ['b2a8a9aa-19cd-3ffd-b02c-0f2a47d1d0eb_315967944259809000.npz',
                        'b2d9d8a5-847b-3c3b-aed1-c414319d20af_315978619960272000.npz',
                        '1886b0d1-9c5e-326f-99df-30b64044638f_315966427660131000.npz',
                        '0749e9e0-ca52-3546-b324-d704138b11b5_315972778959661000.npz',
                        'ba737c78-2ef2-3643-a5b2-4804dfff9d93_315966271159849000.npz',
                        'ff8e7fdb-1073-3592-ba5e-8111bc3ce48b_315968526659956000.npz',
                        'a4400a38-bc38-391c-b102-ba385d7e475e_315973828960075000.npz',
                        'd9530d0a-b83e-44a3-910a-2b5bb8f1fb80_315973124960080000.npz',
                        '05fa5048-f355-3274-b565-c0ddc547b315_315976477759901000.npz',
                        '4935629c-fd9e-3b2f-b68e-9489c89585df_315967430659851000.npz',
                        'a059b6b9-ca26-4881-bcf7-d202433de0c2_315972538759881000.npz',
                        'fd4e2c4c-f7e9-3110-8e32-28d3add3937d_315977003959690000.npz',
                        'e42aa296-0e5d-4733-87ec-131a82f917bc_315966258860240000.npz',
                        '0fb7276f-ecb5-3e5b-87a8-cc74c709c715_315968092259864000.npz',
                        '83faae69-e37e-4804-b7a9-684d4a900320_315966720059907000.npz',
                        'e1f37027-6a39-3eb1-b38a-3f2836b84735_315968028860195000.npz',
                        '0749e9e0-ca52-3546-b324-d704138b11b5_315972781759856000.npz',
                        '7a1412d3-5a53-378f-85df-ba58b2408f46_315968803559575000.npz',
                        'f648b945-6c70-3105-bd23-9502894e37d4_315969807859925000.npz',
                        '47358aac-2ec0-3d45-a837-f2069ca7cee3_315966619459660000.npz']
    random_files_old = ['ded5ef6e-46ea-3a66-9180-18a6fa0a2db4_315976262659983000.npz',
                        'cf5aaa11-4f92-3377-a7a2-861f305023eb_315972109559792000.npz',
                        '27be7d34-ecb4-377b-8477-ccfd7cf4d0bc_315969631360189000.npz',
                        '20dd185d-b4eb-3024-a17a-b4e5d8b15b65_315969737960496000.npz',
                        'bf382949-3515-3c16-b505-319442937a43_315967939859906000.npz',
                        'fbee355f-8878-31fa-8ac8-b9a45a3f130a_315967354059758000.npz',
                        '1f434d15-8745-3fba-9c3e-ccb026688397_315972873660033000.npz',
                        '24642607-2a51-384a-90a7-228067956d05_315970917960388000.npz',
                        'ded5ef6e-46ea-3a66-9180-18a6fa0a2db4_315976259859774000.npz',
                        'de9cf513-a0cd-3389-bc79-3f9f6f261317_315971620760118000.npz',
                        '7606de8d-486c-4916-9cbb-002ee966f834_315978505060382000.npz',
                        'e2e921fe-e489-3656-a0a2-5e17bd399ddf_315966473659811000.npz',
                        'fbd62533-2d32-3c95-8590-7fd81bd68c87_315978109759782000.npz',
                        'e50e7698-de3d-355f-aca2-eddd09c09533_315971203860272000.npz',
                        '91aab547-1912-3b8e-8e7f-df3b202147bf_315968308059737000.npz',
                        'fbd62533-2d32-3c95-8590-7fd81bd68c87_315978124560111000.npz',
                        '0aa4e8f5-2f9a-39a1-8f80-c2fdde4405a2_315971850660699000.npz',
                        '52971a8a-ed62-3bfd-bcd4-ca3308b594e0_315968617359614000.npz',
                        '52971a8a-ed62-3bfd-bcd4-ca3308b594e0_315968618559987000.npz',
                        '05fa5048-f355-3274-b565-c0ddc547b315_315976479960236000.npz']
    if "geo" in anno_path:
        random_files = random_files_geo
    else:
        random_files = random_files_old
    sample_list_path = '/home/sun/Bev/BeMapNet/zknecht/vis_unmerge/22dcf96c-ef5e-376b-9db5-dc9f91040f5e.txt'
    sample_list = []
    with open(sample_list_path, 'r') as file:
        for i, line in enumerate(file):
            if i % 4 == 0:
                sample_list.append(line.strip() + '.npz')

    for file_name in tqdm(random_files):
        anno_data_path = osp.join(anno_path, file_name)
        anno_data = np.load(anno_data_path, allow_pickle=True)
        # ['input_dict', 'instance_mask', 'instance_mask8', 'semantic_mask', 'ctr_points', 'ego_points', 'map_vectors']

        anno_data_dict = {key: anno_data[key].tolist() for key in anno_data.files}
        input_dict = anno_data_dict["input_dict"]
        # ['timestamp', 'img_filename', 'camera_intrinsics', 'ego2cam', 'camera2ego', 'lidar2img', 'map_geoms',
        # 'ego2global_translation', 'ego2global_rotation', 'sample_idx', 'scene_name', 'lidar_path', 'map_location',
        # 'split', 'seq_info']
        # map_geoms = input_dict["map_geoms"]
        # ped_crossing = map_geoms["ped_crossing"]
        # pdb.set_trace()
        timestamp = input_dict["timestamp"]
        scene_token = input_dict["scene_name"]
        local_index = input_dict["seq_info"][1]  # sample 在 scene中

        sample_path = os.path.join(vis_path, f"{scene_token}_{timestamp}")
        if not os.path.exists(sample_path):
            os.makedirs(sample_path, exist_ok=True)

        # save_images
        img_filename = input_dict["img_filename"]
        # ('data/argoverse2/sensor/val/201fe83b-768638a9/sensors/cameras/ring_front_center/315969617449927219', '.jpg')
        cams_dict = {}
        for img in img_filename:
            path = Path(img)
            img_name = path.parts[-2]
            img_path = osp.join(project_path, img.replace('av2', 'argoverse2/sensor'))
            cams_dict[img_name] = img_path
        save_surround(cams_dict, sample_path, timestamp)

        all_eval = glob.glob(osp.join(output_path, 'evaluation*'))
        for path in all_eval:
            pred_path = os.path.join(path, 'results', file_name)
            save_pred_visual(pred_path, sample_path, anno_data_dict)

        save_gt_visual(anno_data_dict, car_img, sample_path)
        concat(sample_path)

        # if cam_img is not None:
        #     cv2.imshow("cam_img", cam_img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # else:
        #     print("Error: Processed image for key  is None.")

        # index += 1
        # if index >= num_visual:
        #     break


if __name__ == "__main__":
    main()
