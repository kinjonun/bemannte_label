import pdb
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import os.path as osp
import cv2
from tqdm import tqdm
import glob


eval_ohne_depth_loss = '/home/sun/Bev/BeMapNet/outputs/bemapnet_nuscenes_res50_lss/2024-07-03/visual'
eval_mit_depth_loss = '/home/sun/Bev/BeMapNet/outputs/bemapnet_nuscenes_res50_lss/2024-07-16/visual'

def concat(eval_ohne_depth_loss, eval_mit_depth_loss, epoch):
    gt_path = osp.join(eval_ohne_depth_loss, 'GT.png')
    surroud_path = osp.join(eval_ohne_depth_loss, 'surroud_view.jpg')

    pred_path_ohne = glob.glob(osp.join(eval_ohne_depth_loss, f'*{epoch}*'))[0]
    pred_path_mit = glob.glob(osp.join(eval_mit_depth_loss, f'*{epoch}*'))[0]

    gt = cv2.imread(gt_path)
    surroud = cv2.imread(surroud_path)
    pred_mit = cv2.imread(pred_path_mit)
    pred_ohne = cv2.imread(pred_path_ohne)

    text = "mit"
    text_ = "ohne"
    org = (2600, 270)  # 文本位置 (x, y)
    font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
    font_scale = 8  # 字体大小
    color = (255, 0, 0)  # 字体颜色 (B, G, R)
    thickness = 20  # 字体粗细
    # 在图像上添加文本
    cv2.putText(pred_mit, text, org, font, font_scale, color, thickness)
    cv2.putText(pred_ohne, text_, org, font, font_scale, color, thickness)


    surroud_h, surroud_w, _ = surroud.shape
    pred_h, pred_w, _ = pred_mit.shape
    resize_ratio = surroud_h / pred_h

    resized_w = pred_w * resize_ratio
    resized_pred_mit = cv2.resize(pred_mit, (int(resized_w), int(surroud_h)))
    resized_pred_ohne = cv2.resize(pred_ohne, (int(resized_w), int(surroud_h)))
    resized_gt_map_img = cv2.resize(gt, (int(resized_w), int(surroud_h)))

    img = cv2.hconcat([surroud, resized_gt_map_img, resized_pred_ohne, resized_pred_mit])

    cams_img_path = osp.join(eval_mit_depth_loss, 'comp_vis_epoch_'+str(epoch)+'.png')
    cv2.imwrite(cams_img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 80])


epoch_index = [16, ]
samples = os.listdir(eval_mit_depth_loss)
for sample in tqdm(samples):
    eval_mit = os.path.join(eval_mit_depth_loss, sample)
    eval_ohne = os.path.join(eval_ohne_depth_loss, sample)
    for epoch in epoch_index:
        concat(eval_ohne, eval_mit, epoch)
