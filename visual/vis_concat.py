import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
import os.path as osp
from nuscenes import NuScenes
import cv2


img1_path = "/home/sun/Bev/BeMapNet/visual_global/geosplit/av2/06e5ac08-f4cb-34ae-9406-3496f7cadc62/local_gt_/30.png"
img2_path = "/home/sun/Bev/BeMapNet/visual_global/geosplit/av2/06e5ac08-f4cb-34ae-9406-3496f7cadc62/local_gt_/39.png"
save_path = "/home/sun/Bev/BeMapNet/paper_imgs"
# hCon = True
hCon = False

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

img1_h, img1_w = img1.shape[:2]
img2_h, img2_w = img2.shape[:2]

if hCon:
    resize_ratio = img1_h / img2_h
    resized_w = int(img2_w * resize_ratio)
    resized_img2 = cv2.resize(img2, (resized_w, img1_h))
    img = cv2.hconcat([img1, resized_img2])

else:
    resize_ratio = img1_w / img2_w
    resized_h = int(img2_h * resize_ratio)
    resized_img2 = cv2.resize(img2, (img1_w, resized_h))
    img = cv2.vconcat([img1, resized_img2])

cams_img_path = osp.join(save_path, 'vis_concat.jpg')
cv2.imwrite(cams_img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
# print(img1_h, img1_w)
