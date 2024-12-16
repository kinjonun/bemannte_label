import pdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from projects.mmdet3d_plugin.maptr.modules.ops.diff_ras.polygon import SoftPolygon


class DecodeRaster:
    def __init__(self, bev_h, bev_w):
        self.num_classes = 3
        self.max_num = 50
        self.map_size = [60, 30]
        self.num_ins = [0, 20, 45, 60]
        # self.raster_threshold = 0.4    # 自带门槛 0.5
        self.bev_h, self.bev_w = bev_h, bev_w

    def rasterize_preds(self, pts, labels, height=200, width=100, inv_smoothness=2.0, use_dilate=False):
        """
        Args:
            pts: shape [batch_size * num_pred, num_pts, 2]  height: bev_h  width: bev_w
        """
        new_pts = pts.clone()  # [0, 20, 2]
        new_pts[..., 1:2] = (1 - pts[..., 0:1]) * width / 2   # av2: * 200,  NuS: * 100   # x,y换序 output的问题 /2什么吊bug
        new_pts[..., 0:1] = (1 - pts[..., 1:2]) * height     # av2: * 100,   NuS: * 200

        divider_index = torch.nonzero(labels == 0, as_tuple=True)
        ped_crossing_index = torch.nonzero(labels == 1, as_tuple=True)
        boundary_index = torch.nonzero(labels == 2, as_tuple=True)
        divider_pts = new_pts[divider_index]
        ped_crossing_pts = new_pts[ped_crossing_index]
        boundary_pts = new_pts[boundary_index]

        rasterized_results = torch.zeros(3, height, width, device=pts.device)  # [3, 200, 100]
        if divider_pts.shape[0] > 0:
            HARD_CUDA_RASTERIZER = SoftPolygon(mode="boundary", inv_smoothness=inv_smoothness)
            rasterized_line = HARD_CUDA_RASTERIZER(divider_pts, int(width), int(height), 1.0)
            rasterized_line, _ = torch.max(rasterized_line, 0)
            rasterized_results[0] = rasterized_line

        if ped_crossing_pts.shape[0] > 0:
            HARD_CUDA_RASTERIZER = SoftPolygon(mode="mask", inv_smoothness=inv_smoothness)
            rasterized_poly = HARD_CUDA_RASTERIZER(ped_crossing_pts, int(width), int(height), 1.0)
            rasterized_poly, _ = torch.max(rasterized_poly, 0)
            rasterized_results[1] = rasterized_poly

        if boundary_pts.shape[0] > 0:
            HARD_CUDA_RASTERIZER = SoftPolygon(mode="boundary", inv_smoothness=inv_smoothness)
            rasterized_line = HARD_CUDA_RASTERIZER(boundary_pts, int(width), int(height), 1.0)
            rasterized_line, _ = torch.max(rasterized_line, 0)
            rasterized_results[2] = rasterized_line

        if use_dilate:
            max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            rasterized_results = max_pool(rasterized_results)

        return rasterized_results

    def decode_raster_single(self, cls_scores, pts_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
            pts_preds (Tensor):
                Shape [num_query, fixed_num_pts, 2]
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num  # 50
        # pdb.set_trace()
        labels, instances = [], []
        for i in range(self.num_classes):
            pred_scores, pred_labels = torch.max(F.softmax(cls_scores[self.num_ins[i]:self.num_ins[i + 1]], dim=-1),
                                                 dim=-1)
            keep_ids = torch.where((pred_labels == 0).int())[0]
            if keep_ids.shape[0] == 0:
                continue
            pred_scores = pred_scores[keep_ids]
            curve_pts = pts_preds[self.num_ins[i]:self.num_ins[i + 1]][keep_ids]  # [60, 100, 2]
            # curve_pts[:, :, 0] *= self.map_size[1]     # * 30
            # curve_pts[:, :, 1] *= self.map_size[0]     # * 60
            instances.append(curve_pts)
            label = torch.zeros_like(pred_scores) + i
            labels.append(label)

        if len(labels) > 0:
            instances = torch.cat(instances, dim=0)
            labels = torch.cat(labels, dim=0)
            raster_preds = self.rasterize_preds(instances, labels, self.bev_h, self.bev_w)
            # raster_preds_cpu = raster_preds.cpu().numpy()
            # combined_mask = np.zeros((100, 200))
            # # 为每个类别赋予不同的整数值 (例如 1, 2, 3)
            # combined_mask[raster_preds_cpu[0] > 0.1] = 1  # 第1类
            # combined_mask[raster_preds_cpu[1] > 0.1] = 2  # 第2类
            # combined_mask[raster_preds_cpu[2] > 0.1] = 3  # 第3类
            # plt.imshow(combined_mask, cmap='gray')
            # plt.savefig('combined_mask.png', bbox_inches='tight', dpi=300)
            # pdb.set_trace()
        else:
            raster_preds = torch.zeros(3, self.bev_h, self.bev_w, device=pts_preds.device)

        return raster_preds

    def decode_raster(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        # divider = preds_dicts['curve_points'][-1][0][0]   # [20, 100, 2]
        # divider_obj = preds_dicts['obj_logits'][-1][0][0]
        # div_scores, div_labels = torch.max(F.softmax(divider_obj, dim=-1), dim=-1)
        # # [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0]      9, 12, 13, 16 19
        # pdb.set_trace()
        cls_scores = torch.cat(preds_dicts['obj_logits'][-1], dim=1)  # list([bs, 20, 2], [bs, 25, 2] ... -> [bs, 60, 2
        pts_preds = torch.cat(preds_dicts['curve_points'][-1], dim=1)  # list([bs, 20, 100, 2], [bs, 25, 100, 2] ...
        batch_size = cls_scores.shape[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_raster_single(cls_scores[i], pts_preds[i]))
        return predictions_list
