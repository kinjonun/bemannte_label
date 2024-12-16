import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import comb as n_over_k


class FFN(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, basic_type='linear'):
        super().__init__()
        self.basic_type = basic_type
        if output_dim == 0:
            self.basic_type = "identity"
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(self.basic_layer(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

    def basic_layer(self, n, k):
        if self.basic_type == 'linear':
            return nn.Linear(n, k)
        elif self.basic_type == 'conv':
            return nn.Conv2d(n, k, kernel_size=1, stride=1)
        elif self.basic_type == 'identity':
            return nn.Identity()
        else:
            raise NotImplementedError


class PiecewiseBezierMapOutputHead(nn.Module):
    def __init__(self, in_channel, num_queries, tgt_shape, num_degree, max_pieces, bev_channels=-1, ins_channel=64):
        super(PiecewiseBezierMapOutputHead, self).__init__()
        self.num_queries = num_queries
        self.num_classes = len(num_queries)
        self.tgt_shape = tgt_shape
        self.bev_channels = bev_channels
        self.semantic_heads = None
        if self.bev_channels > 0:
            self.semantic_heads = nn.ModuleList(
                nn.Sequential(nn.Conv2d(bev_channels, 2, kernel_size=1, stride=1)) for _ in range(self.num_classes)
            )
        self.num_degree = num_degree
        self.max_pieces = max_pieces
        self.num_ctr_im = [(n + 1) for n in self.max_pieces]
        self.num_ctr_ex = [n * (d - 1) for n, d in zip(self.max_pieces, self.num_degree)]
        _N = self.num_classes

        _C = ins_channel
        self.im_ctr_heads = nn.ModuleList(FFN(in_channel, 256, (self.num_ctr_im[i] * 2) * _C, 3) for i in range(_N))
        self.ex_ctr_heads = nn.ModuleList(FFN(in_channel, 256, (self.num_ctr_ex[i] * 2) * _C, 3) for i in range(_N))
        self.npiece_heads = nn.ModuleList(FFN(in_channel, 256, self.max_pieces[i], 3) for i in range(_N))
        self.gap_layer = nn.AdaptiveAvgPool2d((1, 1))           # 平均池化
        self.coords = self.compute_locations(device='cuda')     # [1, 2, 400, 200]
        self.coords_head = FFN(2, 256, _C, 3, 'conv')
        self.curve_size = 100
        self.bezier_coefficient_np = self._get_bezier_coefficients()
        self.bezier_coefficient = [torch.from_numpy(x).float().cuda() for x in self.bezier_coefficient_np]

    def forward(self, inputs):
        num_decoders = len(inputs["mask_features"])
        dt_obj_logit = [[[] for _ in range(self.num_classes)] for _ in range(num_decoders)]
        dt_ins_masks = [[[] for _ in range(self.num_classes)] for _ in range(num_decoders)]
        im_ctr_coord = [[[] for _ in range(self.num_classes)] for _ in range(num_decoders)]
        ex_ctr_coord = [[[] for _ in range(self.num_classes)] for _ in range(num_decoders)]
        dt_end_logit = [[[] for _ in range(self.num_classes)] for _ in range(num_decoders)]
        # pdb.set_trace()
        coords_feats = self.coords_head.forward(self.coords.repeat((inputs["mask_features"][0].shape[0], 1, 1, 1)))  # [1, 64, 400, 200]

        for i in range(num_decoders):
            x_ins_cw = inputs["mask_features"][i].split(self.num_queries, dim=1)    # ([1, 20, 200, 100], [1, 25, 200, 100], [1, 15, 200, 100] )
            x_obj_cw = inputs["obj_scores"][i].split(self.num_queries, dim=1)       # ([1, 20, 2], [1, 25, 2], [1, 15, 2])
            x_qry_cw = inputs["decoder_outputs"][i].split(self.num_queries, dim=1)  # ([1, 20, 512], ([1, 25, 512]), ([1, 15, 512]))
            # pdb.set_trace()
            batch_size = x_qry_cw[0].shape[0]

            for j in range(self.num_classes):
                num_qry = self.num_queries[j]
                # if self.training:
                dt_ins_masks[i][j] = self.up_sample(x_ins_cw[j])
                dt_obj_logit[i][j] = x_obj_cw[j]              
                dt_end_logit[i][j] = self.npiece_heads[j](x_qry_cw[j])         # ([1, 20, 3],  )

                # im
                im_feats = self.im_ctr_heads[j](x_qry_cw[j])
                im_feats = im_feats.reshape(batch_size, num_qry, self.num_ctr_im[j] * 2, -1).flatten(1, 2)
                im_coords_map = torch.einsum("bqc,bchw->bqhw", im_feats, coords_feats)
                im_coords = self.gap_layer(im_coords_map)
                im_ctr_coord[i][j] = im_coords.reshape(batch_size, num_qry, self.max_pieces[j] + 1, 2)

                # ex
                if self.num_ctr_ex[j] == 0:
                    ex_ctr_coord[i][j] = torch.zeros(batch_size, num_qry, self.max_pieces[j], 0, 2).cuda()
                else:
                    ex_feats = self.ex_ctr_heads[j](x_qry_cw[j])    # [1, 20, 384]
                    ex_feats = ex_feats.reshape(batch_size, num_qry, self.num_ctr_ex[j] * 2, -1).flatten(1, 2)  # [1, 120, 64]
                    ex_coords_map = torch.einsum("bqc,bchw->bqhw", ex_feats, coords_feats)    # [1, 120, 400, 200]
                    ex_coords = self.gap_layer(ex_coords_map)             # [1, 120, 1, 1]
                    ex_ctr_coord[i][j] = ex_coords.reshape(batch_size, num_qry, self.max_pieces[j], self.num_degree[j] - 1, 2)   # [1, 20, 3, 1, 2]

        ret = {"outputs": {"obj_logits": dt_obj_logit, "ins_masks": dt_ins_masks, "ctr_im": im_ctr_coord,
                           "ctr_ex": ex_ctr_coord, "end_logits": dt_end_logit}}

        # pdb.set_trace()
        if self.semantic_heads is not None:
            num_decoders = len(inputs["bev_enc_features"])
            dt_sem_masks = [[[] for _ in range(self.num_classes)] for _ in range(num_decoders)]
            for i in range(num_decoders):
                x_sem = inputs["bev_enc_features"][i]      # [1, 512, 200, 100]
                # pdb.set_trace()
                for j in range(self.num_classes):
                    dt_sem_masks[i][j] = self.up_sample(self.semantic_heads[j](x_sem))
            ret["outputs"].update({"sem_masks": dt_sem_masks})

        ret["outputs"].update(self.bezier_curve_outputs(ret["outputs"]))       # add curve_points in "outputs"

        if 'lidar_depth' in inputs:
            ret['outputs'].update({"pred_depth": inputs['pred_depth'], "lidar_depth": inputs['lidar_depth']})
        return ret

    def up_sample(self, x, tgt_shape=None):
        tgt_shape = self.tgt_shape if tgt_shape is None else tgt_shape
        if tuple(x.shape[-2:]) == tuple(tgt_shape):
            return x
        return F.interpolate(x, size=tgt_shape, mode="bilinear", align_corners=True)

    def compute_locations(self, stride=1, device='cpu'):
        fh, fw = self.tgt_shape       # 400, 200
        shifts_x = torch.arange(0, fw * stride, step=stride, dtype=torch.float32, device=device)          # 200
        shifts_y = torch.arange(0, fh * stride, step=stride, dtype=torch.float32, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)    # [400, 200], [400, 200]
        # [[  0.,   0.,   0.,  ...,   0.,   0.,   0.],  [  1.,   1.,   1.,  ...,   1.,   1.,   1.],
        # [[  0.,   1.,   2.,  ..., 197., 198., 199.],  [  0.,   1.,   2.,  ..., 197., 198., 199.],
        shift_x = shift_x.reshape(-1)          # [80000]
        shift_y = shift_y.reshape(-1)          # [80000]
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2         # [80000, 2]

        locations = locations.unsqueeze(0).permute(0, 2, 1).contiguous().float().view(1, 2, fh, fw)   # [1, 2, 400, 200]
        # (Pdb) locations[0][0]
        # tensor([[  0.,   1.,   2.,  ..., 197., 198., 199.],
        #         [  0.,   1.,   2.,  ..., 197., 198., 199.],
        #         [  0.,   1.,   2.,  ..., 197., 198., 199.],
        #         ...,
        #         [  0.,   1.,   2.,  ..., 197., 198., 199.],
        #         [  0.,   1.,   2.,  ..., 197., 198., 199.],
        #         [  0.,   1.,   2.,  ..., 197., 198., 199.]], device='cuda:0')
        # (Pdb) locations[0][1]
        # tensor([[  0.,   0.,   0.,  ...,   0.,   0.,   0.],
        #         [  1.,   1.,   1.,  ...,   1.,   1.,   1.],
        #         [  2.,   2.,   2.,  ...,   2.,   2.,   2.],
        #         ...,
        #         [397., 397., 397.,  ..., 397., 397., 397.],
        #         [398., 398., 398.,  ..., 398., 398., 398.],
        #         [399., 399., 399.,  ..., 399., 399., 399.]], device='cuda:0')
        locations[:, 0, :, :] /= fw
        locations[:, 1, :, :] /= fh

        return locations

    def bezier_curve_outputs(self, outputs):
        dt_ctr_im, dt_ctr_ex, dt_ends = outputs["ctr_im"], outputs["ctr_ex"], outputs["end_logits"]
        num_decoders, num_classes = len(dt_ends), len(dt_ends[0])                                  # 6, 3
        ctr_points = [[[] for _ in range(self.num_classes)] for _ in range(num_decoders)]
        curve_points = [[[] for _ in range(self.num_classes)] for _ in range(num_decoders)]
        for i in range(num_decoders):
            for j in range(num_classes):
                batch_size, num_queries = dt_ctr_im[i][j].shape[:2]

                im_coords = dt_ctr_im[i][j].sigmoid()                                                 # [1, 20, 4, 2]
                ex_offsets = dt_ctr_ex[i][j].sigmoid() - 0.5                                          # [1, 20, 3, 1, 2]
                im_center_coords = ((im_coords[:, :, :-1] + im_coords[:, :, 1:]) / 2).unsqueeze(-2)   # [1, 20, 3, 1, 2]
                # pdb.set_trace()
                ex_coords = torch.stack([im_center_coords[:, :, :, :, 0] + ex_offsets[:, :, :, :, 0],
                                         im_center_coords[:, :, :, :, 1] + ex_offsets[:, :, :, :, 1]], dim=-1)  # [1, 20, 3, 1, 2]
                im_coords = im_coords.unsqueeze(-2)             # [1, 20, 4, 1, 2]
                ctr_coords = torch.cat([im_coords[:, :, :-1], ex_coords], dim=-2).flatten(2, 3)  # [1, 20, 6, 2]
                ctr_coords = torch.cat([ctr_coords, im_coords[:, :, -1:, 0, :]], dim=-2)     # [1, 20, 7, 2]
                ctr_points[i][j] = ctr_coords.clone()

                end_inds = torch.max(torch.softmax(dt_ends[i][j].flatten(0, 1), dim=-1), dim=-1)[1]
                curve_pts = self.curve_recovery_with_bezier(ctr_coords.flatten(0, 1), end_inds, j)  # [20, 100, 2]
                curve_points[i][j] = curve_pts.reshape(batch_size, num_queries, *curve_pts.shape[-2:])     # [1, 20, 100, 2]
        curve_point = curve_points[-1][0][0]   # num_decoders, num_classes, bs
        # pdb.set_trace()
        return {"curve_points": curve_points, 'ctr_points': ctr_points}

    def curve_recovery_with_bezier(self, ctr_points, end_indices, cid):
        device = ctr_points.device
        curve_pts_ret = torch.zeros((0, self.curve_size, 2), dtype=torch.float, device=device)
        # pdb.set_trace()
        num_instances, num_pieces = ctr_points.shape[0], ctr_points.shape[1]
        pieces_ids = [[i+j for j in range(self.num_degree[cid]+1)] for i in range(0, num_pieces - 1, self.num_degree[cid])]
        # [[0, 1, 2], [2, 3, 4], [4, 5, 6]]
        pieces_ids = torch.tensor(pieces_ids).long().to(device)

        points_ids = torch.tensor(list(range(self.curve_size))).long().to(device)
        points_ids = (end_indices + 1).unsqueeze(1) * points_ids.unsqueeze(0)             # [20, 100]
        # ([[  0,   2,   4,   6,   8,  10,  12,  14,   ... 196, 198], [  0,   1,   2,   3, ...  98,  99], [ ...

        if num_instances > 0:
            ctr_points_flatten = ctr_points[:, pieces_ids, :].flatten(0, 1)               # [20, 3, 3, 2] -> [60, 3, 2]
            curve_pts = torch.matmul(self.bezier_coefficient[cid], ctr_points_flatten)    # [60, 100, 2]
            # pdb.set_trace()
            curve_pts = curve_pts.reshape(num_instances, pieces_ids.shape[0], *curve_pts.shape[-2:])   # [20, 3, 100, 2]
            curve_pts = curve_pts.flatten(1, 2)                                                               # [20, 300, 2]
            curve_pts_ret = torch.stack([curve_pts[i][points_ids[i]] for i in range(points_ids.shape[0])])    # [20, 100, 2]
        return curve_pts_ret

    def _get_bezier_coefficients(self):

        def bernstein_func(n, t, k):
            return (t ** k) * ((1 - t) ** (n - k)) * n_over_k(n, k)

        ts = np.linspace(0, 1, self.curve_size)
        bezier_coefficient_list = []
        for nn in self.num_degree:
            bezier_coefficient_list.append(np.array([[bernstein_func(nn, t, k) for k in range(nn + 1)] for t in ts]))
        return bezier_coefficient_list
