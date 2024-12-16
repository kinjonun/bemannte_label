import torch.nn as nn
import pdb
from bemapnet.models import backbone, bev_decoder, ins_decoder, output_head
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = "INFO"
# warnings.filterwarnings('ignore')
from projects.mmdet3d_plugin.hrmap.global_map import GlobalMap
import torch
import torch.nn.functional as F
from .decode_raster import DecodeRaster
import matplotlib.pyplot as plt
import numpy as np


class BeMapNet(nn.Module):
    def __init__(self, model_config, *args, **kwargs):
        super(BeMapNet, self).__init__()
        self.im_backbone = self.create_backbone(**model_config["im_backbone"])
        self.bev_decoder = self.create_bev_decoder(**model_config["bev_decoder"])
        if "map_decoder" in model_config:
            self.map_decoder = self.create_map_decoder(**model_config["map_decoder"])
        else:
            self.map_decoder = None
        self.ins_decoder = self.create_ins_decoder(**model_config["ins_decoder"])
        self.output_head = self.create_output_head(**model_config["output_head"])
        self.post_processor = self.create_post_processor(**model_config["post_processor"])
        self.use_depth_loss = model_config["use_depth_loss"]

        self.epoch = -1
        if "global_map_cfg" in model_config:
            self.global_map = GlobalMap(model_config["global_map_cfg"])
            self.update_map = model_config["global_map_cfg"]['update_map']  # True
            self.DecodeRaster = DecodeRaster(**model_config["raster_cfg"])
        else:
            self.global_map = None
            self.update_map = False

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, inputs, status):
        outputs = {}
        outputs.update({k: inputs[k] for k in ["images", "extra_infos"]})
        outputs.update({k: inputs[k].float()for k in ["extrinsic", "intrinsic", "ida_mats"]})
        # pdb.set_trace()
        if self.use_depth_loss:
            outputs.update({k: inputs[k].float() for k in ['lidar_depth', 'lidar2ego']})  # ['lidar_depth']: [1, 6, 384, 896]

        if self.global_map is not None:
            self.global_map.check_map(inputs['images'].device, self.epoch, status)
            local_map = self.obtain_global_map(inputs['extra_infos']['img_metas'], status)     # [20000, 1, 3]
        else:
            local_map = None
        # vis_local_map = local_map.view(100, 200, 1, 3).permute(2, 3, 0, 1).cpu().numpy()   # [1, 3, 100 ,200]
        # combined_mask = np.zeros((100, 200))
        # combined_mask[vis_local_map[0][0] > 0.1] = 1
        # combined_mask[vis_local_map[0][1] > 0.1] = 2
        # combined_mask[vis_local_map[0][2] > 0.1] = 3
        # plt.imshow(combined_mask, cmap='gray')
        # plt.savefig('local_map_mask.png', bbox_inches='tight', dpi=300)
        # pdb.set_trace()
        outputs.update({"local_map": local_map})

        # dtype = mlvl_feats[0].dtype
        # bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
        # bev_pos = self.positional_encoding(bev_mask).to(dtype)
        # local_map_input = local_map.permute(1, 0, 2).view(bs, self.bev_h, self.bev_w, -1).permute(0, 3, 1, 2)
        # local_map_input = F.interpolate(local_map_input, [self.bev_h // self.map_query_scale,
        #                                                   self.bev_w // self.map_query_scale]).permute(0, 2, 3,
        #                                                                                                1)  # local_map下采样
        # local_map_pos = F.interpolate(bev_pos, [self.bev_h // self.map_query_scale,
        #                                         self.bev_w // self.map_query_scale]).permute(0, 2, 3, 1)
        #
        # local_map_valid, _ = torch.max(local_map_input, dim=-1)

        # pdb.set_trace()
        outputs.update(self.im_backbone(outputs))
        outputs.update(self.bev_decoder(outputs))
        if self.map_decoder is not None:
            outputs.update(self.map_decoder(outputs))
        outputs.update(self.ins_decoder(outputs))
        outputs.update(self.output_head(outputs))

        if self.update_map:
            new_map = self.get_pred_mask(outputs['outputs'])
            self.update_global_map(inputs['extra_infos']['img_metas'], new_map, status)

        return outputs

    def get_pred_mask(self, preds_dicts):
        with torch.no_grad():
            raster_lists = self.DecodeRaster.decode_raster(preds_dicts)
            raster_tensor = torch.stack(raster_lists, dim=0)
            # bs, n, bev_h, bev_w = raster_tensor.shape
            raster_tensor = raster_tensor.permute(0, 2, 3, 1)
            return raster_tensor

    def update_global_map(self, img_metas, raster, status):
        bs = len(img_metas['map_location'])
        for i in range(bs):
            # metas = img_metas[i]
            city_name = img_metas['map_location'][i]
            trans = img_metas['lidar2global'][i]
            self.global_map.update_map(city_name, trans, raster[i], status)

    def obtain_global_map(self, img_metas, status):
        bs = len(img_metas['map_location'])
        bev_maps = []
        for i in range(bs):
            # metas = img_metas[i]
            city_name = img_metas['map_location'][i]
            trans = img_metas['lidar2global'][i]
            local_map = self.global_map.get_map(city_name, trans, status)
            bev_maps.append(local_map)
        bev_maps = torch.stack(bev_maps)
        bev_maps = bev_maps.permute(1, 0, 2)
        return bev_maps

    def return_map(self):
        if self.update_map:
            self.global_map.save_global_map()
        # return self.global_map.get_global_map()

    @staticmethod
    def create_backbone(arch_name, ret_layers, bkb_kwargs, fpn_kwargs, up_shape=None):
        __factory_dict__ = {
            "resnet": backbone.ResNetBackbone,
            "efficient_net": backbone.EfficientNetBackbone,
            "swin_transformer": backbone.SwinTRBackbone,
        }
        return __factory_dict__[arch_name](bkb_kwargs, fpn_kwargs, up_shape, ret_layers)

    @staticmethod
    def create_bev_decoder(arch_name, net_kwargs):
        __factory_dict__ = {
            "transformer": bev_decoder.TransformerBEVDecoder,
            "transformer_depth": bev_decoder.TransformerBEVDecoderDepth,
            "LSS_transform": bev_decoder.TransformerBEVDecoderLSS,
        }
        return __factory_dict__[arch_name](**net_kwargs)

    @staticmethod
    def create_map_decoder(arch_name, net_kwargs):
        __factory_dict__ = {
            "transformer": bev_decoder.TransformerBEVDecoder,
            "transformer_depth": bev_decoder.TransformerBEVDecoderDepth,
            "LSS_transform": bev_decoder.TransformerBEVDecoderLSS,
        }
        return __factory_dict__[arch_name](**net_kwargs)

    @staticmethod
    def create_ins_decoder(arch_name, net_kwargs):
        __factory_dict__ = {
            "mask2former": ins_decoder.Mask2formerINSDecoder,
        }

        return __factory_dict__[arch_name](**net_kwargs)

    @staticmethod
    def create_output_head(arch_name, net_kwargs):
        __factory_dict__ = {
            "bezier_output_head": output_head.PiecewiseBezierMapOutputHead,
        }
        return __factory_dict__[arch_name](**net_kwargs)

    @staticmethod
    def create_post_processor(arch_name, net_kwargs):
        __factory_dict__ = {
            "bezier_post_processor": output_head.PiecewiseBezierMapPostProcessor,
        }
        return __factory_dict__[arch_name](**net_kwargs)
