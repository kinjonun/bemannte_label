import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from bemapnet.models.ins_decoder.mask2former import MultiScaleMaskedTransformerDecoder
import time


class Mask2formerINSDecoder(nn.Module):
    def __init__(self, decoder_ids=(5, ), tgt_shape=None, use_local_map=False, **kwargs):
        super(Mask2formerINSDecoder, self).__init__()
        self.decoder_ids = tuple(decoder_ids)  # [0, 1, 2, 3, 4, 5]
        self.tgt_shape = tgt_shape
        self.bev_decoder = MultiScaleMaskedTransformerDecoder(**kwargs)
        self.use_local_map = use_local_map
        self.in_channels = kwargs.get('in_channels', 512)
        if self.use_local_map:
            self.map_feats_conv = nn.Sequential(
                nn.Conv2d(self.in_channels + 3, self.in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(True))
        self.epoch = -1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, inputs):
        assert "bev_enc_features" in inputs
        bev_enc_features = inputs["bev_enc_features"]

        if self.tgt_shape is not None:
            bev_enc_features = [self.up_sample(x) for x in inputs["bev_enc_features"]]   # [4, 1, 512, 200, 100]
        bs, dim, bev_w, bev_h = bev_enc_features[0].shape

        if 'lss_bev_enc_features' in inputs:
            lss_bev_enc_features = inputs["lss_bev_enc_features"][0]
            bev_enc_features[-1] += lss_bev_enc_features
        # pdb.set_trace()
        if inputs["local_map"] is not None:  # and self.epoch > 0:
            local_map_reshape = inputs["local_map"].permute(1, 0, 2).view(bs, bev_h, bev_w, -1).permute(0, 3, 2, 1).contiguous()

            bev_fuse = torch.cat((bev_enc_features[-1], local_map_reshape), dim=1)
            bev_embed = self.map_feats_conv(bev_fuse)
            # bev_embed = bev_embed.flatten(2).permute(0, 2, 1).contiguous()
        # pdb.set_trace()
        if inputs["local_map"] is not None:  # and self.epoch > 0:
            out = self.bev_decoder([bev_embed], bev_embed)
        else:
            out = self.bev_decoder(bev_enc_features[-1:], bev_enc_features[-1])

        return {"mask_features": [out["pred_masks"][1:][i] for i in self.decoder_ids],
                "obj_scores": [out["pred_logits"][1:][i] for i in self.decoder_ids],
                "decoder_outputs": [out["decoder_outputs"][1:][i] for i in self.decoder_ids],
                "bev_enc_features": bev_enc_features}

    def up_sample(self, x, tgt_shape=None):
        tgt_shape = self.tgt_shape if tgt_shape is None else tgt_shape
        if tuple(x.shape[-2:]) == tuple(tgt_shape):
            return x
        return F.interpolate(x, size=tgt_shape, mode="bilinear", align_corners=True)
