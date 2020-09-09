#
# @author:charlotte.Song
# @file: pano_fpn_v2.py
# @Date: 2019/10/11 20:08
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule
"""change from 4x to 1x, and add ASPP into model 
considering the Fast Panoptic model, and use two 
branch to produce semantic segmentation and instance segmentation.
"""
""" complete the test scripts this night. """


@NECKS.register_module
class PanoFPNV2(nn.Module):

    def __init__(self,
                 in_channels,
                 in_embeds,
                 out_channels,
                 stuff_num_classes,
                 num_outs,      # the initial convs.
                 num_proto_convs=3,     # just put it on here.
                 num_segm_convs=3,
                 upsample_ratio=2,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(PanoFPNV2, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.in_embeds = in_embeds
        self.out_channels = out_channels
        self.stuff_num_classes = stuff_num_classes
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.num_proto_convs = num_proto_convs
        self.num_segm_convs = num_segm_convs
        self.upsample_ratio = upsample_ratio
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        """use bn and relu for semantic segmentation and relu """

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level

        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        # for bbox prediction.
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # a U-Net module for seg feats and proto feats
        self.seg_lateral_convs = nn.ModuleList()
        self.seg_fuse_convs = nn.ModuleList()
        self.inst_lateral_convs = nn.ModuleList()
        self.inst_fuse_convs = nn.ModuleList()
        for i in range(self.num_outs):
            l_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=dict(type='BN'),
                inplace=False)
            self.seg_lateral_convs.append(l_conv)
            l_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=dict(type='BN'),
                inplace=False)
            self.inst_lateral_convs.append(l_conv)
        for i in range(self.num_outs - 1):
            fuse_conv = ConvModule(
                2 * out_channels,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=dict(type='BN'),
                inplace=False)
            self.seg_fuse_convs.append(fuse_conv)
            fuse_conv = ConvModule(
                2 * out_channels,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=dict(type='BN'),
                inplace=False)
            self.inst_fuse_convs.append(fuse_conv)

        # aspp for the 1/8 feature maps, num is defined in nums.
        # after lateral then ASPP.
        self.seg_aspp_convs = nn.ModuleList()
        self.seg_aspp_smooth_convs = nn.ModuleList()
        self.inst_aspp_convs = nn.ModuleList()
        self.inst_aspp_smooth_convs = nn.ModuleList()

        for i in range(self.num_proto_convs):
            self.inst_aspp_convs.append(
                ConvModule(  # add bn and relu.
                    out_channels,
                    out_channels,
                    3,
                    padding=6 * i,
                    dilation=6 * i,
                    conv_cfg=conv_cfg,
                    norm_cfg=dict(type='BN'),
                    inplace=False))
            self.inst_aspp_smooth_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=dict(type='BN'),
                    inplace=False))

        for i in range(self.num_segm_convs):
            # element-sum ASPP feature maps.
            self.seg_aspp_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=6 * i,
                    dilation=6 * i,
                    conv_cfg=conv_cfg,
                    norm_cfg=dict(type='BN'),
                    inplace=False))
            self.seg_aspp_smooth_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=dict(type='BN'),
                    inplace=False))

        # smooth aspp feature maps.
        # do not lateral again.
        self.seg_aspp_smooth = ConvModule(
            self.num_segm_convs * out_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=dict(type='BN'),
            inplace=False)
        self.inst_aspp_smooth = ConvModule(
            self.num_proto_convs * out_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=dict(type='BN'),
            inplace=False)

        self.seg_smooth = ConvModule(
            self.num_outs * out_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=dict(type='BN'),
            inplace=False)
        self.inst_smooth = ConvModule(
            self.num_outs * out_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=dict(type='BN'),
            inplace=False)

        # add extra conv layers ( e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        self.up = nn.Upsample(
            scale_factor=self.upsample_ratio, mode='bilinear')
        self.up_1x = nn.Upsample(
            scale_factor=4., mode='bilinear')
        self.proto_logits = nn.Conv2d(out_channels, self.in_embeds, 1)
        self.segm_logits = nn.Conv2d(out_channels, self.stuff_num_classes, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        for m in [self.proto_logits, self.segm_logits]:
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on tol of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))

                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        # first proto segmentation feats
        seg_laterals = [
            seg_lateral_conv(outs[i])
            for i, seg_lateral_conv in enumerate(self.seg_lateral_convs)
        ]
        for i in range(self.num_outs - 1, 0, -1):
            seg_laterals[i - 1] = torch.cat(
                (seg_laterals[i - 1], F.interpolate(seg_laterals[i],
                                                    size=(seg_laterals[i - 1].size()[2],
                                                          seg_laterals[i - 1].size()[3]))),
                dim=1)
            seg_laterals[i - 1] = self.seg_fuse_convs[i - 1](seg_laterals[i - 1])

        up_seg_outs = [
            F.interpolate(seg_laterals[i], size=(seg_laterals[0].size()[2],
                                                 seg_laterals[0].size()[3]),
                          mode='bilinear')
            for i in range(len(seg_laterals))]
        # doing ASPP on 1/8 scale feature maps
        seg_feats = torch.cat(up_seg_outs, dim=1)
        seg_feats = self.seg_smooth(seg_feats)
        seg_aspp_feats = [
            # dilation conv + 1x1 smooth
            self.seg_aspp_smooth_convs[i](self.seg_aspp_convs[i](seg_feats))
            for i in range(len(self.seg_aspp_smooth_convs))
        ]
        for i in range(1, len(seg_aspp_feats)):
            seg_aspp_feats[0] = seg_aspp_feats[0] + seg_aspp_feats[i]

        seg_feat = self.up(seg_aspp_feats[0])
        seg_feat = self.segm_logits(seg_feat)
        # upsample from 4x to 1x.
        segm_mask = self.up_1x(seg_feat)

        # proto segmentation feats
        proto_laterals = [
            proto_lateral_conv(outs[i])
            for i, proto_lateral_conv in enumerate(self.inst_lateral_convs)
        ]
        for i in range(self.num_outs - 1, 0, -1):
            proto_laterals[i - 1] = torch.cat(
                (proto_laterals[i - 1], F.interpolate(proto_laterals[i],
                                                      size=(proto_laterals[i - 1].size()[2],
                                                            proto_laterals[i - 2].size()[3]))),
                dim=1)
            proto_laterals[i - 1] = self.inst_fuse_convs[i - 1](proto_laterals[i - 1])

        up_proto_outs = [
            F.interpolate(proto_laterals[i], size=(proto_laterals[0].size()[2],
                                                   proto_laterals[0].size()[3]),
                          mode='bilinear')
        ]

        # doing ASPP on 1/8 scale feature maps
        proto_feats = torch.cat(up_proto_outs, dim=1)
        proto_feats = self.inst_smooth(proto_feats)
        proto_aspp_feats = [
            self.inst_aspp_smooth_convs[i](self.inst_aspp_convs[i](seg_feats))
            for i in range(len(self.inst_aspp_smooth_convs))
        ]
        for i in range(1, len(proto_aspp_feats)):
            proto_aspp_feats[0] = proto_aspp_feats[0] + proto_aspp_feats[i]

        proto_feat = self.up(proto_aspp_feats[0])
        proto_feat = self.proto_logits(proto_feat)
        proto_mask = self.up_1x(proto_feat)

        return segm_mask, proto_mask, tuple(outs)















