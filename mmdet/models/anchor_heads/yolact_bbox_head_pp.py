#
# @author:charlotte.Song
# @file: yolact_bbox_head_pp.py
# @Date: 2019/9/23 15:05
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""a version that share weights between cls and cls_embeds"""
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import (auto_fp16, force_fp32, delta2bbox)
from ..builder import build_loss
from mmdet.ops.nms import nms_wrapper
from ..registry import HEADS
from ..utils import ConvModule, bias_init_with_prob
from .anchor_head import AnchorHead


@HEADS.register_module
class YolactBboxHeadPP(AnchorHead):

    def __init__(self,
                 num_classes,
                 inst_embeds,
                 in_channels,
                 num_cls_convs=4,
                 num_reg_convs=4,
                 num_emb_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        self.inst_embeds = inst_embeds
        self.num_cls_convs = num_cls_convs
        self.num_reg_convs = num_reg_convs
        self.num_emb_convs = num_emb_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(YolactBboxHeadPP, self).__init__(
            num_classes=num_classes, in_channels=in_channels,
            **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inpalce=True)
        self.tanh = nn.Tanh()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.num_cls_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(ConvModule(
                chn,
                self.feat_channels,
                3,
                stride=1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg))

        for i in range(self.num_reg_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.reg_convs.append(ConvModule(
                chn,
                self.feat_channels,
                3,
                stride=1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg))

        self.yolact_cls = nn.Conv2d(
            self.feat_channels, self.num_anchors * self.cls_out_channels, 3, padding=1)

        self.yolact_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

        self.yolact_emb = nn.Conv2d(
            self.feat_channels, self.num_anchors * self.num_classes * self.cls_out_channels)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.emb_convs:
            normal_init(m.conv, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.yolact_cls, std=0.01, bias=bias_cls)
        normal_init(self.yolact_reg, std=0.01)
        normal_init(self.yolact_emb, std=0.01)

    def forward_single(self, x):
        cls_emb_feat = x
        reg_feat = x

        for cls_conv in self.cls_convs:
            cls_emb_feat = cls_conv(cls_emb_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        cls_score = self.yolact_cls(cls_emb_feat)
        bbox_pred = self.yolact_reg(reg_feat)
        bbox_embs = self.yolact_emb(cls_emb_feat)
        bbox_embs = self.tanh(bbox_embs)

        return cls_score, bbox_pred, bbox_embs

    def multiclass_nms_emb(self,
                           multi_bboxes,
                           multi_scores,
                           multi_embeds,
                           score_thr,
                           nms_cfg,
                           max_num=-1,
                           score_factors=None):
        """NMS for multi_class bboxes, select the cls_embeds for bboxes """
        num_classes = multi_scores.shape[1]
        bboxes, labels, embeds = [], [], []
        nms_cfg_ = nms_cfg.copy()
        nms_type = nms_cfg_.pop('type', 'nms')
        nms_op = getattr(nms_wrapper, nms_type)
        for i in range(1, num_classes):
            cls_inds = multi_scores[:, i] > score_thr
            if not cls_inds.any():
                continue
            if multi_bboxes.shape[1] == 4:
                _bboxes = multi_bboxes[cls_inds, :]
            else:
                _bboxes = multi_bboxes[cls_inds, i * 4: (i + 1) * 4]
            _scores = multi_scores[cls_inds, i]
            # select corresponding embeds
            # only keep the embeds corresponding to predicted class.
            _embeds = multi_embeds[cls_inds, i * self.inst_embeds: (i + 1) * self.inst_embeds]
            if score_factors is not None:
                _scores *= score_factors[cls_inds]
            cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
            cls_dets, cls_det_inds = nms_op(cls_dets, **nms_cfg_)
            cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                               i - 1,
                                               dtype=torch.long)
            cls_embeds = _embeds[cls_det_inds, :]
            bboxes.append(cls_dets)
            labels.append(cls_labels)
            embeds.append(cls_embeds)

        if bboxes:
            bboxes = torch.cat(bboxes)
            labels = torch.cat(labels)
            embeds = torch.cat(embeds)
            if bboxes.shape[0] > max_num:
                _, inds = bboxes[:, -1].sort(descending=True)
                inds = inds[:max_num]
                bboxes = bboxes[inds]
                labels = labels[inds]
                embeds = embeds[inds]
        else:
            bboxes = multi_bboxes.new_zeros((0, 5))
            labels = multi_bboxes.new_zeros((0,), dtype=torch.long)
            # only keep one group embeds.
            embeds = multi_embeds.new_zeros((0, self.inst_embeds))
        return bboxes, labels, embeds


    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          bbox_embeds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_embeds = []
        # no need to filter out any embeds.
        for cls_score, bbox_pred, bbox_embed, anchors in zip(cls_scores, bbox_preds,
                                                            bbox_embeds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)

            bbox_pred = bbox_pred.permute(1, 2, 0).contiguous().reshape(-1, 4)
            bbox_embed = bbox_embed.permute(1, 2, 0).contiguous().reshape(-1, self.num_classes * self.inst_embeds)
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(cfg.nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds, :]
                bbox_embed = bbox_embed[topk_inds, :]

            bboxes = delta2bbox(anchors, bbox_pred, self.target_means,
                                self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_embeds.append(bbox_embed)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            # do not use scale_factor during training.
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_embeds = torch.cat(mlvl_embeds)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels, det_embeds = self.multiclass_nms_emb(
            mlvl_bboxes, mlvl_scores, mlvl_embeds,
            cfg.score_thr, cfg.nms, cfg.max_per_img)
        return det_bboxes, det_labels, det_embeds

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'bbox_embeds'))
    def get_proposals(self, cls_scores, bbox_preds, bbox_embeds, img_metas, cfg,
                      rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(bbox_embeds)
        num_levels = len(cls_scores)

        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:],
                                                   self.anchor_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_scores_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_preds_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_embeds_list = [
                bbox_embeds[i][img_id] for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_scores_list, bbox_preds_list, bbox_embeds_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor, cfg, rescale=rescale)
            result_list.append(proposals)
        return result_list
