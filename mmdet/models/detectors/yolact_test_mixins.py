#
# @author:charlotte.Song
# @file: yolact_test_mixins.py
# @Date: 2019/9/20 12:02
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, merge_aug_proposals, multiclass_nms_emb)
""" add an mask pred test mixins function. """


class YolactMaskTestMixin(object):

    def simple_test_mask(self,
                         proto_pred,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         det_embeds,
                         cfg,
                         rescale=False):
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            inst_mask_results = [[] for _ in range(self.bbox_head.num_classes)]
        else:
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            _embeds = torch.cat(det_embeds)
            assert _bboxes.shape[0] == _embeds.shape[0]
            # a list of inst_masks.
            mask_preds = self.extract_proto(proto_pred, mask_rois, _embeds,
                                            stride=cfg.stride)
            inst_mask_results = self.get_inst_masks(mask_preds, _bboxes, det_labels,
                                                    cfg, ori_shape, scale_factor,
                                                    rescale)
        return inst_mask_results

