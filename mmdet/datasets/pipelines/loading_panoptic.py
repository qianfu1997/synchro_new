#
# @author:charlotte.Song
# @file: loading_panoptic.py
# @Date: 2019/9/26 15:02
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""rewrite the loading pipeline for panoptic annotations """
import os.path as osp
import warnings
import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from ..registry import PIPELINES


@PIPELINES.register_module
class LoadPanopticAnnotations(object):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=True,
                 with_seg=True,
                 poly2mask=True,
                 skip_img_with_anno=True):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.skip_img_with_anno = skip_img_with_anno

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes']
        if len(results['gt_bboxes']) == 0 and self.skip_img_with_anno:
            file_path = osp.join(results['img_prefix'],
                                 results['img_info']['filename'])
            warnings.warn(
                'Skip the image "{}" that has no valid gt bbox'.format(
                    file_path))
            return None
        results['gt_bboxes_ignore'] = ann_info.get('bboxes_ignore', None)
        results['bbox_fields'].extend(['gt_bboxes', 'gt_bboxes_ignore'])
        return results

    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['labels']
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _load_masks(self, results):
        """load instance masks"""
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        cat_info = results['cat_info']
        seg = mmcv.imread(
            osp.join(results['seg_prefix'], results['ann_info']['seg_map']),
            flag='unchanged').squeeze()
        gt_semantic_seg = np.zeros(seg.shape, dtype=np.int32)
        for cat_id in cat_info['cat_stuff_ids']:
            new_label = cat_info['stuffcat2stufflabel'][str(cat_id)] if int(cat_id) != 183 else 0
            mask = seg == int(cat_id)
            gt_semantic_seg[mask] = int(new_label)
        assert np.max(gt_semantic_seg) < 54
        results['gt_semantic_seg'] = gt_semantic_seg
        return results

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_label={}, with_mask={},'
                     ' with_seg={})').format(self.with_bbox, self.with_label,
                                             self.with_mask, self.with_seg)
        return repr_str






