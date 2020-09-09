#
# @author:charlotte.Song
# @file: panoptic_detector.py
# @Date: 2019/9/19 16:28
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import mmcv

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmdet.core import force_fp32, auto_fp16
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
import torch.nn.functional as F
import pycocotools.mask as mask_util
import os.path as osp


@DETECTORS.register_module
class PanopticDetector(BaseDetector):
    """Add Panoptic Segmentation """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PanopticDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg  = test_cfg

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(PanopticDetector, self).init_weights(pretrained=pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        assert self.with_neck
        segm, proto, x = self.neck(x)
        return segm, proto, x

    # extract_proto
    def extract_proto(self,
                      proto_masks,
                      pos_rois,
                      pos_embeds,
                      stride=4.,
                      final_stride=1.):
        """
        :param proto_masks: proto_masks of 1/4 scales
        :param pos_rois:    the gt_rois sampled by sampler.
        :param pos_embeds:  the pos_embeds produced by bbox_head.
        :param cfg: configs, that tells the strides of rois
        :return: a list of proto_inst_rois
        """
        assert pos_rois.size(0) == pos_embeds.size(0)
        rois_img_inds = pos_rois[:, 0].detach().long()
        # [n, in_embeds, H, W]
        rois_proto_masks = proto_masks.index_select(dim=0, index=rois_img_inds)
        in_embeds, map_h, map_w = rois_proto_masks.shape[1:]
        rois_proto_masks = rois_proto_masks.permute(0, 2, 3, 1).contiguous().view(
            rois_proto_masks.shape[0], map_h * map_w, in_embeds)
        # [n, in_embeds, 1]
        pos_embeds = pos_embeds[..., None]
        # rois_binary_masks: [n, H * W, 1] -> [n, H, W]
        rois_binary_masks = torch.matmul(rois_proto_masks, pos_embeds).view(
            rois_proto_masks.shape[0], map_h, map_w)
        binary_rois = []  # each roi mask: [1, roi_h, roi_w]
        pos_proposals = (pos_rois[:, 1:].detach() / stride).long()
        pos_final_proposals = (pos_rois[:, 1:].detach() / final_stride).long()
        for i, bbox in enumerate(pos_proposals):
            b_mask = rois_binary_masks[i]
            x1, y1, x2, y2 = bbox
            final_bbox = pos_final_proposals[i, :]
            f_x1, f_y1, f_x2, f_y2 = final_bbox
            w = max(x2 - x1 + 1, 1)
            h = max(y2 - y1 + 1, 1)
            f_w = max(f_x2 - f_x1 + 1, 1)
            f_h = max(f_y2 - f_y1 + 1, 1)
            preds = b_mask[y1:y1 + h, x1:x1 + w].unsqueeze(0).unsqueeze(0)
            preds = F.interpolate(preds, size=(f_h, f_w), mode='bilinear')
            binary_rois.append(preds.squeeze(0))
        if not binary_rois:
            binary_rois = pos_rois.new_zeros((0, 0, 0))
        return binary_rois

    def get_target(self, sampling_results, gt_masks, stride=4):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results]
        mask_targets = self.mask_target(pos_proposals, pos_assigned_gt_inds, gt_masks,
                                        stride)
        return mask_targets

    def mask_target_single(self, pos_proposals, pos_assigned_gt_inds, gt_masks, stride):
        num_pos = pos_proposals.size(0)
        rois_gt_targets = []
        if num_pos > 0:
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
            pos_bboxes = (pos_proposals.detach() / stride).long()
            pos_bboxes = pos_bboxes.cpu().numpy()
            for i in range(num_pos):
                gt_mask = gt_masks[pos_assigned_gt_inds[i]]
                gt_mask = mmcv.imrescale(gt_mask, scale=1. / stride, interpolation='nearest')

                bbox = pos_bboxes[i]
                x1, y1, x2, y2 = bbox
                w = np.maximum(x2 - x1 + 1, 1)
                h = np.maximum(y2 - y1 + 1, 1)
                target = gt_mask[y1:y1 + h, x1:x1 + w][None, ...]
                target = torch.from_numpy(target).float().to(pos_proposals.device)
                rois_gt_targets.append(target)
        else:
            # rois_gt_targets.append(pos_proposals.new_zeros((0, 0, 0)))
            rois_gt_targets = pos_proposals.new_zeros((0, 0, 0))
        return rois_gt_targets

    def mask_target(self, pos_proposal_list, pos_assigned_gt_inds_list, gt_masks_list,
                    stride):
        stride_list = [stride for _ in range(len(pos_proposal_list))]
        mask_target_list = map(self.mask_target_single, pos_proposal_list,
                               pos_assigned_gt_inds_list, gt_masks_list, stride_list)
        mask_targets = []
        for l in mask_target_list:
            mask_targets.extend(l)
        return mask_targets

    # will be deal with during loading...
    def segm_target(self, segms_preds, gt_semantic_segms):
        """
        :param gt_semantic_segms: [n_classes + 1, H, W]
        :param stride:
        :return:
        """
        rescaled_semantic_segms = F.interpolate(
            gt_semantic_segms, size=(segms_preds.size(2), segms_preds.size(3)),
            mode='nearest')
        # list of [stuff_classes + 1, H, W]
        return rescaled_semantic_segms

    @force_fp32(apply_to=('mask_pred', 'segm_pred', ))
    def mask_loss(self, mask_pred, mask_targets, segm_pred, segm_targets):
        loss = dict()
        assert len(mask_pred) == len(mask_targets)
        losses_list = []
        for pred_i, target_i in zip(mask_pred, mask_targets):
            loss_i = F.binary_cross_entropy_with_logits(
                pred_i, target_i.float(), reduction='mean')[None]
            losses_list.append(loss_i)
        losses_list = torch.cat(losses_list)
        loss_mask = torch.mean(losses_list)

        # segm_pred: [N, classes, H, W], segm_targets: [N, classes, H, W]
        # segm_targets: from [N, 1, H, W] -> [N, H, W]
        segm_targets = segm_targets.long().squeeze(1)
        loss_segm = F.cross_entropy(segm_pred, segm_targets)
        loss['loss_mask'] = loss_mask
        loss['loss_segm'] = loss_segm
        return loss

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None):
        # change the neck
        segm_pred, proto, x = self.extract_feat(img)
        losses = dict()

        cls_scores, bbox_preds, bbox_embeds = self.bbox_head(x)
        outs = (cls_scores, bbox_preds)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas,
                              self.train_cfg.bbox_head)
        loss_bbox = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        losses.update(loss_bbox)

        # proposal
        proposal_cfg = self.train_cfg.get('proposals')
        proposal_inputs = outs + (bbox_embeds, img_metas, proposal_cfg)
        proposal_list = self.bbox_head.get_proposals(*proposal_inputs)
        proposal_bbox, proposal_labels, proposal_embeds = map(list, zip(*proposal_list))

        # assign gts and sample proposals
        assigner = build_assigner(self.train_cfg.mask_head.assigner)
        sampler = build_sampler(self.train_cfg.mask_head.sampler,
                                context=self)
        num_imgs = img.size(0)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = assigner.assign(
                proposal_bbox[i],
                gt_bboxes[i],
                gt_bboxes_ignore[i],
                gt_labels[i])
            sampling_result = sampler.sample(
                assign_result,
                proposal_bbox[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        # mask_rois
        pos_rois = bbox2roi(
            res.pos_bboxes for res in sampling_results)
        pos_gt_rois = bbox2roi(
            res.pos_gt_bboxes for res in sampling_results)
        pos_embed_list = [
            proposal_embeds[i][res.pos_inds] for i, res in enumerate(sampling_results)]
        pos_embeds = torch.cat(pos_embed_list)
        assert pos_embeds.size(0) == pos_gt_rois.size(0)

        # crop the rois from proto_masks, then calculate the loss
        # using pos_bboxes but not pos_gt_rois
        # crop the rois from proto_masks, then calculate the loss
        # using pos_bboxes but not pos_gt_rois
        mask_targets = self.get_target(sampling_results, gt_masks,
                                       stride=self.train_cfg.proto.final_stride)
        mask_preds = self.extract_proto(proto, pos_rois, pos_embeds,
                                        stride=self.train_cfg.proto.stride,
                                        final_stride=self.train_cfg.proto.final_stride)

        # for segm
        # segm_targets = self.segm_target(segm_pred, gt_semantic_seg)
        loss_mask = self.mask_loss(mask_preds, mask_targets,
                                   segm_pred, gt_semantic_seg)
        losses.update(loss_mask)
        return losses

    def get_inst_masks(self, inst_preds, det_bboxes, det_labels, proto_test_cfg,
                       ori_shape, scale_factor, rescale):
        inst_preds_np = []
        for inst_pred in inst_preds:
            if isinstance(inst_pred, torch.Tensor):
                inst_pred = inst_pred.sigmoid().cpu().numpy()
                inst_pred = inst_pred.astype(np.float32)
            assert isinstance(inst_pred, np.ndarray)
            inst_preds_np.append(inst_pred)

        cls_segms = [[] for _ in range(self.bbox_head.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        for i in range(bboxes.shape[0]):
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            mask_pred_ = inst_preds_np[i][0, :, :]
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            bbox_mask = mmcv.imresize(mask_pred_, (w, h))
            bbox_mask = (bbox_mask > proto_test_cfg.mask_thr_binary).astype(
                np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            rle = mask_util.encode(
            np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[label - 1].append(rle)

        return cls_segms

    def get_instmask_preds(self, proto, img_meta, det_bboxes, det_labels, det_embeds,
                           cfg, rescale):
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            mask_preds = []
        else:
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])

            assert _bboxes.shape[0] == det_embeds.shape[0]
            # mask_preds are
            mask_preds = self.extract_proto(proto, mask_rois, det_embeds,
                                            stride=cfg.stride,
                                            final_stride=cfg.final_stride)
        return mask_preds

    # produce instance segmentation masks
    def simple_test_instmask(self, mask_preds, img_meta, det_bboxes, det_labels,
                             det_embeds, cfg, rescale=False):
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            inst_mask_results = [[] for _ in range(self.bbox_head.num_classes)]
        else:
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            assert _bboxes.shape[0] == det_embeds.shape[0] == det_labels.shape[0]
            # return RLE
            inst_mask_results = self.get_inst_masks(mask_preds, _bboxes, det_labels, cfg,
                                                    ori_shape, scale_factor, rescale)
        return inst_mask_results

    def id2rgb(self, id_map):
        if isinstance(id_map, np.ndarray):
            # //256
            id_map_copy = id_map.copy()
            rgb_shape = tuple(list(id_map.shape) + [3])
            rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
            for i in range(3):
                rgb_map[..., i] = id_map_copy % 256
                id_map_copy //= 256
            return rgb_map
        color = []
        for _ in range(3):
            color.append(id_map % 256)
            id_map //= 256
        return color

    def rgb2id(self, color):
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
        return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

        # produce instance segmentation masks
    def simple_test_panomask(self, segm_pred, inst_preds, img_meta,
                                 det_bboxes, det_labels, det_embeds, cfg,
                                 rescale=False):
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        cat_info = img_meta[0]['cat_info']
        segments_info = []

        # for instance segmentation, rescale mask_preds to inst mask preds.
        # and then paste it by scores.
        inst_preds_np = []
        for inst_pred in inst_preds:
            if isinstance(inst_pred, torch.Tensor):
                inst_pred = inst_pred.sigmoid().cpu().numpy()
                inst_pred = inst_pred.astype(np.float32)
            assert isinstance(inst_pred, np.ndarray)
            inst_preds_np.append(inst_pred)

        bboxes = det_bboxes.cpu().numpy()[:, :4]
        inst_labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        im_mask = np.zeros((img_h, img_w), dtype=np.int32)
        for i in range(bboxes.shape[0]):
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            mask_pred_ = inst_preds_np[i][0, :, :]
            # from 4x upsample to 1x
            # insterpolate the 4x size to 1x scale
            bbox_mask = mmcv.imresize(mask_pred_, (w, h))
            # label each pixel with instance id.
            # i + 1 == instance id
            bbox_mask = (bbox_mask > cfg.mask_thr_binary).astype(np.uint8) * (i + 1)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask

        if isinstance(segm_pred, torch.Tensor):
            segm_pred = F.interpolate(
                segm_pred, size=(img_h, img_w), mode='nearest')
            segm_pred = F.softmax(segm_pred, 1).cpu().numpy()
        assert isinstance(segm_pred, np.ndarray)
        assert segm_pred.shape[2:] == im_mask.shape

        segm_label = np.argmax(segm_pred[:, 1:, :, :], axis=1) + 1
        segment_segm_label = segm_label + int(bboxes.shape[0])
        segment_segm_label = segment_segm_label.reshape((img_h, img_w))
        assert segment_segm_label.shape == im_mask.shape

        mask = im_mask == 0
        im_mask[mask] = segment_segm_label[mask]
        # calculate the mask id.
        labels, labels_cnt = np.unique(im_mask, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label <= bboxes.shape[0]:
                segments_info.append({
                    'id': int(label),
                    'category_id': int(cat_info['thinglabel2thingcat'][
                                        # label is the num_id of instance, then transfer to corresponding num classes.
                                        str(inst_labels[label - 1])])})
            else:
                cat_label = label - int(bboxes.shape[0])
                segments_info.append({
                    'id': int(label),
                    'category_id': int(
                        cat_info['stufflabel2stuffcat'][str(cat_label)])})

        im_mask = self.id2rgb(im_mask)
        filename = img_meta[0]['filename'].replace('.jpg', '.png')
        filename = osp.split(filename)[-1]
        ann = dict({
            'image_id': img_meta[0]['id'],
            'file_name': filename,
            'segments_info': segments_info})

        return im_mask, ann

    def simple_test(self, img, img_meta, rescale=False):
        """for single image testing. """
        segm_pred, proto, x = self.extract_feat(img)
        cls_scores, bbox_preds, bbox_embeds = self.bbox_head(x)
        outs = (cls_scores, bbox_preds)
        bbox_inputs = outs + (bbox_embeds, img_meta, self.test_cfg.bbox_head)
        bbox_list = self.bbox_head.get_proposals(*bbox_inputs, rescale=rescale)

        det_bboxes, det_labels, det_embeds = map(list, zip(*bbox_list))
        det_bboxes = torch.cat(det_bboxes)
        det_labels = torch.cat(det_labels)
        det_embeds = torch.cat(det_embeds)

        # use det_bboxes to produce instance masks
        bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        # instance results, give back in cls format.
        mask_preds = self.get_instmask_preds(proto, img_meta, det_bboxes, det_labels,
                                             det_embeds, self.test_cfg.proto, rescale=rescale)
        segm_results = self.simple_test_instmask(mask_preds, img_meta, det_bboxes, det_labels,
                                                 det_embeds, self.test_cfg.proto, rescale=rescale)
        # panoptic segmentation, give back in pano mask format
        # pano_results: annotations with the format same with the
        # pano_images: the pano images.
        # pano_results, pano_images = self.simple_test_panomask(segm_pred, proto, img_meta, det_bboxes, det_labels, det_embeds,
        #                                          self.test_cfg.proto, rescale=rescale)
        pano_images, pano_results = self.simple_test_panomask(segm_pred, mask_preds, img_meta, det_bboxes, det_labels,
                                                               det_embeds, self.test_cfg.proto, rescale=rescale)
        # return bbox_results, segm_results, pano_results, pano_images
        # return bbox_results, segm_results
        # for panoptic testing
        # pano_images: png; pano_results: ann.
        return pano_images, pano_results

    def aut_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError




        

